# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config

from src.utils import *
from src.models.risk_models import *

class OffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        self._reset_log()
        if self._cfgs.risk_cfgs.use_risk:
            self.risk_size = self._cfgs.risk_cfgs.quantile_num
            self._init_risk_model()

            if self._cfgs.risk_cfgs.fine_tune_risk:
                self._init_risk_update()
                self.f_costs = None

    def _init_risk_model(self) -> None:
        """Initialize the risk model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """

        risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                    "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

        self.risk_model = BayesRiskEst(obs_size=self._env.observation_space.shape[0], batch_norm=True, out_size=self.risk_size)
        if os.path.exists(self._cfgs.risk_cfgs.risk_model_path):
            self.risk_model.load_state_dict(torch.load(self._cfgs.risk_cfgs.risk_model_path, map_location=self._device))

        self.risk_model.to(self._device)
        self.risk_model.eval()


    def _init_risk_update(self) -> None:

        self.opt_risk = torch.optim.Adam(self.risk_model.parameters(), lr=self._cfgs.risk_cfgs.risk_lr, eps=1e-10)

        self.risk_rb = ReplayBuffer()

        if self._cfgs.risk_cfgs.risk_type == "quantile":
            weight_tensor = torch.Tensor([1]*self._cfgs.risk_cfgs.quantile_num).to(self._device)
            weight_tensor[0] = self._cfgs.risk_cfgs.risk_weight
        elif self._cfgs.risk_cfgs.risk_type == "binary":
            weight_tensor = torch.Tensor([1., self._cfgs.risk_cfgs.risk_weight]).to(self._device)
        self.risk_criterion = nn.NLLLoss(weight=weight_tensor)

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                risk = self.risk_model(obs) if self._cfgs.risk_cfgs.use_risk else None
                print(obs)
                act = agent.step(obs, risk,  deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """

        risk_bins = np.array([i*self._cfgs.risk_cfgs.quantile_size for i in range(self._cfgs.risk_cfgs.quantile_num+1)])

        for step in range(rollout_step):
            with torch.no_grad():
                risk = None if not self._cfgs.risk_cfgs.use_risk else self.risk_model(self._current_obs)

            if use_rand_action:
                act = torch.as_tensor(self._env.sample_action(), dtype=torch.float32).to(
                    self._device,
                )
            else:
                act = agent.step(self._current_obs, risk, deterministic=False)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            if self._cfgs.risk_cfgs.use_risk and self._cfgs.risk_cfgs.fine_tune_risk:
                self.f_costs = cost.unsqueeze(0) if self.f_costs is None else torch.cat([self.f_costs, cost.unsqueeze(0)], axis=0)


            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=real_next_obs,
            )

            if "final_observation" in info:
                if self._cfgs.risk_cfgs.use_risk and self._cfgs.risk_cfgs.fine_tune_risk:
                    f_risks = torch.empty_like(self.f_costs)
                    for i in range(self._cfgs.train_cfgs.vector_env_nums):
                        f_risks[:, i] = compute_fear(self.f_costs[:, i])

                    # f_risks = f_risks.view(-1, 1)
                    
                    f_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, np.expand_dims(f_risks.cpu().numpy(), 1)))
                    buffer.store_risk(f_risks_quant)

                self.f_costs = None

            ## Risk model update 

            if self._cfgs.risk_cfgs.use_risk and self._cfgs.risk_cfgs.fine_tune_risk:
                if buffer.size > self._cfgs.risk_cfgs.risk_batch_size and step % self._cfgs.risk_cfgs.risk_update_period == 0:
                    data = buffer.sample_batch(self._cfgs.risk_cfgs.risk_batch_size)
                    pred = self.risk_model(data["next_obs"])
                    risk_loss = self.risk_criterion(pred, torch.argmax(data["risk"].squeeze(), axis=1))
                    self.opt_risk.zero_grad()
                    risk_loss.backward()
                    self.opt_risk.step()

                    logger.store(
                        {
                           'Risk/risk_loss': risk_loss.item(),
                        },
                    )



            self._current_obs = next_obs

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
