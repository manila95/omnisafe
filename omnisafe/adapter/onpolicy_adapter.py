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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config

from src.models.risk_models import * 
from src.utils import * 

class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

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
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._reset_log()

        self._num_envs = num_envs
    
        if self._cfgs.risk_cfgs.use_risk:
            self.obs_size = self._env.observation_space.shape[0]
            self.risk_model = BayesRiskEst(self.obs_size, out_size=self._cfgs.risk_cfgs.quantile_num)

            if os.path.exists(self._cfgs.risk_cfgs.risk_model_path):
                self.risk_model.load_state_dict(torch.load(self._cfgs.risk_cfgs.risk_model_path))
            
            if self._cfgs.risk_cfgs.fine_tune_risk:
                self.risk_rb = ReplayBuffer()
                self.risk_optim = torch.optim.Adam(self.risk_model.parameters(), lr=self._cfgs.risk_cfgs.risk_lr)
                self.risk_bins =  np.array([i*self._cfgs.risk_cfgs.quantile_size for i in range(self._cfgs.risk_cfgs.quantile_num+1)])
                self.risk_criterion = torch.nn.NLLLoss()
            self.risk_model.eval()


    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()

        obs, _ = self.reset()
        f_next_obs, f_costs = None, None
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            
            with torch.no_grad():
                risk = None if not self._cfgs.risk_cfgs.use_risk else self.risk_model(obs)
            act, value_r, value_c, logp = agent.step(obs, risk)

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            if self._cfgs.risk_cfgs.use_risk and self._cfgs.risk_cfgs.fine_tune_risk:
                f_next_obs = next_obs.unsqueeze(0) if f_next_obs is None else torch.concat([f_next_obs, next_obs.unsqueeze(0)], axis=0)
                f_costs = cost.unsqueeze(0) if f_costs is None else torch.concat([f_costs, cost.unsqueeze(0)], axis=0)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            ## Updating the risk model 
            if self._cfgs.risk_cfgs.use_risk and self._cfgs.risk_cfgs.fine_tune_risk:
                self.risk_model.train()
                if len(self.risk_rb) > self._cfgs.risk_cfgs.risk_batch_size and step % self._cfgs.risk_cfgs.risk_update_freq == 0:
                    risk_data = self.risk_rb.sample(self._cfgs.risk_cfgs.risk_batch_size)
                    pred = self.risk_model(risk_data["next_obs"].to(self._device))
                    risk_loss = self.risk_criterion(pred, torch.argmax(risk_data["risks"].squeeze(), axis=1).to(self._device))
                    self.risk_optim.zero_grad()
                    risk_loss.backward()
                    self.risk_optim.step()
                    logger.store({'Loss/risk_loss': risk_loss})
                self.risk_model.eval()
                


            if "final_observation" in info:
                if self._cfgs.risk_cfgs.use_risk and self._cfgs.risk_cfgs.fine_tune_risk:
                    f_risks = torch.empty_like(f_costs)
                    for i in range(self._num_envs):
                        f_risks[:, i] = compute_fear(f_costs[:, i])
                    
                    e_risks = f_risks.view(-1, 1).cpu().numpy()
                    e_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=self.risk_bins)[0], 1, np.expand_dims(e_risks, 1))).to(self._device)
                    self.risk_rb.add(None, f_next_obs.view(-1, self.obs_size), None, None, None, None, e_risks_quant, f_risks.view(-1, 1))

                    f_next_obs, f_costs = None, None
                with torch.no_grad():
                    final_risk = self.risk_model(info["final_observation"]) if self._cfgs.risk_cfgs.use_risk  else None

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            with torch.no_grad():
                risk = self.risk_model(obs) if self._cfgs.risk_cfgs.use_risk else None

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx], risk[idx] if risk is not None else None)
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx], final_risk[idx] if final_risk is not None else None,
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)

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
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
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
