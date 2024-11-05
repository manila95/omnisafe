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
"""Online Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any
import os
import torch

from omnisafe.envs.core import CMDP, make, support_envs
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.common.logger import Logger

from omnisafe.utils.tools import get_device
from src.utils import *
from src.models.risk_models import *

class OnlineAdapter:
    """Online Adapter for OmniSafe.

    OmniSafe is a framework for safe reinforcement learning. It is designed to be compatible with
    any existing RL algorithms. The online adapter is used to adapt the environment to the
    framework.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnlineAdapter`."""
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._cfgs: Config = cfgs
        self._device: torch.device = get_device(cfgs.train_cfgs.device)
        self._env_id: str = env_id

        env_cfgs = {}

        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()

        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device, **env_cfgs)
        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._eval_env: CMDP | None = None
        if self._env.need_evaluation:
            self._eval_env = make(env_id, num_envs=1, device=self._device, **env_cfgs)
            self._wrapper_eval(obs_normalize=cfgs.algo_cfgs.obs_normalize)

        self._env.set_seed(seed)

        if self._cfgs.risk_cfgs.use_risk:
            # Creating variables for storing data for the risk model (episodic)
            self.f_next_obs, self.f_costs = None, None 

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


    def _store_risk_data(self, next_obs, costs):
        self.f_next_obs = next_obs.unsqueeze(0) if self.f_next_obs is None else torch.concat([self.f_next_obs, next_obs.unsqueeze(0)], axis=0)
        self.f_costs = costs.unsqueeze(0) if self.f_costs is None else torch.concat([self.f_costs, costs.unsqueeze(0)], axis=0)


    def _update_risk(self, step, logger: Logger):
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


    def _populate_risk_rb(self):
        f_risks = torch.empty_like(self.f_costs)
        for i in range(self._num_envs):
            f_risks[:, i] = compute_fear(self.f_costs[:, i])
        
        e_risks = f_risks.view(-1, 1).cpu().numpy()
        e_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=self.risk_bins)[0], 1, np.expand_dims(e_risks, 1))).to(self._device)
        self.risk_rb.add(None, self.f_next_obs.view(-1, self.obs_size), None, None, None, None, e_risks_quant, f_risks.view(-1, 1))

        self.f_next_obs, self.f_costs = None, None


    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. hint::
            OmniSafe supports the following wrappers:

        +-----------------+--------------------------------------------------------+
        | Wrapper         | Description                                            |
        +=================+========================================================+
        | TimeLimit       | Limit the time steps of the environment.               |
        +-----------------+--------------------------------------------------------+
        | AutoReset       | Reset the environment when the episode is done.        |
        +-----------------+--------------------------------------------------------+
        | ObsNormalize    | Normalize the observation.                             |
        +-----------------+--------------------------------------------------------+
        | RewardNormalize | Normalize the reward.                                  |
        +-----------------+--------------------------------------------------------+
        | CostNormalize   | Normalize the cost.                                    |
        +-----------------+--------------------------------------------------------+
        | ActionScale     | Scale the action.                                      |
        +-----------------+--------------------------------------------------------+
        | Unsqueeze       | Unsqueeze the step result for single environment case. |
        +-----------------+--------------------------------------------------------+


        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        if self._env.need_time_limit_wrapper:
            assert (
                self._env.max_episode_steps
            ), 'You must define max_episode_steps as an integer\
                \nor cancel the use of the time_limit wrapper.'
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def _wrapper_eval(
        self,
        obs_normalize: bool = True,
    ) -> None:
        """Wrapper the environment for evaluation.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        assert self._eval_env, 'Your environment for evaluation does not exist!'
        if self._env.need_time_limit_wrapper:
            assert (
                self._eval_env.max_episode_steps
            ), 'You must define max_episode_steps as an\
                \ninteger or cancel the use of the time_limit wrapper.'
            self._eval_env = TimeLimit(
                self._eval_env,
                time_limit=self._eval_env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if obs_normalize:
            self._eval_env = ObsNormalize(self._eval_env, device=self._device)
        self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment."""
        return self._env.observation_space

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        return self._env.step(action)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        return self._env.reset(seed=seed, options=options)

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be an empty dict. common wrappers are
            ``obs_normalize``, ``reward_normalize``, and ``cost_normalize``.

        Returns:
            The saved components of environment, e.g., ``obs_normalizer``.
        """
        return self._env.save()

    def close(self) -> None:
        """Close the environment after training."""
        self._env.close()

    @property
    def env_spec_keys(self) -> list[str]:
        """Return the environment specification log."""
        if hasattr(self._env, 'env_spec_log'):
            return list(self._env.env_spec_log.keys())
        return []
