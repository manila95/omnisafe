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
"""Implementation of ConstraintActorCritic."""

from __future__ import annotations

import itertools

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.critic.successor_representation_critic import (
    SuccessorRepresentationLinearReadout,
    SuccessorRepresentationReadout,
    SuccessorRepresentationTrunk,
    TDRidgeSuccessorRepresentationTrunk,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+
    | Cost V Critic   | Input is observation. Output is cost value.   |
    +-----------------+-----------------------------------------------+

    .. note::
        When ``model_cfgs.use_successor_representation`` is ``True``, ``reward_critic`` and
        ``cost_critic`` are both read-out heads over a single shared successor-representation
        value function (see ``model_cfgs.sr_cfgs.sr_mode``) instead of two independent
        networks. Every other consumer of ``reward_critic`` / ``cost_critic`` (rollout, GAE,
        the critic training loop) is unaffected, since both modes still expose the standard
        ``Critic`` interface (``forward(obs) -> [value]``).

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        cost_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self._use_sr: bool = bool(model_cfgs.get('use_successor_representation', False))
        self._sr_mode: str | None = model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') if self._use_sr else None

        if not self._use_sr:
            self.cost_critic: Critic = CriticBuilder(
                obs_space=obs_space,
                act_space=act_space,
                hidden_sizes=model_cfgs.critic.hidden_sizes,
                activation=model_cfgs.critic.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                num_critics=1,
                use_obs_encoder=False,
            ).build_critic('v')
            self.add_module('cost_critic', self.cost_critic)

            if model_cfgs.critic.lr is not None:
                self.cost_critic_optimizer: optim.Optimizer
                self.cost_critic_optimizer = optim.Adam(
                    self.cost_critic.parameters(),
                    lr=model_cfgs.critic.lr,
                )
            return

        self._build_successor_representation_critics(obs_space, act_space, model_cfgs)

    def _build_successor_representation_critics(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
    ) -> None:
        """Replace the plain reward/cost critics with a shared successor-representation critic.

        Discards the plain ``reward_critic`` built by :class:`ActorCritic` in favor of a
        successor-representation-based one; ``reward_critic_optimizer`` and
        ``cost_critic_optimizer`` are set to the *same* optimizer instance so that the shared
        trunk's Adam momentum state is not split across two independent optimizers (each of
        :meth:`PolicyGradient._update_reward_critic` / :meth:`_update_cost_critic` still does
        its own ``zero_grad`` / ``backward`` / ``step`` call, so the shared trunk still only
        ever moves along one loss's gradient at a time).

        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            model_cfgs (ModelConfig): The model configurations.
        """
        sr_cfgs = model_cfgs.sr_cfgs
        obs_dim = obs_space.shape[0]

        if self._sr_mode == 'shared_trunk':
            trunk = SuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                hidden_sizes=list(sr_cfgs.hidden_sizes),
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
            )
            self.reward_critic = SuccessorRepresentationReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                model_cfgs.weight_initialization_mode,
            )
            self.cost_critic: Critic = SuccessorRepresentationReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                model_cfgs.weight_initialization_mode,
            )
            self.sr_trunk: SuccessorRepresentationTrunk | TDRidgeSuccessorRepresentationTrunk = trunk
            trainable_params = itertools.chain(
                trunk.parameters(),
                self.reward_critic.head.parameters(),
                self.cost_critic.head.parameters(),
            )
        elif self._sr_mode == 'td_ridge':
            trunk = TDRidgeSuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                hidden_sizes=list(sr_cfgs.hidden_sizes),
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
            )
            self.reward_critic = SuccessorRepresentationLinearReadout(
                obs_space,
                act_space,
                trunk,
                'w_r',
                model_cfgs.weight_initialization_mode,
            )
            self.cost_critic = SuccessorRepresentationLinearReadout(
                obs_space,
                act_space,
                trunk,
                'w_c',
                model_cfgs.weight_initialization_mode,
            )
            self.sr_trunk = trunk
            # w_r / w_c are buffers (ridge-solved), so the trunk's own parameters (trunk +
            # phi_head + psi_head) are the complete set of SGD-trainable SR parameters.
            trainable_params = trunk.parameters()
        else:
            raise NotImplementedError(
                f'Unknown sr_cfgs.sr_mode "{self._sr_mode}". '
                'Available successor-representation modes are: "shared_trunk", "td_ridge".',
            )

        self.add_module('reward_critic', self.reward_critic)
        self.add_module('cost_critic', self.cost_critic)
        self.add_module('sr_trunk', self.sr_trunk)

        if sr_cfgs.lr is not None:
            sr_optimizer = optim.Adam(list(trainable_params), lr=sr_cfgs.lr)
            self.reward_critic_optimizer: optim.Optimizer = sr_optimizer
            self.cost_critic_optimizer: optim.Optimizer = sr_optimizer
            self.sr_optimizer: optim.Optimizer = sr_optimizer

    def sr_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phi, psi)`` successor-representation features (``td_ridge`` mode only).

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            phi: The one-step feature of the observation.
            psi: The successor feature of the observation.
        """
        assert self._sr_mode == 'td_ridge', (
            'sr_features() is only available when model_cfgs.sr_cfgs.sr_mode == "td_ridge".'
        )
        with torch.no_grad():
            z = self.sr_trunk.features(obs)
            phi = self.sr_trunk.phi(obs, z=z)
            psi = self.sr_trunk.psi(obs, z=z)
        return phi, psi

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs)

            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_c[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)
