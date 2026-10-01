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

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.critic.successor_representation_critic import (
    SuccessorRepresentationLinearReadout,
    TDRidgeSuccessorTrunk,
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
        if self._use_sr:
            # Replaces both critics with read-outs of one shared trunk. Built after super(),
            # which has already made the actor and a plain reward critic -- the latter is
            # discarded here, as in MICE.
            self._build_sr_critics(obs_space, act_space, model_cfgs)
            return
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

    def _build_sr_critics(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
    ) -> None:
        """Build the ``td_ridge`` successor-representation trunk and its two read-outs."""
        sr_cfgs = model_cfgs.sr_cfgs
        sr_mode = sr_cfgs.get('sr_mode', 'td_ridge')
        if sr_mode != 'td_ridge':
            raise NotImplementedError(
                f'Unknown sr_cfgs.sr_mode "{sr_mode}"; this port supports "td_ridge" only.',
            )
        obs_dim = obs_space.shape[0]  # type: ignore[union-attr]
        self._sr_phi_source: str = sr_cfgs.get('phi_source', 'random')
        self._sr_phi_trained: bool = self._sr_phi_source == 'contrastive'
        trunk = TDRidgeSuccessorTrunk(
            obs_dim=obs_dim,
            hidden_sizes=list(sr_cfgs.hidden_sizes),
            sr_dim=sr_cfgs.sr_dim,
            activation=sr_cfgs.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            phi_source=self._sr_phi_source,
            phi_hidden_sizes=list(sr_cfgs.get('phi_hidden_sizes', []) or []),
        )
        self.reward_critic = SuccessorRepresentationLinearReadout(
            obs_space, act_space, trunk, 'w_r', model_cfgs.weight_initialization_mode,
        )
        self.cost_critic: Critic = SuccessorRepresentationLinearReadout(
            obs_space, act_space, trunk, 'w_c', model_cfgs.weight_initialization_mode,
        )
        self.sr_trunk = trunk
        # phi has its own loss and its own optimizer, so it is kept out of both the shared SR
        # optimizer and the critic-norm penalty that optimizer's loss carries.
        phi_param_ids = (
            {id(p) for p in trunk.phi_net.parameters()} if self._sr_phi_trained else set()
        )
        self._sr_critic_norm_excluded_ids: set[int] = phi_param_ids
        trainable_params = [
            param
            for param in trunk.parameters()
            if param.requires_grad and id(param) not in phi_param_ids
        ]
        self.add_module('reward_critic', self.reward_critic)
        self.add_module('cost_critic', self.cost_critic)
        self.add_module('sr_trunk', self.sr_trunk)
        self.reward_critic.eval()
        self.cost_critic.eval()

        if sr_cfgs.lr is not None:
            sr_optimizer = optim.AdamW(trainable_params, lr=sr_cfgs.lr, weight_decay=0.0)
            self.cost_critic_optimizer: optim.Optimizer = sr_optimizer
            self.sr_optimizer: optim.Optimizer = sr_optimizer
            self.reward_critic_optimizer: optim.Optimizer = sr_optimizer
        if self._sr_phi_trained:
            # Its own Adam state: the InfoNCE loss runs on a very different cadence from the
            # per-minibatch psi TD loss, so sharing momentum between them helps neither.
            self.sr_phi_optimizer: optim.Optimizer = optim.Adam(
                trunk.phi_net.parameters(),
                lr=sr_cfgs.get('phi_lr', None) or sr_cfgs.lr,
            )

    def sr_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phi, psi)`` for ``obs``, as cached by the rollout."""
        with torch.no_grad():
            return self.sr_trunk.phi(obs), self.sr_trunk.psi(obs)

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
