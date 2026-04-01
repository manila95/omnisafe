# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Quantile Critic for TQC (Truncated Quantile Critics) used in COX-Q."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class QuantileCritic(Critic):
    """N independent MLP critics each outputting M quantile atoms.

    Output shape: ``[batch, num_critics, num_quantiles]``.

    Args:
        obs_space: Observation space.
        act_space: Action space.
        hidden_sizes: Hidden layer sizes.
        activation: Activation function.
        weight_initialization_mode: Weight init scheme.
        num_critics: Number of independent critic heads N.
        num_quantiles: Number of quantile atoms M per critic.
        use_obs_encoder: Unused, kept for API compatibility.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 5,
        num_quantiles: int = 25,
        use_obs_encoder: bool = False,
    ) -> None:
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder,
        )
        self.num_quantiles: int = num_quantiles
        self.net_lst: list[nn.Sequential] = []
        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim + self._act_dim, *hidden_sizes, num_quantiles],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            critic = nn.Sequential(net)
            self.net_lst.append(critic)
            self.add_module(f'critic_{idx}', critic)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: ``[batch, obs_dim]``
            act: ``[batch, act_dim]``

        Returns:
            ``[batch, num_critics, num_quantiles]``
        """
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([critic(x) for critic in self.net_lst], dim=1)
