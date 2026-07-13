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
"""Successor-representation critic architectures shared by reward and cost value functions.

Two modes are provided, selected by ``model_cfgs.sr_cfgs.sr_mode``:

* ``shared_trunk``: a single feature trunk followed by two independent linear read-out heads
  (reward, cost). The whole network is trained end-to-end by the ordinary value-target MSE
  loss -- there is no dedicated successor-representation loss, so this mode reuses the stock
  training loop (:meth:`PolicyGradient._update_reward_critic` /
  :meth:`PolicyGradient._update_cost_critic`) completely unchanged.
* ``td_ridge``: a literal successor representation. A trunk produces a one-step feature
  ``phi(s)`` (l2-normalized linear read-out) and a successor feature ``psi(s)`` (MLP read-out)
  trained by TD to satisfy ``psi(s_t) ~= phi(s_t) + gamma * psi(s_{t+1})``. The reward/cost
  read-out weights ``w_r`` / ``w_c`` are solved in closed form by ridge regression of the
  one-step reward/cost onto ``phi`` once per update (see :meth:`ridge_update`) and are stored
  as buffers, not parameters.
"""

from __future__ import annotations

import torch
from torch import nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network, initialize_layer


class SuccessorRepresentationTrunk(nn.Module):
    """Shared feature trunk producing the successor-representation vector directly.

    Args:
        obs_dim (int): Observation dimension.
        hidden_sizes (list of int): Hidden layer sizes of the trunk.
        sr_dim (int): Dimensionality of the shared feature vector.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationTrunk`."""
        super().__init__()
        self.net = build_mlp_network(
            sizes=[obs_dim, *hidden_sizes, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the shared feature vector for ``obs``."""
        return self.net(obs)


class SuccessorRepresentationReadout(Critic):
    """A linear read-out head over a shared successor-representation trunk.

    Mirrors the standard :class:`Critic` interface (``forward`` returns a length-1 list holding
    a ``(B,)`` value tensor), so it drops in place of the stock ``reward_critic`` /
    ``cost_critic`` without changing any of their call sites.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (SuccessorRepresentationTrunk): The shared trunk (same instance passed to both the
            reward and the cost head, so its parameters are trained by both losses).
        sr_dim (int): Dimensionality of the shared feature vector.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: SuccessorRepresentationTrunk,
        sr_dim: int,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationReadout`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes=[],
            activation='identity',
            weight_initialization_mode=weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        )
        self.trunk = trunk
        self.head = nn.Linear(sr_dim, 1)
        initialize_layer(weight_initialization_mode, self.head)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Read out a scalar value from the shared trunk's feature vector."""
        feat = self.trunk(obs)
        value = torch.squeeze(self.head(feat), -1)
        return [value]


class TDRidgeSuccessorRepresentationTrunk(nn.Module):
    """phi / psi bundle for the literal successor-representation (``td_ridge``) mode.

    Args:
        obs_dim (int): Observation dimension.
        hidden_sizes (list of int): Hidden layer sizes of the shared trunk.
        sr_dim (int): Dimensionality of ``phi`` and ``psi``.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`TDRidgeSuccessorRepresentationTrunk`."""
        super().__init__()
        self.sr_dim = sr_dim
        trunk_out = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.trunk: nn.Module = (
            build_mlp_network(
                sizes=[obs_dim, *hidden_sizes],
                activation=activation,
                output_activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            if hidden_sizes
            else nn.Identity()
        )
        self.phi_head = nn.Linear(trunk_out, sr_dim)
        initialize_layer(weight_initialization_mode, self.phi_head)
        self.psi_head = build_mlp_network(
            sizes=[trunk_out, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        # w_r / w_c are solved by closed-form ridge regression (see ridge_update), never by
        # SGD, so they are buffers rather than parameters.
        self.register_buffer('w_r', torch.zeros(sr_dim))
        self.register_buffer('w_c', torch.zeros(sr_dim))

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared trunk features."""
        return self.trunk(obs)

    def phi(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalized one-step feature ``phi(s)``."""
        z = self.features(obs) if z is None else z
        p = self.phi_head(z)
        return p / (p.norm(dim=-1, keepdim=True) + 1e-8)

    def psi(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Successor feature ``psi(s)``."""
        z = self.features(obs) if z is None else z
        return self.psi_head(z)

    @torch.no_grad()
    def ridge_update(
        self,
        phi: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        ridge_kappa: float,
        ema_tau: float,
    ) -> dict[str, float]:
        r"""Refresh ``w_r`` and ``w_c`` by closed-form ridge regression on the fresh batch.

        .. math::

            w \leftarrow (1 - \tau) w + \tau (\Phi^T \Phi + \kappa I)^{-1} \Phi^T y

        solved in float64 for numerical stability, then EMA-blended into the stored buffer.

        Args:
            phi (torch.Tensor): One-step features of shape ``(N, sr_dim)`` for the fresh batch.
            reward (torch.Tensor): One-step rewards of shape ``(N,)``.
            cost (torch.Tensor): One-step costs of shape ``(N,)``.
            ridge_kappa (float): Ridge regularization coefficient (scales the mean diagonal of
                the Gram matrix).
            ema_tau (float): EMA blending coefficient; ``1.0`` means no smoothing (replace).

        Returns:
            A dict of diagnostic statistics for logging.
        """
        p = phi.double()
        gram = p.T @ p
        kappa = ridge_kappa * torch.diagonal(gram).mean().clamp(min=1e-12)
        mat = gram + kappa * torch.eye(self.sr_dim, dtype=torch.float64, device=p.device)

        w_r_new = torch.linalg.solve(mat, p.T @ reward.double())
        w_c_new = torch.linalg.solve(mat, p.T @ cost.double())
        self.w_r.mul_(1.0 - ema_tau).add_(ema_tau * w_r_new.float())
        self.w_c.mul_(1.0 - ema_tau).add_(ema_tau * w_c_new.float())

        resid_r = (p @ w_r_new - reward.double()).pow(2).mean().sqrt().item()
        resid_c = (p @ w_c_new - cost.double()).pow(2).mean().sqrt().item()
        return {
            'Misc/RidgeResidualReward': resid_r,
            'Misc/RidgeResidualCost': resid_c,
            'Misc/WrNorm': self.w_r.norm().item(),
            'Misc/WcNorm': self.w_c.norm().item(),
            'Misc/GramCond': torch.linalg.cond(mat).item(),
        }


class SuccessorRepresentationLinearReadout(Critic):
    """A read-out head whose weight vector is solved by ridge regression, not by SGD.

    ``value(s) = psi(s)^T w``, where ``psi`` is the trunk's successor-feature output and ``w``
    is a non-learnable buffer refreshed by :meth:`TDRidgeSuccessorRepresentationTrunk.ridge_update`.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (TDRidgeSuccessorRepresentationTrunk): The shared phi/psi trunk.
        weight_name (str): Name of the trunk buffer to read out (``'w_r'`` or ``'w_c'``).
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: TDRidgeSuccessorRepresentationTrunk,
        weight_name: str,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationLinearReadout`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes=[],
            activation='identity',
            weight_initialization_mode=weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        )
        self.trunk = trunk
        self._weight_name = weight_name

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Read out a scalar value as ``psi(s)^T w``, with gradient flowing into ``psi`` only."""
        psi = self.trunk.psi(obs)
        weight = getattr(self.trunk, self._weight_name)
        value = (psi * weight).sum(-1)
        return [value]
