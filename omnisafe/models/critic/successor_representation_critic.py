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
r"""Successor-representation critic, ``sr_mode='td_ridge'``.

Factorizes the value function as :math:`V(s) = \psi(s)^T w`, where :math:`\psi` is the discounted
sum of a one-step feature stream :math:`\phi` and :math:`w` is the ridge solution of the immediate
reward (or cost) onto :math:`\phi`. Only the pieces this port uses are here: ``phi_source`` is
``'random'`` (a frozen linear projection) or ``'contrastive'`` (an MLP trained by InfoNCE), the
read-out weights are always ridge-solved, and ``psi`` is always trained by its TD target.

``phi`` is frozen under ``'random'`` because ``psi`` is *defined* as its discounted sum: a ``phi``
that moves leaves ``psi`` chasing a feature map that no longer exists and the ridge solve fitting
a basis that shifts underneath it. ``'contrastive'`` moves it deliberately, and pays for that by
relabelling the affected tensors (see ``PolicyGradient._relabel_after_phi_update``).
"""

from __future__ import annotations

import torch
from torch import nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class FrozenPhiFeatures(nn.Module):
    """L2-normalized one-step feature map held fixed at initialization.

    Backs ``phi_source='random'`` with ``hidden_sizes=[]``, i.e. a single random linear projection.

    Args:
        in_dim: Input dimension (``obs_dim``).
        hidden_sizes: Hidden layer sizes; empty gives a single linear projection.
        sr_dim: Dimensionality of ``phi``.
        activation: Activation function.
        weight_initialization_mode: Weight initialization mode.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        super().__init__()
        self.net = build_mlp_network(
            sizes=[in_dim, *hidden_sizes, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.net.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized frozen feature of ``x``."""
        p = self.net(x)
        return p / (p.norm(dim=-1, keepdim=True) + 1e-8)


class ContrastivePhiFeatures(FrozenPhiFeatures):
    """Same network as :class:`FrozenPhiFeatures`, left trainable.

    Backs ``phi_source='contrastive'``. Trained by ``PolicyGradient._contrastive_update_phi``
    through its own ``sr_phi_optimizer``, so the InfoNCE loss's Adam state never mixes with the
    differently-cadenced ``psi`` TD loss's.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        super().__init__(in_dim, hidden_sizes, sr_dim, activation, weight_initialization_mode)
        self.net.requires_grad_(True)  # undo FrozenPhiFeatures' freeze


class TDRidgeSuccessorTrunk(nn.Module):
    r"""Owns ``phi``, ``psi`` and the ridge-solved read-out weights ``w_r`` / ``w_c``.

    ``w_r`` / ``w_c`` are buffers, never parameters: they are refreshed in closed form by
    :meth:`ridge_update` rather than by any gradient step.

    Args:
        obs_dim: Observation dimension.
        hidden_sizes: Trunk hidden sizes; empty makes the trunk an identity.
        sr_dim: Dimensionality of ``phi`` / ``psi``.
        activation: Activation function.
        weight_initialization_mode: Weight initialization mode.
        phi_source: ``'random'`` or ``'contrastive'``.
        phi_hidden_sizes: Hidden sizes of the standalone ``phi`` network.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
        phi_source: str = 'random',
        phi_hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.sr_dim = sr_dim
        self.register_buffer('w_r', torch.zeros(sr_dim))
        self.register_buffer('w_c', torch.zeros(sr_dim))

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
        if phi_source == 'contrastive':
            phi_cls: type[FrozenPhiFeatures] = ContrastivePhiFeatures
        elif phi_source == 'random':
            phi_cls = FrozenPhiFeatures
        else:
            raise NotImplementedError(
                f'Unknown sr_cfgs.phi_source "{phi_source}"; this port supports "random" and '
                '"contrastive".',
            )
        self.phi_net = phi_cls(
            in_dim=obs_dim,
            hidden_sizes=list(phi_hidden_sizes or []) if phi_source == 'contrastive' else [],
            sr_dim=sr_dim,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.psi_head = build_mlp_network(
            sizes=[trunk_out, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def phi(self, obs: torch.Tensor) -> torch.Tensor:
        """L2-normalized one-step feature ``phi(s)``, read off the raw observation."""
        return self.phi_net(obs)

    def psi(self, obs: torch.Tensor) -> torch.Tensor:
        """Successor feature ``psi(s)``."""
        return self.psi_head(self.trunk(obs))

    @torch.no_grad()
    def ridge_update(
        self,
        phi: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        ridge_kappa: float,
        ema_tau: float,
        ridge_kappa_cost: float | None = None,
    ) -> dict[str, float]:
        r"""Refresh ``w_r`` / ``w_c`` by closed-form ridge regression of reward/cost onto ``phi``.

        .. math:: w \leftarrow (1 - \tau) w + \tau (X^T X + \kappa I)^{-1} X^T y

        Solved in float64 for stability, then EMA-blended into the stored buffer. Reward and cost
        share the design matrix but not necessarily ``kappa``: a sparse cost the basis fits poorly
        wants more shrinkage than a dense reward, so ``ridge_kappa_cost`` can set it separately.

        Args:
            phi: ``(N, sr_dim)`` one-step features.
            reward: ``(N,)`` one-step rewards.
            cost: ``(N,)`` one-step costs.
            ridge_kappa: Ridge coefficient, scaling the Gram matrix's mean diagonal.
            ema_tau: EMA blend; ``1.0`` replaces outright.
            ridge_kappa_cost: Cost-only ridge coefficient; ``None`` reuses ``ridge_kappa``.

        Returns:
            Diagnostic statistics for logging.
        """
        x = phi.double()
        gram = x.T @ x
        scale = torch.diagonal(gram).mean().clamp(min=1e-12)
        eye = torch.eye(self.sr_dim, dtype=torch.float64, device=x.device)
        mat_r = gram + ridge_kappa * scale * eye
        mat_c = mat_r if ridge_kappa_cost is None else gram + ridge_kappa_cost * scale * eye

        w_r_new = torch.linalg.solve(mat_r, x.T @ reward.double())
        w_c_new = torch.linalg.solve(mat_c, x.T @ cost.double())
        self.w_r.mul_(1.0 - ema_tau).add_(ema_tau * w_r_new.float())
        self.w_c.mul_(1.0 - ema_tau).add_(ema_tau * w_c_new.float())

        return {
            'Misc/RidgeResidualReward': (x @ w_r_new - reward.double()).pow(2).mean().sqrt().item(),
            'Misc/RidgeResidualCost': (x @ w_c_new - cost.double()).pow(2).mean().sqrt().item(),
            'Misc/WrNorm': self.w_r.norm().item(),
            'Misc/WcNorm': self.w_c.norm().item(),
        }


class SuccessorRepresentationLinearReadout(Critic):
    """Read-out head computing ``value(s) = psi(s)^T w``.

    ``w`` is detached, so the value-target loss trains ``psi`` but never the ridge-solved weights.

    Args:
        obs_space: Observation space.
        act_space: Action space.
        trunk: The shared phi/psi trunk.
        weight_name: Which trunk buffer to read out, ``'w_r'`` or ``'w_c'``.
        weight_initialization_mode: Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: TDRidgeSuccessorTrunk,
        weight_name: str,
        weight_initialization_mode: InitFunction,
    ) -> None:
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
        """Read out ``psi(s)^T w``, with gradient flowing into ``psi`` only."""
        psi = self.trunk.psi(obs)
        return [(psi * getattr(self.trunk, self._weight_name).detach()).sum(-1)]


def discount_cumsum_segments(
    vector_x: torch.Tensor,
    lengths: list[int],
    discount: float,
) -> torch.Tensor:
    """Per-episode discounted cumulative sum over a batch of concatenated segments."""
    if not lengths:
        return vector_x.clone()
    device = vector_x.device
    n_seg, t_max = len(lengths), max(lengths)
    feature_shape = vector_x.shape[1:]
    len_t = torch.as_tensor(lengths, device=device).reshape(n_seg, *([1] * len(feature_shape)))

    padded = vector_x.new_zeros((n_seg, t_max, *feature_shape))
    offset = 0
    for i, length in enumerate(lengths):
        padded[i, :length] = vector_x[offset : offset + length]
        offset += length

    out = torch.zeros_like(padded)
    running = padded.new_zeros((n_seg, *feature_shape))
    for t in reversed(range(t_max)):
        active = t < len_t
        running = torch.where(active, padded[:, t] + discount * running, torch.zeros_like(running))
        out[:, t] = running

    gathered = vector_x.new_empty(vector_x.shape)
    offset = 0
    for i, length in enumerate(lengths):
        gathered[offset : offset + length] = out[i, :length]
        offset += length
    return gathered


def gae_lambda_targets_segments(
    values: torch.Tensor,
    rewards: torch.Tensor,
    lengths: list[int],
    lam: float,
    gamma: float,
) -> torch.Tensor:
    """Segment-wise TD(lambda) targets, used to relabel ``target_sr`` after ``phi`` moves."""
    assert values.shape == rewards.shape, (
        f'values {tuple(values.shape)} and rewards {tuple(rewards.shape)} must have the same shape.'
    )
    assert sum(lengths) == values.shape[0], (
        f'segment lengths sum to {sum(lengths)} but values has {values.shape[0]} rows.'
    )
    if not lengths:
        return values.clone()

    feature_shape = values.shape[1:]
    deltas = values.new_empty(values.shape)
    offset = 0
    for length in lengths:
        seg_values = values[offset : offset + length]
        seg_rewards = rewards[offset : offset + length]
        next_values = torch.cat(
            [seg_values[1:], seg_values.new_zeros((1, *feature_shape))],
            dim=0,
        )
        deltas[offset : offset + length] = seg_rewards + gamma * next_values - seg_values
        offset += length

    return discount_cumsum_segments(deltas, lengths, gamma * lam) + values
