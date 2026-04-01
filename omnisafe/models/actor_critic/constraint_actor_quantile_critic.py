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
"""Actor + TQC reward/cost critics for COX-Q (Constrained Optimistic eXploration Q-learning).

Reference:
    Title: Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration
    Authors: Guopeng Li, Matthijs T.J. Spaan, Julian F.P. Kooij
    URL: https://openreview.net/forum?id=EHs3tSukHC
    Published: ICLR 2026
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.distributions as D
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR

from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.critic.quantile_critic import QuantileCritic
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorQuantileCritic(nn.Module):
    """Gaussian SAC actor paired with two TQC quantile critic ensembles (reward + cost).

    Attributes:
        actor: Gaussian SAC actor.
        reward_critic / cost_critic: QuantileCritic ``[batch, N, M]``.
        target_reward_critic / target_cost_critic: Frozen polyak targets.
        use_cox: If True, ``step()`` uses COX exploration.
        cox_delta: Adaptive trust-region size δ.
        cox_cost_limit_q: Episode cost limit d converted to Q-value space.
        cox_lambda: Current Lagrangian multiplier λ.
        cox_beta_r / cox_beta_c: Optimism/pessimism scales for Q bounds.
        cox_alpha_cvar: Number of top-α quantiles for cost CVaR.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        super().__init__()

        # ── Actor ──────────────────────────────────────────────────────────────
        self.actor = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            activation=model_cfgs.actor.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
        ).build_actor(actor_type=model_cfgs.actor_type)

        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False

        # ── Quantile critics ───────────────────────────────────────────────────
        num_critics: int = model_cfgs.critic.num_critics
        num_quantiles: int = model_cfgs.critic.num_quantiles

        self.reward_critic = QuantileCritic(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=num_critics,
            num_quantiles=num_quantiles,
        )
        self.target_reward_critic = deepcopy(self.reward_critic)
        for p in self.target_reward_critic.parameters():
            p.requires_grad = False

        self.cost_critic = QuantileCritic(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=num_critics,
            num_quantiles=num_quantiles,
        )
        self.target_cost_critic = deepcopy(self.cost_critic)
        for p in self.target_cost_critic.parameters():
            p.requires_grad = False

        # ── Optimizers ─────────────────────────────────────────────────────────
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        self.reward_critic_optimizer = optim.Adam(
            self.reward_critic.parameters(), lr=model_cfgs.critic.lr
        )
        self.cost_critic_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=model_cfgs.critic.lr
        )

        # ── LR scheduler ───────────────────────────────────────────────────────
        if model_cfgs.linear_lr_decay:
            self.actor_scheduler: LinearLR | ConstantLR = LinearLR(
                self.actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs
            )
        else:
            self.actor_scheduler = ConstantLR(
                self.actor_optimizer, factor=1.0, total_iters=epochs
            )

        # ── COX exploration state ──────────────────────────────────────────────
        self.use_cox: bool = False
        self.cox_delta: float = 6.0
        self.cox_cost_limit_q: float = 1.0
        self.cox_lambda: float = 0.0
        self.cox_beta_r: float = 3.0
        self.cox_beta_c: float = 3.0
        self.cox_alpha_cvar: int = 13

    # ── Q-bound computation ────────────────────────────────────────────────────

    def _compute_q_bounds(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Q_r^UB, Q_c^LB (CVaR), Q_c^mean — all shape ``[batch]``."""
        M = self.reward_critic.num_quantiles
        alpha = min(self.cox_alpha_cvar, M)

        # Reward: optimistic UB = μ + β_r * σ averaged over quantiles
        r_q = self.reward_critic(obs, action)          # [B, N, M]
        r_mu = r_q.mean(dim=1)                          # [B, M]
        r_std = r_q.std(dim=1)                          # [B, M]
        q_r_ub = (r_mu + self.cox_beta_r * r_std).mean(dim=-1)   # [B]

        # Cost: pessimistic LB = μ - β_c * σ; CVaR over top-α quantiles
        c_q = self.cost_critic(obs, action)             # [B, N, M]
        c_mu = c_q.mean(dim=1)                          # [B, M]
        c_std = c_q.std(dim=1)                          # [B, M]
        c_lb_per = c_mu - self.cox_beta_c * c_std       # [B, M]
        q_c_lb = c_lb_per[:, (M - alpha):].mean(dim=-1) # [B]

        q_c_mean = c_q.mean(dim=(1, 2))                 # [B]
        return q_r_ub, q_c_lb, q_c_mean

    # ── COX exploration step ───────────────────────────────────────────────────

    def _cox_step(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute COX exploration action (Algorithm 1 from paper).

        Steps:
        1. Get policy mean μ_T and std σ_T.
        2. Compute Q gradients ∇Q_r^UB and ∇Q_c^LB w.r.t. pre-tanh mean.
        3. Apply Policy-MGDA to resolve gradient conflicts (Lemma 1, Eq. 14).
        4. Compute adaptive step length η* (Lemma 2, Eq. 18).
        5. Shift mean: μ_E = μ_T + η* Σ_T g*; sample action.
        """
        with torch.no_grad():
            dist = self.actor._distribution(obs)
            mu_T_val = dist.mean          # [B, A], no grad
            sigma_T = dist.stddev         # [B, A]
            var_T = sigma_T ** 2

        # Leaf tensor through tanh for gradient computation
        mu_T = mu_T_val.detach().requires_grad_(True)
        action_T = torch.tanh(mu_T)

        q_r_ub, q_c_lb, q_c_mean = self._compute_q_bounds(obs.detach(), action_T)

        g_r = torch.autograd.grad(q_r_ub.sum(), mu_T, retain_graph=True)[0].detach()
        g_c = torch.autograd.grad(q_c_lb.sum(), mu_T, retain_graph=True)[0].detach()
        g_m = torch.autograd.grad(q_c_mean.sum(), mu_T)[0].detach()
        q_c_mean_val = q_c_mean.detach()

        # ── Policy-MGDA ────────────────────────────────────────────────────────
        lam = self.cox_lambda
        d_q = self.cox_cost_limit_q
        g_raw = g_r - lam * g_c   # [B, A]

        def si(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Σ-inner product."""
            return (a * v * b).sum(dim=-1)

        g_star = g_raw.clone()
        unsafe = ~(q_c_mean_val <= d_q)

        if unsafe.any():
            gr = g_r[unsafe]; gc = g_c[unsafe]
            vt = var_T[unsafe]; gr_raw = g_raw[unsafe]

            def _si(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return (a * vt * b).sum(dim=-1)

            s_rr = _si(gr, gr); s_cc = _si(gc, gc); s_rc = _si(gr, gc)
            v_r = _si(gr_raw, gr); v_c = _si(gr_raw, gc)

            g_mgda = gr_raw.clone()

            # Case 1: v_r < 0, v_c ≤ 0 → remove g_r component
            m1 = (v_r < 0) & (v_c <= 0)
            if m1.any():
                g_mgda[m1] = gr_raw[m1] - (v_r[m1] / s_rr[m1]).unsqueeze(-1) * gr[m1]

            # Case 2: v_r ≥ 0, v_c > 0 → remove g_c component
            m2 = (v_r >= 0) & (v_c > 0)
            if m2.any():
                g_mgda[m2] = gr_raw[m2] - (v_c[m2] / s_cc[m2]).unsqueeze(-1) * gc[m2]

            # Case 3: v_r < 0, v_c > 0 → solve full 2×2 system (Eq. 13)
            m3 = (v_r < 0) & (v_c > 0)
            if m3.any():
                det = (s_rr[m3] * s_cc[m3] - s_rc[m3] ** 2).clamp(min=1e-8)
                mu_r = (-s_cc[m3] * v_r[m3] + s_rc[m3] * v_c[m3]) / det
                mu_c = (-s_rr[m3] * v_c[m3] + s_rc[m3] * v_r[m3]) / det
                g_mgda[m3] = (
                    gr_raw[m3]
                    - mu_r.unsqueeze(-1) * gr[m3]
                    + mu_c.unsqueeze(-1) * gc[m3]
                )
            g_star[unsafe] = g_mgda

        # ── Adaptive step length η* (Eq. 18) ──────────────────────────────────
        g_norm_sq = si(g_star, g_star, var_T).clamp(min=1e-8)
        eta_kl = torch.sqrt(2.0 * self.cox_delta / g_norm_sq)

        s_val = si(g_m, g_star, var_T)
        r_val = d_q - q_c_mean_val

        eta_star = torch.where(
            s_val < 0,
            eta_kl,
            torch.where(
                (r_val < 0) | (s_val == 0),
                torch.zeros_like(eta_kl),
                torch.minimum(eta_kl, r_val / s_val.clamp(min=1e-8)),
            ),
        )

        # ── Sample exploration action ──────────────────────────────────────────
        mu_delta = eta_star.unsqueeze(-1) * var_T * g_star
        mu_E = mu_T_val + mu_delta
        eps = torch.randn_like(sigma_T)
        raw_action = mu_E + sigma_T * eps       # pre-tanh sample
        action = torch.tanh(raw_action)

        # Set actor internal state so adapter can call actor.log_prob() afterwards.
        # log_prob() uses _current_dist and _current_raw_action set by predict().
        self.actor._current_dist = D.Normal(mu_E, sigma_T)
        self.actor._current_raw_action = raw_action
        self.actor._after_inference = True

        return action.clamp(-1.0, 1.0)

    # ── Public interface ───────────────────────────────────────────────────────

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action for environment interaction."""
        if deterministic or not self.use_cox:
            with torch.no_grad():
                return self.actor.predict(obs, deterministic=deterministic)
        self.reward_critic.eval()
        self.cost_critic.eval()
        action = self._cox_step(obs)
        self.reward_critic.train()
        self.cost_critic.train()
        return action.detach()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.step(obs, deterministic=deterministic)

    def polyak_update(self, tau: float) -> None:
        """Soft-update all target networks."""
        for p, tp in [
            *zip(self.actor.parameters(), self.target_actor.parameters()),
            *zip(self.reward_critic.parameters(), self.target_reward_critic.parameters()),
            *zip(self.cost_critic.parameters(), self.target_cost_critic.parameters()),
        ]:
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
