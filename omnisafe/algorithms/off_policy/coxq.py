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
"""COX-Q: Constrained Optimistic eXploration Q-learning.

Reference:
    Title: Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration
    Authors: Guopeng Li, Matthijs T.J. Spaan, Julian F.P. Kooij
    URL: https://openreview.net/forum?id=EHs3tSukHC
    Published: ICLR 2026
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_quantile_critic import (
    ConstraintActorQuantileCritic,
)


@registry.register
class COXQ(BaseAlgo):
    """Constrained Optimistic eXploration Q-learning (COX-Q).

    Combines:
    - TQC (Truncated Quantile Critics) for distributional reward/cost estimation.
    - COX exploration: cost-constrained optimistic action selection via Policy-MGDA.
    - ALM (Augmented Lagrangian Method) actor update.
    - Adaptive trust-region δ based on recent buffer costs.
    - Lagrangian multiplier updated from episode costs.
    """

    _epoch: int

    # ── Environment ────────────────────────────────────────────────────────────

    def _init_env(self) -> None:
        self._env: OffPolicyAdapter = OffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'steps_per_epoch must be divisible by vector_env_nums'
        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'total_steps must be divisible by steps_per_epoch'

        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert self._steps_per_epoch % self._update_cycle == 0
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    # ── Model ──────────────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        # Sync num_critics / num_quantiles into model_cfgs.critic
        self._cfgs.model_cfgs.critic['num_critics'] = self._cfgs.algo_cfgs.num_critics
        self._cfgs.model_cfgs.critic['num_quantiles'] = self._cfgs.algo_cfgs.num_quantiles

        self._actor_critic: ConstraintActorQuantileCritic = ConstraintActorQuantileCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    # ── Initialization ─────────────────────────────────────────────────────────

    def _init(self) -> None:
        self._buf: VectorOffPolicyBuffer = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            penalty_coefficient=0.0,
            device=self._device,
        )

        # Lagrange multiplier
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        # SAC temperature α
        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy: float = -torch.prod(
                torch.tensor(self._env.action_space.shape, dtype=torch.float32)
            ).item()
            self._log_alpha: torch.Tensor = torch.zeros(
                1, requires_grad=True, device=self._device
            )
            self._alpha_optimizer: optim.Optimizer = optim.Adam(
                [self._log_alpha], lr=self._cfgs.model_cfgs.critic.lr
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(self._cfgs.algo_cfgs.alpha, device=self._device)
            )

        # Cost limit in Q-value space: d_q = d_ep * (1-γ^T) / (T*(1-γ))  [Eq. B.2]
        gamma = self._cfgs.algo_cfgs.gamma
        T = self._cfgs.algo_cfgs.episode_len
        d_ep = self._cfgs.lagrange_cfgs.cost_limit
        self._cost_limit_q: float = float(
            d_ep * (1.0 - gamma ** T) / (T * (1.0 - gamma))
        )

        # Adaptive δ state
        self._delta: float = self._cfgs.algo_cfgs.delta_max
        self._delta_max: float = self._cfgs.algo_cfgs.delta_max
        self._delta_lr: float = self._cfgs.algo_cfgs.delta_lr

    # ── Logging ────────────────────────────────────────────────────────────────

    def _init_log(self) -> None:
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {'pi': self._actor_critic.actor}
        if self._cfgs.algo_cfgs.obs_normalize:
            what_to_save['obs_normalizer'] = self._env.save()['obs_normalizer']
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        wl = self._cfgs.logger_cfgs.window_lens
        for key in ('Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen'):
            self._logger.register_key(key, window_length=wl)
        self._logger.register_key('Metrics/TotalCost')

        if self._cfgs.train_cfgs.eval_episodes > 0:
            for key in ('Metrics/TestEpRet', 'Metrics/TestEpCost', 'Metrics/TestEpLen'):
                self._logger.register_key(key, window_length=wl)

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')
        self._logger.register_key('TotalEnvSteps')
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Loss/Loss_cost_critic', delta=True)
        self._logger.register_key('Value/reward_critic')
        self._logger.register_key('Value/cost_critic')
        self._logger.register_key('Value/alpha')
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.register_key('Loss/alpha_loss')
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Metrics/COX_delta')
        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')
        for env_spec_key in self._env.env_spec_keys:
            self._logger.register_key(env_spec_key)

    @property
    def _alpha(self) -> float:
        return self._log_alpha.exp().item()

    # ── Sync COX params to model ───────────────────────────────────────────────

    def _sync_cox_params(self) -> None:
        """Push current algorithm state into the model for use in step()."""
        self._actor_critic.cox_delta = self._delta
        self._actor_critic.cox_cost_limit_q = self._cost_limit_q
        self._actor_critic.cox_lambda = self._lagrange.lagrangian_multiplier.item()
        self._actor_critic.cox_beta_r = self._cfgs.algo_cfgs.beta_r
        self._actor_critic.cox_beta_c = self._cfgs.algo_cfgs.beta_c
        self._actor_critic.cox_alpha_cvar = self._cfgs.algo_cfgs.alpha_cvar

    # ── Main training loop ─────────────────────────────────────────────────────

    def learn(self) -> tuple[float, float, float]:
        """Main training entry point."""
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0

        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                # Enable COX after warm-up
                self._actor_critic.use_cox = (
                    self._cfgs.algo_cfgs.use_cox
                    and step > self._cfgs.algo_cfgs.start_learning_steps
                )
                self._sync_cox_params()

                rollout_start = time.time()
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            # Update Lagrange and δ after each epoch
            Jc = self._logger.get_stats('Metrics/EpCost')[0]
            self._lagrange.update_lagrange_multiplier(Jc)
            if self._cfgs.algo_cfgs.adapt_delta:
                self._update_delta()

            self._logger.store({
                'Time/Update': update_time,
                'Time/Rollout': rollout_time,
                'Time/Evaluate': eval_time,
                'TotalEnvSteps': step + 1,
                'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                'Time/Total': time.time() - start_time,
                'Time/Epoch': time.time() - epoch_time,
                'Train/Epoch': epoch,
                'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.item(),
                'Metrics/COX_delta': self._delta,
            })

            if self._cfgs.model_cfgs.linear_lr_decay:
                self._actor_critic.actor_scheduler.step()

            self._logger.dump_tabular()
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()
        return ep_ret, ep_cost, ep_len

    # ── Parameter updates ──────────────────────────────────────────────────────

    def _update(self) -> None:
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs = data['obs']
            act = data['act']
            reward = data['reward']
            cost = data['cost']
            done = data['done']
            next_obs = data['next_obs']

            self._update_reward_critic(obs, act, reward, done, next_obs)
            self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """TQC reward critic update: target truncates top k_r atoms."""
        kr = self._cfgs.algo_cfgs.truncate_kr
        N = self._cfgs.algo_cfgs.num_critics
        M = self._cfgs.algo_cfgs.num_quantiles

        with torch.no_grad():
            next_act = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_act)
            # Target quantiles [B, N, M] → sort and truncate top k_r per critic
            next_q = self._actor_critic.target_reward_critic(next_obs, next_act)  # [B,N,M]
            # Flatten to [B, N*M], sort ascending, take first N*M-kr atoms
            n_keep = N * M - kr
            next_q_flat = next_q.reshape(next_q.size(0), -1)  # [B, N*M]
            next_q_sorted, _ = next_q_flat.sort(dim=-1)        # ascending
            next_q_trunc = next_q_sorted[:, :n_keep]           # [B, n_keep]
            target = (
                reward.unsqueeze(-1)
                + self._cfgs.algo_cfgs.gamma * (1 - done.unsqueeze(-1))
                * (next_q_trunc - self._alpha * next_logp.unsqueeze(-1))
            )  # [B, n_keep]

        current_q = self._actor_critic.reward_critic(obs, act)  # [B, N, M]
        loss = self._quantile_huber_loss(current_q, target)

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store({
            'Loss/Loss_reward_critic': loss.item(),
            'Value/reward_critic': current_q.mean().item(),
        })

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """TQC cost critic update: target truncates bottom k_c atoms (conservative)."""
        kc = self._cfgs.algo_cfgs.truncate_kc
        N = self._cfgs.algo_cfgs.num_critics
        M = self._cfgs.algo_cfgs.num_quantiles

        with torch.no_grad():
            next_act = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_q = self._actor_critic.target_cost_critic(next_obs, next_act)  # [B,N,M]
            # Sort ascending, discard bottom k_c (keep highest = conservative)
            n_keep = N * M - kc
            next_q_flat = next_q.reshape(next_q.size(0), -1)
            next_q_sorted, _ = next_q_flat.sort(dim=-1)
            next_q_trunc = next_q_sorted[:, kc:]   # keep top n_keep
            target = (
                cost.unsqueeze(-1)
                + self._cfgs.algo_cfgs.gamma * (1 - done.unsqueeze(-1)) * next_q_trunc
            )  # [B, n_keep]

        current_q = self._actor_critic.cost_critic(obs, act)    # [B, N, M]
        loss = self._quantile_huber_loss(current_q, target)

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()
        self._logger.store({
            'Loss/Loss_cost_critic': loss.item(),
            'Value/cost_critic': current_q.mean().item(),
        })

    def _quantile_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Quantile Huber loss ρ_τ(u) = |τ - I(u<0)| * L_1(u).

        Args:
            pred:   ``[B, N, M]`` predicted quantiles.
            target: ``[B, n_target]`` target atoms.

        Returns:
            Scalar mean loss.
        """
        B, N, M = pred.shape
        n_t = target.size(-1)
        tau = (torch.arange(1, M + 1, device=pred.device).float() * 2 - 1) / (2.0 * M)

        # [B, N, M, n_t]
        pred_e = pred.unsqueeze(-1).expand(B, N, M, n_t)
        tgt_e = target.unsqueeze(1).unsqueeze(1).expand(B, N, M, n_t)

        errors = tgt_e - pred_e   # [B, N, M, n_t]
        huber = F.smooth_l1_loss(errors, torch.zeros_like(errors), reduction='none')
        tau_w = (tau.view(1, 1, M, 1) - (errors.detach() < 0).float()).abs()
        return (tau_w * huber).mean()

    def _update_actor(self, obs: torch.Tensor) -> None:
        """ALM actor update (Eq. B.1 from paper)."""
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()

        # Alpha update
        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -(self._log_alpha * (log_prob + self._target_entropy).mean())
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store({'Loss/alpha_loss': alpha_loss.item()})

        self._logger.store({
            'Loss/Loss_pi': loss.item(),
            'Value/alpha': self._alpha,
        })

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        """ALM actor loss: SAC entropy + Lagrangian + optional quadratic penalty.

        L = α*logπ - Q_r^UB + λ*Q_c^UB + (c/2)*(Q_c^UB - d)^2  [when active]
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        # Q bounds from online critics
        q_r = self._actor_critic.reward_critic(obs, action)   # [B, N, M]
        q_c = self._actor_critic.cost_critic(obs, action)     # [B, N, M]

        # Reward UB: μ + β_r * σ, mean over quantiles
        r_mu = q_r.mean(dim=1); r_std = q_r.std(dim=1)
        q_r_ub = (r_mu + self._cfgs.algo_cfgs.beta_r * r_std).mean(dim=-1)  # [B]

        # Cost UB for ALM: μ + β_c * σ, CVaR over top-α
        c_mu = q_c.mean(dim=1); c_std = q_c.std(dim=1)
        c_ub_per = c_mu + self._cfgs.algo_cfgs.beta_c * c_std
        alpha = min(self._cfgs.algo_cfgs.alpha_cvar, c_ub_per.size(-1))
        M = c_ub_per.size(-1)
        q_c_ub = c_ub_per[:, (M - alpha):].mean(dim=-1)   # [B]

        lam = self._lagrange.lagrangian_multiplier.item()
        d_q = self._cost_limit_q
        c_alm = self._cfgs.algo_cfgs.alm_coeff

        violation = q_c_ub - d_q   # [B]

        # ALM: add quadratic penalty when λ/c ≥ d - E[Q_c^UB]  (Eq. B.1)
        use_alm = (lam / c_alm) >= (d_q - q_c_ub.mean().item())
        if use_alm:
            actor_loss = (
                self._alpha * log_prob
                - q_r_ub
                + lam * violation
                + 0.5 * c_alm * violation ** 2
            ).mean()
        else:
            actor_loss = (self._alpha * log_prob - q_r_ub + lam * violation).mean()

        return actor_loss

    def _update_delta(self) -> None:
        """Adaptive δ update (Eq. 19): increase when safe, decrease when unsafe."""
        data = self._buf.sample_batch()
        mean_cost = data['cost'].mean().item()
        T = self._cfgs.algo_cfgs.episode_len
        d_per_step = self._cfgs.lagrange_cfgs.cost_limit / T
        self._delta = float(
            max(1e-4, min(self._delta_max, self._delta + self._delta_lr * (d_per_step - mean_cost)))
        )

    def _log_when_not_update(self) -> None:
        self._logger.store({
            'Loss/Loss_pi': 0.0,
            'Loss/Loss_reward_critic': 0.0,
            'Loss/Loss_cost_critic': 0.0,
            'Value/reward_critic': 0.0,
            'Value/cost_critic': 0.0,
            'Value/alpha': self._alpha,
        })
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.store({'Loss/alpha_loss': 0.0})
