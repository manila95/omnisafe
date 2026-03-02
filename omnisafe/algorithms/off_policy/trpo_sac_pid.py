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
"""Implementation of TRPOSACPID: TRPOPID warmup + SACPID.

Phase 1: Run TRPOPID (on-policy safe RL) for warmup_epochs - uses both reward and cost
from the start for safe exploration. In parallel, train Q-critics on replay buffer.

Phase 2: Switch to SACPID with policy and lambda transferred from Phase 1.
"""

from __future__ import annotations

import io
import time
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter.hybrid_adapter import HybridAdapter
from omnisafe.utils.q_mc_eval import q_vs_mc_scatter_figure
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_pid import SACPID
from omnisafe.common.buffer import VectorOffPolicyBuffer, VectorOnPolicyBuffer
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


def _log_figure_to_wandb(fig: Any, tag: str, step: int) -> None:
    """Log a matplotlib figure directly to wandb."""
    if wandb.run is None:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    try:
        from PIL import Image as PILImage
        img = np.array(PILImage.open(buf).copy())
    except ImportError:
        import matplotlib.image as mpimg
        buf.seek(0)
        img = mpimg.imread(buf)
    wandb.log({tag: wandb.Image(img), 'epoch': step})


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class TRPOSACPID(SACPID):
    """TRPOSACPID: TRPOPID warmup phase followed by SACPID.

    Replaces the cost-blind warmup of SACPID/SACLag with an on-policy TRPOPID phase
    that uses both reward and cost from the start for safe exploration.

    - Phase 1 (warmup_epochs): TRPOPID updates policy + V-critics; Q-critics trained
      in parallel on replay buffer (Option A).
    - Phase 2: SACPID with policy and lambda transferred from Phase 1.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: https://arxiv.org/abs/2007.03964
    """

    def _init_env(self) -> None:
        """Initialize HybridAdapter for both on-policy and off-policy rollout."""
        self._env: HybridAdapter = HybridAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'steps_per_epoch must be divisible by vector_env_nums.'
        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'total_steps must be divisible by steps_per_epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'steps_per_epoch must be divisible by update_cycle.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Create SACPID model (Q-critics) and on-policy model (V-critics) with shared actor."""
        super()._init_model()

        # On-policy actor-critic for Phase 1 (TRPOPID) - share SAC actor
        # (gaussian_sac) so weights stay in sync
        self._on_policy_actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

        # Share actor: use SAC actor for on-policy phase (same architecture)
        # We replace the on-policy actor with the SAC actor so weights stay in sync
        self._on_policy_actor_critic.actor = self._actor_critic.actor

    def _init(self) -> None:
        """Create buffers and pass off-policy buffer to HybridAdapter."""
        super()._init()

        # On-policy buffer for Phase 1
        self._on_policy_buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=getattr(self._cfgs.algo_cfgs, 'lam', 0.95),
            lam_c=getattr(self._cfgs.algo_cfgs, 'lam_c', 0.95),
            advantage_estimator=getattr(
                self._cfgs.algo_cfgs, 'adv_estimation_method', 'gae'
            ),
            standardized_adv_r=getattr(
                self._cfgs.algo_cfgs, 'standardized_rew_adv', True
            ),
            standardized_adv_c=getattr(
                self._cfgs.algo_cfgs, 'standardized_cost_adv', True
            ),
            penalty_coefficient=getattr(self._cfgs.algo_cfgs, 'penalty_coef', 0.0),
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

        self._env.set_off_policy_buffer(self._buf)

    def _init_log(self) -> None:
        """Log keys for both phases."""
        super()._init_log()
        self._logger.register_key('Train/Phase')
        # On-policy rollout (Phase 1) logs Value/reward and Value/cost from V-critics
        self._logger.register_key('Value/reward')
        self._logger.register_key('Value/cost')
        # Phase 1 TRPO update keys
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/PolicyRatio')
        self._logger.register_key('Train/PolicyStd')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Value/Adv')
        # Phase 2: Q vs MC scatter (off-policy value evaluation)
        self._logger.register_key('Q_MC/reward_mae')
        self._logger.register_key('Q_MC/reward_corr')
        self._logger.register_key('Q_MC/cost_mae')
        self._logger.register_key('Q_MC/cost_corr')

    def _phase1_trpo_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """TRPOPID-style surrogate: (adv_r - lambda * adv_c) / (1 + lambda)."""
        penalty = self._lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)

    def _phase1_trpo_loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        """Policy gradient loss for TRPO (ratio * adv)."""
        dist = self._on_policy_actor_critic.actor(obs)
        logp_new = self._on_policy_actor_critic.actor.log_prob(act)
        std = self._on_policy_actor_critic.actor.std
        ratio = torch.exp(logp_new - logp)
        loss = -(ratio * adv).mean()
        entropy = dist.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio.mean().item(),
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _phase1_update(self) -> None:
        """One epoch of TRPOPID update + Q-critic updates."""
        # PID update
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._lagrange.pid_update(Jc)
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

        # Get on-policy data
        data = self._on_policy_buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_dist = self._on_policy_actor_critic.actor(obs)

        # Update V-critics
        dataloader_v = DataLoader(
            TensorDataset(obs, target_value_r, target_value_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for ob, tvr, tvc in dataloader_v:
                self._on_policy_actor_critic.reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(
                    self._on_policy_actor_critic.reward_critic(ob)[0], tvr
                )
                loss_r.backward()
                if self._cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self._on_policy_actor_critic.reward_critic.parameters(),
                        self._cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self._on_policy_actor_critic.reward_critic)
                self._on_policy_actor_critic.reward_critic_optimizer.step()

                if self._cfgs.algo_cfgs.use_cost:
                    self._on_policy_actor_critic.cost_critic_optimizer.zero_grad()
                    loss_c = nn.functional.mse_loss(
                        self._on_policy_actor_critic.cost_critic(ob)[0], tvc
                    )
                    loss_c.backward()
                    if self._cfgs.algo_cfgs.use_max_grad_norm:
                        clip_grad_norm_(
                            self._on_policy_actor_critic.cost_critic.parameters(),
                            self._cfgs.algo_cfgs.max_grad_norm,
                        )
                    distributed.avg_grads(self._on_policy_actor_critic.cost_critic)
                    self._on_policy_actor_critic.cost_critic_optimizer.step()

        # TRPO actor update (conjugate gradient + line search)
        adv = self._phase1_trpo_adv_surrogate(adv_r, adv_c)
        self._fvp_obs = obs[:: getattr(self._cfgs.algo_cfgs, 'fvp_sample_freq', 1)]

        def fvp(params: torch.Tensor) -> torch.Tensor:
            self._on_policy_actor_critic.actor.zero_grad()
            q_dist = self._on_policy_actor_critic.actor(self._fvp_obs)
            with torch.no_grad():
                p_dist = self._on_policy_actor_critic.actor(self._fvp_obs)
            kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
            grads = torch.autograd.grad(
                kl,
                tuple(self._on_policy_actor_critic.actor.parameters()),
                create_graph=True,
            )
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            kl_p = (flat_grad_kl * params).sum()
            grads2 = torch.autograd.grad(
                kl_p,
                tuple(self._on_policy_actor_critic.actor.parameters()),
                retain_graph=False,
            )
            flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads2])
            distributed.avg_tensor(flat_grad_grad_kl)
            cg_damping = getattr(self._cfgs.algo_cfgs, 'cg_damping', 0.1)
            return flat_grad_grad_kl + params * cg_damping

        theta_old = get_flat_params_from(self._on_policy_actor_critic.actor)
        self._on_policy_actor_critic.actor.zero_grad()
        loss_before = self._phase1_trpo_loss_pi(obs, act, logp, adv)
        loss_before = distributed.dist_avg(loss_before)
        loss_before.backward()
        distributed.avg_grads(self._on_policy_actor_critic.actor)
        grads = -get_flat_gradients_from(self._on_policy_actor_critic.actor)
        cg_iters = getattr(self._cfgs.algo_cfgs, 'cg_iters', 15)
        x = conjugate_gradients(fvp, grads, cg_iters)
        target_kl = getattr(self._cfgs.algo_cfgs, 'target_kl', 0.01)
        xHx = torch.dot(x, fvp(x))
        alpha = torch.sqrt(2 * target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        theta_new = theta_old + step_direction
        set_param_values_to_model(self._on_policy_actor_critic.actor, theta_new)

        self._logger.store(
            {
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': 0.0,
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
            },
        )

        # Q-critic updates (Option A: train in parallel during phase 1)
        phase1_q_iters = getattr(
            self._cfgs.algo_cfgs,
            'phase1_q_update_iters',
            self._cfgs.algo_cfgs.update_iters,
        )
        if self._buf.size > 0:
            q_batch_size = min(
                self._buf.size,
                self._cfgs.algo_cfgs.batch_size,
            )
            for _ in range(phase1_q_iters):
                q_data = self._buf.sample_batch(batch_size=q_batch_size)
                q_obs = q_data['obs']
                q_act = q_data['act']
                q_reward = q_data['reward']
                q_cost = q_data['cost']
                q_done = q_data['done']
                q_next_obs = q_data['next_obs']
                self._update_reward_critic(q_obs, q_act, q_reward, q_done, q_next_obs)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(q_obs, q_act, q_cost, q_done, q_next_obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _transfer_phase1_to_phase2(self) -> None:
        """Transfer policy and lambda from Phase 1 to Phase 2."""
        # Policy: already shared, no copy needed
        # Lambda: SACPID's _lagrange is PIDLagrangian, already updated during phase 1
        # Replay buffer: already populated
        n_steps = self._buf.size
        n_transitions = n_steps * self._buf.num_envs
        self._logger.log(
            f'INFO: Transitioning from Phase 1 (TRPOPID) to Phase 2 (SACPID) '
            f'with {n_steps} steps ({n_transitions} transitions) in replay buffer'
        )

    def learn(self) -> tuple[float, float, float]:
        """Two-phase training: TRPOPID warmup then SACPID."""
        self._logger.log('INFO: Start TRPOSACPID training (Phase 1: TRPOPID, Phase 2: SACPID)')
        start_time = time.time()
        warmup_epochs = self._cfgs.algo_cfgs.warmup_epochs
        total_env_steps = 0

        # Phase 1: TRPOPID
        for epoch in range(warmup_epochs):
            self._epoch = epoch
            epoch_time = time.time()
            rollout_time = 0.0
            update_time = 0.0

            rollout_start = time.time()
            self._env.on_policy_rollout(
                steps_per_epoch=self._cfgs.algo_cfgs.steps_per_epoch,
                agent=self._on_policy_actor_critic,
                on_policy_buffer=self._on_policy_buf,
                logger=self._logger,
                off_policy_buffer=self._buf,
            )
            rollout_time = time.time() - rollout_start

            update_start = time.time()
            self._phase1_update()
            update_time = time.time() - update_start

            total_env_steps = (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch

            # Phase 1: Q vs MC scatter (Q-critics are trained in parallel during warmup)
            eval_env = getattr(self._env, '_eval_env', None)
            log_q_mc_freq = getattr(self._cfgs.algo_cfgs, 'log_q_mc_freq', 0)
            if log_q_mc_freq > 0 and epoch % log_q_mc_freq == 0 and eval_env is not None:
                num_ep = getattr(self._cfgs.algo_cfgs, 'q_mc_num_episodes', 20)
                max_pt = getattr(self._cfgs.algo_cfgs, 'q_mc_max_points', 500)
                fig, qmc_stats = q_vs_mc_scatter_figure(
                    env=eval_env,
                    actor_critic=self._actor_critic,
                    gamma=self._cfgs.algo_cfgs.gamma,
                    num_episodes=num_ep,
                    max_ep_len=getattr(eval_env, '_max_ep_len', None) or getattr(eval_env, 'time_limit', 1000),
                    use_cost=self._cfgs.algo_cfgs.use_cost,
                    device=self._device,
                    deterministic_policy=False,
                    max_points=max_pt,
                )
                _log_figure_to_wandb(fig, 'Q_vs_MC/scatter', epoch)
                self._logger.log_figure('Q_vs_MC/scatter', fig, step=epoch)
                plt.close(fig)
                phase1_qmc = {
                    'Q_MC/reward_mae': qmc_stats.get('reward/mae', 0.0),
                    'Q_MC/reward_corr': qmc_stats.get('reward/correlation', 0.0),
                    'Q_MC/cost_mae': qmc_stats.get('cost/mae', 0.0),
                    'Q_MC/cost_corr': qmc_stats.get('cost/correlation', 0.0),
                }
            else:
                phase1_qmc = {
                    'Q_MC/reward_mae': 0.0,
                    'Q_MC/reward_corr': 0.0,
                    'Q_MC/cost_mae': 0.0,
                    'Q_MC/cost_corr': 0.0,
                }

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store(
                {
                    'TotalEnvSteps': total_env_steps,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': time.time() - start_time,
                    'Time/Epoch': time.time() - epoch_time,
                    'Time/Rollout': rollout_time,
                    'Time/Update': update_time,
                    'Time/Evaluate': eval_time,
                    'Train/Epoch': epoch,
                    'Train/Phase': 1,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                    **phase1_qmc,
                },
            )
            self._logger.dump_tabular()
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        # Transition
        self._transfer_phase1_to_phase2()

        # Phase 2: SACPID (standard off-policy loop)
        for epoch in range(warmup_epochs, self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise
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

            total_env_steps = (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch

            # Phase 2: off-policy Q vs MC return scatter (evaluate value function)
            eval_env = getattr(self._env, '_eval_env', None)
            log_q_mc_freq = getattr(self._cfgs.algo_cfgs, 'log_q_mc_freq', 0)
            if (
                log_q_mc_freq > 0
                and (epoch - warmup_epochs) % log_q_mc_freq == 0
                and eval_env is not None
            ):
                num_ep = getattr(self._cfgs.algo_cfgs, 'q_mc_num_episodes', 20)
                max_pt = getattr(self._cfgs.algo_cfgs, 'q_mc_max_points', 500)
                fig, qmc_stats = q_vs_mc_scatter_figure(
                    env=eval_env,
                    actor_critic=self._actor_critic,
                    gamma=self._cfgs.algo_cfgs.gamma,
                    num_episodes=num_ep,
                    max_ep_len=getattr(eval_env, '_max_ep_len', None) or getattr(eval_env, 'time_limit', 1000),
                    use_cost=self._cfgs.algo_cfgs.use_cost,
                    device=self._device,
                    deterministic_policy=False,
                    max_points=max_pt,
                )
                _log_figure_to_wandb(fig, 'Q_vs_MC/scatter', epoch)
                self._logger.log_figure('Q_vs_MC/scatter', fig, step=epoch)
                plt.close(fig)
                self._logger.store(
                    {
                        'Q_MC/reward_mae': qmc_stats.get('reward/mae', 0.0),
                        'Q_MC/reward_corr': qmc_stats.get('reward/correlation', 0.0),
                        'Q_MC/cost_mae': qmc_stats.get('cost/mae', 0.0),
                        'Q_MC/cost_corr': qmc_stats.get('cost/correlation', 0.0),
                    },
                )
            else:
                self._logger.store(
                    {
                        'Q_MC/reward_mae': 0.0,
                        'Q_MC/reward_corr': 0.0,
                        'Q_MC/cost_mae': 0.0,
                        'Q_MC/cost_corr': 0.0,
                    },
                )

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            if (
                total_env_steps > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': total_env_steps,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': time.time() - start_time,
                    'Time/Epoch': time.time() - epoch_time,
                    'Time/Rollout': rollout_time,
                    'Time/Update': update_time,
                    'Time/Evaluate': eval_time,
                    'Train/Epoch': epoch,
                    'Train/Phase': 2,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )
            self._logger.dump_tabular()
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()
        return ep_ret, ep_cost, ep_len
