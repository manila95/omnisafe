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
"""Implementation of the SACPID (PID version of SACLag) algorithm."""


import warnings

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.pid_lagrange import PIDLagrangian


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACPID(SAC):
    """The SACPID (PID version of SACLag) algorithm.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: `SACPID <https://arxiv.org/abs/2007.03964>`_
    """

    def _init(self) -> None:
        """The initialization of the SACPID algorithm.

        The SACPID algorithm uses a PID-Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        lagrange_cfgs = self._cfgs.lagrange_cfgs.todict()
        if self._cfgs.algo_cfgs.get('cost_limit_normalize', False):
            lagrange_cfgs['cost_limit'] = 1.0
        self._lagrange: PIDLagrangian = PIDLagrangian(**lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the SACPID specific information.

        +----------------------------+------------------------------+
        | Things to log              | Description                  |
        +============================+==============================+
        | Metrics/LagrangeMultiplier | The PID-Lagrange multiplier. |
        +----------------------------+------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`pid_update` method.

        When ``cost_critic_recency_decay > 0`` the cost critic is updated with a
        separately sampled batch whose time-step indices are drawn with exponential
        recency weights, while the reward critic and actor continue to use the
        standard uniform replay-buffer sample.  Setting the decay to ``0`` (the
        default) recovers the original behaviour.
        """
        recency_decay: float = self._cfgs.algo_cfgs.get('recency_decay', 0.001)
        recency_reward: bool = self._cfgs.algo_cfgs.get('recency_reward_critic', False)
        recency_cost: bool = self._cfgs.algo_cfgs.get('recency_cost_critic', False)
        n_step_cost: int = self._cfgs.algo_cfgs.get('n_step_cost', 1)
        retrace_lambda: float = self._cfgs.algo_cfgs.get('retrace_lambda', 0.0)
        retrace_n_steps: int = self._cfgs.algo_cfgs.get('retrace_n_steps', 5)
        retrace_reward: bool = self._cfgs.algo_cfgs.get('retrace_reward_critic', False)
        retrace_cost: bool = self._cfgs.algo_cfgs.get('retrace_cost_critic', True)

        if retrace_lambda > 0.0 and not retrace_reward and not retrace_cost:
            warnings.warn(
                'retrace_lambda > 0 but both retrace_reward_critic and retrace_cost_critic '
                'are False — Retrace is disabled. Set at least one flag to True.',
                UserWarning,
                stacklevel=2,
            )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            if retrace_lambda > 0.0 and retrace_reward:
                r_data, r_idx, r_env_idx = self._buf.sample_batch_nstep_safe(retrace_n_steps)
                retrace_r_targets = self._buf.compute_retrace_reward_targets(
                    idx=r_idx,
                    env_idx=r_env_idx,
                    n_steps=retrace_n_steps,
                    retrace_lambda=retrace_lambda,
                    gamma=self._cfgs.algo_cfgs.gamma,
                    target_reward_critic=self._actor_critic.target_reward_critic,
                    actor=self._actor_critic.actor,
                    alpha=self._alpha,
                )
                self._update_reward_critic_with_targets(
                    r_data['obs'], r_data['act'], retrace_r_targets
                )
            elif recency_reward:
                r_data = self._buf.sample_batch_recency_weighted(recency_decay)
                self._update_reward_critic(
                    r_data['obs'],
                    r_data['act'],
                    r_data['reward'],
                    r_data['done'],
                    r_data['next_obs'],
                )
            else:
                self._update_reward_critic(obs, act, reward, done, next_obs)

            if self._cfgs.algo_cfgs.use_cost:
                if retrace_lambda > 0.0 and retrace_cost:
                    c_data, c_idx, c_env_idx = self._buf.sample_batch_nstep_safe(retrace_n_steps)
                    retrace_targets = self._buf.compute_retrace_cost_targets(
                        idx=c_idx,
                        env_idx=c_env_idx,
                        n_steps=retrace_n_steps,
                        retrace_lambda=retrace_lambda,
                        gamma=self._cfgs.algo_cfgs.gamma,
                        target_cost_critic=self._actor_critic.target_cost_critic,
                        actor=self._actor_critic.actor,
                    )
                    self._update_cost_critic_with_targets(
                        c_data['obs'], c_data['act'], retrace_targets
                    )
                elif n_step_cost > 1:
                    # Biased n-step approximation (no IS correction) — kept for ablations.
                    c_data, c_idx, c_env_idx = self._buf.sample_batch_nstep_safe(n_step_cost)
                    nstep_targets = self._buf.compute_nstep_cost_targets(
                        idx=c_idx,
                        env_idx=c_env_idx,
                        n_steps=n_step_cost,
                        gamma=self._cfgs.algo_cfgs.gamma,
                        target_cost_critic=self._actor_critic.target_cost_critic,
                        next_action_fn=lambda o: self._actor_critic.actor.predict(
                            o, deterministic=False
                        ),
                    )
                    self._update_cost_critic_with_targets(
                        c_data['obs'], c_data['act'], nstep_targets
                    )
                elif recency_cost:
                    c_data = self._buf.sample_batch_recency_weighted(recency_decay)
                    self._update_cost_critic(
                        c_data['obs'],
                        c_data['act'],
                        c_data['cost'],
                        c_data['done'],
                        c_data['next_obs'],
                    )
                else:
                    self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

        pid_update_freq: int = self._cfgs.algo_cfgs.get('pid_update_freq', 1)
        if self._update_count % pid_update_freq == 0:
            Jc = self._logger.get_stats('Metrics/EpCost')[0]
            if self._cfgs.algo_cfgs.get('cost_limit_normalize', False):
                Jc = Jc / self._cfgs.lagrange_cfgs.cost_limit
            if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                self._lagrange.pid_update(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )
    def _update_reward_critic_with_targets(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update the reward critic against precomputed targets (e.g. Retrace returns).

        Args:
            obs (torch.Tensor): Observations sampled from the buffer.
            action (torch.Tensor): Actions sampled from the buffer.
            targets (torch.Tensor): Precomputed reward return targets, shape ``(B,)``.
        """
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_r, targets) + nn.functional.mse_loss(
            q2_value_r, targets
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            },
        )

    def _update_cost_critic_with_targets(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update the cost critic against precomputed targets (e.g. n-step returns).

        Skips the TD target computation done in the base
        :meth:`_update_cost_critic` and goes straight to the MSE loss.

        Args:
            obs (torch.Tensor): Observations sampled from the buffer.
            action (torch.Tensor): Actions sampled from the buffer.
            targets (torch.Tensor): Precomputed cost return targets, shape ``(B,)``.
        """
        q_value_c = self._actor_critic.cost_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_c, targets)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SACPID is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)
        loss_q_c = self._actor_critic.cost_critic(obs, action)[0]
        loss_c = self._lagrange.lagrangian_multiplier * loss_q_c

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier)

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )
