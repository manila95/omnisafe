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


import torch

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
        self._lagrange: PIDLagrangian = PIDLagrangian(**self._cfgs.lagrange_cfgs)
        self._last_q_diag_epoch: int = -1

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

        # Q-critic calibration keys (replay-buffer pred vs Bellman target)
        self._logger.register_key('Value/QRewardCriticCorr')
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.register_key('Value/QCostCriticCorr')

        # Q-critic calibration keys (eval-rollout pred vs true return)
        self._logger.register_key('Value/QRewardTrueReturnCorr')
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.register_key('Value/QCostTrueReturnCorr')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`pid_update` method.
        """
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.pid_update(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )

        # Log Q-critic diagnostics once per epoch (at the first _update() call of each new epoch)
        if self._epoch != self._last_q_diag_epoch:
            self._last_q_diag_epoch = self._epoch
            eval_freq = getattr(self._cfgs.algo_cfgs, 'value_eval_freq', 10)
            if self._epoch % eval_freq == 0:
                self._log_q_critic_diagnostics()

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

    # ------------------------------------------------------------------
    # Q-critic calibration diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_sac_q_targets(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SAC Bellman targets and current Q predictions for a batch.

        Returns:
            q_pred_r: min(Q1, Q2)(obs, act)  — reward critic prediction
            q_target_r: SAC Bellman target for reward critic
            q_pred_c: Q_c(obs, act)  — cost critic prediction (or zeros if not use_cost)
            q_target_c: Bellman target for cost critic (or zeros)
        """
        gamma = self._cfgs.algo_cfgs.gamma

        next_act = self._actor_critic.actor.predict(next_obs, deterministic=False)
        next_logp = self._actor_critic.actor.log_prob(next_act)

        # reward critic
        next_q1_r, next_q2_r = self._actor_critic.target_reward_critic(next_obs, next_act)
        next_q_r = torch.min(next_q1_r, next_q2_r) - self._alpha * next_logp
        q_target_r = reward + gamma * (1.0 - done) * next_q_r

        q1_r, q2_r = self._actor_critic.reward_critic(obs, act)
        q_pred_r = torch.min(q1_r, q2_r)

        if self._cfgs.algo_cfgs.use_cost:
            next_q_c = self._actor_critic.target_cost_critic(next_obs, next_act)[0]
            q_target_c = cost + gamma * (1.0 - done) * next_q_c
            q_pred_c = self._actor_critic.cost_critic(obs, act)[0]
        else:
            q_target_c = torch.zeros_like(q_target_r)
            q_pred_c = torch.zeros_like(q_pred_r)

        return q_pred_r, q_target_r, q_pred_c, q_target_c

    def _q_eval_rollout(
        self, n_episodes: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Roll out ``n_episodes`` and score every (s_t, a_t) against its discounted-to-go,
        not just the episode-start pair.

        Uses ``self._env`` — the *same* (wrapped, reward/cost-normalized) environment used to
        collect the replay-buffer transitions the critic is trained on — rather than a separate
        eval env. A separate eval env skips the ``RewardNormalize``/``CostNormalize`` wrappers,
        which would put true returns on a different scale than what the critic was trained to
        predict, corrupting the comparison. We only track index 0 of the vectorized env as a
        single serial trajectory; the other parallel envs are stepped in lockstep but their data
        is discarded. The adapter's ``_current_obs`` bookkeeping is restored afterward so the
        next real training rollout isn't disrupted.

        For each step we record ``Q(s_t, a_t)`` under the live critic, then once the episode
        ends we backfill the true discounted-to-go ``G_t = r_t + gamma * G_{t+1}`` via backward
        recursion. On true termination the recursion is seeded with 0; on timeout (truncated,
        not terminated) it is seeded with the critic's own bootstrap estimate at the true final
        observation (``info['final_observation']``, since the env auto-resets) — the same
        convention the on-policy buffer uses for ``discounted_ret``.

        Returns:
            q_r: Q_r(s_t, a_t) for every visited step across all episodes
            q_c: Q_c(s_t, a_t) for every visited step, or None
            G_r: true discounted reward-to-go for every visited step
            G_c: true discounted cost-to-go for every visited step, or None
        """
        gamma = self._cfgs.algo_cfgs.gamma
        use_cost = self._cfgs.algo_cfgs.use_cost
        all_q_r, all_q_c, all_G_r, all_G_c = [], [], [], []

        for _ in range(n_episodes):
            obs, _ = self._env.reset()
            obs = obs.to(self._device)

            ep_q_r, ep_q_c, ep_rewards, ep_costs = [], [], [], []
            done = False
            terminated_flag = False
            final_obs = None
            while not done:
                with torch.no_grad():
                    act = self._actor_critic.actor.predict(obs, deterministic=True)
                    q_r = torch.min(*self._actor_critic.reward_critic(obs, act))[0].squeeze()
                    q_c = (
                        self._actor_critic.cost_critic(obs, act)[0][0].squeeze()
                        if use_cost
                        else None
                    )

                next_obs, reward, cost, terminated, truncated, info = self._env.step(act)

                ep_q_r.append(q_r.item())
                ep_rewards.append(float(reward[0].item()))
                if use_cost:
                    ep_q_c.append(q_c.item())
                    ep_costs.append(float(cost[0].item()))

                terminated_flag = bool(terminated[0].item())
                done = terminated_flag or bool(truncated[0].item())
                if done and 'final_observation' in info:
                    final_obs = info['final_observation'][0].unsqueeze(0).to(self._device)
                obs = next_obs.to(self._device)

            # bootstrap at timeout, zero at true termination (mirrors on-policy convention)
            bootstrap_r, bootstrap_c = 0.0, 0.0
            if not terminated_flag and final_obs is not None:
                with torch.no_grad():
                    f_act = self._actor_critic.actor.predict(final_obs, deterministic=True)
                    bootstrap_r = torch.min(
                        *self._actor_critic.target_reward_critic(final_obs, f_act),
                    ).item()
                    if use_cost:
                        bootstrap_c = self._actor_critic.target_cost_critic(final_obs, f_act)[
                            0
                        ].item()

            G_r = bootstrap_r
            ep_G_r = [0.0] * len(ep_rewards)
            for t in reversed(range(len(ep_rewards))):
                G_r = ep_rewards[t] + gamma * G_r
                ep_G_r[t] = G_r
            all_q_r.extend(ep_q_r)
            all_G_r.extend(ep_G_r)

            if use_cost:
                G_c = bootstrap_c
                ep_G_c = [0.0] * len(ep_costs)
                for t in reversed(range(len(ep_costs))):
                    G_c = ep_costs[t] + gamma * G_c
                    ep_G_c[t] = G_c
                all_q_c.extend(ep_q_c)
                all_G_c.extend(ep_G_c)

        # restore the adapter's bookkeeping so the next real training rollout() is consistent
        self._env._current_obs, _ = self._env.reset()  # noqa: SLF001

        q_r = torch.tensor(all_q_r)
        G_r = torch.tensor(all_G_r)
        q_c = torch.tensor(all_q_c) if use_cost else None
        G_c = torch.tensor(all_G_c) if use_cost else None
        return q_r, q_c, G_r, G_c

    def _log_q_critic_diagnostics(self) -> None:
        """Log Q-critic calibration scatter plots and Pearson correlations.

        Two diagnostics are logged:

        1. **Replay buffer** — Q_pred(s,a) vs SAC Bellman target Q_target(s,a).
           Uses a large random sample from the replay buffer.

        2. **Eval rollout** — Q_pred(s_t, a_t) vs true discounted-to-go G_t, scored at every
           visited step (not just episode start).  Tells us how calibrated the critic is
           against true returns (not just bootstrap targets).
        """
        diag_batch = getattr(self._cfgs.algo_cfgs, 'q_diag_batch_size', 4096)
        diag_episodes = getattr(self._cfgs.algo_cfgs, 'q_diag_eval_episodes', 20)

        # ---- 1. Replay-buffer pred vs Bellman target ----
        data = self._buf.sample_batch()
        # Grab a larger sample by calling sample_batch multiple times if needed
        obs_list, act_list, rew_list, cost_list, done_list, nobs_list = (
            [data['obs']], [data['act']], [data['reward']],
            [data['cost']], [data['done']], [data['next_obs']],
        )
        while sum(o.shape[0] for o in obs_list) < diag_batch:
            d = self._buf.sample_batch()
            obs_list.append(d['obs']); act_list.append(d['act'])
            rew_list.append(d['reward']); cost_list.append(d['cost'])
            done_list.append(d['done']); nobs_list.append(d['next_obs'])

        obs  = torch.cat(obs_list)[:diag_batch]
        act  = torch.cat(act_list)[:diag_batch]
        rew  = torch.cat(rew_list)[:diag_batch]
        cost = torch.cat(cost_list)[:diag_batch]
        done = torch.cat(done_list)[:diag_batch]
        nobs = torch.cat(nobs_list)[:diag_batch]

        q_pred_r, q_target_r, q_pred_c, q_target_c = self._compute_sac_q_targets(
            obs, act, rew, cost, done, nobs,
        )

        n = q_pred_r.shape[0]
        idx = torch.randperm(n)[:min(n, 2000)]

        corr_r = torch.corrcoef(torch.stack([q_pred_r, q_target_r]))[0, 1].item()
        self._logger.store({'Value/QRewardCriticCorr': corr_r})
        self._logger.log_scatter_image(
            'Value/QRewardCriticScatter',
            x_values=q_target_r[idx],
            y_values=q_pred_r[idx],
            xlabel='Q target (Bellman)',
            ylabel='Q predicted',
        )

        if self._cfgs.algo_cfgs.use_cost:
            corr_c = torch.corrcoef(torch.stack([q_pred_c, q_target_c]))[0, 1].item()
            self._logger.store({'Value/QCostCriticCorr': corr_c})
            self._logger.log_scatter_image(
                'Value/QCostCriticScatter',
                x_values=q_target_c[idx],
                y_values=q_pred_c[idx],
                xlabel='Q_c target (Bellman)',
                ylabel='Q_c predicted',
            )

        # ---- 2. Eval rollout pred vs true discounted-to-go (every visited step) ----
        q_r_eval, q_c_eval, G_r_eval, G_c_eval = self._q_eval_rollout(diag_episodes)

        if q_r_eval.numel() > 1:
            eval_idx = torch.randperm(q_r_eval.shape[0])[:min(q_r_eval.shape[0], 2000)]
            corr_true_r = torch.corrcoef(torch.stack([q_r_eval, G_r_eval]))[0, 1].item()
            self._logger.store({'Value/QRewardTrueReturnCorr': corr_true_r})
            self._logger.log_scatter_image(
                'Value/QRewardTrueReturnScatter',
                x_values=G_r_eval[eval_idx],
                y_values=q_r_eval[eval_idx],
                xlabel='True G_r (reward-to-go)',
                ylabel='Q_r(s_t, a_t)',
            )

        if self._cfgs.algo_cfgs.use_cost and q_c_eval is not None and q_c_eval.numel() > 1:
            corr_true_c = torch.corrcoef(torch.stack([q_c_eval, G_c_eval]))[0, 1].item()
            self._logger.store({'Value/QCostTrueReturnCorr': corr_true_c})
            self._logger.log_scatter_image(
                'Value/QCostTrueReturnScatter',
                x_values=G_c_eval[eval_idx],
                y_values=q_c_eval[eval_idx],
                xlabel='True G_c (cost-to-go)',
                ylabel='Q_c(s_t, a_t)',
            )
