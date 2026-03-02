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
"""Evaluate learned Q(s,a) against on-rollout Monte Carlo returns.

Run multiple episodes; for each episode use only the **first** state-action pair
(s0, a0) and the empirical discounted return G from that step to the end. Compare
G with Q(s0, a0) from the critic. Use this to judge whether the Q-function is
learning the right values.

Scatter plot: use q_vs_mc_scatter_figure() to get a figure of MC return (G) vs
predicted Q for reward and cost; log with logger.log_figure().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted returns from the end of the episode backward."""
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


def evaluate_q_vs_mc(
    env: Any,
    actor_critic: ConstraintActorQCritic,
    gamma: float = 0.99,
    num_episodes: int = 20,
    max_ep_len: int | None = None,
    use_cost: bool = True,
    device: torch.device | None = None,
    deterministic_policy: bool = False,
) -> dict[str, Any]:
    """Evaluate Q(s,a) against Monte Carlo returns from on-policy rollouts.

    Runs num_episodes full episodes with the current policy. For each episode
    only the **first** state-action pair (s0, a0) is used: compute the empirical
    discounted return G from that step to the end of the episode, then compare
    G with Q(s0, a0) from the learned critic.

    Args:
        env: OmniSafe-style adapter with reset() and step(action). Should expose
            single env (num_envs=1) so that step returns 1-d tensors.
        actor_critic: ConstraintActorQCritic (actor + reward_critic + cost_critic).
        gamma: Discount factor for return computation.
        num_episodes: Number of episodes to roll out.
        max_ep_len: Max steps per episode. If None, use env default (run until done).
        use_cost: Whether to evaluate cost Q as well.
        device: Device for tensors. If None, use actor_critic parameters' device.
        deterministic_policy: If True, use mean action for rollout; else sample.

    Returns:
        Dict with:
        - reward/mae: mean absolute error |Q_r - G_r|
        - reward/correlation: Pearson correlation between Q_r and G_r
        - reward/mean_q, reward/mean_g: means
        - cost/mae, cost/correlation, ... (if use_cost)
        - n_samples: number of (s,a) pairs used
        - q_reward, g_reward, q_cost, g_cost: numpy arrays for custom plotting (if needed)
    """
    if device is None:
        device = next(actor_critic.parameters()).device

    obs_list: list[torch.Tensor] = []
    act_list: list[torch.Tensor] = []
    g_reward_list: list[float] = []
    g_cost_list: list[float] = []

    max_steps = max_ep_len if max_ep_len is not None else getattr(
        env, '_max_ep_len', 1000
    )

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_obs: list[torch.Tensor] = []
        ep_act: list[torch.Tensor] = []
        ep_reward: list[float] = []
        ep_cost: list[float] = []

        for _ in range(max_steps):
            with torch.no_grad():
                act = actor_critic.actor.predict(obs, deterministic=deterministic_policy)
            next_obs, reward, cost, terminated, truncated, _ = env.step(act)

            ep_obs.append(obs)
            ep_act.append(act)
            r_val = reward.mean().item() if reward.numel() > 1 else reward.item()
            c_val = cost.mean().item() if cost.numel() > 1 else cost.item()
            ep_reward.append(r_val)
            ep_cost.append(c_val)

            done = torch.logical_or(terminated, truncated).any().item()
            if done:
                break
            obs = next_obs

        if not ep_obs:
            continue

        G_r = _discounted_returns(ep_reward, gamma)
        G_c = _discounted_returns(ep_cost, gamma)

        # Keep only the first state-action pair and its return (from start of episode)
        obs_list.append(ep_obs[0])
        act_list.append(ep_act[0])
        g_reward_list.append(G_r[0])
        g_cost_list.append(G_c[0])

    if not obs_list:
        return {
            'reward/mae': float('nan'),
            'reward/correlation': float('nan'),
            'reward/mean_q': float('nan'),
            'reward/mean_g': float('nan'),
            'n_samples': 0,
        }

    def _to_batch(x_list: list[torch.Tensor]) -> torch.Tensor:
        out = []
        for x in x_list:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out.append(x)
        return torch.cat(out, dim=0).to(device)

    obs_batch = _to_batch(obs_list)
    act_batch = _to_batch(act_list)
    g_reward = np.array(g_reward_list, dtype=np.float64)
    g_cost = np.array(g_cost_list, dtype=np.float64)

    with torch.no_grad():
        q_r_out = actor_critic.reward_critic(obs_batch, act_batch)
        if isinstance(q_r_out, (list, tuple)):
            q_reward = torch.minimum(q_r_out[0], q_r_out[1]).squeeze(-1).cpu().numpy()
        else:
            q_reward = q_r_out.squeeze(-1).cpu().numpy()
        if use_cost:
            q_cost = actor_critic.cost_critic(obs_batch, act_batch)[0].squeeze(-1).cpu().numpy()

    n = len(g_reward_list)
    result: dict[str, Any] = {
        'reward/mae': float(np.abs(q_reward - g_reward).mean()),
        'reward/correlation': float(np.corrcoef(q_reward, g_reward)[0, 1]) if n > 1 and np.std(q_reward) > 0 and np.std(g_reward) > 0 else float('nan'),
        'reward/mean_q': float(q_reward.mean()),
        'reward/mean_g': float(g_reward.mean()),
        'n_samples': n,
        'q_reward': q_reward,
        'g_reward': g_reward,
    }
    if use_cost:
        result['cost/mae'] = float(np.abs(q_cost - g_cost).mean())
        result['cost/correlation'] = float(np.corrcoef(q_cost, g_cost)[0, 1]) if n > 1 and np.std(q_cost) > 0 and np.std(g_cost) > 0 else float('nan')
        result['cost/mean_q'] = float(q_cost.mean())
        result['cost/mean_g'] = float(g_cost.mean())
        result['q_cost'] = q_cost
        result['g_cost'] = g_cost
    return result


def q_vs_mc_scatter_figure(
    env: Any,
    actor_critic: ConstraintActorQCritic,
    gamma: float = 0.99,
    num_episodes: int = 20,
    max_ep_len: int | None = None,
    use_cost: bool = True,
    device: torch.device | None = None,
    deterministic_policy: bool = False,
    max_points: int = 500,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Scatter plot: MC return (G) vs predicted Q(s,a) for reward and cost.

    Runs the same on-policy rollouts as evaluate_q_vs_mc (first (s,a) per episode
    only), then builds a figure with two subplots: (1) G_reward vs Q_reward,
    (2) G_cost vs Q_cost. Points along the diagonal indicate the Q-function
    matches the empirical return.

    Args:
        env: Adapter with reset() and step(action).
        actor_critic: ConstraintActorQCritic.
        gamma: Discount for return computation.
        num_episodes: Number of episodes to roll out.
        max_ep_len: Max steps per episode.
        use_cost: Include cost scatter (second subplot).
        device: Device for tensors.
        deterministic_policy: Use mean action in rollouts if True.
        max_points: Subsample to this many points for the plot if more collected.

    Returns:
        figure: Matplotlib figure with 1 or 2 subplots (reward; reward+cost if use_cost).
        stats: Same dict as evaluate_q_vs_mc (reward/mae, reward/correlation, ...).
    """
    stats = evaluate_q_vs_mc(
        env=env,
        actor_critic=actor_critic,
        gamma=gamma,
        num_episodes=num_episodes,
        max_ep_len=max_ep_len,
        use_cost=use_cost,
        device=device,
        deterministic_policy=deterministic_policy,
    )
    n = stats['n_samples']
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
        return fig, stats

    g_reward = stats['g_reward']
    q_reward = stats['q_reward']
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        g_reward = g_reward[idx]
        q_reward = q_reward[idx]
    if use_cost:
        g_cost = stats['g_cost']
        q_cost = stats['q_cost']
        if stats['n_samples'] > max_points:
            g_cost = g_cost[idx]
            q_cost = q_cost[idx]

    ncols = 2 if use_cost else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Reward: scatter G (x) vs Q (y)
    ax = axes[0]
    ax.scatter(g_reward, q_reward, alpha=0.5, s=12)
    lo = min(g_reward.min(), q_reward.min())
    hi = max(g_reward.max(), q_reward.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, label='Q=G')
    ax.set_xlabel('MC return (G) reward')
    ax.set_ylabel('Q(s,a) reward')
    ax.set_title('Reward: MC return vs predicted Q')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    if use_cost:
        ax = axes[1]
        ax.scatter(g_cost, q_cost, alpha=0.5, s=12)
        lo = min(g_cost.min(), q_cost.min())
        hi = max(g_cost.max(), q_cost.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, label='Q=G')
        ax.set_xlabel('MC return (G) cost')
        ax.set_ylabel('Q(s,a) cost')
        ax.set_title('Cost: MC return vs predicted Q')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig, stats
