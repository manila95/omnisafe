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
"""Monte-Carlo evaluation of a critic against the return it is meant to predict.

Two studies, differing only in which states they probe: :func:`estimate_true_value_same_state_mc`
at episode starts (reproduced by re-seeding) and :func:`estimate_value_from_snapshots` at
on-policy mid-episode states (reproduced by simulator snapshot/restore). Both re-roll the same
state ``mc_repeats`` times, so the sample mean is a genuine estimate of V^pi with a measurable
variance. Metrics live in :mod:`omnisafe.utils.value_utils`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from omnisafe.utils.value_utils import (
    calibration_stats,
    effective_rollout_horizon,
    rollout_target,
    sync_obs_normalizer,
    to_tensor,
)


def _estimator_cfg(cfgs) -> tuple:
    """``(adv_r, adv_c, lam_r, lam_c, penalty_coef)`` as training itself reads them."""
    adv_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    return (
        adv_r,
        getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_r,
        lam_r,
        getattr(cfgs.algo_cfgs, 'lam_c', lam_r),
        getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0),
    )


def _resolve_horizon(full_horizon, discount_r, discount_c, bootstrap_threshold, tail_mode, who):
    """Truncated horizon and whether the tail is bootstrapped.

    Raises:
        ValueError: If the horizon is truncated without an explicit ``tail_mode``.
    """
    horizon = effective_rollout_horizon(full_horizon, discount_r, discount_c, bootstrap_threshold)
    if horizon < full_horizon and tail_mode not in ('bootstrap', 'drop'):
        raise ValueError(
            f'{who}: bootstrap_threshold truncates the rollout at {horizon} of {full_horizon} '
            "steps, so the tail must be handled explicitly. tail_mode='drop' stops accumulating "
            '(the value is the exact truncated discounted sum, low by at most the threshold of '
            "its own scale, and still simulation-only); tail_mode='bootstrap' estimates it with "
            "the critic's own value, which is the quantity these studies exist to validate. "
            'bootstrap_threshold=None keeps the full horizon.',
        )
    return horizon, tail_mode == 'bootstrap'


def _roll_probes(
    agent,
    env,
    tasks: list,
    begin_wave: Callable[[list], torch.Tensor],
    horizon: int,
    full_horizon: int,
    discount_r: float,
    discount_c: float,
    bootstrap_tail: bool,
    estimators: tuple,
    who: str,
) -> dict:
    """Roll every task out in waves of ``env.num_envs`` and accumulate per-task results.

    The only thing the two studies differ in is ``begin_wave``, which puts the env into each
    wave's start states and returns the batched observation.

    Assumes every episode ends at ``full_horizon`` uniformly across the batch, which is checked
    after each wave; an env that terminates early would need a per-slot done mask instead.

    Args:
        agent: Actor-critic exposing ``step(obs) -> (act, value_r, value_c, logp)``.
        env: Already-wrapped env.
        tasks: One entry per rollout (``probes * mc_repeats``), in probe-major order.
        begin_wave: ``(wave_tasks) -> obs``, padded to ``num_envs``.
        horizon: Steps actually simulated.
        full_horizon: Untruncated horizon, for the terminal check.
        discount_r: Reward discount.
        discount_c: Cost discount.
        bootstrap_tail: Add the critic's value at the truncation point to the return.
        estimators: From :func:`_estimator_cfg`.
        who: Caller name, for error messages.

    Returns:
        Per-task lists keyed ``pred_r``/``pred_c``/``obs``/``act``/``ret_r``/``ret_c``/
        ``target_r``/``target_c``.

    Raises:
        RuntimeError: If not every probe reached ``done`` at ``full_horizon``.
    """
    adv_r, adv_c, lam_r, lam_c, penalty_coef = estimators
    n_envs = env.num_envs
    n_tasks = len(tasks)
    out = {k: [None] * n_tasks for k in ('pred_r', 'pred_c', 'obs', 'act')}
    out.update({k: [0.0] * n_tasks for k in ('ret_r', 'ret_c', 'target_r', 'target_c')})

    for start in range(0, n_tasks, n_envs):
        idxs = list(range(start, min(start + n_envs, n_tasks)))
        wave = [tasks[i] for i in idxs]
        if len(wave) < n_envs:
            wave += [wave[0]] * (n_envs - len(wave))

        obs = begin_wave(wave)
        act, value_r, value_c, log_prob = agent.step(obs)
        pred_r = value_r.reshape(-1).detach().cpu().numpy()
        pred_c = value_c.reshape(-1).detach().cpu().numpy()
        obs_s0, act_a0 = obs.detach().cpu().clone(), act.detach().cpu().clone()

        r_seq = np.zeros((horizon + 1, n_envs))
        c_seq = np.zeros((horizon + 1, n_envs))
        v_r_seq = np.zeros((horizon + 1, n_envs))
        v_c_seq = np.zeros((horizon + 1, n_envs))
        logp_seq = np.zeros((horizon, n_envs))
        v_r_seq[0], v_c_seq[0] = pred_r, pred_c

        g_r = np.zeros(n_envs)
        g_c = np.zeros(n_envs)
        disc_r = disc_c = 1.0
        terminated = truncated = None
        for t in range(horizon):
            obs, r, c, terminated, truncated, _ = env.step(act)
            # Accumulate from the float32 arrays, not the float64 rows they are stored into:
            # numpy's value-based casting keeps `disc * r_np` in float32, and matching that
            # exactly is what keeps these returns bit-identical to the pre-refactor ones.
            r_np = r.reshape(-1).detach().cpu().numpy()
            c_np = c.reshape(-1).detach().cpu().numpy()
            r_seq[t], c_seq[t] = r_np, c_np
            g_r += disc_r * r_np
            g_c += disc_c * c_np
            disc_r *= discount_r
            disc_c *= discount_c
            if t < horizon - 1:
                act, value_r, value_c, log_prob = agent.step(obs)
                v_r_seq[t + 1] = value_r.reshape(-1).detach().cpu().numpy()
                v_c_seq[t + 1] = value_c.reshape(-1).detach().cpu().numpy()
                logp_seq[t] = log_prob.reshape(-1).detach().cpu().numpy()

        if horizon == full_horizon:
            done = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
            if not bool(done.all()):
                raise RuntimeError(
                    f'{who}: not every probe reached done at horizon={full_horizon} '
                    f'(got done={done.tolist()}). This assumes homogeneous, fixed-length '
                    'episodes across the batch; an env that terminates early needs a per-slot '
                    'done mask added here.',
                )
            is_terminated = terminated.reshape(-1).bool().detach().cpu().numpy()
        else:
            is_terminated = np.zeros(n_envs, dtype=bool)

        _, boot_r_t, boot_c_t, _ = agent.step(obs)
        boot_r = np.where(is_terminated, 0.0, boot_r_t.reshape(-1).detach().cpu().numpy())
        boot_c = np.where(is_terminated, 0.0, boot_c_t.reshape(-1).detach().cpu().numpy())
        v_r_seq[horizon], v_c_seq[horizon] = boot_r, boot_c
        if horizon < full_horizon and bootstrap_tail:
            g_r += disc_r * boot_r
            g_c += disc_c * boot_c

        for local, task_idx in enumerate(idxs):
            out['pred_r'][task_idx] = float(pred_r[local])
            out['pred_c'][task_idx] = float(pred_c[local])
            out['obs'][task_idx] = obs_s0[local].clone()
            out['act'][task_idx] = act_a0[local].clone()
            out['ret_r'][task_idx] = float(g_r[local])
            out['ret_c'][task_idx] = float(g_c[local])

            r_this = torch.from_numpy(r_seq[:horizon, local]).float()
            c_this = torch.from_numpy(c_seq[:horizon, local]).float()
            logp_this = torch.from_numpy(logp_seq[:, local]).float()
            term = bool(is_terminated[local])
            out['target_r'][task_idx] = rollout_target(
                r_this - penalty_coef * c_this,
                torch.from_numpy(v_r_seq[:, local]).float(),
                term, lam_r, discount_r, adv_r, logp_seq=logp_this,
            )
            out['target_c'][task_idx] = rollout_target(
                c_this,
                torch.from_numpy(v_c_seq[:, local]).float(),
                term, lam_c, discount_c, adv_c, logp_seq=logp_this,
            )
    return out


def _probe_stats(per_task: dict, n_probes: int, mc_repeats: int, return_raw: bool):
    """Group per-task rollouts back into probes and score them.

    ``per_task`` is in probe-major order, so probe ``p`` owns the ``mc_repeats`` consecutive
    entries starting at ``p * mc_repeats``.
    """
    cols: dict = {k: [] for k in (
        'pred_r', 'pred_c', 'mc_mean_r', 'mc_mean_c', 'mc_var_r', 'mc_var_c',
        'target_r', 'target_c', 'returns_r', 'returns_c',
        'target_repeats_r', 'target_repeats_c', 'obs', 'action',
    )}
    for p in range(n_probes):
        idxs = range(p * mc_repeats, (p + 1) * mc_repeats)
        cols['pred_r'].append(per_task['pred_r'][idxs[0]])
        cols['pred_c'].append(per_task['pred_c'][idxs[0]])
        cols['obs'].append([per_task['obs'][i] for i in idxs])
        cols['action'].append([per_task['act'][i] for i in idxs])
        for stream in ('r', 'c'):
            rets = [per_task[f'ret_{stream}'][i] for i in idxs]
            tgts = [per_task[f'target_{stream}'][i] for i in idxs]
            cols[f'mc_mean_{stream}'].append(float(np.mean(rets)))
            cols[f'mc_var_{stream}'].append(float(np.var(rets)))
            cols[f'returns_{stream}'].append(rets)
            cols[f'target_{stream}'].append(float(np.mean(tgts)))
            cols[f'target_repeats_{stream}'].append(tgts)

    stats: dict = {}
    for stream in ('r', 'c'):
        stats.update({
            f'{k}_{stream}': v
            for k, v in calibration_stats(
                to_tensor(cols[f'pred_{stream}']),
                to_tensor(cols[f'mc_mean_{stream}']),
                to_tensor(cols[f'target_{stream}']),
                returns=cols[f'returns_{stream}'],
            ).items()
        })
    if not return_raw:
        return stats
    raw = {
        'obs': cols['obs'],
        'action': cols['action'],
        **{
            stream: {
                'pred': cols[f'pred_{stream}'],
                'mc_mean': cols[f'mc_mean_{stream}'],
                'mc_var': cols[f'mc_var_{stream}'],
                'target': cols[f'target_{stream}'],
                'returns': cols[f'returns_{stream}'],
                'target_repeats': cols[f'target_repeats_{stream}'],
            }
            for stream in ('r', 'c')
        },
    }
    return stats, raw


def estimate_true_value_same_state_mc(
    agent,
    env,
    cfgs,
    discount_r,
    discount_c,
    probe_seeds,
    mc_repeats=5,
    epoch=None,
    sync_normalizer_from=None,
    max_episode_steps=None,
    return_raw=False,
    bootstrap_threshold=None,
    tail_mode=None,
):
    """Score V(s0) against a same-layout Monte-Carlo estimate.

    A Safety-Gymnasium layout is reproduced exactly by ``reset(seed=X)``, so each probe seed can
    be re-rolled ``mc_repeats`` times under the current stochastic policy.

    Args:
        agent: Actor-critic exposing ``step(obs) -> (act, value_r, value_c, logp)``.
        env: Already-wrapped env, any ``num_envs >= 1``.
        cfgs: Resolved algorithm config.
        discount_r: Reward discount.
        discount_c: Cost discount.
        probe_seeds: Fixed layouts to probe, held constant across eval epochs.
        mc_repeats: Independent rollouts averaged per probe.
        epoch: Unused; accepted for call-site symmetry.
        sync_normalizer_from: Env whose observation statistics to snapshot before probing.
        max_episode_steps: Episode horizon; read from ``env`` when omitted.
        return_raw: Also return the per-probe arrays.
        bootstrap_threshold: Truncate once the discounted tail falls below this.
        tail_mode: ``'drop'`` or ``'bootstrap'``; required when truncating.

    Returns:
        ``stats``, or ``(stats, raw)`` when ``return_raw``.
    """
    del epoch
    if sync_normalizer_from is not None:
        sync_obs_normalizer(env, sync_normalizer_from)
    if max_episode_steps is None:
        max_episode_steps = env.max_episode_steps
    assert max_episode_steps and max_episode_steps > 0

    horizon, bootstrap_tail = _resolve_horizon(
        max_episode_steps, discount_r, discount_c, bootstrap_threshold, tail_mode,
        'estimate_true_value_same_state_mc',
    )
    n_envs = env.num_envs

    def begin_wave(seeds: list) -> torch.Tensor:
        obs, _ = env.reset(seed=seeds if n_envs > 1 else seeds[0])
        return obs

    per_task = _roll_probes(
        agent, env,
        tasks=[seed for seed in probe_seeds for _ in range(mc_repeats)],
        begin_wave=begin_wave,
        horizon=horizon, full_horizon=max_episode_steps,
        discount_r=discount_r, discount_c=discount_c,
        bootstrap_tail=bootstrap_tail, estimators=_estimator_cfg(cfgs),
        who='estimate_true_value_same_state_mc',
    )
    result = _probe_stats(per_task, len(probe_seeds), mc_repeats, return_raw)
    if return_raw:
        result[1]['probe_seeds'] = list(probe_seeds)
    return result


def estimate_value_from_snapshots(
    agent,
    env,
    cfgs,
    discount_r,
    discount_c,
    snapshots,
    horizon,
    mc_repeats=5,
    epoch=None,
    return_raw=False,
    bootstrap_threshold=None,
    tail_mode=None,
):
    """Score V(s) at on-policy mid-episode states, restored from simulator snapshots.

    Same protocol as :func:`estimate_true_value_same_state_mc`, but probes come from
    :mod:`omnisafe.utils.state_snapshot` rather than ``reset(seed=X)``, which cannot reach them.
    Each restored state is treated as a fresh start and given the full ``horizon``, so the
    question asked is the same one asked of s0.

    Args:
        agent: Actor-critic exposing ``step(obs) -> (act, value_r, value_c, logp)``.
        env: Already-wrapped vectorized env.
        cfgs: Resolved algorithm config.
        discount_r: Reward discount.
        discount_c: Cost discount.
        snapshots: Pre-captured states, one per probe.
        horizon: Steps to roll each probe out for.
        mc_repeats: Independent rollouts averaged per probe.
        epoch: Unused; accepted for call-site symmetry.
        return_raw: Also return the per-probe arrays.
        bootstrap_threshold: Truncate once the discounted tail falls below this.
        tail_mode: ``'drop'`` or ``'bootstrap'``; required when truncating.

    Returns:
        ``stats``, or ``(stats, raw)`` when ``return_raw``.
    """
    from omnisafe.utils.state_snapshot import restore_and_get_obs  # noqa: PLC0415

    del epoch
    assert horizon and horizon > 0
    device = torch.device(cfgs.train_cfgs.device)
    nominal_horizon = horizon
    horizon, bootstrap_tail = _resolve_horizon(
        nominal_horizon, discount_r, discount_c, bootstrap_threshold, tail_mode,
        'estimate_value_from_snapshots',
    )

    def begin_wave(snaps: list) -> torch.Tensor:
        return restore_and_get_obs(env, snaps, device, reset_elapsed_steps=True)

    per_task = _roll_probes(
        agent, env,
        tasks=[snap for snap in snapshots for _ in range(mc_repeats)],
        begin_wave=begin_wave,
        horizon=horizon, full_horizon=nominal_horizon,
        discount_r=discount_r, discount_c=discount_c,
        bootstrap_tail=bootstrap_tail, estimators=_estimator_cfg(cfgs),
        who='estimate_value_from_snapshots',
    )
    return _probe_stats(per_task, len(snapshots), mc_repeats, return_raw)
