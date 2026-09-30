"""Monte-Carlo evaluation of a critic against the return it is meant to predict.

Two studies, differing only in which states they probe: :func:`estimate_true_value_same_state_mc`
at episode starts (reproduced by re-seeding) and :func:`estimate_value_from_snapshots` at
on-policy mid-episode states (reproduced by simulator snapshot/restore). Both re-roll the same
state ``mc_repeats`` times, so the sample mean is a genuine estimate of V^pi with a measurable
variance. Metrics live in :mod:`omnisafe.utils.value_utils`.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from rich.progress import Progress

from omnisafe.utils.value_utils import (
    calibration_stats,
    effective_rollout_horizon,
    rollout_target,
    sync_obs_normalizer,
    to_tensor,
)


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
    be re-rolled ``mc_repeats`` times under the current stochastic policy. ``env`` may be
    vectorized: the rollouts are packed into waves of ``num_envs``. Assumes every episode ends at
    ``max_episode_steps``, which is asserted after each wave.

    Args:
        agent: Actor-critic exposing ``step(obs) -> (act, value_r, value_c, logp)``.
        env: Already-wrapped env, any ``num_envs >= 1``.
        cfgs: Resolved algorithm config.
        discount_r: Reward discount.
        discount_c: Cost discount.
        probe_seeds: Fixed layouts to probe, held constant across eval epochs.
        mc_repeats: Independent rollouts averaged per probe.
        epoch: For the progress bar only.
        sync_normalizer_from: Env whose observation statistics to snapshot before probing.
        max_episode_steps: Episode horizon; read from ``env`` when omitted.
        return_raw: Also return the per-probe arrays.
        bootstrap_threshold: Truncate once the discounted tail falls below this.
        tail_mode: ``'drop'`` or ``'bootstrap'``; required when truncating.

    Returns:
        ``stats``, or ``(stats, raw)`` when ``return_raw``.

    Raises:
        ValueError: If truncating without a ``tail_mode``.
    """
    if sync_normalizer_from is not None:
        sync_obs_normalizer(env, sync_normalizer_from)
    device = torch.device(cfgs.train_cfgs.device)

    n_envs = env.num_envs
    if max_episode_steps is None:
        assert n_envs == 1, (
            'max_episode_steps must be passed explicitly when env.num_envs > 1 (a vectorized '
            'env has no .spec to read it off).'
        )
        max_episode_steps = env.max_episode_steps
    assert max_episode_steps and max_episode_steps > 0
    horizon = effective_rollout_horizon(max_episode_steps, discount_r, discount_c, bootstrap_threshold)
    bootstrap_tail = tail_mode == 'bootstrap'
    if horizon < max_episode_steps and tail_mode not in ('bootstrap', 'drop'):
        raise ValueError(
            'estimate_true_value_same_state_mc: bootstrap_threshold truncates the rollout at '
            f'{horizon} of {max_episode_steps} steps, so the tail must be handled explicitly. '
            "Set tail_mode='drop' (stop accumulating; the value is the exact truncated "
            'discounted sum, low by at most bootstrap_threshold of its own scale, and still a '
            "simulation-only quantity) or tail_mode='bootstrap' (estimate the tail with the "
            "critic's own value -- note that is the quantity these studies exist to validate, "
            'and an untrained V_c makes cost returns negative). bootstrap_threshold=None keeps '
            'the full horizon with no approximation.',
        )

    adv_estimator_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    adv_estimator_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_estimator_r
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
    penalty_coef = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

    tasks = [seed for seed in probe_seeds for _ in range(mc_repeats)]
    n_tasks = len(tasks)
    pred_r_of: list[float | None] = [None] * n_tasks
    pred_c_of: list[float | None] = [None] * n_tasks
    obs_of: list[torch.Tensor | None] = [None] * n_tasks
    act_of: list[torch.Tensor | None] = [None] * n_tasks
    ret_r_of: list[float] = [0.0] * n_tasks
    ret_c_of: list[float] = [0.0] * n_tasks
    target_r_of: list[float] = [0.0] * n_tasks
    target_c_of: list[float] = [0.0] * n_tasks

    n_waves = (n_tasks + n_envs - 1) // n_envs
    for w in range(n_waves):
        start = w * n_envs
        wave_task_idxs = list(range(start, min(start + n_envs, n_tasks)))
        wave_size = len(wave_task_idxs)
        wave_seeds = [tasks[i] for i in wave_task_idxs]
        if wave_size < n_envs:
            wave_seeds += [wave_seeds[0]] * (n_envs - wave_size)

        obs, _ = env.reset(seed=wave_seeds if n_envs > 1 else wave_seeds[0])
        act, value_r, value_c, log_prob = agent.step(obs)
        pred_r_batch = value_r.reshape(-1).detach().cpu().numpy()
        pred_c_batch = value_c.reshape(-1).detach().cpu().numpy()
        obs_s0_cpu = obs.detach().cpu().clone()
        act_a0_cpu = act.detach().cpu().clone()

        r_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        c_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        v_r_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        v_c_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        logp_seq = np.zeros((horizon, n_envs), dtype=np.float64)
        v_r_seq[0] = pred_r_batch
        v_c_seq[0] = pred_c_batch

        g_r = np.zeros(n_envs, dtype=np.float64)
        g_c = np.zeros(n_envs, dtype=np.float64)
        disc_r, disc_c = 1.0, 1.0
        terminated = truncated = None
        for t in range(horizon):
            obs, r, c, terminated, truncated, _ = env.step(act)
            r_np = r.reshape(-1).detach().cpu().numpy()
            c_np = c.reshape(-1).detach().cpu().numpy()
            r_seq[t] = r_np
            c_seq[t] = c_np
            g_r += disc_r * r_np
            g_c += disc_c * c_np
            disc_r *= discount_r
            disc_c *= discount_c
            if t < horizon - 1:
                act, value_r, value_c, log_prob = agent.step(obs)
                v_r_seq[t + 1] = value_r.reshape(-1).detach().cpu().numpy()
                v_c_seq[t + 1] = value_c.reshape(-1).detach().cpu().numpy()
                logp_seq[t] = log_prob.reshape(-1).detach().cpu().numpy()

        if horizon == max_episode_steps:
            done_at_end = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
            if not bool(done_at_end.all()):
                raise RuntimeError(
                    'estimate_true_value_same_state_mc: not every probe reached done at '
                    f'max_episode_steps={max_episode_steps} (got done={done_at_end.tolist()}). '
                    'This function assumes homogeneous, fixed-length episodes across the '
                    "vectorized batch (see docstring) -- this environment doesn't satisfy that "
                    'and needs a per-slot done mask added instead.',
                )
            is_terminated = terminated.reshape(-1).bool().detach().cpu().numpy()
        else:
            is_terminated = np.zeros(n_envs, dtype=bool)

        _, boot_value_r, boot_value_c, _ = agent.step(obs)
        boot_r = np.where(is_terminated, 0.0, boot_value_r.reshape(-1).detach().cpu().numpy())
        boot_c = np.where(is_terminated, 0.0, boot_value_c.reshape(-1).detach().cpu().numpy())
        v_r_seq[horizon] = boot_r
        v_c_seq[horizon] = boot_c
        if horizon < max_episode_steps and bootstrap_tail:
            g_r += disc_r * boot_r
            g_c += disc_c * boot_c

        for local_i, task_idx in enumerate(wave_task_idxs):
            pred_r_of[task_idx] = float(pred_r_batch[local_i])
            pred_c_of[task_idx] = float(pred_c_batch[local_i])
            obs_of[task_idx] = obs_s0_cpu[local_i].clone()
            act_of[task_idx] = act_a0_cpu[local_i].clone()
            ret_r_of[task_idx] = float(g_r[local_i])
            ret_c_of[task_idx] = float(g_c[local_i])

            r_this = torch.from_numpy(r_seq[:horizon, local_i]).float()
            c_this = torch.from_numpy(c_seq[:horizon, local_i]).float()
            v_r_this = torch.from_numpy(v_r_seq[:, local_i]).float()
            v_c_this = torch.from_numpy(v_c_seq[:, local_i]).float()
            logp_this = torch.from_numpy(logp_seq[:, local_i]).float()
            r_this_penalized = r_this - penalty_coef * c_this
            target_r_of[task_idx] = rollout_target(
                r_this_penalized, v_r_this, bool(is_terminated[local_i]), lam_r, discount_r,
                adv_estimator_r, logp_seq=logp_this,
            )
            target_c_of[task_idx] = rollout_target(
                c_this, v_c_this, bool(is_terminated[local_i]), lam_c, discount_c,
                adv_estimator_c, logp_seq=logp_this,
            )

    pred_r_list: list[float] = []
    pred_c_list: list[float] = []
    mc_mean_r_list: list[float] = []
    mc_mean_c_list: list[float] = []
    mc_var_r_list: list[float] = []
    mc_var_c_list: list[float] = []
    target_r_list: list[float] = []
    target_c_list: list[float] = []
    returns_r_list: list[list[float]] = []
    returns_c_list: list[list[float]] = []
    target_repeats_r_list: list[list[float]] = []
    target_repeats_c_list: list[list[float]] = []
    obs_list: list[list[torch.Tensor]] = []
    act_list: list[list[torch.Tensor]] = []
    idx = 0
    for _seed in probe_seeds:
        idxs = range(idx, idx + mc_repeats)
        idx += mc_repeats
        pred_r_list.append(pred_r_of[idxs[0]])
        pred_c_list.append(pred_c_of[idxs[0]])
        obs_list.append([obs_of[i] for i in idxs])
        act_list.append([act_of[i] for i in idxs])
        returns_r = [ret_r_of[i] for i in idxs]
        returns_c = [ret_c_of[i] for i in idxs]
        mc_mean_r_list.append(float(np.mean(returns_r)))
        mc_mean_c_list.append(float(np.mean(returns_c)))
        mc_var_r_list.append(float(np.var(returns_r)))
        mc_var_c_list.append(float(np.var(returns_c)))
        returns_r_list.append(returns_r)
        returns_c_list.append(returns_c)
        targets_r = [target_r_of[i] for i in idxs]
        targets_c = [target_c_of[i] for i in idxs]
        target_r_list.append(float(np.mean(targets_r)))
        target_c_list.append(float(np.mean(targets_c)))
        target_repeats_r_list.append(targets_r)
        target_repeats_c_list.append(targets_c)


    pred_r_t, pred_c_t = to_tensor(pred_r_list), to_tensor(pred_c_list)
    mc_mean_r_t, mc_mean_c_t = to_tensor(mc_mean_r_list), to_tensor(mc_mean_c_list)
    target_r_t, target_c_t = to_tensor(target_r_list), to_tensor(target_c_list)


    stats = {}
    for stream, (pred_t, mc_t, tgt_t, rets) in (
        ('r', (pred_r_t, mc_mean_r_t, target_r_t, returns_r_list)),
        ('c', (pred_c_t, mc_mean_c_t, target_c_t, returns_c_list)),
    ):
        stats.update({
            f'{k}_{stream}': v
            for k, v in calibration_stats(pred_t, mc_t, tgt_t, returns=rets).items()
        })
    del epoch  # accepted for call-site symmetry with estimate_true_value; not used here
    if not return_raw:
        return stats
    raw = {
        'probe_seeds': list(probe_seeds),
        'obs': obs_list, 'action': act_list,
        'r': {
            'pred': pred_r_list, 'mc_mean': mc_mean_r_list, 'mc_var': mc_var_r_list,
            'target': target_r_list,
            'returns': returns_r_list, 'target_repeats': target_repeats_r_list,
        },
        'c': {
            'pred': pred_c_list, 'mc_mean': mc_mean_c_list, 'mc_var': mc_var_c_list,
            'target': target_c_list,
            'returns': returns_c_list, 'target_repeats': target_repeats_c_list,
        },
    }
    return stats, raw


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
        epoch: For the progress bar only.
        return_raw: Also return the per-probe arrays.
        bootstrap_threshold: Truncate once the discounted tail falls below this.
        tail_mode: ``'drop'`` or ``'bootstrap'``; required when truncating.

    Returns:
        ``stats``, or ``(stats, raw)`` when ``return_raw``.

    Raises:
        ValueError: If truncating without a ``tail_mode``.
    """
    from omnisafe.utils.state_snapshot import restore_and_get_obs  # noqa: PLC0415

    device = torch.device(cfgs.train_cfgs.device)
    n_envs = env.num_envs
    assert horizon and horizon > 0
    nominal_horizon = horizon
    horizon = effective_rollout_horizon(nominal_horizon, discount_r, discount_c, bootstrap_threshold)
    bootstrap_tail = tail_mode == 'bootstrap'
    if horizon < nominal_horizon and tail_mode not in ('bootstrap', 'drop'):
        raise ValueError(
            'estimate_value_from_snapshots: bootstrap_threshold truncates the rollout at '
            f'{horizon} of {nominal_horizon} steps, so the tail must be handled explicitly. '
            "Set tail_mode='drop' or tail_mode='bootstrap' (see "
            'estimate_true_value_same_state_mc), or bootstrap_threshold=None for the full '
            'horizon.',
        )

    adv_estimator_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    adv_estimator_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_estimator_r
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
    penalty_coef = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

    tasks = [snap for snap in snapshots for _ in range(mc_repeats)]
    n_tasks = len(tasks)
    pred_r_of: list[float | None] = [None] * n_tasks
    pred_c_of: list[float | None] = [None] * n_tasks
    obs_of: list[torch.Tensor | None] = [None] * n_tasks
    act_of: list[torch.Tensor | None] = [None] * n_tasks
    ret_r_of: list[float] = [0.0] * n_tasks
    ret_c_of: list[float] = [0.0] * n_tasks
    target_r_of: list[float] = [0.0] * n_tasks
    target_c_of: list[float] = [0.0] * n_tasks

    n_waves = (n_tasks + n_envs - 1) // n_envs
    for w in range(n_waves):
        start = w * n_envs
        wave_task_idxs = list(range(start, min(start + n_envs, n_tasks)))
        wave_size = len(wave_task_idxs)
        wave_snaps = [tasks[i] for i in wave_task_idxs]
        if wave_size < n_envs:
            wave_snaps += [wave_snaps[0]] * (n_envs - wave_size)

        obs = restore_and_get_obs(env, wave_snaps, device, reset_elapsed_steps=True)
        act, value_r, value_c, log_prob = agent.step(obs)
        pred_r_batch = value_r.reshape(-1).detach().cpu().numpy()
        pred_c_batch = value_c.reshape(-1).detach().cpu().numpy()
        obs_s0_cpu = obs.detach().cpu().clone()
        act_a0_cpu = act.detach().cpu().clone()

        r_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        c_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        v_r_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        v_c_seq = np.zeros((horizon + 1, n_envs), dtype=np.float64)
        logp_seq = np.zeros((horizon, n_envs), dtype=np.float64)
        v_r_seq[0] = pred_r_batch
        v_c_seq[0] = pred_c_batch

        g_r = np.zeros(n_envs, dtype=np.float64)
        g_c = np.zeros(n_envs, dtype=np.float64)
        disc_r, disc_c = 1.0, 1.0
        terminated = truncated = None
        for t in range(horizon):
            obs, r, c, terminated, truncated, _ = env.step(act)
            r_np = r.reshape(-1).detach().cpu().numpy()
            c_np = c.reshape(-1).detach().cpu().numpy()
            r_seq[t] = r_np
            c_seq[t] = c_np
            g_r += disc_r * r_np
            g_c += disc_c * c_np
            disc_r *= discount_r
            disc_c *= discount_c
            if t < horizon - 1:
                act, value_r, value_c, log_prob = agent.step(obs)
                v_r_seq[t + 1] = value_r.reshape(-1).detach().cpu().numpy()
                v_c_seq[t + 1] = value_c.reshape(-1).detach().cpu().numpy()
                logp_seq[t] = log_prob.reshape(-1).detach().cpu().numpy()

        if horizon == nominal_horizon:
            done_at_end = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
            if not bool(done_at_end.all()):
                raise RuntimeError(
                    'estimate_value_from_snapshots: not every probe reached done after '
                    f'horizon={nominal_horizon} steps (got done={done_at_end.tolist()}). '
                    'horizon must equal the env\'s own max_episode_steps for every probe to '
                    'reach the time limit exactly here (reset_elapsed_steps=True means the '
                    'restored state always starts counting from 0).',
                )
            is_terminated = terminated.reshape(-1).bool().detach().cpu().numpy()
        else:
            is_terminated = np.zeros(n_envs, dtype=bool)

        _, boot_value_r, boot_value_c, _ = agent.step(obs)
        boot_r = np.where(is_terminated, 0.0, boot_value_r.reshape(-1).detach().cpu().numpy())
        boot_c = np.where(is_terminated, 0.0, boot_value_c.reshape(-1).detach().cpu().numpy())
        v_r_seq[horizon] = boot_r
        v_c_seq[horizon] = boot_c
        if horizon < nominal_horizon and bootstrap_tail:
            g_r += disc_r * boot_r
            g_c += disc_c * boot_c

        for local_i, task_idx in enumerate(wave_task_idxs):
            pred_r_of[task_idx] = float(pred_r_batch[local_i])
            pred_c_of[task_idx] = float(pred_c_batch[local_i])
            obs_of[task_idx] = obs_s0_cpu[local_i].clone()
            act_of[task_idx] = act_a0_cpu[local_i].clone()
            ret_r_of[task_idx] = float(g_r[local_i])
            ret_c_of[task_idx] = float(g_c[local_i])

            r_this = torch.from_numpy(r_seq[:horizon, local_i]).float()
            c_this = torch.from_numpy(c_seq[:horizon, local_i]).float()
            v_r_this = torch.from_numpy(v_r_seq[:, local_i]).float()
            v_c_this = torch.from_numpy(v_c_seq[:, local_i]).float()
            logp_this = torch.from_numpy(logp_seq[:, local_i]).float()
            r_this_penalized = r_this - penalty_coef * c_this
            target_r_of[task_idx] = rollout_target(
                r_this_penalized, v_r_this, bool(is_terminated[local_i]), lam_r, discount_r,
                adv_estimator_r, logp_seq=logp_this,
            )
            target_c_of[task_idx] = rollout_target(
                c_this, v_c_this, bool(is_terminated[local_i]), lam_c, discount_c,
                adv_estimator_c, logp_seq=logp_this,
            )

    pred_r_list: list[float] = []
    pred_c_list: list[float] = []
    mc_mean_r_list: list[float] = []
    mc_mean_c_list: list[float] = []
    mc_var_r_list: list[float] = []
    mc_var_c_list: list[float] = []
    target_r_list: list[float] = []
    target_c_list: list[float] = []
    returns_r_list: list[list[float]] = []
    returns_c_list: list[list[float]] = []
    target_repeats_r_list: list[list[float]] = []
    target_repeats_c_list: list[list[float]] = []
    obs_list: list[list[torch.Tensor]] = []
    act_list: list[list[torch.Tensor]] = []
    idx = 0
    for _snap in snapshots:
        idxs = range(idx, idx + mc_repeats)
        idx += mc_repeats
        pred_r_list.append(pred_r_of[idxs[0]])
        pred_c_list.append(pred_c_of[idxs[0]])
        obs_list.append([obs_of[i] for i in idxs])
        act_list.append([act_of[i] for i in idxs])
        returns_r = [ret_r_of[i] for i in idxs]
        returns_c = [ret_c_of[i] for i in idxs]
        mc_mean_r_list.append(float(np.mean(returns_r)))
        mc_mean_c_list.append(float(np.mean(returns_c)))
        mc_var_r_list.append(float(np.var(returns_r)))
        mc_var_c_list.append(float(np.var(returns_c)))
        returns_r_list.append(returns_r)
        returns_c_list.append(returns_c)
        targets_r = [target_r_of[i] for i in idxs]
        targets_c = [target_c_of[i] for i in idxs]
        target_r_list.append(float(np.mean(targets_r)))
        target_c_list.append(float(np.mean(targets_c)))
        target_repeats_r_list.append(targets_r)
        target_repeats_c_list.append(targets_c)


    pred_r_t, pred_c_t = to_tensor(pred_r_list), to_tensor(pred_c_list)
    mc_mean_r_t, mc_mean_c_t = to_tensor(mc_mean_r_list), to_tensor(mc_mean_c_list)
    target_r_t, target_c_t = to_tensor(target_r_list), to_tensor(target_c_list)


    stats = {}
    for stream, (pred_t, mc_t, tgt_t, rets) in (
        ('r', (pred_r_t, mc_mean_r_t, target_r_t, returns_r_list)),
        ('c', (pred_c_t, mc_mean_c_t, target_c_t, returns_c_list)),
    ):
        stats.update({
            f'{k}_{stream}': v
            for k, v in calibration_stats(pred_t, mc_t, tgt_t, returns=rets).items()
        })
    del epoch  # accepted for call-site symmetry; not used here
    if not return_raw:
        return stats
    raw = {
        'obs': obs_list, 'action': act_list,
        'r': {
            'pred': pred_r_list, 'mc_mean': mc_mean_r_list, 'mc_var': mc_var_r_list,
            'target': target_r_list,
            'returns': returns_r_list, 'target_repeats': target_repeats_r_list,
        },
        'c': {
            'pred': pred_c_list, 'mc_mean': mc_mean_c_list, 'mc_var': mc_var_c_list,
            'target': target_c_list,
            'returns': returns_c_list, 'target_repeats': target_repeats_c_list,
        },
    }
    return stats, raw

