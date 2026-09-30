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
"""Monte-Carlo evaluation of a value function against the return it is meant to predict.

The critic's training loss says how well it fits its own regression target; it does not say
whether that target tracks the true discounted return. These studies answer the second question
directly, by rolling the *current* policy out and comparing V(s) against a genuine Monte-Carlo
estimate of the return from s.

Two studies, differing only in which states they probe:

* :func:`estimate_true_value_same_state_mc` -- episode starts. A Safety-Gymnasium layout is
  reproduced exactly by resetting with the same seed, so the same state can be re-rolled
  ``mc_repeats`` times and the sample mean is a real MC estimate of V^pi(s0), with a measurable
  sample variance.
* :func:`estimate_value_from_snapshots` -- states the policy actually visits mid-episode, which
  ``reset(seed=X)`` cannot reach. They are captured by simulator snapshot/restore (see
  :mod:`omnisafe.utils.state_snapshot`) and then scored the same way. This is the question the
  critic's accuracy actually needs to answer for GAE/TD to work.

:func:`~omnisafe.utils.value_utils.pool_correlation_stats` pools both into one correlation over
evaluated.

Only the estimation error and the prediction/true correlation are computed here. Everything else
these probes could support -- the target-vs-true decomposition, gradient alignment, residual
distributions -- is a function of the raw per-probe arrays, which the caller dumps in full every
eval epoch, so it is recoverable offline without re-running training.
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
    r"""Compare the critic's V(s0) against a genuine same-layout Monte-Carlo estimate.

    ``estimate_true_value`` above scores the critic against one MC sample per visited state --
    each state is seen once, under whatever action the current policy happened to sample there,
    so its "true" return is a single noisy draw, not an estimate with a known variance. This
    function instead fixes a state (a Safety-Gymnasium *layout*, reproduced exactly by resetting
    with the same seed -- procedural generation is deterministic in the seed) and re-rolls the
    current *stochastic* policy out from it ``mc_repeats`` times, so the resulting sample mean is
    a genuine Monte-Carlo estimate of :math:`V^\pi(s_0)` / :math:`V_c^\pi(s_0)` with a directly
    measurable sample variance -- the closest thing to an oracle this codebase can produce without
    an analytic model of the environment.

    ``env`` may be vectorized (``num_envs = N > 1``): the same-layout requirement only constrains
    *one env instance* per probe (each repeat needs a clean, uninterrupted ``reset(seed=X)`` ->
    rollout on the *same* underlying env slot), not the whole call to run on a single env. With
    N > 1, the ``len(probe_seeds) * mc_repeats`` independent rollouts are packed into
    ``ceil(total / N)`` waves of N concurrent (subprocess-parallel, since
    ``safety_gymnasium.vector.make`` defaults to ``asynchronous=True``) rollouts instead of
    running one at a time -- close to an N-x speedup on what was the dominant cost of the whole
    training loop. Pass a dedicated eval env (see ``sync_normalizer_from``) built via
    ``omnisafe.envs.core.make(..., num_envs=N)`` with the same wrapper recipe as training
    (``TimeLimit``/``AutoReset``/``ObsNormalize``/``ActionScale``, plus ``Unsqueeze`` only when
    N == 1) -- training itself does not need to give up its own vectorization for this.

    Assumes every probe episode reaches ``done`` at exactly ``max_episode_steps`` (uniformly
    across the whole vectorized batch) -- true for Safety-Gymnasium's Goal/Push/etc. tasks, which
    only end via the time limit and never terminate early (verified empirically: ``Metrics/EpLen``
    is exactly ``max_episode_steps`` in every logged epoch of every run in this codebase). This
    lets the rollout loop below always run for exactly ``max_episode_steps`` steps rather than
    tracking a per-slot done mask, which keeps the vectorized loop simple; it is checked with an
    assertion after each wave, so a future environment that terminates early would fail loudly
    here instead of silently producing wrong (early-terminated-episode-truncated) returns -- such
    an environment would need a done mask added to freeze each slot's accumulation independently.

    Args:
        agent: The actor-critic; must expose ``step(obs) -> (action, value_r, value_c, log_prob)``.
        env: An already-wrapped environment (any ``num_envs >= 1``) to roll the probes out in.
            May be a dedicated eval env (see ``sync_normalizer_from``) or the training env itself.
        cfgs: The resolved algorithm config.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        probe_seeds (list of int): The fixed set of layouts (env reset seeds) to probe. Held
            constant across calls so the same states are re-evaluated at every eval epoch.
        mc_repeats (int): Number of independent same-layout rollouts averaged per probe seed.
            Defaults to 5 -- a deliberately modest budget: this routine is
            ``len(probe_seeds) * mc_repeats`` full episodes *per call*, so the cost scales
            directly with this number (before accounting for the ``env.num_envs``-way speedup).
        epoch (int or None): Current epoch, for logging only.
        sync_normalizer_from: Optional live env (e.g. the training env) whose ``ObsNormalize``
            running statistics are copied into ``env``'s ``ObsNormalize`` (via
            :func:`_find_obs_normalizer` and a ``state_dict`` copy) before probing. A no-op if
            either env has no ``ObsNormalize`` wrapper (``algo_cfgs.obs_normalize=False``).
            ``None`` (the default) skips syncing -- appropriate only if ``env`` already *is* the
            live training env, or normalization is off.
        max_episode_steps (int or None): Episode horizon to roll every probe out for. Required
            when ``env.num_envs > 1`` (a vectorized ``AsyncVectorEnv`` has no ``.spec`` to read
            this off directly -- the caller must supply it, e.g. from a disposable single-env
            instance). Optional when ``env.num_envs == 1``, where it defaults to
            ``env.max_episode_steps``.
        return_raw (bool): If True, also return the per-probe arrays the aggregate stats were
            computed from (predictions, MC-mean returns, per-repeat variances) -- e.g. for
            pickling alongside the aggregates for later offline analysis, or for scatter plots
            (each probe is one point). Defaults to False (the original, stats-dict-only return).
        bootstrap_threshold (float or None): If set, roll each probe out only until both streams'
            discounted tail contribution drops below this fraction (see
            :func:`_effective_rollout_horizon`), bootstrapping the remainder with the critic's own
            value there instead of continuing to simulate. Trades a small, bounded bias (the
            substituted quantity is exactly what this function exists to validate) for wall-clock
            -- e.g. ``0.01`` with ``gamma=0.99`` cuts ``max_episode_steps=1000`` down to ~460
            steps. ``None`` (default) preserves the exact original full-horizon behavior.
        tail_mode (str or None): What to do with the tail once ``bootstrap_threshold``
            truncates the rollout -- ``'drop'`` stops accumulating (the value stays a
            simulation-only quantity, low by at most the threshold), ``'bootstrap'`` fills it
            in with the critic's own
            value. ``False`` (default) keeps the MC "true" return strictly simulation-only: it is
            then a genuine sample of the discounted return, with the properties the real return
            has -- in particular a cost return is non-negative whenever the per-step cost is,
            which a critic-bootstrapped one is not (an untrained ``V_c`` predicts negative values
            and those flow straight into the "ground truth"). ``True`` restores the older
            behaviour and is only usable together with ``bootstrap_threshold``; the two are
            checked for consistency, since truncating *without* bootstrapping would drop the tail
            rather than estimate it.

    Returns:
        A dict of aggregate statistics over the ``len(probe_seeds)`` probes: for each of
        ``r`` (reward) and ``c`` (cost), ``EstimationError_{r,c}`` (mean of MC estimate minus
        critic prediction), ``Correlation_{r,c}`` (Pearson correlation between the
        ``len(probe_seeds)`` critic predictions and MC-mean estimates), and ``MeanVar_{r,c}``
        (the sample variance of the ``mc_repeats`` returns at each probe, averaged over probes --
        how noisy the MC "oracle" itself is at this point in training). If ``return_raw``, a
        ``(stats, raw)`` tuple instead, where ``raw`` is a dict with ``probe_seeds`` and, for each
        of ``r``/``c``: ``pred`` (critic's V(s) at each probe), ``mc_mean`` (MC-estimated true
        value at each probe), ``mc_var`` (MC sample variance at each probe) -- all
        ``len(probe_seeds)``-length lists, in ``probe_seeds`` order.
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
    # See _effective_rollout_horizon's docstring -- horizon <= max_episode_steps always;
    # strictly less only when bootstrap_threshold is set, in which case the tail past `horizon`
    # is bootstrapped with the critic's own value instead of simulated.
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

    # Which advantage estimator/lambda each stream's target should mirror -- the same config
    # training itself reads (cost null-falls-back to reward's, same convention as
    # cost_gamma/critic_norm_coef_cost/etc. elsewhere in this codebase).
    adv_estimator_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    adv_estimator_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_estimator_r
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
    penalty_coef = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

    # Flat task list, mc_repeats consecutive entries per probe seed -- waves below cut across
    # this list without regard to seed boundaries; the seed-grouping happens afterward, purely
    # by array-index arithmetic, so it doesn't matter which wave a given repeat lands in.
    tasks = [seed for seed in probe_seeds for _ in range(mc_repeats)]
    n_tasks = len(tasks)
    pred_r_of: list[float | None] = [None] * n_tasks
    pred_c_of: list[float | None] = [None] * n_tasks
    # s0/a0 for each task -- the probe's starting observation and the action the current
    # stochastic policy actually sampled there, captured once per repeat (not just per probe,
    # unlike pred_r_of/pred_c_of: a0 genuinely varies repeat-to-repeat since the policy is
    # stochastic, and more independent (s0, a0, return) samples means a lower-variance gradient
    # estimate downstream -- see compute_gradient_alignment). Needed to reconstruct
    # grad_theta log pi(a0|s0) later; storing the tensors themselves (not just floats) since a
    # score-function gradient needs autograd through the actor, not a rebuilt Normal/Categorical
    # from summary statistics.
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
            # Last, partial wave: pad with a repeated seed so every env slot still gets a valid
            # reset -- the padding slots' results are simply never read back out below.
            wave_seeds += [wave_seeds[0]] * (n_envs - wave_size)

        obs, _ = env.reset(seed=wave_seeds if n_envs > 1 else wave_seeds[0])
        act, value_r, value_c, log_prob = agent.step(obs)
        # V(s0) is deterministic given s0 (no action taken yet), so it only needs computing once
        # per rollout -- cheap regardless (one batched NN forward pass vs. hundreds of env steps).
        pred_r_batch = value_r.reshape(-1).detach().cpu().numpy()
        pred_c_batch = value_c.reshape(-1).detach().cpu().numpy()
        # s0/a0 capture for compute_gradient_alignment -- must happen here, before the per-step
        # loop below overwrites obs/act with later timesteps.
        obs_s0_cpu = obs.detach().cpu().clone()
        act_a0_cpu = act.detach().cpu().clone()

        # Per-step sequences, needed (in addition to the running discounted sum below, which is
        # all the *true* MC estimate needs) to compute the training-style target after the
        # rollout -- see _rollout_target. Row `horizon` is the bootstrap slot, filled in after the
        # loop -- either the env's own final obs (horizon == max_episode_steps, no truncation) or
        # the early-stop point (horizon < max_episode_steps, bootstrap_threshold truncation).
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
            # No truncation this call -- every probe must have reached the env's own time limit,
            # exactly as before bootstrap_threshold existed. Still enforced here (not just when
            # bootstrap_threshold is set) so an environment that terminates early keeps failing
            # loudly instead of silently producing wrong returns -- see the docstring.
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
            # Deliberately stopped short of the env's own time limit (bootstrap_threshold
            # truncation) -- every slot is by construction still mid-episode (Safety-Gymnasium's
            # Goal/Push/etc. tasks never terminate early, see docstring), so bootstrap
            # unconditionally with the critic's value rather than checking terminated/truncated
            # flags that don't mean "done" here.
            is_terminated = np.zeros(n_envs, dtype=bool)

        # Bootstrap value at the final observation -- 0 for a true terminal, the critic's own
        # prediction there otherwise -- mirroring OnPolicyAdapter.rollout's last_value_r/c
        # construction exactly (see omnisafe.utils.gae's docstring). Queried for every slot
        # uniformly (cheap: one more batched forward pass) rather than branching per-env; the
        # terminated slots' values are simply zeroed out afterward.
        _, boot_value_r, boot_value_c, _ = agent.step(obs)
        boot_r = np.where(is_terminated, 0.0, boot_value_r.reshape(-1).detach().cpu().numpy())
        boot_c = np.where(is_terminated, 0.0, boot_value_c.reshape(-1).detach().cpu().numpy())
        v_r_seq[horizon] = boot_r
        v_c_seq[horizon] = boot_c
        if horizon < max_episode_steps and bootstrap_tail:
            # Only reachable with bootstrap_tail=True (see this function's docstring): the MC
            # "true" return is otherwise kept strictly simulation-only, so it stays a genuine
            # sample of the return and can never inherit a critic artifact -- most visibly, a
            # negative discounted cost-to-go, which is impossible for a non-negative cost signal
            # but is exactly what an untrained V_c contributes here.
            # disc_r/disc_c already equal discount**horizon at this point.
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
            # Mirrors finish_path's `rewards -= penalty_coefficient * costs` -- an intrinsic-cost
            # penalty folded into the reward stream's target only; a no-op at the (default) 0.0.
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
    # Per-repeat raw values (mc_repeats-length list per probe), not just their mean/var --
    # needed for anything downstream that wants the actual repeat-to-repeat distribution (a
    # variance-distribution plot, a bootstrap CI, etc.), which mc_mean/mc_var alone can't
    # reconstruct (they're already a lossy reduction of exactly this).
    returns_r_list: list[list[float]] = []
    returns_c_list: list[list[float]] = []
    target_repeats_r_list: list[list[float]] = []
    target_repeats_c_list: list[list[float]] = []
    # s0/a0 per repeat (mc_repeats-length list of tensors per probe) -- unlike pred_list, which
    # only reads idxs[0] since V(s0) doesn't depend on the repeat, every repeat's own (s0, a0)
    # pair is kept here: a0 is a fresh stochastic-policy sample each repeat, so each one is an
    # independent (s0, a0, return) triple usable by compute_gradient_alignment -- discarding all
    # but the first would throw away most of the available gradient-estimation samples.
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
        # Averaged across the mc_repeats independent rollouts, same as mc_mean -- the target
        # genuinely varies per repeat (it depends on the realized trajectory, unlike pred which
        # only depends on s0), so this is the same kind of reduction, not a different one.
        targets_r = [target_r_of[i] for i in idxs]
        targets_c = [target_c_of[i] for i in idxs]
        target_r_list.append(float(np.mean(targets_r)))
        target_c_list.append(float(np.mean(targets_c)))
        target_repeats_r_list.append(targets_r)
        target_repeats_c_list.append(targets_c)


    pred_r_t, pred_c_t = to_tensor(pred_r_list), to_tensor(pred_c_list)
    mc_mean_r_t, mc_mean_c_t = to_tensor(mc_mean_r_list), to_tensor(mc_mean_c_list)
    target_r_t, target_c_t = to_tensor(target_r_list), to_tensor(target_c_list)


    # Both streams, through the shared metric helper -- see value_utils.calibration_stats for
    # what the five numbers mean and why the correlation is reported alongside its two factors.
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
        # Shared between the r/c streams (same rollout, same s0/a0) -- see
        # compute_gradient_alignment. probe_seeds-length list of mc_repeats-length lists of
        # tensors (obs_dim / act_dim respectively), not stacked into a single tensor, since
        # different algorithms/envs can have different obs/act shapes and this keeps the pickle
        # simple to inspect probe-by-probe.
        'obs': obs_list, 'action': act_list,
        'r': {
            'pred': pred_r_list, 'mc_mean': mc_mean_r_list, 'mc_var': mc_var_r_list,
            'target': target_r_list,
            # Per-repeat raw values (mc_repeats-length list per probe) -- see the comment where
            # these are collected, above.
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
    r"""Like :func:`estimate_true_value_same_state_mc`, but for arbitrary on-policy *intermediate*
    states captured via :mod:`omnisafe.utils.state_snapshot`, instead of states reachable by
    ``env.reset(seed=X)``. Where the s0 study asks "is the critic accurate at episode starts", this
    asks the question the critic's accuracy actually needs to answer for TD-learning/GAE/advantages
    to work: is it accurate at states the *current* policy actually visits mid-episode?

    Structurally almost identical to :func:`estimate_true_value_same_state_mc` -- same
    wave-batched rollout, same per-probe ``mc_repeats`` averaging, same aggregate stats -- with one
    real difference: probes are restored from pre-captured snapshots
    (:func:`omnisafe.utils.state_snapshot.restore_and_get_obs`) instead of reset from seeds.

    The restored state is treated as a fresh *start* state, exactly like s0 -- it gets the full
    ``horizon`` budget ahead of it (``reset_elapsed_steps=True`` on the restore call), not the
    physically-remaining steps of the one particular episode instance it happened to be captured
    from. This is deliberate: the point of the intermediate-state study is to sample a diverse set
    of *states* the policy might visit at any point in an episode, then ask the same question the
    s0 study asks of s0 -- "what's this state's value as a start state" -- not "what's this state's
    value given only the wall-clock time left in the specific episode it was captured from". An
    earlier version of this function used ``remaining_horizon = max_episode_steps -
    step_index_captured_at`` instead, which answered a different (and for a state visited late in
    an episode, much less useful -- e.g. only 100 steps of budget for a step-900 snapshot) question
    than the one this study is actually meant to answer.

    Args:
        agent: The actor-critic; must expose ``step(obs) -> (action, value_r, value_c, log_prob)``.
        env: An already-wrapped, vectorized (``num_envs >= 1``) env to roll the probes out in --
            same recipe as ``estimate_true_value_same_state_mc``'s ``env`` argument, but there is
            no ``sync_normalizer_from`` here: normalizer syncing has to happen once, before
            *capturing* the snapshots (so the on-policy actions that produced them were sampled
            under realistic normalization), not per scoring call -- see the state-collection call
            site for where that sync happens.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        snapshots (list of dict): Pre-captured states (one entry per probe). Unlike before, these
            no longer need to share a single within-episode step index -- since every snapshot now
            gets the same fresh ``horizon`` budget regardless of where it was captured, probes
            captured at different original step indices can be scored together in one call.
        horizon (int): Steps to roll every probe out for, treating it as a fresh start state --
            typically ``max_episode_steps`` (the same full budget s0 gets), not a
            position-dependent "steps remaining" value.
        mc_repeats (int): Independent rollouts averaged per probe state.
        epoch (int or None): Current epoch, for logging only.
        return_raw (bool): See :func:`estimate_true_value_same_state_mc` -- same meaning, same
            per-probe ``raw`` dict shape (just no ``probe_seeds`` key, since these probes came
            from pre-captured snapshots rather than reset seeds).
        bootstrap_threshold (float or None): See :func:`estimate_true_value_same_state_mc` --
            same meaning, applied against ``horizon``.
        tail_mode (str or None): See :func:`estimate_true_value_same_state_mc` -- same meaning,
            same default of ``False``.

    Returns:
        Same shape/semantics as :func:`estimate_true_value_same_state_mc`'s return dict, just
        without the ``MCStudy/`` key prefix -- callers should apply their own prefix (e.g. tagging
        which within-episode position these snapshots came from). If ``return_raw``, a
        ``(stats, raw)`` tuple -- see :func:`estimate_true_value_same_state_mc`'s docstring.
    """
    # Local import: mirrors state_snapshot.py's own local import of _find_obs_normalizer from
    # here -- avoids import-time coupling between the two modules in either direction.
    from omnisafe.utils.state_snapshot import restore_and_get_obs  # noqa: PLC0415

    device = torch.device(cfgs.train_cfgs.device)
    n_envs = env.num_envs
    assert horizon and horizon > 0
    nominal_horizon = horizon
    # See _effective_rollout_horizon's docstring (and estimate_true_value_same_state_mc's use of
    # it) -- horizon <= nominal_horizon always; strictly less only when bootstrap_threshold is
    # set.
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

    # Flat task list, mc_repeats consecutive entries per probe state -- same rationale as
    # estimate_true_value_same_state_mc's `tasks` list.
    tasks = [snap for snap in snapshots for _ in range(mc_repeats)]
    n_tasks = len(tasks)
    pred_r_of: list[float | None] = [None] * n_tasks
    pred_c_of: list[float | None] = [None] * n_tasks
    # s0/a0 capture -- see estimate_true_value_same_state_mc's matching comment.
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
            # Last, partial wave: pad with a repeated snapshot so every env slot still gets a
            # valid restore -- the padding slots' results are simply never read back out below.
            wave_snaps += [wave_snaps[0]] * (n_envs - wave_size)

        # reset_elapsed_steps=True: treat the restored state as a fresh start state with a full
        # horizon budget ahead, not one bound by the physically-remaining steps of the episode it
        # was captured from -- see this function's docstring.
        obs = restore_and_get_obs(env, wave_snaps, device, reset_elapsed_steps=True)
        act, value_r, value_c, log_prob = agent.step(obs)
        pred_r_batch = value_r.reshape(-1).detach().cpu().numpy()
        pred_c_batch = value_c.reshape(-1).detach().cpu().numpy()
        obs_s0_cpu = obs.detach().cpu().clone()
        act_a0_cpu = act.detach().cpu().clone()

        # Per-step sequences, needed to compute the training-style target after the rollout --
        # see estimate_true_value_same_state_mc's matching comment. Sized to the *effective*
        # `horizon`, not `nominal_horizon` -- see that function's matching comment on why.
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
            # No truncation this call -- see estimate_true_value_same_state_mc's matching branch.
            # Since the restore reset the env's own elapsed-steps counter to 0
            # (reset_elapsed_steps=True above), the wrapped TimeLimit will fire "done" here iff
            # `horizon` actually equals that env's own configured max_episode_steps -- if a caller
            # passes something shorter (deliberately partial, not via bootstrap_threshold), this
            # check would correctly fail rather than silently under-counting.
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
            # See estimate_true_value_same_state_mc's matching branch -- off by default so the
            # MC return stays simulation-only.
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


    # Both streams, through the shared metric helper -- see value_utils.calibration_stats for
    # what the five numbers mean and why the correlation is reported alongside its two factors.
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
        # See estimate_true_value_same_state_mc's matching comment.
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

