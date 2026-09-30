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
"""Metrics and helpers for the value-evaluation studies.

Split out of :mod:`omnisafe.utils.value_eval` so that module is left holding only the two things
that are genuinely about *rolling the policy out*; everything here is either a pure function of
arrays already collected (the metrics) or a small piece of env//horizon bookkeeping those
rollouts need.

Four groups:

* metrics -- :func:`corr`, :func:`calibration_stats`, :func:`pool_correlation_stats`
* targets -- :func:`rollout_target`, the training-style regression target at a probe state
* horizon -- :func:`effective_rollout_horizon`
* env     -- :func:`find_obs_normalizer`, :func:`sync_obs_normalizer`
"""

from __future__ import annotations

import numpy as np
import torch

from omnisafe.utils.gae import calculate_adv_and_value_targets


def to_tensor(values, device: torch.device | None = None) -> torch.Tensor:
    """Pack a list of per-probe floats into a 1-D float tensor."""
    return torch.tensor(values, device=device, dtype=torch.float32)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, NaN when it is undefined rather than raising.

    Undefined here means fewer than two points, or either side constant -- both of which happen
    legitimately: a study with one probe, or a cost stream whose probes all returned zero.
    """
    if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def avg_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x``, with ties sharing their average rank."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a positive state is scored above a negative one (Mann-Whitney U).

    Computed from rank sums rather than by sweeping thresholds, which is exact and O(n log n);
    the average-rank treatment of ties contributes 0.5 per tied pair, as it should.

    0.5 means no discrimination, 1.0 a perfect ranking. NaN when either class is empty, which
    happens legitimately -- a cost stream whose probes all returned zero has no positives.

    Why this is worth having next to ``Correlation``: AUROC is invariant to any monotone
    rescaling of the score, so it isolates *ordering* from scale and bias. A critic can be
    badly mis-scaled and still rank states correctly (high AUROC, poor EstimationError), or be
    well-centred and rank at chance (low AUROC, small EstimationError). Those are different
    failures with different fixes, and neither number alone separates them.

    Args:
        score: The value being evaluated as a ranking, one entry per probe.
        label: Boolean positives.

    Returns:
        The AUROC, or NaN if it is undefined.
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    n1 = int(label.sum())
    n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((avg_ranks(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ceiling(returns: list[list[float]]) -> float:
    """How well the Monte-Carlo reference ranks *itself*, at this repeat count.

    The label the critic is scored against is an MC estimate, so it carries sampling noise; with
    finitely many repeats even a perfect critic cannot reach AUROC 1. Splitting the repeats in
    half and scoring each half against the other estimates that achievable ceiling, so a critic's
    AUROC can be read against what is attainable rather than against 1.0.

    This matters here specifically: the cost stream is the noisy one, and a split-half analysis
    of these probes found n=5 repeats badly under-samples it. Without the ceiling, a mediocre
    AUROC is indistinguishable from a well-ranked critic measured against a noisy label.

    Args:
        returns: ``n_probes x n_repeats`` per-rollout returns.

    Returns:
        The ceiling, or NaN with fewer than two repeats.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return float('nan')
    half = arr.shape[1] // 2
    h1, h2 = arr[:, :half].mean(1), arr[:, half:].mean(1)
    return float(np.nanmean([
        auroc(h1, h2 >= np.median(h2)),
        auroc(h2, h1 >= np.median(h1)),
    ]))


def spearman(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Spearman rank correlation -- Pearson applied to the ranks.

    The threshold-free counterpart of :func:`auroc`. AUROC marginalizes the threshold applied to
    the *score*, but it still needs a binary label, so a cut on the continuous ground truth has
    to be invented (here a median split). Spearman needs no label at all: it compares the two
    full orderings directly, so no cut is chosen and none can be argued with.

    It also differs usefully from the Pearson ``Correlation`` reported beside it. Pearson is
    sensitive to the heavy tail of a discounted cost-to-go -- a handful of very costly probes can
    carry it on their own -- whereas ranks bound every probe's influence equally. A large gap
    between the two is itself informative: it says the linear relationship is being driven by the
    extremes rather than holding across the range.

    Ties take their average rank, matching :func:`auroc`'s convention.

    Args:
        a: First series.
        b: Second series, same length.

    Returns:
        The rank correlation, or NaN when it is undefined (fewer than two points, or either
        ordering constant -- e.g. a cost stream whose probes all returned zero).
    """
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=float)
    b = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b, dtype=float)
    return corr(to_tensor(avg_ranks(a)), to_tensor(avg_ranks(b)))


def calibration_stats(
    pred: torch.Tensor,
    mc_true: torch.Tensor,
    target: torch.Tensor | None = None,
    returns: list[list[float]] | None = None,
    prefix: str = '',
) -> dict[str, float]:
    """The five calibration numbers, for one stream.

    ``Correlation`` is the end-to-end question -- does the critic's prediction track the true
    discounted return -- but on its own it cannot say *which* of two independent things went
    wrong, so it is reported alongside the factors it decomposes into:

    * ``Correlation_pred_target`` -- is the critic fitting what it is actually trained to fit?
      A low value here is a critic-fitting problem: optimizer, capacity, too few epochs.
    * ``Correlation_target_true`` -- is that target itself a good proxy for the true return?
      A low value here is a property of the *estimator* (adv_estimation_method /
      value_target_method), not of the critic, and no amount of critic training fixes it.

    ``EstimationError`` and ``EstimationError_target_true`` are the bias counterparts: signed
    mean gaps rather than correlations, so a critic that tracks the shape of the return but sits
    uniformly above it shows up here and not above.

    Args:
        pred: Critic predictions, one per probe.
        mc_true: Monte-Carlo estimate of the true value at the same probes.
        target: The training-style regression target at those probes, when available.
        prefix: Prepended to every key (e.g. ``'ValueEval/'``).

    Returns:
        ``{key: value}``, five entries when ``target`` is given and two when it is not.
    """
    stats = {
        f'{prefix}EstimationError': (mc_true - pred).mean().item(),
        f'{prefix}Correlation': corr(pred, mc_true),
    }
    if target is not None:
        stats[f'{prefix}Correlation_pred_target'] = corr(pred, target)
        stats[f'{prefix}Correlation_target_true'] = corr(target, mc_true)
        stats[f'{prefix}EstimationError_target_true'] = (mc_true - target).mean().item()

    # Ranking quality, independent of scale and bias. The positive class is a median split of
    # the MC reference rather than a fixed value threshold: median-splitting is scale-free, so
    # one definition works for both the reward and cost streams and stays comparable across
    # epochs as the value range moves, and it always yields both classes unless every probe ties.
    truth = mc_true.detach().cpu().numpy()
    stats[f'{prefix}AUROC'] = auroc(pred.detach().cpu().numpy(), truth >= np.median(truth))
    # Rank correlation, alongside AUROC and the Pearson one above. Unlike AUROC it needs no
    # positive class at all, so it carries no threshold choice to defend; unlike Pearson it is
    # not dominated by the tail of the cost distribution.
    stats[f'{prefix}Spearman'] = spearman(pred, mc_true)
    if target is not None:
        stats[f'{prefix}Spearman_pred_target'] = spearman(pred, target)
        stats[f'{prefix}Spearman_target_true'] = spearman(target, mc_true)
    if returns is not None:
        stats[f'{prefix}AUROC_ceiling'] = auroc_ceiling(returns)
    return stats


def find_obs_normalizer(env):
    """Walk an env's wrapper chain (outermost first) for its ``ObsNormalize`` instance, if any.

    ``Wrapper.__getattr__`` only forwards non-underscore names (see ``omnisafe.envs.core.Wrapper``),
    so an outer wrapper (``ActionScale``, ``Unsqueeze``, ...) can't reach an inner
    ``ObsNormalize``'s ``_obs_normalizer`` through attribute delegation alone -- this walks the
    ``._env`` chain explicitly instead. Returns ``None`` if no ``ObsNormalize`` wrapper is present
    (``algo_cfgs.obs_normalize=False``), in which case there is nothing to sync.
    """
    # Local import: value_eval.py must not import omnisafe.envs.wrapper at module load time (it
    # would create a circular import, since that module's callers import from here too).
    from omnisafe.envs.wrapper import ObsNormalize  # noqa: PLC0415

    while True:
        if isinstance(env, ObsNormalize):
            return env._obs_normalizer  # noqa: SLF001
        inner = getattr(env, '_env', None)
        if inner is None:
            return None
        env = inner


def sync_obs_normalizer(target_env, source_env) -> None:
    """Copy ``source_env``'s ``ObsNormalize`` running statistics into ``target_env``'s.

    A no-op if either env has no ``ObsNormalize`` wrapper (``algo_cfgs.obs_normalize=False``).
    Used to snapshot a dedicated eval-only env's normalizer from the live training env's current
    statistics before probing with it -- see ``estimate_true_value_same_state_mc``'s
    ``sync_normalizer_from`` and the on-policy intermediate-state study's collection env, which
    both need the critic's inputs to be processed exactly as they would be during real training,
    without the dedicated env's own normalizer (typically built with ``update_stats=False``, see
    ``omnisafe.envs.wrapper.ObsNormalize``) needing to independently accumulate its own history to
    get there.
    """
    target_norm = find_obs_normalizer(target_env)
    source_norm = find_obs_normalizer(source_env)
    if target_norm is not None and source_norm is not None:
        target_norm.load_state_dict(source_norm.state_dict())


def rollout_target(
    r_seq,
    v_seq_incl_boot,
    terminated,
    lam,
    gamma,
    advantage_estimator,
    logp_seq=None,
):
    r"""The training-style regression target for one already-completed rollout.

    Runs the *exact same* GAE/plain/vtrace/etc. formula (:func:`omnisafe.utils.gae.
    calculate_adv_and_value_targets`) the training buffer uses on its own on-policy trajectories
    -- applied here to a probe rollout's own reward/value sequence -- and returns the target at
    that rollout's own first timestep (index 0), which is the value the current policy/critic
    combination would actually have been trained to predict for that starting state, had this
    rollout been part of a training batch.

    Mirrors ``OnPolicyAdapter.rollout``'s bootstrap convention exactly (see
    ``omnisafe.utils.gae``'s docstring): 0 if the rollout ended in a true terminal, otherwise the
    critic's own value at the final observation, appended as both the last "value" *and* the last
    "reward" entry (the latter so rewards-to-go-style targets fold the bootstrap in the same way
    GAE's ``values[1:]`` does).

    Args:
        r_seq: ``(T,)`` per-step reward (or cost) sequence for this one rollout.
        v_seq_incl_boot: ``(T+1,)`` per-step critic values *including* the bootstrap value as the
            final entry (already zeroed by the caller for terminated rollouts).
        terminated: Whether this rollout ended in a true terminal (vs. a horizon truncation) --
            only used to decide whether the appended bootstrap "reward" should be 0 or the
            bootstrap value itself (it's always the same value already in ``v_seq_incl_boot[-1]``
            either way, this only controls the pseudo-reward, matching ``finish_path``'s
            ``rewards = torch.cat([..., last_value_r])`` regardless of terminated/truncated --
              the bootstrap value is 0 for a true terminal, so the appended pseudo-reward is 0
              too automatically; kept as an explicit arg for clarity at call sites, not because
              the formula branches on it).
        lam: GAE lambda for this stream.
        gamma: Discount factor for this stream.
        advantage_estimator: ``algo_cfgs.adv_estimation_method`` / ``cost_adv_estimation_method``.
        logp_seq: ``(T,)`` log-probabilities, only read under ``advantage_estimator == 'vtrace'``.

    Returns:
        The scalar target value at this rollout's first timestep.
    """
    del terminated  # see docstring -- already folded into v_seq_incl_boot[-1]
    rewards = torch.cat([r_seq, v_seq_incl_boot[-1:]])
    action_probs = logp_seq.exp() if advantage_estimator == 'vtrace' else None
    _, target = calculate_adv_and_value_targets(
        values=v_seq_incl_boot,
        rewards=rewards,
        lam=lam,
        gamma=gamma,
        advantage_estimator=advantage_estimator,
        action_probs=action_probs,
        behavior_action_probs=action_probs,
    )
    return target[0].item()


def effective_rollout_horizon(
    nominal_horizon: int,
    discount_r: float,
    discount_c: float,
    bootstrap_threshold: float | None,
) -> int:
    r"""How many steps a same-state MC rollout actually needs to run, given how fast the discount
    decays.

    The tail past step :math:`t` contributes at most a factor of :math:`\gamma^t` of its own scale
    to the discounted return -- for ``gamma=0.99``, that's already down to ~0.66% by ``t=500`` and
    ~0.09% by ``t=700``. Simulating all the way to ``nominal_horizon`` (900-1000 steps for the
    positions this codebase studies) to capture a shrinking fraction of a percent of the answer is
    wasted compute; this instead finds the smallest ``t <= nominal_horizon`` such that BOTH
    streams' residual weight is already below ``bootstrap_threshold``, so the rollout can stop
    there and bootstrap the tail with the critic's own (already-computed every step) value
    estimate at that point -- exactly the same 0-if-terminated/critic-otherwise convention already
    used for a rollout's *natural* end (see the callers' ``is_terminated`` handling), just reached
    early on purpose instead of by the env's own time limit.

    Uses ``max(discount_r, discount_c)`` (whichever decays slower) so the returned horizon is
    conservative for both streams at once, not just whichever happens to be smaller.

    This trades a small, bounded, quantifiable bias for wall-clock: the substituted quantity is
    the critic's own value estimate -- exactly what these functions exist to independently
    validate -- so a nonzero ``bootstrap_threshold`` means the resulting "true" MC value is no
    longer bias-free ground truth, just approximately so, to within `bootstrap_threshold` of the
    terminal value's own scale. ``bootstrap_threshold=None`` (the default everywhere this is
    called) preserves the exact prior behavior -- full ``nominal_horizon``, zero approximation.

    Args:
        nominal_horizon: The horizon that would be used with no truncation -- ``max_episode_steps``
            for every caller (both :func:`estimate_true_value_same_state_mc`'s ``s0`` probes and
            :func:`estimate_value_from_snapshots`'s restored intermediate-state probes, since the
            latter are now scored with the same full fresh-start budget as ``s0``, not a
            position-dependent "steps remaining" value).
        discount_r: Reward discount factor.
        discount_c: Cost discount factor.
        bootstrap_threshold: Target residual discount weight (e.g. ``0.01`` = stop once both
            streams' remaining contribution is below 1% of the bootstrapped value's own scale).
            ``None`` or ``<= 0`` disables truncation entirely (returns ``nominal_horizon``
            unchanged).

    Returns:
        The horizon to actually roll out to, in ``[1, nominal_horizon]``.
    """
    if not bootstrap_threshold or bootstrap_threshold <= 0:
        return nominal_horizon
    gamma_max = max(discount_r, discount_c)
    if gamma_max <= 0:
        return nominal_horizon
    if gamma_max >= 1:
        # Undiscounted (or discount > 1, degenerate) -- the tail never shrinks, so there is no
        # valid truncation point; fall back to the untruncated horizon rather than silently
        # returning something arbitrary.
        return nominal_horizon
    # gamma_max ** t <= threshold  <=>  t >= log(threshold) / log(gamma_max) (log(gamma_max) < 0)
    import math  # noqa: PLC0415 -- only needed on this (opt-in, off by default) code path

    needed = math.ceil(math.log(bootstrap_threshold) / math.log(gamma_max))
    return max(1, min(nominal_horizon, needed))


def pool_correlation_stats(raw_list: list[dict], prefix: str = '') -> tuple[dict, dict]:
    """Pool several ``return_raw=True`` dicts into one correlation, as if every probe from every
    source (e.g. s0 plus every on-policy intermediate position) had been scored together.

    The per-category ``Correlation_r``/``Correlation_c`` that ``estimate_true_value_same_state_mc``
    and ``estimate_value_from_snapshots`` report are each computed over only their own narrow slice
    of states (just resets, or just one fixed within-episode position) -- useful for spotting
    *where* the critic is worse, but neither one answers "how accurate is the critic across the
    actual diversity of states we evaluate on". This does: it concatenates the raw
    predicted/MC-true pairs across every source first, then computes one correlation over the
    pooled set, which is not the same as averaging the per-category correlations (pooling can
    surface a relationship -- or wash one out -- that no individual narrow slice shows on its own).

    Args:
        raw_list: The ``raw`` dicts to pool (each has ``'r'``/``'c'`` keys, each with
            ``'pred'``/``'mc_mean'`` lists of equal length, as returned by
            ``estimate_true_value_same_state_mc``/``estimate_value_from_snapshots`` with
            ``return_raw=True``).
        prefix: Optional key prefix for the returned stats dict (e.g. ``'PooledMC/'``).

    Returns:
        ``(stats, raw)`` -- ``stats`` has the same key shape as the per-category stats dicts
        (``Correlation_r/c``, ``EstimationError_r/c``, ``MeanTrue_r/c``, ``MeanPred_r/c``,
        ``NumProbes``), and ``raw`` is the merged ``{'r': {...}, 'c': {...}}`` dict (no
        ``mc_var``/``probe_seeds`` -- pooling those across heterogeneous sources isn't meaningful),
        suitable for passing straight into ``eval_data_dump.save_scatter_grid`` as one more series.
    """



    stats: dict = {}
    raw: dict = {}
    num_probes = None
    # obs/action are shared between streams (same rollout), pooled once here rather than inside
    # the per-stream loop below -- see estimate_true_value_same_state_mc's matching comment for
    # why these are kept (compute_gradient_alignment needs them). Only pooled when every source
    # has them (older raw dicts, or ones built before this field existed, simply omit it).
    if all('obs' in src and 'action' in src for src in raw_list):
        obs_pooled: list = []
        act_pooled: list = []
        for src in raw_list:
            obs_pooled.extend(src['obs'])
            act_pooled.extend(src['action'])
        raw['obs'] = obs_pooled
        raw['action'] = act_pooled
    for stream in ('r', 'c'):
        pred_list: list[float] = []
        mc_mean_list: list[float] = []
        target_list: list[float] = []
        returns_list: list[list[float]] = []
        target_repeats_list: list[list[float]] = []
        has_target = all('target' in src[stream] for src in raw_list)
        has_repeats = all('returns' in src[stream] for src in raw_list)
        for src in raw_list:
            pred_list.extend(src[stream]['pred'])
            mc_mean_list.extend(src[stream]['mc_mean'])
            if has_target:
                target_list.extend(src[stream]['target'])
            if has_repeats:
                returns_list.extend(src[stream]['returns'])
                target_repeats_list.extend(src[stream].get('target_repeats', []))
        pred_t, mc_mean_t = to_tensor(pred_list), to_tensor(mc_mean_list)
        raw[stream] = {'pred': pred_list, 'mc_mean': mc_mean_list}
        target_t = None
        if has_target:
            target_t = to_tensor(target_list)
            raw[stream]['target'] = target_list
        stats.update({
            f'{k}_{stream}': v
            for k, v in calibration_stats(
                pred_t, mc_mean_t, target_t,
                returns=returns_list if has_repeats else None,
                prefix=prefix,
            ).items()
        })
        if has_repeats:
            raw[stream]['returns'] = returns_list
            raw[stream]['target_repeats'] = target_repeats_list
    return stats, raw
