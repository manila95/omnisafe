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
"""Metrics and helpers for the value-evaluation studies in :mod:`omnisafe.utils.value_eval`."""

from __future__ import annotations

import math

import numpy as np
import torch

from omnisafe.utils.gae import calculate_adv_and_value_targets


def to_tensor(values, device: torch.device | None = None) -> torch.Tensor:
    """Pack per-probe floats into a 1-D float tensor."""
    return torch.tensor(values, device=device, dtype=torch.float32)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation; NaN when undefined (<2 points, or either side constant)."""
    if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def avg_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x``, ties sharing their average rank."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def spearman(a, b) -> float:
    """Rank correlation: Pearson on the ranks. Needs no threshold and ignores the tail."""
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=float)
    b = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b, dtype=float)
    return corr(to_tensor(avg_ranks(a)), to_tensor(avg_ranks(b)))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(positive ranked above negative), via Mann-Whitney U. NaN if either class is empty."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    n1 = int(label.sum())
    n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((avg_ranks(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ceiling(returns: list[list[float]]) -> float:
    """AUROC of the MC reference against itself (split-half).

    The label is an estimate, so a perfect critic cannot reach 1.0 at finite repeats; this is
    what is actually attainable. ``returns`` is ``n_probes x n_repeats``.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return float('nan')
    half = arr.shape[1] // 2
    h1, h2 = arr[:, :half].mean(1), arr[:, half:].mean(1)
    return float(np.nanmean([auroc(h1, h2 >= np.median(h2)), auroc(h2, h1 >= np.median(h1))]))


def calibration_stats(
    pred: torch.Tensor,
    mc_true: torch.Tensor,
    target: torch.Tensor | None = None,
    returns: list[list[float]] | None = None,
    prefix: str = '',
) -> dict[str, float]:
    """Calibration of ``pred`` against the MC value ``mc_true``, for one stream.

    ``*_pred_target`` asks whether the critic fits what it is trained to fit; ``*_target_true``
    whether that target is a good proxy for the truth -- a property of the estimator, not the
    critic. The plain metric is the end-to-end result the two factor.
    """
    stats = {
        f'{prefix}EstimationError': (mc_true - pred).mean().item(),
        f'{prefix}Correlation': corr(pred, mc_true),
        f'{prefix}Spearman': spearman(pred, mc_true),
    }
    # Median split, not a fixed value: scale-free, so one definition serves both streams and
    # stays comparable as the value range moves.
    truth = mc_true.detach().cpu().numpy()
    stats[f'{prefix}AUROC'] = auroc(pred.detach().cpu().numpy(), truth >= np.median(truth))
    if returns is not None:
        stats[f'{prefix}AUROC_ceiling'] = auroc_ceiling(returns)
    if target is not None:
        stats[f'{prefix}Correlation_pred_target'] = corr(pred, target)
        stats[f'{prefix}Correlation_target_true'] = corr(target, mc_true)
        stats[f'{prefix}Spearman_pred_target'] = spearman(pred, target)
        stats[f'{prefix}Spearman_target_true'] = spearman(target, mc_true)
        stats[f'{prefix}EstimationError_target_true'] = (mc_true - target).mean().item()
    return stats


def pool_correlation_stats(raw_list: list[dict], prefix: str = '') -> tuple[dict, dict]:
    """Pool several ``return_raw=True`` dicts and score them as one set.

    Not the per-source numbers averaged: pooling can surface a relationship, or wash one out,
    that no individual slice shows.
    """
    stats: dict = {}
    raw: dict = {}
    if all('obs' in src and 'action' in src for src in raw_list):
        raw['obs'] = [o for src in raw_list for o in src['obs']]
        raw['action'] = [a for src in raw_list for a in src['action']]

    for stream in ('r', 'c'):
        has_target = all('target' in src[stream] for src in raw_list)
        has_repeats = all('returns' in src[stream] for src in raw_list)

        def gather(field: str, stream: str = stream) -> list:
            return [v for src in raw_list for v in src[stream][field]]

        raw[stream] = {'pred': gather('pred'), 'mc_mean': gather('mc_mean')}
        target_t = None
        if has_target:
            raw[stream]['target'] = gather('target')
            target_t = to_tensor(raw[stream]['target'])
        if has_repeats:
            raw[stream]['returns'] = gather('returns')
            raw[stream]['target_repeats'] = [
                v for src in raw_list for v in src[stream].get('target_repeats', [])
            ]
        stats.update({
            f'{k}_{stream}': v
            for k, v in calibration_stats(
                to_tensor(raw[stream]['pred']),
                to_tensor(raw[stream]['mc_mean']),
                target_t,
                returns=raw[stream]['returns'] if has_repeats else None,
                prefix=prefix,
            ).items()
        })
    return stats, raw


def find_obs_normalizer(env):
    """The ``ObsNormalize`` wrapper's normalizer in ``env``'s chain, or None.

    Walked explicitly because ``Wrapper.__getattr__`` does not forward underscore names.
    """
    from omnisafe.envs.wrapper import ObsNormalize  # noqa: PLC0415

    while env is not None:
        if isinstance(env, ObsNormalize):
            return env._obs_normalizer  # noqa: SLF001
        env = getattr(env, '_env', None)
    return None


def sync_obs_normalizer(target_env, source_env) -> None:
    """Copy ``source_env``'s observation statistics into ``target_env``. No-op if either lacks one."""
    target_norm = find_obs_normalizer(target_env)
    source_norm = find_obs_normalizer(source_env)
    if target_norm is not None and source_norm is not None:
        target_norm.load_state_dict(source_norm.state_dict())


def rollout_target(
    r_seq: torch.Tensor,
    v_seq_incl_boot: torch.Tensor,
    terminated: bool,
    lam: float,
    gamma: float,
    advantage_estimator: str,
    logp_seq: torch.Tensor | None = None,
) -> float:
    """The training-style regression target at the first state of a probe rollout.

    Whatever ``adv_estimation_method`` computes, so the studies can compare the critic against
    its own target as well as against the truth. ``terminated`` is already folded into
    ``v_seq_incl_boot[-1]`` by the caller.
    """
    del terminated
    action_probs = logp_seq.exp() if advantage_estimator == 'vtrace' else None
    _, target = calculate_adv_and_value_targets(
        values=v_seq_incl_boot,
        rewards=torch.cat([r_seq, v_seq_incl_boot[-1:]]),
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
    """Smallest horizon whose remaining discounted weight is below ``bootstrap_threshold``.

    Uses the slower-decaying discount so the result is valid for both streams. Returns
    ``nominal_horizon`` unchanged when truncation is off or the discount does not decay.
    """
    if not bootstrap_threshold or bootstrap_threshold <= 0:
        return nominal_horizon
    gamma_max = max(discount_r, discount_c)
    if not 0 < gamma_max < 1:
        return nominal_horizon
    needed = math.ceil(math.log(bootstrap_threshold) / math.log(gamma_max))
    return max(1, min(nominal_horizon, needed))
