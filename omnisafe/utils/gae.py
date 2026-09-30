"""Shared advantage/value-target computation, used by both the training buffer and the eval
value studies.

Previously this logic only existed as a method on :class:`~omnisafe.common.buffer.onpolicy_buffer
.OnPolicyBuffer` (``_calculate_adv_and_value_targets``). Pulled out here so
``omnisafe.utils.value_eval`` can compute the *exact* training-style target (whichever
``algo_cfgs.adv_estimation_method`` a run actually uses -- GAE, plain, vtrace, etc.) for the MC
value study's own probe rollouts, not just the true MC return -- see
:func:`omnisafe.utils.value_eval.estimate_true_value_same_state_mc`'s ``target`` field. Without
one shared implementation, the buffer and the eval studies would each carry their own copy of this
math, free to silently drift apart.
"""

from __future__ import annotations

import torch

from omnisafe.typing import AdvatageEstimator
from omnisafe.utils.math import discount_cumsum


#: Legacy single-name estimators, expanded into the two axes they always secretly were:
#: ``(advantage, value_target)``. ``advantage_estimator`` picked *both* the policy gradient's
#: advantage and the critic's regression target, so only 6 of the 3x3 combinations were reachable
#: -- and notably "TD(0) advantage with a TD(lambda) target" was not expressible at all. Keeping
#: the old names as presets means every existing config resolves to exactly what it resolved to
#: before, while `value_target` can now be set independently.
ADV_TARGET_PRESETS: dict[str, tuple[str, str]] = {
    'gae': ('gae', 'td_lambda'),
    'gae-rtg': ('gae', 'mc'),
    'td_zero_gae': ('gae', 'td_zero'),
    'plain': ('td_zero', 'mc'),
    'td_zero': ('td_zero', 'td_zero'),
    'reinforce': ('mc', 'mc'),
    'vtrace': ('vtrace', 'vtrace'),
}

#: Valid values for each axis when set explicitly.
ADVANTAGE_METHODS = ('gae', 'td_zero', 'mc', 'vtrace')
VALUE_TARGET_METHODS = ('td_lambda', 'mc', 'td_zero', 'vtrace')


def resolve_adv_and_target(
    advantage_estimator: str,
    value_target: str | None = None,
) -> tuple[str, str]:
    """Resolve the (advantage, value_target) pair actually in force.

    ``value_target=None`` reproduces the legacy coupling by expanding ``advantage_estimator``
    through :data:`ADV_TARGET_PRESETS`. Naming an axis explicitly overrides just that axis, so
    ``advantage_estimator='td_zero', value_target='td_lambda'`` -- previously unreachable --
    works.

    Args:
        advantage_estimator: A legacy preset name, or an :data:`ADVANTAGE_METHODS` value.
        value_target: An explicit :data:`VALUE_TARGET_METHODS` value, or None to use the preset.

    Returns:
        ``(advantage, value_target)``.

    Raises:
        NotImplementedError: If either axis is not recognised.
    """
    if value_target is None:
        if advantage_estimator not in ADV_TARGET_PRESETS:
            raise NotImplementedError(
                f'Unknown advantage_estimator {advantage_estimator!r}. Either use a preset '
                f'{sorted(ADV_TARGET_PRESETS)} or set value_target explicitly alongside one of '
                f'{list(ADVANTAGE_METHODS)}.',
            )
        return ADV_TARGET_PRESETS[advantage_estimator]
    adv = ADV_TARGET_PRESETS[advantage_estimator][0] if advantage_estimator in ADV_TARGET_PRESETS \
        else advantage_estimator
    if adv not in ADVANTAGE_METHODS:
        raise NotImplementedError(f'Unknown advantage method {adv!r}; expected {list(ADVANTAGE_METHODS)}')
    if value_target not in VALUE_TARGET_METHODS:
        raise NotImplementedError(
            f'Unknown value_target {value_target!r}; expected {list(VALUE_TARGET_METHODS)}',
        )
    if (adv == 'vtrace') != (value_target == 'vtrace'):
        # V-trace derives both from one recursion; mixing half of it with another axis would
        # silently use a target that does not correspond to the advantage's own correction.
        raise NotImplementedError(
            "'vtrace' must be used on both axes or neither (got advantage="
            f'{adv!r}, value_target={value_target!r}).',
        )
    return adv, value_target


def calculate_adv_and_value_targets(
    values: torch.Tensor,
    rewards: torch.Tensor,
    lam: float,
    gamma: float,
    advantage_estimator: AdvatageEstimator,
    value_target: str | None = None,
    action_probs: torch.Tensor | None = None,
    behavior_action_probs: torch.Tensor | None = None,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the estimated advantage and value-function regression target.

    See :meth:`omnisafe.common.buffer.onpolicy_buffer.OnPolicyBuffer._calculate_adv_and_value_
    targets` for the full formula docstrings (gae/gae-rtg/vtrace/plain) -- this is that same
    logic, as a standalone function so it has exactly one implementation.

    Args:
        values: ``(T+1,)`` critic values along the path, including the bootstrap value appended
            as the final entry (0 if the path ended in a true terminal, otherwise the critic's
            own prediction at the truncation point -- see ``OnPolicyAdapter.rollout``'s
            ``last_value_r``/``last_value_c`` construction, which this must mirror exactly to
            produce a genuinely comparable target).
        rewards: ``(T+1,)`` rewards along the path, with that same bootstrap value appended as
            the pseudo-final "reward" too (so rewards-to-go-style targets fold in the bootstrap
            the same way GAE's use of ``values[1:]`` does).
        lam: GAE lambda.
        gamma: Discount factor.
        advantage_estimator: A legacy preset (``'gae'``, ``'gae-rtg'``, ``'vtrace'``,
            ``'plain'``, ``'reinforce'``, ``'td_zero'``, ``'td_zero_gae'``) or, when
            ``value_target`` is also given, an advantage method: ``'gae'``, ``'td_zero'``,
            ``'mc'``, ``'vtrace'``.
        value_target: The critic's regression target, chosen independently of the advantage:
            ``'td_lambda'``, ``'mc'``, ``'td_zero'``, ``'vtrace'``. ``None`` keeps the legacy
            coupling, expanding ``advantage_estimator`` through :data:`ADV_TARGET_PRESETS`.
        action_probs: Policy action probabilities along the path (``'vtrace'`` only).
        behavior_action_probs: Behavior-policy action probabilities (``'vtrace'`` only; equal to
            ``action_probs`` for a genuinely on-policy call, giving an importance ratio of 1).
        rho_bar: V-trace truncation level for the importance ratio (``'vtrace'`` only).
        c_bar: V-trace truncation level for the trace coefficient (``'vtrace'`` only).

    Returns:
        ``(advantage, target_value)``, each ``(T,)``.
    """
    adv_method, target_method = resolve_adv_and_target(advantage_estimator, value_target)

    # V-trace is a single recursion producing both quantities together, so it cannot be split
    # across the two axes (resolve_adv_and_target enforces that).
    if adv_method == 'vtrace':
        assert action_probs is not None and behavior_action_probs is not None, (
            "advantage_estimator == 'vtrace' requires action_probs/behavior_action_probs"
        )
        target_value, adv, _ = calculate_v_trace(
            policy_action_probs=action_probs,
            values=values,
            rewards=rewards,
            behavior_action_probs=behavior_action_probs,
            gamma=gamma,
            rho_bar=rho_bar,
            c_bar=c_bar,
        )
        return adv, target_value

    # One-step TD residuals, the shared building block of both the GAE advantage and the TD
    # targets below. deltas_t = r_t + gamma * V(s_{t+1}) - V(s_t).
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    # --- advantage axis: what the policy gradient is weighted by -------------------------
    if adv_method == 'gae':
        adv = discount_cumsum(deltas, gamma * lam)
    elif adv_method == 'td_zero':
        adv = deltas
    else:  # 'mc'
        adv = discount_cumsum(rewards, gamma)[:-1]

    # --- value-target axis: what the critic regresses toward -----------------------------
    # Independent of the advantage above. Previously these were welded together, which made
    # combinations like "TD(0) advantage with a TD(lambda) target" unreachable, and meant a
    # comparison between two estimators silently varied both axes at once.
    if target_method == 'td_lambda':
        # TD(lambda) = GAE(lambda) + V, computed from its own GAE regardless of which advantage
        # the policy is using -- so this no longer implies adv_method == 'gae'.
        target_value = discount_cumsum(deltas, gamma * lam) + values[:-1]
    elif target_method == 'td_zero':
        target_value = rewards[:-1] + gamma * values[1:]
    else:  # 'mc'
        target_value = discount_cumsum(rewards, gamma)[:-1]

    return adv, target_value


# pylint: disable-next=too-many-arguments,too-many-locals
def calculate_v_trace(
    policy_action_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    behavior_action_probs: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute V-trace targets -- byte-for-byte the same recursion as ``OnPolicyBuffer``'s
    original (now-removed) ``_calculate_v_trace`` static method, moved here verbatim rather than
    reimplemented, specifically to avoid a second, independently-written copy of this recursion
    silently drifting from the training path's actual behavior. See Espeholt et al. 2018, IMPALA:
    https://arxiv.org/abs/1802.01561.

    Args:
        policy_action_probs: Action probabilities of the policy.
        values: The value of states, including the bootstrap value.
        rewards: The reward of states, including the bootstrap value.
        behavior_action_probs: Action probabilities of the behavior policy.
        gamma: The discount factor. Defaults to 0.99.
        rho_bar: The maximum value of importance weights. Defaults to 1.0.
        c_bar: The maximum value of clipped importance weights. Defaults to 1.0.

    Returns:
        V-trace targets, shape=(batch_size, sequence_length)

    Raises:
        AssertionError: If the input tensors are scalars.
        AssertionError: If c_bar is greater than rho_bar.
    """
    assert values.ndim in (1, 2), 'Please provide arrays instead of scalars'
    assert rewards.ndim in (1, 2), 'Please provide arrays instead of scalars'
    assert policy_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
    assert behavior_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
    assert c_bar <= rho_bar, 'c_bar should be less than or equal to rho_bar'

    sequence_length = policy_action_probs.shape[0]
    # pylint: disable-next=assignment-from-no-return
    rhos = torch.div(policy_action_probs, behavior_action_probs)
    clip_rhos = torch.min(rhos, torch.as_tensor(rho_bar))  # pylint: disable=assignment-from-no-return
    clip_cs = torch.min(rhos, torch.as_tensor(c_bar))  # pylint: disable=assignment-from-no-return
    if values.ndim == 2:
        # broadcast the per-timestep scalar importance ratio against a (T, D) feature stream.
        clip_rhos = clip_rhos.unsqueeze(-1)
        clip_cs = clip_cs.unsqueeze(-1)
    v_s = values[:-1].clone()  # copy all values except bootstrap value
    last_v_s = values[-1]  # bootstrap from last state

    for index in reversed(range(sequence_length)):
        delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
        v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
        last_v_s = v_s[index]  # accumulate current v_s for next iteration

    v_s_plus_1 = torch.cat((v_s[1:], values[-1:]))
    policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

    return v_s, policy_advantage, clip_rhos
