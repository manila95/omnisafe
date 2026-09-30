"""Exact environment-state snapshot/restore for Safety-Gymnasium ``Builder`` envs.

Lets us capture an on-policy-visited *intermediate* state during a rollout and later restore it
exactly (bit-for-bit observation match, and bit-for-bit reward/cost on every subsequent step when
replayed -- verified) in a different env instance, so the same same-state repeated-rollout
Monte-Carlo trick :func:`omnisafe.utils.value_eval.estimate_true_value_same_state_mc` uses for
:math:`s_0` can be applied to arbitrary states, not just resets.

Mechanism
---------
Started out as ``copy.deepcopy(task)`` -- Safety-Gymnasium's ``Task`` holds everything that
defines "the state that matters" (the mujoco model/data, the task's own RNG, reward-shaping
bookkeeping like ``last_dist_goal``), and both support ``copy.deepcopy`` natively, so this looked
like a complete, low-risk snapshot with no hand-picked fields to get subtly wrong.

It was wrong for a different reason: ``task.model`` carries ``model.tex_rgb``, a **~191 MB**
texture pixel buffer -- static rendering data, byte-identical across every layout of a given
``env_id`` (verified), utterly irrelevant to the physical state, and yet deep-copied *and*
pickled across the multiprocessing IPC boundary on every single snapshot/restore. That's fine for
one snapshot; at the actual call volume here (dozens of snapshot/restore events per eval epoch,
each fanning out across N worker subprocesses) it was the dominant cost by 1-2 orders of
magnitude, discovered when the intermediate-state study took >400s for what should have been a
~1-minute eval epoch.

So the snapshot is now a hand-picked set of exactly the fields that vary at runtime -- everything
static (geometry, textures, the kinematic tree) is simply never touched, on the reasoning that the
*restore target* already has its own byte-identical copy of anything static, from its own
construction:

- ``data.qpos`` / ``data.qvel`` / ``data.time`` / ``data.act`` -- all of mujoco's own dynamical
  state (tiny: a few floats).
- ``data.ctrl`` -- the last-applied actuator control signal. Easy to miss: it's not part of
  qpos/qvel, but ``mj_forward`` uses it to compute the *current* acceleration and everything
  downstream of that (e.g. the accelerometer sensor) as a "what if this force were applied right
  now" query -- found by an actual restored-observation mismatch (accelerometer off by ~2-22,
  everything else bit-exact) that traced to exactly this being left at whatever the *target*
  env's own prior rollout last set it to.
- ``model.body_pos`` / ``model.body_quat`` (the *whole* arrays, not just the goal's slice) --
  covers every body that can be runtime-repositioned (confirmed the goal is; captured
  unconditionally for hazards/vase/etc. too rather than assuming which ones currently are, since
  getting this wrong silently produces a wrong-but-plausible-looking restored layout instead of
  an error).
- ``task.last_dist_goal`` -- reward-shaping state specific to the Goal task family this codebase
  actually uses (SafetyPointGoal1-v0); a genuinely general version would need a per-task-type
  hook here, out of scope for the one env this module is used with.
- ``task.random_generator.random_generator.get_state()`` -- the task's own RNG state, so any
  future stochastic event during the *scored* rollout (e.g. a goal respawn) replays with the same
  quality of randomness as a real trajectory would have had, not a freshly-reseeded stream.
- ``builder.steps`` / ``.terminated`` / ``.truncated``, same as before.

Restoring writes these fields onto the *existing* ``data``/``model`` objects in place (rather than
replacing them wholesale) and calls ``mujoco.mj_forward`` to recompute every derived quantity
(``xpos``, sensor readings, contacts) consistent with the restored state -- mirroring exactly what
Safety-Gymnasium's own ``build_goal_position`` does after mutating ``body_pos``. Verified against
the original deepcopy-based design's own bit-exact-match / full-continuation-match tests before
being adopted (see the git history on this file for that comparison) -- still bit-exact, ~4 orders
of magnitude smaller payload.

A single restored-state snapshot is a few KB (dominated by the RNG state vector), not ~191 MB. The
mujoco-state fields (``qpos``/``qvel``/etc.) are individually cheap to copy -- what actually costs
anything is calling ``.copy()``/``mj_forward`` at all, on the order of the same ~0.2ms as a single
``env.step()``, not the ~130ms the old deepcopy-based version cost. Capturing/restoring dozens of
times per eval epoch is no longer a meaningfully separate cost from the rollout stepping itself.

Reaching into a vectorized env
-------------------------------
The actual ``Task`` objects live inside an ``AsyncVectorEnv``'s worker subprocesses, unreachable
directly from the main process. :func:`enable_state_snapshots` monkeypatches ``Builder.step`` at
*import time*, before any vectorized env's workers are forked. This repo's multiprocessing context
defaults to ``'fork'`` on Linux (confirmed empirically), so a subprocess forked after the patch is
applied inherits it automatically -- no per-worker setup needed. Each worker then checks its own
``snapshot_trigger_steps`` attribute and, when its own step counter hits one of those trigger
points, stashes a snapshot into ``info['state_snapshot']``. That flows back to the main process
for free, via the exact same info-dict pickling ``AsyncVectorEnv`` already uses for everything
else (e.g. ``info['final_observation']``).

Configuring that attribute across a vectorized env is one layer trickier than it looks:
``VectorEnv.set_attr`` only reaches the object each worker's ``env_fn()`` directly returned --
which, per ``safety_gymnasium.vector.make``'s own implementation, calls the very same
``safety_gymnasium.make()`` used for a single env, i.e. a ``Builder`` wrapped in
``SafeTimeLimit``. ``set_attr`` would set the attribute on that *wrapper*, not on the inner
``Builder`` our patched ``step`` actually checks -- and there's no cross-process way to reach
*into* a nested attribute directly. Two more wrinkles compound this: gymnasium's
``Wrapper.__getattr__`` proxies attribute *reads* down to the wrapped env, but that's reads only
(``setattr`` always lands on the wrapper itself, never proxied), and it explicitly refuses to
proxy any name starting with ``_`` (raises ``AttributeError``) -- so this can't be worked around
with a leading-underscore "private" name either. The fix used here:
:func:`enable_state_snapshots` also monkeypatches a small ``Builder.set_snapshot_trigger_steps``
*method* (not a bare attribute) onto the class, and :func:`configure_snapshot_triggers` invokes it
via the vector env's ``call()`` RPC (not ``set_attr``) -- since ``SafeTimeLimit`` doesn't define a
method by that name itself, its (non-underscore, so unblocked) ``__getattr__`` proxies the call
down to the wrapped ``Builder``, landing exactly where ``step`` looks for it.

Portability note: this relies on fork-based multiprocessing (the Linux default). It will not work
under a spawn/forkserver context, where a subprocess re-imports modules fresh rather than
inheriting patched state -- ``enable_state_snapshots()`` would need to run inside each worker's
own init in that case.

A second, non-obvious wrinkle in restoring (not just capturing): ``SafeTimeLimit`` -- the *same*
wrapper each vector worker's ``Builder`` sits inside -- tracks its own truncation independently,
via its own ``self._elapsed_steps`` counter (incremented every ``step()``, reset to 0 only by its
own ``reset()``). ``restore_builder`` only touches the inner ``Builder`` (``.steps``, ``.task``,
...); it doesn't reset the *wrapper's* separate counter. If left alone, that counter keeps
counting every step ever taken by that worker since its last real reset -- including the steps
spent capturing the snapshot in the first place -- and fires its own truncation at the wrong time,
regardless of what the restored ``Builder`` thinks its own step count is. So
:func:`enable_state_snapshots` also monkeypatches ``SafeTimeLimit.restore_and_get_obs`` (distinct
from ``Builder``'s own, since defining it on the outermost object takes priority over any
``__getattr__`` proxying, letting it get first crack at ``self._elapsed_steps`` before delegating
to the inner ``Builder``'s restore for everything else).
"""

from __future__ import annotations

import copy
import functools

import numpy as np
import torch
from gymnasium.vector.async_vector_env import AsyncState
from safety_gymnasium.builder import Builder
from safety_gymnasium.wrappers.time_limit import SafeTimeLimit

from omnisafe.utils.value_utils import find_obs_normalizer


_PATCHED = False


def enable_state_snapshots() -> None:
    """Monkeypatch ``Builder`` to support conditional state snapshotting.
    """
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return

    original_step = Builder.step

    @functools.wraps(original_step)
    def patched_step(self, action):
        obs, reward, cost, terminated, truncated, info = original_step(self, action)
        trigger_steps = getattr(self, 'snapshot_trigger_steps', None)
        if trigger_steps and self.steps in trigger_steps:
            info['state_snapshot'] = snapshot_builder(self)
        return obs, reward, cost, terminated, truncated, info

    def set_snapshot_trigger_steps(self, trigger_steps: set[int] | None) -> None:
        self.snapshot_trigger_steps = trigger_steps

    def restore_and_get_obs(self, snapshot: dict, reset_elapsed_steps: bool = False):
        restore_builder(self, snapshot)
        if reset_elapsed_steps:
            self.steps = 0
            self.terminated = False
            self.truncated = False
        return self.task.obs()

    def restore_and_get_obs_through_time_limit(self, snapshot: dict, reset_elapsed_steps: bool = False):
        self._elapsed_steps = 0 if reset_elapsed_steps else snapshot['steps']
        return self.env.restore_and_get_obs(snapshot, reset_elapsed_steps=reset_elapsed_steps)

    Builder.step = patched_step
    Builder.set_snapshot_trigger_steps = set_snapshot_trigger_steps
    Builder.restore_and_get_obs = restore_and_get_obs
    SafeTimeLimit.restore_and_get_obs = restore_and_get_obs_through_time_limit
    _PATCHED = True


def configure_snapshot_triggers(env, trigger_steps: set[int] | None) -> None:
    """Configure which within-episode step indices ``env`` should snapshot at.

    Args:
        env: A single (``num_envs == 1``) or vectorized env. Requires
        :func:`enable_state_snapshots` to have already been called (before the vectorized
        case's workers were forked).
        trigger_steps: The set of ``builder.steps`` values (i.e. steps *elapsed this episode*,
        1-indexed since ``Builder.step`` increments ``self.steps`` before returning) at which
        to snapshot. ``None`` or empty disables snapshotting.
    """
    target = _find_safety_gymnasium_target(env)
    if hasattr(target, 'call'):
        target.call('set_snapshot_trigger_steps', trigger_steps)
    else:
        target.set_snapshot_trigger_steps(trigger_steps)


def snapshot_builder(builder: Builder) -> dict:
    """Capture everything needed to exactly restore ``builder``'s current state.

    Hand-picked fields only, not a whole-object deepcopy -- see the module docstring's
    "Mechanism" section for why (``model.tex_rgb``, a ~191 MB static texture buffer, made the
    naive deepcopy-the-whole-task approach unusably slow at the actual call volume here).
    """
    data = builder.task.data
    model = builder.task.model
    return {
        'qpos': data.qpos.copy(),
        'qvel': data.qvel.copy(),
        'time': float(data.time),
        'act': data.act.copy() if model.na > 0 else None,
        'ctrl': data.ctrl.copy(),
        'body_pos': model.body_pos.copy(),
        'body_quat': model.body_quat.copy(),
        'last_dist_goal': builder.task.last_dist_goal,
        'rng_state': builder.task.random_generator.random_generator.get_state(),
        'steps': builder.steps,
        'terminated': builder.terminated,
        'truncated': builder.truncated,
    }


def restore_builder(builder: Builder, snapshot: dict) -> None:
    """Restore ``builder`` (in place) to a previously captured snapshot.
    """
    import mujoco  # noqa: PLC0415

    data = builder.task.data
    model = builder.task.model
    data.qpos[:] = snapshot['qpos']
    data.qvel[:] = snapshot['qvel']
    data.time = snapshot['time']
    if snapshot['act'] is not None:
        data.act[:] = snapshot['act']
    data.ctrl[:] = snapshot['ctrl']
    model.body_pos[:] = snapshot['body_pos']
    model.body_quat[:] = snapshot['body_quat']
    mujoco.mj_forward(model, data)
    builder.task.last_dist_goal = snapshot['last_dist_goal']
    builder.task.random_generator.random_generator.set_state(snapshot['rng_state'])
    builder.steps = snapshot['steps']
    builder.terminated = snapshot['terminated']
    builder.truncated = snapshot['truncated']


def _find_safety_gymnasium_target(env):
    """Walk down omnisafe's own wrapper chain (see :func:`configure_snapshot_triggers`) to
    whatever safety_gymnasium object is underneath: an ``AsyncVectorEnv``/``SyncVectorEnv``
    (num_envs > 1), or a single env wrapped in ``SafeTimeLimit`` (num_envs == 1, resolved via
    ``.unwrapped`` straight to the ``Builder``).
    """
    target = env
    while (
        not hasattr(target, 'parent_pipes')
        and not hasattr(target, 'envs')
        and not isinstance(target, Builder)
    ):
        inner = getattr(target, '_env', None)
        if inner is None:
            target = target.unwrapped
            break
        target = inner
    return target


def restore_and_get_obs(
    env,
    snapshots: list[dict],
    device: torch.device,
    reset_elapsed_steps: bool = False,
) -> torch.Tensor:
    """Restore a (possibly vectorized) env to ``snapshots`` (one entry per env slot -- each slot

    Args:
        reset_elapsed_steps: If True, the restored state's RL-episode bookkeeping (the ``Builder``
        step counter and the ``SafeTimeLimit`` wrapper's own ``_elapsed_steps``) is reset to 0
        instead of restored to its captured value -- i.e. the physical state is treated as a
        fresh start state with a full ``max_episode_steps`` budget ahead of it, not one bound
        by however many steps happened to be physically left in the episode it was captured
        from. See :func:`omnisafe.utils.value_eval.estimate_value_from_snapshots`'s docstring
        for why this is what an intermediate-state value study wants. Defaults to False
        (restore exactly as captured) for any other caller of this function.
    """

    target = _find_safety_gymnasium_target(env)

    if hasattr(target, 'parent_pipes'):  # AsyncVectorEnv
        assert len(snapshots) == target.num_envs
        target._assert_is_running()  # noqa: SLF001
        for pipe, snap in zip(target.parent_pipes, snapshots):
            pipe.send((
                '_call',
                ('restore_and_get_obs', (snap,), {'reset_elapsed_steps': reset_elapsed_steps}),
            ))
        target._state = AsyncState.WAITING_CALL  # noqa: SLF001
        results, successes = zip(*[pipe.recv() for pipe in target.parent_pipes])
        target._raise_if_errors(successes)  # noqa: SLF001
        target._state = AsyncState.DEFAULT  # noqa: SLF001
        raw_obs = np.stack(results)
    elif hasattr(target, 'envs'):  # SyncVectorEnv
        assert len(snapshots) == len(target.envs)
        raw_obs = np.stack([
            e.unwrapped.restore_and_get_obs(s, reset_elapsed_steps=reset_elapsed_steps)
            for e, s in zip(target.envs, snapshots)
        ])
    else:  # single Builder
        assert len(snapshots) == 1
        raw_obs = target.restore_and_get_obs(
            snapshots[0], reset_elapsed_steps=reset_elapsed_steps,
        )[None, :]

    obs = torch.as_tensor(raw_obs, dtype=torch.float32, device=device)
    normalizer = find_obs_normalizer(env)
    if normalizer is not None:
        obs = normalizer.normalize(obs, update=False)
    return obs


def collect_on_policy_snapshots(
    agent,
    env,
    trigger_steps,
    base_seed: int = 0,
) -> dict[int, list[dict]]:
    """Roll out ``env`` (vectorized, ``num_envs = N``) with the current stochastic policy,

    Args:
        agent: The actor-critic; must expose ``step(obs) -> (action, value_r, value_c, log_prob)``.
        env: A vectorized (``num_envs = N``) env to collect from. Its ``ObsNormalize`` should
        already be synced from the live training env (same rationale as
        ``estimate_true_value_same_state_mc``'s ``sync_normalizer_from``) *before* calling
        this, so the on-policy actions sampled here are representative of real training
        behavior -- there's no sync parameter here since that only needs doing once, not per
        collection call, and the caller already owns that env's lifecycle.
        trigger_steps: The within-episode step indices to snapshot at.
        base_seed: First of N consecutive seeds used to reset the N parallel episodes
        (``base_seed .. base_seed + N - 1``). Vary this per call (e.g. by epoch) for a fresh
        batch of on-policy states each time.

    Returns:
        Dict mapping each step index in ``trigger_steps`` to a list of N snapshots (one per env
        slot), in env-slot order.
    """
    trigger_steps = set(trigger_steps)
    configure_snapshot_triggers(env, trigger_steps)
    n_envs = env.num_envs
    max_t = max(trigger_steps)

    obs, _ = env.reset(seed=list(range(base_seed, base_seed + n_envs)))
    collected: dict[int, list[dict | None]] = {t: [None] * n_envs for t in trigger_steps}
    for t in range(max_t):
        act, _, _, _ = agent.step(obs)
        obs, _, _, _, _, info = env.step(act)
        if 'state_snapshot' in info:
            step_idx = t + 1
            if step_idx in collected:
                mask = info['_state_snapshot']
                for i, ok in enumerate(mask):
                    if ok:
                        collected[step_idx][i] = info['state_snapshot'][i]

    for t, snaps in collected.items():
        missing = [i for i, s in enumerate(snaps) if s is None]
        if missing:
            raise RuntimeError(
                f'collect_on_policy_snapshots: missing snapshots at step {t} for env slots '
                f'{missing} -- an episode likely ended early (broke the lockstep assumption; see '
                'docstring), or configure_snapshot_triggers/enable_state_snapshots was not set '
                'up correctly for this env.',
            )
    return collected  # type: ignore[return-value]
