"""Value-function evaluation for environments that cannot host a second env in-process.

The studies in :mod:`omnisafe.utils.value_eval` need their own environment instances, separate
from the training env: the s0 study resets to fixed probe layouts, and the intermediate-state
study rolls its own on-policy episodes. Safety-Gymnasium is happy to have several envs alive at
once, so those studies just build dedicated ones.

MetaDrive is not. Its engine is a process-global singleton, and the failure is not subtle:

- constructing a second ``SafeMetaDriveEnv`` once the first has been reset raises
  ``AssertionError: Can not call this API after engine initialization!``;
- constructing both *before* either is reset gets past that, but then they share the one engine
  and tread on each other (``AssertionError: Agents: 0 != Num_agents: 1``);
- there is no engine state save/restore API, so the mujoco-style snapshot/restore that
  :mod:`omnisafe.utils.state_snapshot` relies on has no equivalent to call.

Reusing the *training* env instead would sidestep the singleton but resets it mid-training, which
changes the very run being measured. So the studies are run in a spawned subprocess with its own
engine: the training env is never touched, and the parent only pays a process spawn plus a
state-dict transfer per eval epoch (both negligible against the rollouts themselves).

Two further things differ from the Safety-Gymnasium path and are handled here rather than by
relaxing the shared code (which would change behaviour for every other env):

- **Ragged episodes.** MetaDrive episodes end on events -- a crash, reaching the destination --
  not on a time limit, so they have different lengths and the shared rollout loop's "every probe
  is done at exactly ``max_episode_steps``" assumption does not hold. The loops here stop at each
  episode's own ``done`` and discount accordingly.
- **No horizon at all.** ``config['horizon']`` is ``None``, and an episode that never crashes
  never ends (a zero-action car simply idles indefinitely), so every rollout needs an explicit
  cap -- ``algo_cfgs.isolated_eval_episode_cap``. A rollout stopped by that cap is not a terminal
  state, so it is bootstrapped with the critic's own value exactly as a time-limit truncation is.

**Snapshots are not used.** The intermediate-state study here is a single on-policy pass: roll an
episode, read ``V(s_t)`` at each configured position, and take the realized discounted
return-to-go from that same ``t`` as the MC target. Prediction and target refer to the same
visited state, so no restore is needed -- which is what makes the study possible on an env with
no state-restore API at all. The cost is that each intermediate probe is a single sample
(``MCRepeats`` is 1), so its ``MeanVar`` is 0 and correlations are noisier at equal probe counts
than the snapshot-based study's repeat-averaged ones.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any

import numpy as np
import torch


__all__ = ['env_needs_isolated_eval', 'run_isolated_value_studies']


def env_needs_isolated_eval(env_id: str) -> bool:
    """Whether ``env_id``'s env class declares that it cannot share a process with a second env.

    Read off the registered env class rather than matched against a list of ids here, so the
    answer lives next to the environment that knows it (see
    ``CMDP.eval_needs_isolated_process``).
    """
    from omnisafe.envs.core import ENV_REGISTRY  # noqa: PLC0415

    try:
        env_class = ENV_REGISTRY.get_class(env_id, None)
    except (KeyError, ValueError):
        return False
    return bool(getattr(env_class, 'eval_needs_isolated_process', False))


def _build_env(env_id: str, env_cfgs: dict, obs_normalize: bool, norm_state: dict | None):
    """Rebuild the training wrapper stack around a fresh single env, inside the child."""
    from omnisafe.envs.core import make as make_env  # noqa: PLC0415
    from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze  # noqa: PLC0415
    try:  # the two checkouts keep this helper in different modules
        from omnisafe.utils.value_utils import find_obs_normalizer  # noqa: PLC0415
    except ImportError:
        from omnisafe.utils.value_eval import _find_obs_normalizer as find_obs_normalizer  # noqa: PLC0415

    device = torch.device('cpu')
    env = make_env(env_id, num_envs=1, device=device, **env_cfgs)
    if env.need_time_limit_wrapper:
        env = TimeLimit(env, time_limit=env.max_episode_steps, device=device)
    if env.need_auto_reset_wrapper:
        env = AutoReset(env, device=device)
    if obs_normalize:
        # update_stats=False, same as every other eval env in this codebase: the statistics are a
        # snapshot of the training env's, copied in below, and must not drift while probing.
        env = ObsNormalize(env, device=device, update_stats=False)
        if norm_state is not None:
            normalizer = find_obs_normalizer(env)
            if normalizer is not None:
                normalizer.load_state_dict(norm_state)
    env = ActionScale(env, low=-1.0, high=1.0, device=device)
    return Unsqueeze(env, device=device)


def _build_agent(obs_space, act_space, model_cfgs, epochs: int, actor_state: dict):
    from omnisafe.models.actor_critic.constraint_actor_critic import (  # noqa: PLC0415
        ConstraintActorCritic,
    )

    agent = ConstraintActorCritic(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=model_cfgs,
        epochs=epochs,
    )
    agent.load_state_dict(actor_state)
    agent.eval()
    return agent


def _scenario_count(env) -> int | None:
    """How many distinct layouts this env will accept as a reset seed, if it is limited.

    MetaDrive indexes a fixed bank of generated scenarios and rejects anything outside
    ``[0, num_scenarios)``, unlike Safety-Gymnasium where the seed is just an RNG seed and any
    integer is valid. Walk the wrapper chain to whichever object knows.
    """
    target = env
    for _ in range(16):
        count = getattr(target, '_num_scenarios', None) or getattr(target, 'num_scenarios', None)
        if count:
            return int(count)
        target = getattr(target, '_env', None)
        if target is None:
            return None
    return None


def _as_reset_seed(seed: int, scenario_count: int | None) -> int:
    """Fold a probe seed into the env's valid layout range, if it has one."""
    return seed if scenario_count is None else int(seed) % scenario_count


def _roll_one_episode(agent, env, seed: int, cap: int, scenario_count: int | None = None) -> dict:
    """Roll a single on-policy episode to its own ``done`` (or ``cap``), recording per-step data.

    Returns the per-step reward/cost/value/log-prob sequences plus the bootstrap value at the
    final observation, which is 0 for a true terminal and the critic's own prediction when the
    episode was cut short by ``cap`` (an artificial truncation, exactly like a time limit).
    """
    obs, _ = env.reset(seed=_as_reset_seed(seed, scenario_count))
    rewards: list[float] = []
    costs: list[float] = []
    values_r: list[float] = []
    values_c: list[float] = []
    logps: list[float] = []
    obs_seq: list[np.ndarray] = []
    act_seq: list[np.ndarray] = []
    terminated_at_end = False
    for _ in range(cap):
        with torch.no_grad():
            act, value_r, value_c, log_prob = agent.step(obs)
        values_r.append(float(value_r.reshape(-1)[0]))
        values_c.append(float(value_c.reshape(-1)[0]))
        logps.append(float(log_prob.reshape(-1)[0]))
        obs_seq.append(obs.detach().cpu().reshape(-1).numpy().copy())
        act_seq.append(act.detach().cpu().reshape(-1).numpy().copy())
        obs, reward, cost, terminated, truncated, _ = env.step(act)
        rewards.append(float(reward.reshape(-1)[0]))
        costs.append(float(cost.reshape(-1)[0]))
        if bool(terminated.reshape(-1)[0]) or bool(truncated.reshape(-1)[0]):
            # A terminal ends the value recursion with 0; a time-limit truncation does not.
            terminated_at_end = bool(terminated.reshape(-1)[0])
            break
    with torch.no_grad():
        _, boot_r, boot_c, _ = agent.step(obs)
    return {
        'rewards': np.asarray(rewards, dtype=np.float64),
        'costs': np.asarray(costs, dtype=np.float64),
        'values_r': np.asarray(values_r, dtype=np.float64),
        'values_c': np.asarray(values_c, dtype=np.float64),
        'logps': np.asarray(logps, dtype=np.float64),
        'obs': obs_seq,
        'act': act_seq,
        'terminated': terminated_at_end,
        'boot_r': 0.0 if terminated_at_end else float(boot_r.reshape(-1)[0]),
        'boot_c': 0.0 if terminated_at_end else float(boot_c.reshape(-1)[0]),
    }


def _return_from(episode: dict, start: int, discount_r: float, discount_c: float) -> tuple[float, float]:
    """Realized discounted return-to-go from step ``start``, including the bootstrap tail.

    The tail term is 0 whenever the episode ended in a true terminal (``boot_*`` is already
    zeroed there), so this is the exact discounted return for a completed episode and a
    critic-bootstrapped one only where the rollout was cut off.
    """
    rewards, costs = episode['rewards'][start:], episode['costs'][start:]
    n = len(rewards)
    disc_r = discount_r ** np.arange(n)
    disc_c = discount_c ** np.arange(n)
    ret_r = float(np.sum(disc_r * rewards) + (discount_r**n) * episode['boot_r'])
    ret_c = float(np.sum(disc_c * costs) + (discount_c**n) * episode['boot_c'])
    return ret_r, ret_c


def _target_from(episode: dict, start: int, cfgs, discount_r: float, discount_c: float) -> tuple[float, float]:
    """The training-style value target at ``start``, for the target-vs-truth decomposition.

    Mirrors ``estimate_true_value_same_state_mc``'s use of the shared rollout-target helper -- same estimator,
    same lambda, same penalty-coefficient folding -- so the ``*_target_*`` statistics mean the
    same thing on this path as on the Safety-Gymnasium one.
    """
    try:  # the two checkouts keep this helper in different modules, under different names
        from omnisafe.utils.value_utils import rollout_target  # noqa: PLC0415
    except ImportError:
        from omnisafe.utils.value_eval import _rollout_target as rollout_target  # noqa: PLC0415

    adv_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    adv_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_r
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
    penalty_coef = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

    rewards = torch.from_numpy(episode['rewards'][start:]).float()
    costs = torch.from_numpy(episode['costs'][start:]).float()
    logps = torch.from_numpy(episode['logps'][start:]).float()
    # Value sequence needs one extra slot at the end for the bootstrap, matching _rollout_target's
    # expected (T+1,) shape.
    values_r = torch.from_numpy(
        np.append(episode['values_r'][start:], episode['boot_r']),
    ).float()
    values_c = torch.from_numpy(
        np.append(episode['values_c'][start:], episode['boot_c']),
    ).float()
    target_r = rollout_target(
        rewards - penalty_coef * costs, values_r, episode['terminated'], lam_r, discount_r,
        adv_r, logp_seq=logps,
    )
    target_c = rollout_target(
        costs, values_c, episode['terminated'], lam_c, discount_c, adv_c, logp_seq=logps,
    )
    return target_r, target_c


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _assemble(probes: list[dict], n_repeats: int, prefix: str) -> tuple[dict, dict]:
    """Build the ``(stats, raw)`` pair in exactly the shape the shared study functions return.

    ``probes`` is one entry per probe, each holding the per-repeat predictions/returns/targets.
    Keeping this identical to ``estimate_true_value_same_state_mc``'s output is what lets the
    isolated path feed the same pooling, logging and eval_data-dump code unchanged.
    """
    pred_r = [p['pred_r'] for p in probes]
    pred_c = [p['pred_c'] for p in probes]
    mc_mean_r = [float(np.mean(p['returns_r'])) for p in probes]
    mc_mean_c = [float(np.mean(p['returns_c'])) for p in probes]
    mc_var_r = [float(np.var(p['returns_r'])) for p in probes]
    mc_var_c = [float(np.var(p['returns_c'])) for p in probes]
    target_r = [float(np.mean(p['targets_r'])) for p in probes]
    target_c = [float(np.mean(p['targets_c'])) for p in probes]

    def _t(lst):
        return torch.tensor(lst, dtype=torch.float32)

    pred_r_t, pred_c_t = _t(pred_r), _t(pred_c)
    mc_r_t, mc_c_t = _t(mc_mean_r), _t(mc_mean_c)
    tgt_r_t, tgt_c_t = _t(target_r), _t(target_c)
    stats = {
        f'{prefix}EstimationError_r': (mc_r_t - pred_r_t).mean().item(),
        f'{prefix}EstimationError_c': (mc_c_t - pred_c_t).mean().item(),
        f'{prefix}Correlation_r': _corr(pred_r_t, mc_r_t),
        f'{prefix}Correlation_c': _corr(pred_c_t, mc_c_t),
        f'{prefix}EstimationError_target_true_r': (mc_r_t - tgt_r_t).mean().item(),
        f'{prefix}EstimationError_target_true_c': (mc_c_t - tgt_c_t).mean().item(),
        f'{prefix}Correlation_target_true_r': _corr(tgt_r_t, mc_r_t),
        f'{prefix}Correlation_target_true_c': _corr(tgt_c_t, mc_c_t),
        f'{prefix}Correlation_pred_target_r': _corr(pred_r_t, tgt_r_t),
        f'{prefix}Correlation_pred_target_c': _corr(pred_c_t, tgt_c_t),
        f'{prefix}MeanTarget_r': tgt_r_t.mean().item(),
        f'{prefix}MeanTarget_c': tgt_c_t.mean().item(),
        f'{prefix}MeanVar_r': float(np.mean(mc_var_r)),
        f'{prefix}MeanVar_c': float(np.mean(mc_var_c)),
        f'{prefix}MeanTrue_r': mc_r_t.mean().item(),
        f'{prefix}MeanTrue_c': mc_c_t.mean().item(),
        f'{prefix}MeanPred_r': pred_r_t.mean().item(),
        f'{prefix}MeanPred_c': pred_c_t.mean().item(),
        f'{prefix}NumProbes': float(len(probes)),
        f'{prefix}MCRepeats': float(n_repeats),
    }
    raw = {
        'probe_seeds': [p['seed'] for p in probes],
        'obs': [p['obs'] for p in probes],
        'action': [p['act'] for p in probes],
        'r': {
            'pred': pred_r, 'mc_mean': mc_mean_r, 'mc_var': mc_var_r, 'target': target_r,
            'returns': [p['returns_r'] for p in probes],
            'target_repeats': [p['targets_r'] for p in probes],
        },
        'c': {
            'pred': pred_c, 'mc_mean': mc_mean_c, 'mc_var': mc_var_c, 'target': target_c,
            'returns': [p['returns_c'] for p in probes],
            'target_repeats': [p['targets_c'] for p in probes],
        },
    }
    return stats, raw


def _child(queue, payload: dict) -> None:
    """Entry point of the spawned process: build env + agent, run both studies, ship results."""
    try:
        cfgs = payload['cfgs']
        discount_r, discount_c = payload['discount_r'], payload['discount_c']
        cap = payload['episode_cap']
        env = _build_env(
            payload['env_id'], payload['env_cfgs'], payload['obs_normalize'], payload['norm_state'],
        )
        agent = _build_agent(
            payload['obs_space'], payload['act_space'], payload['model_cfgs'],
            payload['epochs'], payload['actor_state'],
        )
        scenario_count = _scenario_count(env)
        result: dict[str, Any] = {'episode_lengths': [], 'scenario_count': scenario_count}

        # s0 study: fixed probe layouts, mc_repeats independent rollouts each. Same layout every
        # eval epoch (the seeds are fixed by the caller), so the numbers are comparable over time.
        probes = []
        for seed in payload['probe_seeds']:
            entry = {
                'seed': seed, 'returns_r': [], 'returns_c': [],
                'targets_r': [], 'targets_c': [], 'obs': [], 'act': [],
            }
            for repeat in range(payload['mc_repeats']):
                ep = _roll_one_episode(agent, env, seed, cap, scenario_count)
                result['episode_lengths'].append(int(len(ep['rewards'])))
                ret_r, ret_c = _return_from(ep, 0, discount_r, discount_c)
                tgt_r, tgt_c = _target_from(ep, 0, cfgs, discount_r, discount_c)
                entry['returns_r'].append(ret_r)
                entry['returns_c'].append(ret_c)
                entry['targets_r'].append(tgt_r)
                entry['targets_c'].append(tgt_c)
                entry['obs'].append(ep['obs'][0])
                entry['act'].append(ep['act'][0])
                if repeat == 0:
                    # V(s0) does not depend on the repeat (no action taken yet).
                    entry['pred_r'] = float(ep['values_r'][0])
                    entry['pred_c'] = float(ep['values_c'][0])
            probes.append(entry)
        result['mc_study'] = _assemble(probes, payload['mc_repeats'], 'MCStudy/')

        # Intermediate-state study: one on-policy episode per probe, read at each position. No
        # snapshot/restore is involved -- see this module's docstring.
        positions = payload['positions']
        per_pos: dict[int, list[dict]] = {pos: [] for pos in positions}
        for i in range(payload['interm_probes']):
            seed = payload['interm_base_seed'] + i
            ep = _roll_one_episode(agent, env, seed, cap, scenario_count)
            result['episode_lengths'].append(int(len(ep['rewards'])))
            for pos in positions:
                if pos >= len(ep['rewards']):
                    # This episode ended before the position -- that state was never visited, so
                    # there is nothing to score. Positions deep into the episode therefore get
                    # fewer probes than shallow ones, which NumProbes records per position.
                    continue
                ret_r, ret_c = _return_from(ep, pos, discount_r, discount_c)
                tgt_r, tgt_c = _target_from(ep, pos, cfgs, discount_r, discount_c)
                per_pos[pos].append({
                    'seed': seed,
                    'pred_r': float(ep['values_r'][pos]), 'pred_c': float(ep['values_c'][pos]),
                    'returns_r': [ret_r], 'returns_c': [ret_c],
                    'targets_r': [tgt_r], 'targets_c': [tgt_c],
                    'obs': [ep['obs'][pos]], 'act': [ep['act'][pos]],
                })
        result['intermediate'] = {
            pos: _assemble(entries, 1, '') for pos, entries in per_pos.items() if entries
        }
        result['skipped_positions'] = [pos for pos, entries in per_pos.items() if not entries]
        env.close()
        queue.put(('ok', result))
    except Exception:  # noqa: BLE001 # pylint: disable=broad-except
        queue.put(('error', traceback.format_exc()))


def run_isolated_value_studies(
    *,
    env_id: str,
    env_cfgs: dict,
    cfgs,
    model_cfgs,
    obs_space,
    act_space,
    epochs: int,
    actor_state: dict,
    norm_state: dict | None,
    obs_normalize: bool,
    probe_seeds: list[int],
    mc_repeats: int,
    positions: list[int],
    interm_probes: int,
    interm_base_seed: int,
    discount_r: float,
    discount_c: float,
    episode_cap: int,
    timeout: float | None = None,
) -> dict:
    """Run both value studies in a spawned subprocess and return their results.

    Returns:
        A dict with ``mc_study`` (a ``(stats, raw)`` pair), ``intermediate`` (position ->
        ``(stats, raw)``), ``skipped_positions`` and ``episode_lengths``.

    Raises:
        RuntimeError: If the child fails; the child's traceback is included verbatim.
    """
    payload = {
        'env_id': env_id, 'env_cfgs': env_cfgs, 'cfgs': cfgs, 'model_cfgs': model_cfgs,
        'obs_space': obs_space, 'act_space': act_space, 'epochs': epochs,
        'actor_state': {k: v.cpu() for k, v in actor_state.items()},
        'norm_state': norm_state, 'obs_normalize': obs_normalize,
        'probe_seeds': list(probe_seeds), 'mc_repeats': int(mc_repeats),
        'positions': list(positions), 'interm_probes': int(interm_probes),
        'interm_base_seed': int(interm_base_seed),
        'discount_r': float(discount_r), 'discount_c': float(discount_c),
        'episode_cap': int(episode_cap),
    }
    # spawn, not fork: a forked child would inherit the parent's already-initialised engine,
    # which is the whole thing being avoided here.
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(queue, payload), daemon=True)
    try:
        proc.start()
    except RuntimeError as exc:
        if 'freeze_support' not in str(exc) and 'bootstrapping' not in str(exc):
            raise
        raise RuntimeError(
            'isolated value study could not start its subprocess: spawning re-imports the '
            'entry module, so the training script must guard its top level with '
            "`if __name__ == '__main__':` (omnisafe's own examples/train_policy.py already "
            'does). Only environments that need an isolated eval process are affected.',
        ) from exc
    try:
        # Drain the queue before joining: a large result still in the pipe would deadlock a
        # join() that runs first.
        status, result = queue.get(timeout=timeout)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
    if status != 'ok':
        raise RuntimeError(f'isolated value study failed in its subprocess:\n{result}')
    # Tensors cannot cross the queue (their reducer hands over shared-memory handles that die
    # with the child), so the child ships numpy and they are rebuilt here -- `raw` then has
    # exactly the shape the in-process studies return.
    for _stats, raw in [result['mc_study'], *result['intermediate'].values()]:
        del _stats
        for key in ('obs', 'action'):
            raw[key] = [[torch.from_numpy(a) for a in per_probe] for per_probe in raw[key]]
    return result
