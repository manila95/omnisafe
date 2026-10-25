# Monte-Carlo value evaluation 

## What it does

Scores a critic against a Monte-Carlo estimate of the value it is supposed to
predict, by re-rolling the *same* states many times and averaging the realised
discounted return. Two studies, both run on an epoch's eval tick, before
`_update()` — so the critic measured is the one that produced that epoch's
advantages, not one already fitted to them.

**Start-state study** (`estimate_true_value_same_state_mc`). `reset(seed=X)`
reproduces a procedurally-generated layout exactly, so each probe seed gives a
fixed `s0` that can be rolled out `mc_repeats` times. Measures the critic at
episode starts.

**On-policy intermediate study** (`estimate_value_from_snapshots`). Episode
starts are not where a policy spends its time. This snapshots the simulator at
`intermediate_state_study_positions` into an on-policy rollout, then restores
and re-rolls from each snapshot. Entries are proportions of the episode, so one
config transfers across envs with different horizons: `0.3` is three-tenths of
the way in whether that is Goal's 1000 steps or Circle's 500. An entry `>= 1` is
instead read as an absolute step index, for explicit one-off overrides; any that
the episode never reaches is dropped with a logged note. Measures the critic where the current
policy actually is.

Both feed one pooled block logged under `ValueEval/`. Per-study breakdowns and
every raw array go to `eval_data/epoch_*.pkl` so anything not computed online
can be derived later.

## Metrics

Per stream (`_r`, `_c`), pooled over both studies:

| metric | question |
|---|---|
| `EstimationError` | is the critic biased? |
| `Correlation`, `Spearman` | does it rank states correctly? (Spearman needs no threshold and ignores the tail) |
| `AUROC` | P(a high-value state ranks above a low one), positive class = above the MC median |
| `AUROC_ceiling` | what a *perfect* critic could score at finite repeats, via split-half on the MC reference |
| `Correlation_pred_target`, `Spearman_pred_target` | does the critic fit what it is trained to fit? |
| `Correlation_target_true`, `Spearman_target_true`, `EstimationError_target_true` | is that target a good proxy for the truth? (a property of the estimator, not the critic) |

The last two groups factor the end-to-end result: a critic can rank poorly
because it fits its target badly, or because the target itself is a poor proxy.
`AUROC_ceiling` matters because the label is itself an estimate — at finite
repeats 1.0 is not attainable, so raw AUROC alone understates a good critic.

The median split keeps AUROC scale-free, so one definition serves both streams
and stays comparable as the value range moves during training.

## The no-bootstrap rule

The MC "true" value is a **pure simulated return**. Filling the tail of a
truncated rollout with the critic's own `V` would bake the critic's artifacts
into the very thing it is scored against — most visibly its sign: a discounted
cost-to-go is a sum of non-negative costs and cannot be negative, yet an
untrained `V_c` happily predicts negative values. Measured on a real 40-repeat
study under the old bootstrapping default, **14.4% of per-rollout cost returns
came out negative**.

So `mc_eval_tail: drop` is the default: rollouts truncate once the remaining
discounted weight falls below `mc_eval_bootstrap_threshold` (0.01), and the tail
is *dropped* rather than estimated. The resulting bias is known and bounded —
each value is short by at most the threshold's share of its own scale — rather
than contaminated by the critic.

## Files

| file | role |
|---|---|
| `utils/value_eval.py` | the two studies and their shared wave-batched rollout |
| `utils/value_utils.py` | metrics (`corr`, `spearman`, `auroc`, `auroc_ceiling`, `calibration_stats`, pooling) and env/normalizer helpers |
| `utils/state_snapshot.py` | simulator snapshot/restore behind the intermediate study |
| `utils/eval_data_dump.py` | the `eval_data/` dumps |
| `utils/gae.py` | shared advantage/target math, so the buffer and the studies cannot drift apart |

Plus `_run_eval_studies` / `_log_train_critic_diagnostics` in `policy_gradient`,
`Normalizer.normalize(update=)` and `ObsNormalize(update_stats=)`.


## Config

```yaml
eval_critic: True              # master switch (also gates train-batch diagnostics)
early_eval_freq: 5             # cadence below early_eval_epochs
early_eval_epochs: 50
value_eval_freq: 50            # cadence at or above it
mc_value_study_probes: 17
mc_value_study_repeats: 10
mc_value_study_vector_envs: 5  # parallelism, independent of train vector_env_nums
intermediate_state_study_positions: [0.1, 0.3, 0.5, 0.7, 0.9]  # fractions of the episode
mc_eval_bootstrap_threshold: 0.01
mc_eval_tail: drop
```

The first evaluation is always epoch 1, never epoch 0: at epoch 0 the rollout
comes from the freshly initialised policy and critics, so there is no trained
update to look at.


## Environments that cannot host a second env (MetaDrive)

Both studies above build their own environment instances alongside the training
env. MetaDrive cannot: its engine is a process-global singleton, so a second
`SafeMetaDriveEnv` in the same process either refuses to construct
(`Can not call this API after engine initialization!`) or shares the one engine
and corrupts the first (`Agents: 0 != Num_agents: 1`). It also exposes no engine
state save/restore, so the snapshot/restore the intermediate study relies on has
nothing to call.

Such an environment sets `eval_needs_isolated_process = True` on its env class,
and the studies then run through `omnisafe/utils/isolated_eval.py`, in a spawned
subprocess with its own engine. The training env is never touched. Results are
filed in exactly the same shape the in-process studies produce, so the pooling,
logging and `eval_data/` dumps are unchanged. Every other environment keeps the
in-process path, bit-for-bit.

Two differences on that path:

- **The intermediate study uses no snapshots.** It rolls one on-policy episode
  per probe, reads `V(s_t)` at each position and takes the realized discounted
  return-to-go from that same `t`. Prediction and target refer to the same
  visited state, so no restore is needed -- at the cost of one sample per probe
  (`MCRepeats` is 1, `MeanVar` is 0).
- **Episodes are ragged and unbounded.** They end on events, not a clock, so
  positions are proportions of `isolated_eval_episode_cap` rather than of a time
  limit, and a position no episode reached that epoch is reported and skipped
  (`NumProbes` therefore falls with depth).

```bash
IMAGE=mice:metadrive docker/run.sh python experiments/train_mice.py \
    --algo CPO --env-id SafeMetaDrive --total-steps 6000 --device cpu \
    --vector-env-nums 1 --algo-cfgs:eval_critic True \
    --algo-cfgs:isolated_eval_episode_cap 1000
```

```yaml
isolated_eval_episode_cap: 1500   # rollout cap; unused unless the env needs the isolated path
```

Because the subprocess is spawned rather than forked (a fork would inherit the
initialised engine), the training script must guard its top level with
`if __name__ == '__main__':`. `examples/train_policy.py` already does; a
hand-written script that does not gets an explicit error saying so.
