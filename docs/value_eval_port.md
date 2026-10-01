# Monte-Carlo value evaluation — port notes

Ported from the MICE branch `mc-eval-no-bootstrap` onto `code-release`.
Commits `c1d4b50` (add), `ac2f6c2` (compact), `817744b` (factor), `e64316b`
(parity), `4dae7c8` (defaults + progress bar).

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
`intermediate_state_study_positions` steps into an on-policy rollout, then
restores and re-rolls from each snapshot. Measures the critic where the current
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

The same reasoning applies to the buffer's `discounted_ret`, which upstream
built by appending the bootstrap as a pseudo-final reward (fixed in `e64316b`).
That is correct for the advantage/target estimators and wrong for the quantity
`Value/Train/*_true_r` scores the critic against.

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

Both studies share one rollout loop (`_roll_probes`); they differ only in a
`begin_wave` callback that puts the env into each wave's start states. The three
commits after the initial add cut the eval code roughly in half.

## Config

```yaml
eval_critic: True              # master switch (also gates train-batch diagnostics)
early_eval_freq: 5             # cadence below early_eval_epochs
early_eval_epochs: 50
value_eval_freq: 50            # cadence at or above it
mc_value_study_probes: 17
mc_value_study_repeats: 10
mc_value_study_vector_envs: 5  # parallelism, independent of train vector_env_nums
intermediate_state_study_positions: [100, 300, 500, 700, 900]
mc_eval_bootstrap_threshold: 0.01
mc_eval_tail: drop
```

The first evaluation is always epoch 1, never epoch 0: at epoch 0 the rollout
comes from the freshly initialised policy and critics, so there is no trained
update to look at.

`mc_value_study_repeats: 10` rather than 5 — a split-half reliability study
found n=5 badly under-samples **cost** specifically; paired with the 0.01
truncation threshold, n=10 costs about the same wall-clock as the old n=5 did.

## Cost

Evaluation dominates an eval epoch's wall-clock (roughly an order of magnitude
more than a training epoch: `probes x repeats` full episodes, plus the same
again per intermediate position). It is latency-bound rather than core-bound —
8 cores measured only ~10% worse than 32 for sequential eval. Hence the cadence
knobs, and `transient=False` progress bar added in `4dae7c8`.

## Reproducibility

Two things must hold to reproduce MICE:

1. **`torch_threads` must match.** The thread count changes GEMM row-blocking,
   so the tail row of an eval wave rounds one float32 ULP differently and the
   chaotic rollout amplifies it. MICE uses 4 for CPO/TRPOPID, 16 elsewhere;
   these configs now match per algorithm.
2. **`_consume_scatter_rng`.** MICE's eval diagnostics draw a `randperm` to
   subsample points for scatter plots that are not reproduced here. The draw
   still has to happen: it lands at the end of `_update`, after the critic
   loop's `DataLoader(shuffle=True)` seeds, so skipping it reshuffles every
   minibatch from the first eval epoch onward. The eval itself stays identical
   while training diverges one epoch later — recognise that signature.

Note the corollary: with eval on, **training results depend on whether eval
ran**, because the eval path consumes RNG. The fix is RNG save/restore around
eval (the `eval_rng` approach on MICE's `decouple-eval` branch), which would
also make eval-on and eval-off runs comparable within a repo.

## Parity

CPO / `SafetyPointGoal1-v0` / seed 0, default config, no overrides:

```
CPO      (torch_threads 4)   10 epochs   training identical   10004/10004 eval arrays
TRPOPID  (torch_threads 4)    6 epochs   training identical   10004/10004 eval arrays
PPOLag   (torch_threads 16)   6 epochs   training identical   10004/10004 eval arrays
control: eval_critic=False   10 epochs   training identical (isolates eval RNG as the only delta)
```

## Not done

- `discounted_cost_ret` is absent from the buffer, so there is no
  `Value/Train/Correlation_true_c` to match `Value/Train/Correlation_true_r`.
- The `early_eval_epochs: 50` cadence switch is untested — the longest parity
  run was 10 epochs, so `early_eval_freq` -> `value_eval_freq` never fired.
- `ppo_simmer_pid` / `trpo_simmer_pid` define their own `_update` with no
  `super()` call, so `_consume_scatter_rng` never runs for them. Neither repo
  draws there, so parity should hold, but it is unverified and
  `_pending_scatter_draw` is left set.
