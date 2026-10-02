# MICE and BC-CPO — port notes

Ported from the MICE branch `mc-eval-no-bootstrap` onto `code-release`.
Commits `c948dc7` (MICE), `42ae42f` (BC-CPO).

Both are CPO variants that change *what cost the policy update sees*. MICE adds a
learned, state-dependent intrinsic cost; BC-CPO adds a flat decaying constant.

---

# MICE

## What it does

CPO plus an intrinsic cost that penalises proximity to states where cost was
previously incurred.

1. A per-env **flashbulb memory** (`FlashBulbMemory`, a `deque` of `buf_maxlen`
   entries) stores the embedding of every state whose raw cost was positive.
   Embeddings come from `RandomProjection`, a fixed random linear map to
   `model_cfgs.emb_dim` — never trained, it only has to be consistent.
2. Each step scores novelty against that memory by a k-NN kernel density over the
   `k_knn` nearest entries, the usual NGU-style inverse kernel
   `eps / (d/mean(d) - cluster_distance + eps)` summed and square-rooted.
3. The score is scaled by `intrinsic_factor * cost_gamma ** epoch` and enters the
   **cost** advantage, weighted by `beta`.
4. `beta` adapts online from the TD residual
   (`beta_ = beta - lr * deltas_n / intrinsic_costs`, EMA'd at `beta_lr = 0.1`),
   so the intrinsic term is scaled by how much of the cost TD error it explains.

The epoch-wise `cost_gamma ** epoch` factor is what retires the intrinsic cost as
training proceeds, so MICE converges toward plain CPO.

## Files

`algorithms/on_policy/mice/`: `mice.py` (the algorithm, a `CPO` subclass with its
own `learn`/`_update`/`_update_actor`), `mice_buffer.py` (`FlashBulbMemory`,
`MICEBuffer`, `MICEVectorBuffer`), `mice_rollout.py` (`MICEAdapter`, which computes
the intrinsic cost during the rollout), `utils.py` (`RandomProjection`), plus
`configs/on-policy/MICE.yaml`.

## What was dropped

- **Most of `utils.py`** — a legacy `estimate_true_value` superseded by
  `utils/value_eval.py`, pulling in matplotlib and wandb.
- **The cost adv/target decoupling kwargs** (`cost_adv_estimation_method`,
  `value_target_method`, `cost_value_target_method`). All null at MICE's defaults.
- **The train/val split** (`n_val_episodes`, 0 by default, where the split is a
  pass-through).
- **The scatter/histogram logging** in `learn()`. None of it consumes torch RNG,
  so dropping it does not shift the stream; `_consume_scatter_rng` sits where
  MICE's diagnostics call was.

## Supporting changes outside `mice/`

| change | why |
|---|---|
| `OnPolicyAdapter._epoch_cost_sum` | MICE's `learn()` logs it as `Metrics/TotalCost`; the port's base never tracked it, so it logged 0.0 |
| `OnPolicyBuffer.cost_gamma`, `discounted_cost_ret` | both exist in MICE's base buffer; this also fills in the missing half of `Value/Train`'s true-value diagnostics |
| `_calculate_adv_and_value_targets(gamma=...)` | the cost stream discounts at `cost_gamma` (a no-op while it equals `gamma`, as in every shipped config) |
| SR flags as class-level defaults | MICE overrides `_init()` and never ran the code that set them |
| `algo_wrapper` pins inter-op threads to 1 | matches MICE |

## A MICE-side bug found on the way

`mice.py` passes `value_target_method` / `cost_value_target_method` to
`MICEVectorBuffer`, but neither it nor `MICEBuffer` accepted them, so **the MICE
algorithm could not construct at all** — `TypeError` in `_init`, unrunnable since
`dc6dcd1`. Fixed in the MICE repo (`6aa6784`).

## Inconsistency replicated deliberately

`MICEBuffer.finish_path` computes `discounted_ret` **with** the bootstrap appended
as a pseudo-final reward, overriding a base class that computes it **without**.
That is an internal inconsistency in MICE; it is reproduced here for parity, so
`discounted_ret` means a different thing under MICE than under every other
algorithm in this repo. Worth a deliberate decision later — and the bootstrap is
non-zero on essentially every path, since Safety-Gymnasium goal envs truncate
rather than terminate.

## Parity

CPO-based, `SafetyPointGoal1-v0`, seed 0, eval on, 2 epochs:

```
training metrics      0 of 88 cells differ
raw eval arrays       0 of 5002 differ
plain CPO regression  5002/5002
```

**33 of the logged eval statistics still differ**, by at most 9.5e-07, all in the
intermediate study. The inputs to those reductions are byte-identical and the
per-repeat arrays they aggregate match exactly, so the cause is last-bit summation
order in a 17-element float32 reduction: MICE's values track numpy's pairwise sum,
this port's track torch's. Ruled out: thread counts, inter-op pinning, memory
alignment, tensor layout, autograd state, dtype, global precision flags, import
order. MICE reproduces itself exactly, so the difference is real rather than noise
— but it is confined to diagnostics derived from data that already matches bit for
bit, and can be recomputed from the dumps deterministically.

Also still unverified: the **start-state study's** stats, because MICE names those
keys `MCStudy/*` and this port uses bare names, so nothing compares.

---

# BC-CPO

## What it does

CPO with a constant bias added to the episode cost that the optimization case is
chosen on, decaying as `cost_bias * cost_bias_decay_rate ** epoch`. Early updates
behave as if the policy were more costly than measured; the bias vanishes and
BC-CPO converges to plain CPO.

The whole algorithm:

```python
def _ep_costs(self) -> float:
    bias = self._cfgs.algo_cfgs.cost_bias * (
        self._cfgs.algo_cfgs.cost_bias_decay_rate ** self._current_epoch
    )
    self._logger.store({'Misc/EpCostBias': bias})
    return super()._ep_costs() + bias
```

`algorithms/on_policy/biased_cost/biased_cost.py`, plus `configs/on-policy/BCCPO.yaml`
(CPO's config with `cost_bias` and `cost_bias_decay_rate` added).

## The one change to CPO

CPO computed `ep_costs` inline, leaving nothing to override, so that single
expression moved into an `_ep_costs()` method. A pure extraction — verified by
`cost_bias: 0` reproducing CPO bit for bit, and plain CPO still matching MICE at
5002/5002.

## Relation to MICE's `use_cost_bias`

Same insertion point, same `Misc/EpCostBias` log key, same parameter names. MICE
derives its bias as a *per-step* cost accumulated into a discounted per-episode
total (`set_cost_bias_for_epoch` / `mean_ep_cost_bias`), which needs buffer
accumulators and a `learn()` hook. BC-CPO adds the constant directly. **The two are
not numerically equivalent**: for the same `cost_bias`, MICE's shift is the
discounted sum of a per-step bias and therefore much larger.

## Verification

```
cost_bias: 0   vs CPO                    bit-identical
cost_bias: 50  near the boundary         76 cells differ, optim_case flips 3 -> 0,
                                         EpCost 39 vs 79 (epoch 1), 64 vs 93 (epoch 2)
```

## The bias is inert deep in violation

When `ep_costs` is large and positive CPO lands in the optimization case whose
recovery step ignores its *magnitude*, so the bias changes only `Misc/B` and the
policy is unchanged. It takes effect as the policy approaches or satisfies the
limit. Measured on PointGoal1 at `cost_limit: 25`, where `EpCost` starts near 69,
`cost_bias: 10` left every training metric identical to CPO.

`cost_bias: 5.0` in the shipped config is a **placeholder** chosen to make the
smoke test legible, not a tuned value — and given the above it will do nothing on
PointGoal1 until the agent's cost falls near the limit.
