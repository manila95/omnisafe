# MICE and Biased-Cost(BC)-CPO 

Both are CPO variants that change *what cost the policy update sees*. MICE adds a
learned, state-dependent intrinsic cost; BC-CPO adds a flat decaying constant but only to the
empirical episodic cost used for .

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
expression moved into an `_ep_costs()` method. 
