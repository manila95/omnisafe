# Successor-representation critic

## What it does

Factorizes the value function as

```
V(s) = psi(s) . w
```

- `phi(s)` — an L2-normalized one-step feature map.
- `psi(s)` — the discounted sum of the `phi` stream, trained by MSE against a
  TD(lambda) target built with the same estimator machinery as the scalar
  reward/cost targets.
- `w_r` / `w_c` — read-out weights, re-solved in closed form each epoch as the
  ridge regression of the *immediate* reward/cost onto `phi`. Buffers, never
  parameters; no gradient step ever touches them.

Enable with `model_cfgs.use_successor_representation: True`. Both critics then
become read-outs of one shared trunk.

## Scope

Deliberately the smallest version that runs. Implemented:

| axis | value |
|---|---|
| `sr_mode` | `td_ridge` only |
| read-out | ridge-solved only |
| `psi_objective` | TD only |
| `phi_source` | `random` (frozen linear projection) or `contrastive` (MLP trained by InfoNCE) |

Left out, along with every config key only they need: SGD/learned read-outs
(`readout: sgd`, `w_lr`, `w_weight_decay`), `w_source: psi`, the `buffer` and
`cost_replay` ridge data sources, `psi_objective: contrastive`, `cost_only`,
`cost_value_clip`, dropout / layer-norm / spectral-norm, critic ensembles, the
`trunk` / `separate` / `rff` / `laplacian` / `joint` phi sources, and the SR
diagnostics (phi/psi drift, ridge R^2, explained variance).

Note `random` is MICE's frozen *linear projection*, not `rff` (random Fourier
features) — a different frozen source, not ported.

## Why `phi` is frozen under `random`

`psi` is *defined* as the discounted sum of `phi`. A `phi` that moves leaves
`psi` chasing a feature map that no longer exists, and leaves the ridge solve
fitting a basis that shifts under the `w` it just fitted. `contrastive` moves it
deliberately and pays for it by relabelling: `phi` is recomputed every epoch,
and `target_sr` is rebuilt after the epoch-0 pretraining burst (which moves
`phi` far more than a normal epoch's drift).

## Files

New:

- `omnisafe/models/critic/successor_representation_critic.py` — `FrozenPhiFeatures`,
  `ContrastivePhiFeatures`, `TDRidgeSuccessorTrunk` (phi/psi + `ridge_update`),
  `SuccessorRepresentationLinearReadout`, and two segment-wise helpers used to
  relabel after `phi` moves.
- `omnisafe/utils/contrastive.py` — `sample_temporal_pairs`, `info_nce_loss`.

Modified: `onpolicy_buffer` / `vector_onpolicy_buffer` (the `phi`/`psi`/`target_sr`
fields, `last_psi`, episode boundaries), `onpolicy_adapter` (feature capture during
rollout), `constraint_actor_critic` (trunk, read-outs, optimizers), and `_update` in
`policy_gradient` / `natural_pg` / `focops`.


`torch_threads` must also match between the two repos: the thread count changes
GEMM row-blocking, which moves the tail row of an eval batch by one float32 ULP,
and the chaotic rollout amplifies it.
