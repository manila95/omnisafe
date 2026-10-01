# Successor-representation critic (`td_ridge`) — port notes

Ported from the MICE branch `mc-eval-no-bootstrap` onto `code-release`.
Commit `9342efd`, verified bit-identical to MICE (see [Parity](#parity)).

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

Plus the same 38-line `sr_cfgs` block in all 23 on-policy configs. It is read
only through `.get()` with defaults, so it could be centralized; it is duplicated
to match how the rest of the repo structures per-algorithm config.

## Fixes to existing code that the port required

1. **Critic-norm parameter set.** The penalty summed every parameter reachable
   from a critic. Under a trained `phi` that is a second, unintended gradient
   into `phi` through the shared trunk — `phi` has its own loss and its own
   optimizer. Now skips frozen and SR-excluded parameters, matching MICE.
2. **`_calculate_adv_and_value_targets` gamma override.** The SR stream may
   discount at `gamma_sr` rather than `algo_cfgs.gamma`.
3. **`get()` exposes `reward` / `cost`** when the SR fields are active; the ridge
   solve regresses onto the raw one-step streams.

## Reproducibility: the vestigial RNG draws

Three draws exist *only* to keep this repo's RNG stream aligned with MICE's:

| hook | mirrors |
|---|---|
| `_consume_scatter_rng` | MICE's scatter-plot subsample in the eval diagnostics |
| `_consume_sr_probe_rng` (x2) | MICE's SR drift-diagnostic probe states |

The corresponding metrics are not reproduced here, but the draws have to happen.
They land before the critic loop's `DataLoader(shuffle=True)` seeds, so omitting
them reshuffles every minibatch from the first eval epoch onward — the eval
itself stays identical while training silently diverges one epoch later. That is
exactly how both bugs presented during the port.

**This is brittle by construction.** Parity holds against *this* MICE branch. If
MICE's diagnostics change how much RNG they consume, or a config trips a path
with different draws (`n_val_episodes > 0`, `sr_mode: td_ridge` with the SR
diagnostics re-enabled), the counts drift apart with no error. The robust
alternative is to save/restore RNG state around eval on both sides — the
`eval_rng` approach on MICE's `decouple-eval` branch.

`torch_threads` must also match between the two repos: the thread count changes
GEMM row-blocking, which moves the tail row of an eval batch by one float32 ULP,
and the chaotic rollout amplifies it.

## Parity

CPO / `SafetyPointGoal1-v0` / seed 0 / `eval_critic: True`, 2 epochs:

```
phi_source=random       0 differing training cells   5002/5002 eval arrays
phi_source=contrastive  0 differing training cells   5002/5002 eval arrays
plain CPO (SR off)      regression check             5002/5002 eval arrays
```

"eval arrays" counts every raw array in the epoch-1 `eval_data` dump compared
element-wise. Tests: `test_buffer`, `test_model`, `test_normalizer` pass (run
with `--noconftest`; the repo's `conftest.py` uses an old pytest hook signature).

## Not done

- `discounted_cost_ret` is still absent from the buffer, so there is no
  `Value/Train/Correlation_true_c`.
- Only CPO was parity-checked with the SR critic on. Other algorithms route
  through the same `_update` hooks but were not run.
