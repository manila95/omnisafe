# Successor-Representation Value Function

A guide to the shared successor-representation (SR) critic added to OmniSafe's on-policy
algorithms: how it works, where the code lives, and — since you're seeing slower constraint
satisfaction than the baseline — which knobs actually move that needle.

## 1. The idea in one paragraph

Normally `ConstraintActorCritic` builds two *independent* networks: `reward_critic` and
`cost_critic`, each with its own hidden layers, trained with its own MSE loss against its own
GAE target. The SR feature replaces this with **one shared representation** that both values
are read out of. Two ways of doing that are implemented, toggled by
`model_cfgs.sr_cfgs.sr_mode`:

- **`shared_trunk`** — a feature trunk `phi_trunk(s)` with two independent *linear* heads,
  `V_r(s) = w_r · phi_trunk(s)` and `V_c(s) = w_c · phi_trunk(s)`. Everything (trunk + both
  heads) is trained by ordinary backprop against the ordinary GAE targets — nothing about the
  target math changes, only the network architecture.
- **`td_ridge`** — a literal successor representation. A trunk produces a one-step feature
  `phi(s)` and a successor feature `psi(s)`, trained by TD to satisfy
  `psi(s_t) ≈ phi(s_t) + γ·psi(s_{t+1})`. The reward/cost read-out weights `w_r`/`w_c` are
  **not** learned by gradient descent — they're solved in closed form each update by ridge
  regression of the one-step reward/cost onto `phi`, then `V_r(s) = w_r · psi(s)`,
  `V_c(s) = w_c · psi(s)`.

In both modes, the advantage estimation and value-target math (GAE / GAE-RTG / V-trace / Plain,
`lam`, `lam_c`, standardization, etc.) is **completely unchanged** — the only thing that changes
is what's underneath `reward_critic(obs)` and `cost_critic(obs)`.

## 2. Where the code lives

| File | Role |
|---|---|
| `omnisafe/models/critic/successor_representation_critic.py` | The four network classes: `SuccessorRepresentationTrunk` / `SuccessorRepresentationReadout` (shared_trunk mode), `TDRidgeSuccessorRepresentationTrunk` / `SuccessorRepresentationLinearReadout` (td_ridge mode). |
| `omnisafe/models/actor_critic/constraint_actor_critic.py` | `_build_successor_representation_critics`: builds whichever mode is configured, and wires `reward_critic_optimizer` / `cost_critic_optimizer` to the **same** optimizer instance (see §4). |
| `omnisafe/common/buffer/onpolicy_buffer.py` | `finish_path` also computes `target_sr` (td_ridge only) by calling `_calculate_adv_and_value_targets` a second time — same function, `phi`/`psi` in place of `reward`/`value`. Also where `_calculate_v_trace` lives (generalized to vector streams for this). |
| `omnisafe/adapter/onpolicy_adapter.py` | `rollout()` fetches `agent.sr_features(obs)` each step (td_ridge only) and stores `phi`/`psi`, bootstraps `last_psi` at path end. |
| `omnisafe/algorithms/on_policy/base/policy_gradient.py` | `_ridge_update_successor_weights` (the ridge solve, once per `_update()` call) and `_update_successor_features` (the psi TD loss, once per minibatch). Also `NaturalPG` and `FOCOPS` have their own copies of the hook since they reimplement `_update()`. |
| `omnisafe/configs/on-policy/*.yaml` | `model_cfgs.use_successor_representation` + `model_cfgs.sr_cfgs.*` defaults, present in every on-policy algorithm's config. |

## 3. Config reference

All of these live under `model_cfgs.sr_cfgs` in the yaml (or `--sr-*` flags / `--use-sr` if
you're using `experiments/train_policy.py`).

| Key | Default | Mode | What it does |
|---|---|---|---|
| `sr_mode` | `shared_trunk` | both | `shared_trunk` or `td_ridge`. |
| `sr_dim` | `64` | both | Width of the shared feature vector. In `td_ridge` mode this is also the dimensionality of the linear regression basis for `w_r`/`w_c` — **this is usually the single highest-leverage knob** (§5). |
| `hidden_sizes` | `[64, 64]` | both | Hidden layers of the trunk (shared_trunk) or of the trunk feeding `phi_head`/`psi_head` (td_ridge). |
| `activation` | `tanh` | both | Trunk activation. |
| `lr` | `0.0003` | both | Learning rate of the **single shared** optimizer (trunk + both heads in shared_trunk; trunk + phi_head + psi_head in td_ridge). |
| `lam_sr` | `0.95` | td_ridge | λ for the `psi` TD target — the vector-stream analogue of `lam_c`. Controls the psi target's bias/variance tradeoff. |
| `gamma_sr` | `null` → `algo_cfgs.gamma` | td_ridge | Discount for the `psi` Bellman recursion. **Note:** there is only one `gamma_sr` for the whole shared representation — see §6.4 if your reward and cost naturally want different horizons. |
| `ridge_kappa` | `0.001` | td_ridge | L2 regularization scale for the ridge solve (`κ · mean(diag(Gram))`). Higher = more regularized/stable, lower = tighter fit but more sensitive to ill-conditioned `phi`. |
| `ema_tau` | `1.0` | td_ridge | EMA blend of the ridge solution into `w_r`/`w_c`. `1.0` = replace outright every update (no smoothing); lower = smoother but laggier. |

Shared knobs from `algo_cfgs` that still apply to *both* modes exactly as before:
`update_iters`, `batch_size`, `steps_per_epoch`, `use_critic_norm`/`critic_norm_coef`,
`use_max_grad_norm`/`max_grad_norm`, `lam_c`, `adv_estimation_method`,
`standardized_cost_adv`.

**One pre-existing OmniSafe quirk worth knowing (not introduced by the SR feature):**
`algo_cfgs.cost_gamma` is *not* actually read by the on-policy buffer — reward and cost targets
both use the single `algo_cfgs.gamma`. This is true for the baseline dual-critic setup too, so
it isn't the cause of your slowdown, but it does mean `gamma_sr` defaulting to `gamma` is the
*consistent* choice, not a change in behavior.

## 4. Why constraint satisfaction can be slower than baseline

This is architectural, not a bug — you're trading capacity/independence for parameter sharing.
Three concrete mechanisms, roughly in order of how much they tend to matter:

### 4.1 One optimizer, alternating gradients (both modes)

`reward_critic_optimizer` and `cost_critic_optimizer` are **literally the same Adam instance**
over the shared parameters. Each minibatch does, in order: zero_grad → reward-loss backward →
step, then zero_grad → cost-loss backward → step (and in td_ridge, a third psi-loss step). So
the shared trunk gets pushed by the reward gradient, then immediately by the cost gradient,
using a *single* momentum/second-moment buffer that's constantly being overwritten by whichever
loss just ran. If reward and cost gradients disagree in direction (common — reward wants
"progress", cost wants "avoid this region"), the trunk effectively receives conflicting
alternating updates rather than a coherent gradient for either objective. This is the main
reason a shared representation converges slower *per task* than two independent critics of the
same size, and it affects the cost stream more visibly whenever `Value/reward`'s loss magnitude
dwarfs `Value/cost`'s (very common — reward scales are usually larger than cost scales), since
the larger-magnitude gradient dominates Adam's shared second-moment estimate.

### 4.2 Halved effective capacity per task (shared_trunk especially)

If you kept `sr_cfgs.hidden_sizes` at the same `[64, 64]` you had for `critic.hidden_sizes`
before, you actually gave the cost stream (and the reward stream) *less* dedicated capacity than
baseline: one `[64,64]` trunk is now doing the job two independent `[64,64]` critics used to do.

### 4.3 td_ridge-specific: `phi` barely moves, and `psi` is a whole extra value function

`phi_head` only ever appears in the *target* for the psi TD loss (as frozen, stored-at-rollout
data) — it never appears in a backward pass with gradient enabled, so it only drifts indirectly
through shared-trunk changes. In practice `phi` behaves close to a fixed random projection. That
means:
- The ridge-regression fit `r_t ≈ phi(s_t)·w_r` (and same for cost) is capacity-limited by
  however good that near-fixed projection happens to be — check `Misc/RidgeResidualCost` in the
  logs; if it's large and not shrinking, `phi`'s projection isn't separating cost-relevant states.
- `psi` has to learn an entire Bellman-consistent representation from scratch via TD, which is
  strictly more to learn than a plain scalar value function — early training necessarily has a
  "ramp-up" period the baseline critic doesn't need.
- `ema_tau=1.0` (the default) means `w_c` is **fully replaced** by a fresh ridge solve every
  single update, with no smoothing across epochs. Early on, when `phi`/`psi` are still noisy and
  the policy (hence the data distribution) is changing fast, this can make `w_c` — and therefore
  the cost advantage the policy update actually sees — swing a lot epoch to epoch. For a
  constrained algorithm (CPO's trust region, PPOLag's Lagrange multiplier, ...) a noisy cost
  advantage signal directly translates into slower/less stable constraint satisfaction. This is
  the most likely single culprit for what you're seeing, and it's also the easiest to fix
  (§5.2).

## 5. Tuning playbook

Ordered by expected impact for "cost/constraint learning feels slow":

### 5.1 If you're on `shared_trunk`

1. **Give the trunk more capacity than a single baseline critic** — try
   `sr_cfgs.hidden_sizes: [128, 128]` (or wider) rather than reusing `[64, 64]`. You're asking
   one network to do the work of two.
2. **Raise `sr_cfgs.lr`** a bit (e.g. `0.0005`–`0.001`) — with a shared optimizer taking
   alternating steps for two losses, the effective progress per task is lower than a dedicated
   optimizer at the same LR.
3. If the imbalance is specifically reward-dominating-cost, consider switching to `td_ridge`
   instead — it decouples the linear read-out weights from the representation-learning loss,
   which structurally isolates the cost fit from reward-gradient interference (see §4.3, but the
   ridge solve is *not* subject to the momentum cross-talk problem in §4.1 the way an SGD head
   would be).

### 5.2 If you're on `td_ridge`

1. **Lower `ema_tau`** first — try `0.3`–`0.5`. This directly smooths `w_c`/`w_r` across epochs
   and is the most likely fix for "noisy, slow constraint satisfaction" per §4.3. Watch
   `Misc/WcNorm` in the logs before/after — it should stop oscillating and trend more smoothly.
2. **Increase `sr_dim`** (e.g. `128`) if `Misc/RidgeResidualCost` / `Misc/RidgeResidualReward`
   stay high — a richer feature basis gives the ridge regression more to work with.
3. **Watch `Misc/GramCond`** (condition number of the ridge Gram matrix). If it's very large
   (poorly conditioned `phi`, e.g. features collapsing to near-duplicates), raise `ridge_kappa`
   for stability; if it's small and residuals are still high, you can lower `ridge_kappa` for a
   tighter fit.
4. **Raise `sr_cfgs.lr`** — `psi` is learning a full Bellman-consistent representation from
   scratch (§4.3), so it may need more gradient steps' worth of movement than a plain critic did.
5. **`lam_sr`**: closer to `1.0` reduces psi's bootstrap bias (more Monte-Carlo-like) at the cost
   of variance — worth trying if cost signals in your environment are sparse/rare-event (typical
   in safe RL), since bootstrapping off a still-inaccurate `psi(s')` early on can propagate bad
   estimates. Closer to `0` gives faster bootstrap propagation once `psi` is reasonably trained.
6. **Bigger `steps_per_epoch`**: the ridge solve uses the *entire* epoch's data in one closed-form
   fit — more samples per epoch means a statistically more stable `w_c`/`w_r` each update.

### 5.3 Applies to both modes

- `algo_cfgs.use_critic_norm` / `critic_norm_coef`: note that in `shared_trunk` mode this L2
  penalty is applied to the shared trunk **twice** per minibatch (once from the reward loss'
  parameter loop, once from the cost loss'), i.e. effectively double-strength on shared
  parameters vs. a standalone critic. If cost learning is being over-regularized, try halving
  `critic_norm_coef` when using `shared_trunk`.
- `algo_cfgs.lam_c` / `standardized_cost_adv`: unchanged from baseline, still the right levers
  for the cost *advantage* (as opposed to the SR value function itself).
- If none of the above closes the gap and you need the cost stream to behave as independently
  as possible while still sharing *some* representation, `shared_trunk` with a much wider trunk
  (so each head effectively gets close to full baseline capacity through its own linear
  projection of a big shared feature space) tends to be more forgiving than `td_ridge`, which has
  the extra `phi`/`psi` ramp-up cost.

## 6. Diagnostics logged during training

| Key | Mode | What to look for |
|---|---|---|
| `Loss/Loss_reward_critic`, `Loss/Loss_cost_critic` | both | Should trend down; compare their *relative* magnitude — a big reward/cost imbalance is the §4.1 symptom. |
| `Loss/Loss_sr` | td_ridge | The psi TD loss — should trend down as the representation stabilizes. |
| `Misc/RidgeResidualReward`, `Misc/RidgeResidualCost` | td_ridge | RMS residual of the ridge fit. High & flat ⇒ `phi` isn't expressive enough (raise `sr_dim`). |
| `Misc/WrNorm`, `Misc/WcNorm` | td_ridge | Norm of the solved weight vectors. Oscillating wildly epoch-to-epoch ⇒ lower `ema_tau`. |
| `Misc/GramCond` | td_ridge | Condition number of the ridge Gram matrix. Very large ⇒ raise `ridge_kappa`. |
| `Value/reward`, `Value/cost` | both | Sanity-check the rollout-time value estimates aren't diverging. |

## 7. When SR probably isn't worth it

If your observation space is low-dimensional and the stock two-critic setup already converges
fast, the SR machinery (especially `td_ridge`) adds real learning-curve overhead for a benefit
(parameter/sample sharing between reward and cost) that mostly pays off when the two value
functions have meaningfully related structure and you're sample-constrained. If you're not
sample-constrained, the baseline (`use_successor_representation: False`) is usually the simpler,
faster-converging choice.
