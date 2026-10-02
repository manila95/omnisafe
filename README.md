# On the Reliability of Value Estimation for Safe Exploration

--------------------------------------------------------------------------------

## 1. Value evaluation (CPO, PID-Lagrangian)

Scores a critic against a Monte-Carlo estimate of the value it should predict, by
re-rolling the same states many times and averaging the realised discounted return.
Two studies run on each eval epoch, before the update, so the critic measured is the
one that produced that epoch's advantages:

- **start states** — `reset(seed=X)` reproduces a layout exactly, so each probe seed
  gives a fixed `s0` that can be rolled out `mc_value_study_repeats` times;
- **on-policy intermediate states** — snapshots the simulator part-way through a
  rollout and re-rolls from there, measuring the critic where the policy actually is.

Both feed one pooled block under `ValueEval/` (estimation error, Pearson, Spearman,
AUROC and its finite-sample ceiling), and every raw array is written to
`eval_data/epoch_*.pkl` so anything not computed online can be derived later.

`eval_critic` is **on by default**. It is expensive — roughly an order of magnitude
more than a training epoch — so it runs on a cadence (`early_eval_freq` for the first
`early_eval_epochs`, then `value_eval_freq`).

```bash
python train_policy.py --algo CPO     --env-id SafetyPointGoal1-v0 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 4
python train_policy.py --algo TRPOPID --env-id SafetyPointGoal1-v0 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 4
```

More details in [`value_eval.md`](docs/value_eval.md)

## 2. MICE

CPO plus an intrinsic cost that penalises proximity to states where cost was
previously incurred. A per-environment *flashbulb memory* stores a random projection
of every state whose cost was positive; each step is scored against it by a k-NN
kernel density, scaled by `intrinsic_factor * cost_gamma ** epoch`, and added to the
cost advantage with a coefficient adapted online from the cost TD residual. The
epoch-wise decay retires the intrinsic term, so MICE converges toward plain CPO.

Key settings: `intrinsic_factor` (5.0), `k_knn` (10), `buf_maxlen` (100 per env),
`model_cfgs.emb_dim` (8).

```bash
python train_policy.py --algo MICE --env-id SafetyPointGoal1-v0 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 16
```
More details in [`mice_and_biased_cost.md`](docs/mice_and_biased_cost.md)

## 3. BC-CPO (biased cost)

The much simpler counterpart: a flat constant added to the episode cost that CPO
picks its optimization case on, decaying as `cost_bias * cost_bias_decay_rate ** epoch`.
Early updates behave as if the policy were more costly than measured; the bias
vanishes and BC-CPO becomes plain CPO. `cost_bias: 0` reproduces CPO exactly.

Note the bias only bites near the constraint boundary. Far into violation, CPO takes
a recovery step that ignores the *magnitude* of the cost, so the bias changes nothing
— and the shipped `cost_bias: 5.0` is a placeholder, not a tuned value.

```bash
python train_policy.py --algo BCCPO --env-id SafetyPointGoal1-v0 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 16 \
    --algo_cfgs:cost_bias 5.0 --algo_cfgs:cost_bias_decay_rate 0.985
```

More details in [`mice_and_biased_cost.md`](docs/mice_and_biased_cost.md)

## 4. Successor-representation critic

Replaces both critics with read-outs of one shared trunk, factorizing the value as
`V(s) = psi(s) . w`: `psi` is the discounted sum of a one-step feature stream `phi`,
trained against its own TD target, and `w` is the closed-form ridge solution of the
immediate reward (or cost) onto `phi`, re-solved once per epoch.

`phi_source` selects the feature map:

- **`random`** — a frozen random linear projection. Frozen is the point: `psi` is
  *defined* as the discounted sum of `phi`, so a moving `phi` leaves `psi` chasing a
  map that no longer exists.
- **`contrastive`** — an MLP trained by a time-contrastive InfoNCE loss, pulling
  together states visited close in time within an episode and pushing apart distant
  ones. Pretrained at epoch 0, then adapted each epoch, with the affected tensors
  relabelled after `phi` moves.

It is off by default; enable it on any on-policy algorithm.

```bash
python train_policy.py --algo CPO --env-id SafetyPointGoal1-v0 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 4 \
    --model_cfgs:use_successor_representation True \
    --model_cfgs:sr_cfgs:phi_source contrastive
```
More details in [`sr_critic.md`](docs/sr_critic.md)


--------------------------------------------------------------------------------

## Acknowledgements

This repository is a fork of [**OmniSafe**](https://github.com/PKU-Alignment/omnisafe),
which provides the safe-RL algorithms, environment adapters and training
infrastructure everything here is built on.

```bibtex
@article{omnisafe,
  title   = {OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research},
  author  = {Jiaming Ji, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, Yaodong Yang},
  journal = {arXiv preprint arXiv:2305.09304},
  year    = {2023}
}
```

The MICE algorithm is taken from the code release of the original paper,
[ShiqingGao/MICE](https://github.com/ShiqingGao/MICE):

```bibtex
@inproceedings{gaocontrolling,
  title     = {Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration},
  author    = {Gao, Shiqing and Ding, Jiaxin and Fu, Luoyi and Wang, Xinbing},
  booktitle = {Forty-second International Conference on Machine Learning}
}
```

See the notes in `docs/` for what was kept, what was left out, and where this
implementation still differs from the original.
