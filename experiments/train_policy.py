# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train any on-policy algorithm, optionally with the shared successor-representation critic.

Mirrors examples/train_policy.py but exposes the successor-representation (SR) value-function
hyperparameters (``model_cfgs.use_successor_representation`` / ``model_cfgs.sr_cfgs.*``) as
first-class flags and supports sequential multi-seed runs, so comparison arms are one-liners:

    # SR arm (shared_trunk mode), 3 seeds
    python experiments/train_policy.py --algo PPOLag --env-id SafetyPointGoal1-v0 \\
        --total-steps 2000000 --seeds 0,1,2 --use-sr --sr-mode shared_trunk

    # SR arm (td_ridge mode), 3 seeds
    python experiments/train_policy.py --algo CPO --env-id SafetyPointGoal1-v0 \\
        --total-steps 2000000 --seeds 0,1,2 --use-sr --sr-mode td_ridge --sr-dim 64

    # stock baseline arm (two independent critics) on identical seeds
    python experiments/train_policy.py --algo CPO --env-id SafetyPointGoal1-v0 \\
        --total-steps 2000000 --seeds 0,1,2

    # sr-dim sweep
    for d in 16 32 64 128; do
        python experiments/train_policy.py --algo PPOLag --use-sr --sr-dim $d --seeds 0,1,2 \\
            --env-id SafetyPointGoal1-v0 --total-steps 2000000
    done

The ``--use-sr`` flag works with every on-policy algorithm (PPO, TRPO, all Lagrangian/PID/
penalty/second-order variants, FOCOPS, CUP, ...), since the SR critic is wired in at the base
on-policy infrastructure level, not per-algorithm.

Any additional dotted OmniSafe config keys still pass through unparsed, in the upstream style:
``--algo_cfgs:batch_size 128``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# allow running from a source checkout without `pip install -e .`
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


def build_parser() -> argparse.ArgumentParser:
    """CLI mirroring examples/train_policy.py plus successor-representation flags."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- upstream-compatible core flags ----
    parser.add_argument(
        '--algo',
        type=str,
        default='PPOLag',
        choices=omnisafe.ALGORITHMS['all'],
        help='algorithm to train (any on-policy algorithm supports --use-sr)',
    )
    parser.add_argument('--env-id', type=str, default='SafetyPointGoal1-v0', help='environment id')
    parser.add_argument('--parallel', type=int, default=1, help='parallel processes')
    parser.add_argument('--total-steps', type=int, default=10_000_000, help='total env steps')
    parser.add_argument('--device', type=str, default='cpu', help='cpu | cuda | cuda:k')
    parser.add_argument('--vector-env-nums', type=int, default=1, help='vectorized env count')
    parser.add_argument('--torch-threads', type=int, default=16, help='torch thread count')
    # ---- run management ----
    parser.add_argument('--seeds', type=str, default='0', help='comma-separated seed list')
    parser.add_argument('--steps-per-epoch', type=int, default=None, help='override steps/epoch')
    parser.add_argument('--exp-prefix', type=str, default=None, help='experiment name prefix')
    # ---- successor-representation critic (model_cfgs.use_successor_representation / sr_cfgs.*) ----
    parser.add_argument(
        '--use-sr',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='use the shared successor-representation value function instead of two '
        'independent reward/cost critics',
    )
    parser.add_argument(
        '--sr-mode',
        type=str,
        default=None,
        choices=['shared_trunk', 'td_ridge'],
        help='shared_trunk: shared trunk + linear reward/cost heads, trained end-to-end by the '
        'ordinary value-target loss. td_ridge: literal successor representation (phi/psi '
        'trained by TD, reward/cost read-out weights solved by ridge regression).',
    )
    parser.add_argument('--sr-dim', type=int, default=None, help='successor-feature dimension')
    parser.add_argument('--sr-lr', type=float, default=None, help='learning rate for the SR trunk')
    parser.add_argument(
        '--lam-sr',
        type=float,
        default=None,
        help='lambda for the SR TD target (td_ridge mode only)',
    )
    parser.add_argument(
        '--gamma-sr',
        type=float,
        default=None,
        help='discount factor for the SR TD recursion (td_ridge mode only; defaults to algo gamma)',
    )
    parser.add_argument(
        '--ridge-kappa',
        type=float,
        default=None,
        help='ridge regression regularization coefficient (td_ridge mode only)',
    )
    parser.add_argument(
        '--ema-tau',
        type=float,
        default=None,
        help='EMA smoothing coefficient for the ridge-solved weights (td_ridge mode only)',
    )
    # ---- generic GAE knob ----
    parser.add_argument('--lam-c', type=float, default=None, help='GAE lambda for the cost stream')
    return parser


def build_custom_cfgs(args: argparse.Namespace, unparsed: list[str], seed: int) -> dict:
    """Assemble the OmniSafe custom_cfgs dict for one run."""
    custom_cfgs: dict = {
        'seed': seed,
        'train_cfgs': {
            'total_steps': args.total_steps,
            'vector_env_nums': args.vector_env_nums,
            'torch_threads': args.torch_threads,
            'device': args.device,
        },
        'algo_cfgs': {},
        'logger_cfgs': {},
    }
    if args.steps_per_epoch is not None:
        custom_cfgs['algo_cfgs']['steps_per_epoch'] = args.steps_per_epoch
    if args.lam_c is not None:
        custom_cfgs['algo_cfgs']['lam_c'] = args.lam_c
    if args.exp_prefix is not None:
        custom_cfgs['logger_cfgs']['log_dir'] = f'./runs/{args.exp_prefix}'

    sr_flags = {
        'sr_mode': args.sr_mode,
        'sr_dim': args.sr_dim,
        'lr': args.sr_lr,
        'lam_sr': args.lam_sr,
        'gamma_sr': args.gamma_sr,
        'ridge_kappa': args.ridge_kappa,
        'ema_tau': args.ema_tau,
    }
    sr_set = {k: v for k, v in sr_flags.items() if v is not None}
    if sr_set and not args.use_sr:
        sys.exit('SR hyperparameter flags were given but --use-sr was not passed.')
    if args.use_sr:
        custom_cfgs['model_cfgs'] = {
            'use_successor_representation': True,
            'sr_cfgs': sr_set,
        }

    # upstream-style passthrough for any other dotted keys
    keys = [k[2:] for k in unparsed[0::2]]
    values = list(unparsed[1::2])
    for k, v in dict(zip(keys, values)).items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))
    return custom_cfgs


def main() -> None:
    parser = build_parser()
    args, unparsed = parser.parse_known_args()
    if len(unparsed) % 2 != 0:
        sys.exit(f'Dangling override (expected --key value pairs): {unparsed}')

    seeds = [int(s) for s in args.seeds.split(',') if s.strip() != '']
    terminal_cfgs = {
        'parallel': args.parallel,
        'device': args.device,
        'vector_env_nums': args.vector_env_nums,
        'torch_threads': args.torch_threads,
        'total_steps': args.total_steps,
    }
    for seed in seeds:
        custom_cfgs = build_custom_cfgs(args, unparsed, seed)
        print(f'=== {args.algo} on {args.env_id} | seed {seed} | use_sr={args.use_sr} ===')
        agent = omnisafe.Agent(
            args.algo,
            args.env_id,
            train_terminal_cfgs=terminal_cfgs,
            custom_cfgs=custom_cfgs,
        )
        agent.learn()


if __name__ == '__main__':
    main()
