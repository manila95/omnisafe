# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Train with Weights & Biases sweeps (or standalone with CLI overrides).

Use dot notation in the sweep YAML for nested OmniSafe config:
  lagrange_cfgs.cost_limit, algo_cfgs.warmup_epochs, etc.

Sweep:  wandb sweep examples/benchmarks/wandb_sweep_sacpid.yaml
        wandb agent <entity>/<project>/<sweep_id>

Standalone with CLI (overrides YAML defaults):
  python examples/train_policy_wandb.py --algo SACPID --env SafetyPointGoal1-v0 --algo_cfgs.warmup_epochs 0
"""

from __future__ import annotations

import argparse
import sys

import omnisafe
from omnisafe.utils.tools import flat_config_to_nested, flatten_dict_to_dot

try:
    import wandb
except ImportError:
    raise ImportError('wandb is required for sweep. Install with: pip install wandb')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', default='SACPID', help='algorithm')
    parser.add_argument('--env', '--env-id', dest='env_id', default='SafetyPointGoal1-v0', help='env id')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--total-steps', type=int, default=1000000, dest='total_steps', help='total steps')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--parallel', type=int, default=1, help='parallel processes')
    parser.add_argument('--vector-env-nums', type=int, default=1, dest='vector_env_nums', help='vector envs')
    parser.add_argument('--torch-threads', type=int, default=16, dest='torch_threads', help='torch threads')
    args, unparsed = parser.parse_known_args()

    def cli_flag_passed(name: str) -> bool:
        """True if a CLI flag that maps to this name was explicitly passed (e.g. --total-steps -> total_steps)."""
        normalized = [a.lstrip('-').replace('-', '_') for a in sys.argv if a.startswith('--')]
        return name in normalized

    # Start from sweep config (if any), then override with CLI only when explicitly passed
    run = wandb.init()
    # Normalize wandb.config: skip keys with '=', flatten nested dicts to dot keys, map env -> env_id
    raw = dict(wandb.config)
    config_flat = flatten_dict_to_dot(
        {k: v for k, v in raw.items() if '=' not in str(k)},
        skip_keys_with_eq=True,
    )
    config = dict(config_flat)
    # Prefer sweep (config) values; override with args only when user passed the flag on CLI
    config['algo'] = args.algo if cli_flag_passed('algo') else config.get('algo', args.algo)
    config['env_id'] = (
        args.env_id if cli_flag_passed('env_id') or cli_flag_passed('env') else (config.get('env_id') or config.get('env') or args.env_id)
    )
    config['seed'] = args.seed if cli_flag_passed('seed') else config.get('seed', args.seed)
    config['total_steps'] = (
        args.total_steps if cli_flag_passed('total_steps') else config.get('total_steps', args.total_steps)
    )
    config['device'] = args.device if cli_flag_passed('device') else config.get('device', args.device)
    config['parallel'] = args.parallel if cli_flag_passed('parallel') else config.get('parallel', args.parallel)
    config['vector_env_nums'] = (
        args.vector_env_nums if cli_flag_passed('vector_env_nums') else config.get('vector_env_nums', args.vector_env_nums)
    )
    config['torch_threads'] = (
        args.torch_threads if cli_flag_passed('torch_threads') else config.get('torch_threads', args.torch_threads)
    )

    # CLI overrides for nested keys, e.g. --algo_cfgs.warmup_epochs 0
    if unparsed:
        cli_pairs = dict(zip([k.lstrip('-') for k in unparsed[0::2]], unparsed[1::2]))
        for k, v in cli_pairs.items():
            config[k.replace('-', '_')] = v
        wandb.config.update(config, allow_val_change=True)

    algo = config.get('algo', 'SACPID')
    env_id = config.get('env_id', 'SafetyPointGoal1-v0')
    seed = int(config.get('seed', 42))

    train_terminal_cfgs = {
        'total_steps': int(config.get('total_steps', 1000000)),
        'device': str(config.get('device', 'cpu')),
        'parallel': int(config.get('parallel', 1)),
        'vector_env_nums': int(config.get('vector_env_nums', 1)),
        'torch_threads': int(config.get('torch_threads', 16)),
    }

    custom_cfgs = flat_config_to_nested(
        config,
        omit_keys=(
            'algo',
            'env_id',
            'seed',
            'total_steps',
            'device',
            'parallel',
            'vector_env_nums',
            'torch_threads',
        ),
    )

    agent = omnisafe.Agent(
        algo,
        env_id,
        seed,
        train_terminal_cfgs=train_terminal_cfgs,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
