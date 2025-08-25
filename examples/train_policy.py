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
"""Example of training a policy with OmniSafe."""

import argparse
import random
import numpy as np
import torch

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='PPOLag',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal1-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='SEED',
        help='random seed for reproducibility',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=10000000,
        metavar='STEPS',
        help='total number of steps to train for algorithm',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=5,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=4,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    
    # Additional hyperparameters
    parser.add_argument(
        '--cost-limit',
        type=float,
        default=10,
        metavar='LIMIT',
        help='cost limit for constrained algorithms (default: use config file value)',
    )
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=None,
        metavar='STEPS',
        help='number of steps per epoch (default: use config file value)',
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        default=True,
        help='use Weights & Biases for logging (default: True)',
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='disable Weights & Biases logging',
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='sam-safe-rl',
        metavar='PROJECT',
        help='Weights & Biases project name',
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        metavar='ENTITY',
        help='Weights & Biases entity/username (default: use default entity)',
    )
    parser.add_argument(
        '--wandb-group',
        type=str,
        default=None,
        metavar='GROUP',
        help='Weights & Biases run group name',
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        metavar='NAME',
        help='Weights & Biases run name (default: auto-generated)',
    )
    
    # SAM-specific arguments
    parser.add_argument(
        '--sam-rho',
        type=float,
        default=None,
        metavar='RHO',
        help='SAM neighborhood size for all networks (overrides individual rho values)',
    )
    parser.add_argument(
        '--sam-actor-rho',
        type=float,
        default=0.05,
        metavar='RHO',
        help='SAM neighborhood size for actor network optimization',
    )
    parser.add_argument(
        '--sam-critic-rho',
        type=float,
        default=0.05,
        metavar='RHO',
        help='SAM neighborhood size for critic network optimization',
    )
    parser.add_argument(
        '--sam-cost-critic-rho',
        type=float,
        default=0.05,
        metavar='RHO',
        help='SAM neighborhood size for cost critic network optimization',
    )
    parser.add_argument(
        '--sam-adaptive',
        default="False",
        help='use adaptive SAM (ASAM) instead of standard SAM',
    )
    parser.add_argument(
        '--sam-eps',
        type=float,
        default=1e-12,
        metavar='EPS',
        help='small constant for numerical stability in SAM',
    )
    parser.add_argument(
        '--sam-config-preset',
        type=str,
        default=None,
        choices=['default', 'conservative', 'aggressive', 'ppo', 'trpo', 'cup', 'focops', 'cpo'],
        help='use a preset SAM configuration instead of individual parameters',
    )
    parser.add_argument(
        '--lagrangian-multiplier-init',
        type=float,
        default=0.001,
        metavar='INIT',
        help='initial value of lagrangian multiplier',
    )
    parser.add_argument(
        '--pid-kp',
        type=float,
        default=0.1,
        metavar='KP',
        help='proportional gain for PID controller',
    )
    parser.add_argument(
        '--pid-ki',
        type=float,
        default=0.01,
        metavar='KI',
        help='integral gain for PID controller',
    )
    parser.add_argument(
        '--pid-kd',
        type=float,
        default=0.01,
        metavar='KD',
        help='derivative gain for PID controller',
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))
    
    # Handle additional hyperparameters
    args.sam_adaptive = args.sam_adaptive.lower() == 'true'
    
    if args.steps_per_epoch is not None:
        if 'algo_cfgs' not in custom_cfgs:
            custom_cfgs['algo_cfgs'] = {}
        custom_cfgs['algo_cfgs']['steps_per_epoch'] = args.steps_per_epoch

    if "PID" in args.algo or args.algo in ['CUP', 'FOCOPS', 'SAMCUP', 'SAMFOCOPS']:
        if 'lagrange_cfgs' not in custom_cfgs:
            custom_cfgs['lagrange_cfgs'] = {}
        custom_cfgs['lagrange_cfgs']['cost_limit'] = args.cost_limit
        custom_cfgs['lagrange_cfgs']['lagrangian_multiplier_init'] = args.lagrangian_multiplier_init
    elif "saute" in args.algo.lower():
        print(args.algo)
        if 'algo_cfgs' not in custom_cfgs:
            custom_cfgs['algo_cfgs'] = {}
        custom_cfgs['algo_cfgs']['safety_budget'] = args.cost_limit
    else:
        if 'algo_cfgs' not in custom_cfgs:
            custom_cfgs['algo_cfgs'] = {}
        custom_cfgs['algo_cfgs']['cost_limit'] = args.cost_limit

    if "PID" in args.algo:
        custom_cfgs['lagrange_cfgs']['pid_kp'] = args.pid_kp
        custom_cfgs['lagrange_cfgs']['pid_ki'] = args.pid_ki
        custom_cfgs['lagrange_cfgs']['pid_kd'] = args.pid_kd
    
    # Handle Weights & Biases configuration
    wandb_enabled = args.use_wandb and not args.no_wandb
    if wandb_enabled:
        if 'logger_cfgs' not in custom_cfgs:
            custom_cfgs['logger_cfgs'] = {}
        custom_cfgs['logger_cfgs']['use_wandb'] = True
        custom_cfgs['logger_cfgs']['wandb_project'] = args.wandb_project
        if args.wandb_entity:
            custom_cfgs['logger_cfgs']['wandb_entity'] = args.wandb_entity
        if args.wandb_group:
            custom_cfgs['logger_cfgs']['wandb_group'] = args.wandb_group
        if args.wandb_name:
            custom_cfgs['logger_cfgs']['wandb_name'] = args.wandb_name
    else:
        if 'logger_cfgs' not in custom_cfgs:
            custom_cfgs['logger_cfgs'] = {}
        custom_cfgs['logger_cfgs']['use_wandb'] = False
    
    # Add SAM configuration if using SAM-augmented algorithm
    if args.algo.startswith('SAM'):
        if args.sam_config_preset:
            # Use preset configuration
            try:
                from omnisafe.configs.sam_configs import get_sam_config
                sam_config = get_sam_config(args.algo.lower().replace('sam', ''), args.sam_config_preset)
            except ImportError:
                print(f"Warning: Could not import sam_configs. Using individual parameters.")
                sam_config = {
                    'actor_rho': args.sam_actor_rho,
                    'critic_rho': args.sam_critic_rho,
                    'cost_critic_rho': args.sam_cost_critic_rho,
                    'adaptive': args.sam_adaptive,
                    'eps': args.sam_eps,
                }
        else:
            # Use individual parameters
            sam_config = {
                'actor_rho': args.sam_actor_rho,
                'critic_rho': args.sam_critic_rho,
                'cost_critic_rho': args.sam_cost_critic_rho,
                'adaptive': args.sam_adaptive,
                'eps': args.sam_eps,
            }
        
        # Override individual rho values if --sam-rho is specified
        if args.sam_rho is not None:
            sam_config['actor_rho'] = args.sam_rho
            sam_config['critic_rho'] = args.sam_rho
            sam_config['cost_critic_rho'] = args.sam_rho
            print(f"Using uniform SAM rho value: {args.sam_rho}")
        
        # Add SAM configuration to custom_cfgs
        if 'algo_cfgs' not in custom_cfgs:
            custom_cfgs['algo_cfgs'] = {}
        custom_cfgs['algo_cfgs']['sam_cfgs'] = sam_config
        
        print(f"Using SAM configuration for {args.algo}:")
        for key, value in sam_config.items():
            print(f"  {key}: {value}")

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")
    
    # Add seed to custom configurations
    custom_cfgs['seed'] = args.seed
    
    # Filter out SAM parameters and additional hyperparameters from training config
    train_cfgs = vars(args).copy()
    sam_params = ['sam_rho', 'sam_actor_rho', 'sam_critic_rho', 'sam_cost_critic_rho', 'sam_adaptive', 'sam_eps', 'sam_config_preset', 'seed']
    additional_params = ['cost_limit', 'steps_per_epoch', 'use_wandb', 'no_wandb', 'wandb_project', 'wandb_entity', 'wandb_group', 'wandb_name']
    lagrange_params = ['pid_kp', 'pid_ki', 'pid_kd', 'lagrangian_multiplier_init']
    params_to_remove = sam_params + additional_params + lagrange_params
    for param in params_to_remove:
        if param in train_cfgs:
            del train_cfgs[param]

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=train_cfgs,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
