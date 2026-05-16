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

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


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
        default=1,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    parser.add_argument(
        '--adv-inflation-coeff',
        type=float,
        default=None,
        metavar='COEFF',
        help='constant bonus added to TRPOPID advantage (0.0 disables)',
    )
    parser.add_argument(
        '--adv-inflation-decay',
        type=float,
        default=None,
        metavar='DECAY',
        help='per-epoch multiplicative decay for adv inflation (1.0=no decay, 0.99=slow decay)',
    )
    parser.add_argument(
        '--lagrangian-multiplier-init',
        type=float,
        default=None,
        metavar='LAMBDA',
        help='initial value of the Lagrangian multiplier',
    )
    parser.add_argument(
        '--pid-kp',
        type=float,
        default=None,
        metavar='KP',
        help='proportional gain of the PID Lagrangian controller',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='SEED',
        help='random seed',
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    adv_inflation_coeff = args.adv_inflation_coeff
    adv_inflation_decay = args.adv_inflation_decay
    lagrangian_multiplier_init = args.lagrangian_multiplier_init
    pid_kp = args.pid_kp
    terminal_cfgs = vars(args)
    del terminal_cfgs['adv_inflation_coeff']
    del terminal_cfgs['adv_inflation_decay']
    del terminal_cfgs['lagrangian_multiplier_init']
    del terminal_cfgs['pid_kp']

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))
    if adv_inflation_coeff is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:adv_inflation_coeff', str(adv_inflation_coeff)))
    if adv_inflation_decay is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:adv_inflation_decay', str(adv_inflation_decay)))
    if lagrangian_multiplier_init is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('lagrange_cfgs:lagrangian_multiplier_init', str(lagrangian_multiplier_init)))
    if pid_kp is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('lagrange_cfgs:pid_kp', str(pid_kp)))

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        args.seed,
        train_terminal_cfgs=terminal_cfgs,
        custom_cfgs=custom_cfgs,
    )
    print(agent.cfgs)
    agent.learn()
