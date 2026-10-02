import time
from typing import Dict, Tuple, Optional, Union
import torch
import numpy as np
import torch.nn.functional as F

from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed
from rich.progress import track
from collections import deque
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.utils.math import discount_cumsum

    
class FlashBulbMemory:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        size: int,
        num_envs: int,
    ):
        self.unsafe_states = [deque(maxlen=size) for _ in range(num_envs)]


class MICEBuffer(OnPolicyBuffer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = torch.device('cpu'),
        constant_cost: Optional[float] = None,
        cost_decay_type: Optional[str] = None,
        cost_decay_rate: float = 0.985,
        cost_decay_step_interval: int = 50,
        cost_decay_factor: float = 0.4,
        no_intrinsic_in_deltas: bool = False,
        cost_gamma: Optional[float] = None,
        constant_cost_source: str = 'fixed',
        cost_limit: Optional[float] = None,
    ):
        super().__init__(
            obs_space,
            act_space,
            size,
            gamma,
            lam,
            lam_c,
            advantage_estimator,
            penalty_coefficient,
            standardized_adv_r,
            standardized_adv_c,
            device,
            cost_gamma=cost_gamma,
        )
        self.data['intrinsic_costs'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['ep_discount_ci'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['time_step'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.beta = 1.0
        self.beta_lr = 0.1
        self._beta_list = []
        self._deltas_n_list = []
        self._deltas_n_mc_list = []
        self.constant_cost = constant_cost
        self.cost_decay_type = cost_decay_type
        self.cost_decay_rate = cost_decay_rate
        self.cost_decay_step_interval = cost_decay_step_interval
        self.cost_decay_factor = cost_decay_factor
        self.no_intrinsic_in_deltas = no_intrinsic_in_deltas
        if constant_cost_source not in ('fixed', 'ep_cost', 'excess_cost'):
            raise ValueError(
                f"Unknown constant_cost_source: {constant_cost_source!r}. "
                "Choose 'fixed', 'ep_cost', or 'excess_cost'."
            )
        self.constant_cost_source = constant_cost_source
        self.cost_limit = cost_limit

    def _get_effective_constant_cost(
        self, epoch: int, current_ep_cost: Optional[float] = None,
    ) -> Optional[float]:
        # 'ep_cost' mode: the "constant" this epoch is just the running Metrics/EpCost mean
        # (see MICEAdapter.rollout, which reads it fresh off the logger right before each
        # finish_path call) -- self-tracking, so no decay schedule applies; constant_cost's
        # literal value/cost_decay_type are ignored entirely in this mode. current_ep_cost is
        # None only before any episode has ever completed (empty logger window), in which case
        # there's no cost signal yet -- fall back to 0.0 (no intrinsic penalty) rather than None,
        # so this still routes through the constant-cost branch below instead of unpredictably
        # falling back to the KNN-adaptive branch for the first few calls of a run.
        if self.constant_cost_source == 'ep_cost':
            return current_ep_cost if current_ep_cost is not None else 0.0
        # 'excess_cost' mode: same live-tracking idea, but clamped to how far EpCost currently
        # is *over* cost_limit rather than the raw EpCost value. Self-extinguishing: once the
        # policy satisfies the constraint (EpCost <= cost_limit), this goes to exactly 0 rather
        # than continuing to apply a deflation the policy no longer needs. Always >= 0 by
        # construction (max with 0.0), same as 'ep_cost'/'fixed' with a non-negative constant_cost.
        if self.constant_cost_source == 'excess_cost':
            if current_ep_cost is None:
                return 0.0
            return max(current_ep_cost - (self.cost_limit or 0.0), 0.0)
        if self.constant_cost is None:
            return None
        if self.cost_decay_type is None:
            return self.constant_cost
        elif self.cost_decay_type == 'exponential':
            return self.constant_cost * (self.cost_decay_rate ** epoch)
        elif self.cost_decay_type == 'step':
            steps = epoch // self.cost_decay_step_interval
            return self.constant_cost * (self.cost_decay_factor ** steps)
        else:
            raise ValueError(f"Unknown cost_decay_type: {self.cost_decay_type!r}. Choose 'exponential' or 'step'.")

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data in the buffer."""
        self._last_episode_slices = list(self._episode_slices)
        self._episode_slices = []
        self.ptr, self.path_start_idx = 0, 0

        beta_flat = torch.cat(self._beta_list, dim=0) if self._beta_list else torch.tensor([self.beta], device=self._device, dtype=torch.float32)
        deltas_n_flat = torch.cat(self._deltas_n_list, dim=0) if self._deltas_n_list else torch.zeros(1, device=self._device, dtype=torch.float32)
        deltas_n_mc_flat = torch.cat(self._deltas_n_mc_list, dim=0) if self._deltas_n_mc_list else torch.zeros(1, device=self._device, dtype=torch.float32)
        self._beta_list = []
        self._deltas_n_list = []
        self._deltas_n_mc_list = []

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'discounted_cost_ret': self.data['discounted_cost_ret'],
            'value_r': self.data['value_r'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
            'cost': self.data['cost'],
            'ep_discount_ci': self.data['ep_discount_ci'],
            'value_c': self.data['value_c'],
            'intrinsic_costs': self.data['intrinsic_costs'],
            'time_step': self.data['time_step'],
            'beta_': beta_flat,
            'beta': torch.tensor([self.beta], device=self._device, dtype=torch.float32),
            'deltas_n': deltas_n_flat,
            'deltas_n_mc': deltas_n_mc_flat,
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
        lr: float = 0.001,
        epoch: int = 0,
        current_ep_cost: Optional[float] = None,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs."""
        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(
            self._device
        )
        last_value_c = last_value_c.to(self._device)

        rewards = torch.cat(
            [self.data['reward'][path_slice], last_value_r]
        )
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discounted_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discounted_ret
        self.data['discounted_cost_ret'][path_slice] = discount_cumsum(costs, self._cost_gamma)[:-1]
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r, rewards, lam=self._lam
        )

        intrinsic_costs = self.data['intrinsic_costs'][path_slice]

        adv_c, target_value_c, ep_discount_ci = self._calculate_balancing_intrinsic_adv_and_value_targets(
            values_c, costs, lam=self._lam_c, intrinsic_costs=intrinsic_costs, lr=lr, epoch=epoch,
            current_ep_cost=current_ep_cost,
        )
        self.data['ep_discount_ci'][path_slice] = ep_discount_ci

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self._episode_slices.append((self.path_start_idx, self.ptr))
        self.path_start_idx = self.ptr

    def _calculate_balancing_intrinsic_adv_and_value_targets(
        self, values_c, costs, lam, intrinsic_costs, lr, epoch: int = 0,
        current_ep_cost: Optional[float] = None,
    ):
        effective_constant_cost = self._get_effective_constant_cost(epoch, current_ep_cost)

        if self._advantage_estimator == 'gae':
            if effective_constant_cost is not None:
                # Ablation: fixed, state-independent intrinsic cost with beta frozen at 1.0.
                # Without freezing beta, it adapts to absorb the constant (beta*C converges to
                # the same value regardless of C), making the ablation meaningless.
                intrinsic_costs = torch.full_like(intrinsic_costs, effective_constant_cost)
                self.data['intrinsic_costs'][self.path_start_idx:self.ptr] = intrinsic_costs

                _ic = torch.zeros_like(intrinsic_costs) if self.no_intrinsic_in_deltas else intrinsic_costs
                deltas_n = (
                    costs[:-1] + _ic + self._cost_gamma * values_c[1:] - values_c[:-1]
                )
                self._deltas_n_list.append(deltas_n.detach().flatten())

                path_rewards = costs[:-1] + intrinsic_costs
                last_v = values_c[-1:].reshape(-1)
                R = torch.cat([path_rewards, last_v])
                mc_returns = discount_cumsum(R, self._cost_gamma)[:-1]
                deltas_n_mc = mc_returns - values_c[:-1]
                self._deltas_n_mc_list.append(deltas_n_mc.detach().flatten())

                beta_ = torch.ones_like(deltas_n)  # frozen at 1.0
                self._beta_list.append(beta_.detach().flatten())

                ep_disount_ci = discount_cumsum(intrinsic_costs, self._cost_gamma)
                ep_disount_ci[:] = ep_disount_ci[0].clone()

                adv_c = discount_cumsum(deltas_n, self._cost_gamma * lam)
                target_value_c = adv_c + values_c[:-1]

            else:
                # TD(0) deltas: one-step TD error
                _ic = torch.zeros_like(intrinsic_costs) if self.no_intrinsic_in_deltas else intrinsic_costs
                deltas_n = (
                    costs[:-1] + _ic + self._cost_gamma * values_c[1:] - values_c[:-1]
                )
                self._deltas_n_list.append(deltas_n.detach().flatten())

                # Monte Carlo deltas: true return from trajectory - V(s_t)
                # G_t = sum_{k>=0} gamma^k * (cost_{t+k} + beta*intrinsic_{t+k}) with bootstrap at path end
                path_rewards = costs[:-1] + self.beta * intrinsic_costs
                last_v = values_c[-1:].reshape(-1)  # bootstrap value at path end
                R = torch.cat([path_rewards, last_v])
                mc_returns = discount_cumsum(R, self._cost_gamma)[:-1]  # G_0, ..., G_{n-1}
                deltas_n_mc = mc_returns - values_c[:-1]
                self._deltas_n_mc_list.append(deltas_n_mc.detach().flatten())

                beta_ = torch.where(
                    intrinsic_costs != 0,
                    self.beta - lr * deltas_n / intrinsic_costs,
                    torch.ones_like(deltas_n) * self.beta,
                )
                self._beta_list.append(beta_.detach().flatten())
                deltas = (
                    costs[:-1] + beta_ * intrinsic_costs + self._cost_gamma * values_c[1:] - values_c[:-1]
                )
                ep_disount_ci = discount_cumsum(beta_ * intrinsic_costs, self._cost_gamma)
                ep_disount_ci[:] = ep_disount_ci[0].clone()
                self.beta = (1 - self.beta_lr) * self.beta + self.beta_lr * beta_.mean().item()

                adv_c = discount_cumsum(deltas, self._cost_gamma * lam)
                target_value_c = adv_c + values_c[:-1]  # c_t + c_t_intrinsic + \gamma * V(s_t+1)

        else:
            raise NotImplementedError

        return adv_c, target_value_c, ep_disount_ci


class MICEVectorBuffer(VectorOnPolicyBuffer):
    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = torch.device('cpu'),
        constant_cost: Optional[float] = None,
        cost_decay_type: Optional[str] = None,
        cost_decay_rate: float = 0.985,
        cost_decay_step_interval: int = 50,
        cost_decay_factor: float = 0.4,
        no_intrinsic_in_deltas: bool = False,
        cost_gamma: Optional[float] = None,
        constant_cost_source: str = 'fixed',
        cost_limit: Optional[float] = None,
    ):
        self._num_buffers = num_envs
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers = [
            MICEBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
                constant_cost=constant_cost,
                cost_decay_type=cost_decay_type,
                cost_decay_rate=cost_decay_rate,
                cost_decay_step_interval=cost_decay_step_interval,
                cost_decay_factor=cost_decay_factor,
                no_intrinsic_in_deltas=no_intrinsic_in_deltas,
                cost_gamma=cost_gamma,
                    constant_cost_source=constant_cost_source,
                cost_limit=cost_limit,
            )
            for _ in range(num_envs)
        ]

    def get_effective_constant_cost(
        self, epoch: int, current_ep_cost: Optional[float] = None,
    ) -> Optional[float]:
        return self.buffers[0]._get_effective_constant_cost(epoch, current_ep_cost)

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
        idx: int = 0,
        lr: float = 0.001,
        epoch: int = 0,
        current_ep_cost: Optional[float] = None,
    ) -> None:
        """Finish the path."""
        self.buffers[idx].finish_path(last_value_r, last_value_c, lr, epoch, current_ep_cost)


