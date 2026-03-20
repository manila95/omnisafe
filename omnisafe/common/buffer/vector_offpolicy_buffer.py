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
"""Implementation of VectorOffPolicyBuffer."""

from __future__ import annotations

import torch
from gymnasium.spaces import Box

from omnisafe.common.buffer.offpolicy_buffer import OffPolicyBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class VectorOffPolicyBuffer(OffPolicyBuffer):
    """Vectorized on-policy buffer.

    The vector-off-policy buffer is a vectorized version of the off-policy buffer. It stores the
    data in a single tensor, and the data of each environment is stored in a separate column.

    .. warning::
        The buffer only supports Box spaces.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        batch_size (int): The batch size of the buffer.
        num_envs (int): The number of environments.
        penalty_coefficient (float, optional): The penalty coefficient. Defaults to 0.0.
        device (torch.device, optional): The device of the buffer. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space or the action space is not Box.
        NotImplementedError: If the action space or the action space is not Box.
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        penalty_coefficient: float = 0.0,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorOffPolicyBuffer`."""
        self._num_envs: int = num_envs
        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size
        self._penalty_coefficient: float = penalty_coefficient
        self._device: torch.device = device
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
            next_obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        if isinstance(act_space, Box):
            act_buf = torch.zeros(
                (size, num_envs, *act_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self.data = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'cost': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'done': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'next_obs': next_obs_buf,
            # log π_behavior(a|s) — behavior policy log-prob stored at rollout time,
            # used by Retrace(λ) to compute off-policy importance sampling ratios.
            'logp': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
        }

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._num_envs

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros(
            (self._max_size, self._num_envs, *shape),
            dtype=dtype,
            device=self._device,
        )

    def sample_batch(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer.

        Returns:
            The sampled batch of data.
        """
        if batch_size is None:
            batch_size = self._batch_size
        idx = torch.randint(
            0,
            self._size,
            (batch_size * self._num_envs,),
            device=self._device,
        )
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(batch_size)
        return {key: value[idx, env_idx] for key, value in self.data.items()}

    def sample_batch_nstep_safe(
        self,
        n_steps: int,
        batch_size: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample a batch of starting transitions that are safe for n-step lookahead.

        In the circular buffer, looking ``n_steps`` ahead from slot ``i`` gives
        slot ``(i + n_steps) % max_size``.  This is only valid when those slots
        were written **after** slot ``i`` in the same pass (i.e. they are not
        overwritten newer data).  To guarantee this, we exclude the ``n_steps``
        newest slots (those just before ``_ptr``) from the pool of starting
        indices.

        Args:
            n_steps (int): Number of steps to look ahead.
            batch_size (int | None): Number of transitions per environment.
                Defaults to ``self._batch_size``.

        Returns:
            A tuple of (batch_dict, idx, env_idx) where ``idx`` and ``env_idx``
            are the sampled time and environment indices, needed for
            :meth:`compute_nstep_cost_targets`.
        """
        if batch_size is None:
            batch_size = self._batch_size

        n_safe = self._size - n_steps
        assert n_safe > 0, (
            f'Buffer has {self._size} transitions but n_step_cost={n_steps} '
            f'requires at least {n_steps + 1} transitions.'
        )

        # Safe time indices: [_ptr, _ptr + n_safe) mod _max_size
        # These are the oldest n_safe slots — all have n_steps newer slots ahead.
        safe_offsets = torch.randint(0, n_safe, (batch_size * self._num_envs,), device=self._device)
        idx = (self._ptr + safe_offsets) % self._max_size
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(batch_size)

        batch = {key: value[idx, env_idx] for key, value in self.data.items()}
        return batch, idx, env_idx

    def compute_nstep_cost_targets(
        self,
        idx: torch.Tensor,
        env_idx: torch.Tensor,
        n_steps: int,
        gamma: float,
        target_cost_critic: torch.nn.Module,
        next_action_fn: object,
    ) -> torch.Tensor:
        """Compute n-step cost return targets for the given starting indices.

        For each starting transition ``(idx[i], env_idx[i])``, accumulates
        discounted costs for up to ``n_steps`` forward, stopping early if a
        terminal ``done`` is encountered, then bootstraps with the target cost
        critic at the final step.

        Episode boundaries are respected: once ``done=True`` is seen at step
        ``k``, costs at steps ``k+1, ...`` and the bootstrap are zeroed out.

        Args:
            idx (torch.Tensor): Time indices of starting transitions, shape ``(B,)``.
            env_idx (torch.Tensor): Environment indices, shape ``(B,)``.
            n_steps (int): Number of lookahead steps.
            gamma (float): Discount factor.
            target_cost_critic (torch.nn.Module): Target cost critic network.
                Called as ``target_cost_critic(obs, act)[0]``.
            next_action_fn (callable): Maps ``next_obs -> action`` under the
                current policy. Called with no-grad.

        Returns:
            Tensor of shape ``(B,)`` with n-step cost return targets.
        """
        with torch.no_grad():
            batch_size = idx.shape[0]
            accumulated = torch.zeros(batch_size, device=self._device)
            # mask[i] = 1 if episode i is still alive (no done seen yet)
            mask = torch.ones(batch_size, device=self._device)

            for k in range(n_steps):
                step_idx = (idx + k) % self._max_size
                costs_k = self.data['cost'][step_idx, env_idx]
                dones_k = self.data['done'][step_idx, env_idx]

                accumulated += (gamma ** k) * mask * costs_k
                mask = mask * (1.0 - dones_k)

            # Bootstrap at step n using the target critic
            final_idx = (idx + n_steps) % self._max_size
            final_next_obs = self.data['next_obs'][final_idx, env_idx]
            final_act = next_action_fn(final_next_obs)
            bootstrap = target_cost_critic(final_next_obs, final_act)[0]
            accumulated += (gamma ** n_steps) * mask * bootstrap

        return accumulated

    def compute_retrace_cost_targets(
        self,
        idx: torch.Tensor,
        env_idx: torch.Tensor,
        n_steps: int,
        retrace_lambda: float,
        gamma: float,
        target_cost_critic: torch.nn.Module,
        actor: torch.nn.Module,
    ) -> torch.Tensor:
        """Compute Retrace(λ) cost targets (Munos et al. 2016) for the given indices.

        Unlike plain n-step returns, Retrace corrects for the off-policy mismatch
        between the *behavior policy* (which generated the stored actions) and the
        *current policy* via clipped per-step importance sampling ratios:

        .. math::

            c_k = \\lambda \\cdot \\min\\left(1,\\;
                  \\frac{\\pi_{\\text{current}}(a_k|s_k)}{\\pi_{\\text{behavior}}(a_k|s_k)}
                  \\right)

        The target is accumulated in the forward direction:

        .. math::

            G_t^{\\text{ret}} = Q(s_t, a_t)
            + \\sum_{k=0}^{n-1} \\gamma^k
              \\left(\\prod_{i=1}^{k} c_i\\right) \\delta_k

        where :math:`\\delta_k = c_{t+k} + \\gamma\\,V(s_{t+k+1}) - Q(s_{t+k}, a_{t+k})`
        is the TD error and :math:`V(s) = Q_{\\text{target}}(s, \\pi_{\\text{current}}(s))`.
        :math:`c_0 = 1` (no IS correction at the starting transition).

        Episode boundaries (``done = 1``) zero the running IS-product and all
        subsequent TD errors, truncating the return at the terminal step.

        Args:
            idx (torch.Tensor): Time indices of starting transitions, shape ``(B,)``.
            env_idx (torch.Tensor): Environment indices, shape ``(B,)``.
            n_steps (int): Number of lookahead steps.
            retrace_lambda (float): Trace-decay parameter λ ∈ (0, 1].
            gamma (float): Discount factor.
            target_cost_critic (torch.nn.Module): Target cost critic.
                Called as ``target_cost_critic(obs, act)[0]``.
            actor (torch.nn.Module): Current policy actor with ``predict()`` and
                ``log_prob()`` methods.

        Returns:
            Tensor of shape ``(B,)`` containing the Retrace cost targets.
        """
        with torch.no_grad():
            # --- Seed: Q_target(s_t, a_t) as the base of the Retrace sum ---
            start_obs = self.data['obs'][idx, env_idx]
            start_act = self.data['act'][idx, env_idx]
            g_retrace = target_cost_critic(start_obs, start_act)[0]

            # Running product of IS ratios: Π_{i=1}^{k} c_i
            # (c_0 = 1 by convention — no IS correction at the starting step)
            product_c = torch.ones(idx.shape[0], device=self._device)

            for k in range(n_steps):
                step_idx = (idx + k) % self._max_size
                next_step_idx = (idx + k + 1) % self._max_size

                obs_k = self.data['obs'][step_idx, env_idx]
                act_k = self.data['act'][step_idx, env_idx]
                cost_k = self.data['cost'][step_idx, env_idx]
                done_k = self.data['done'][step_idx, env_idx]

                # Q_target(s_k, a_k) — stored action evaluated under target network
                q_k = target_cost_critic(obs_k, act_k)[0]

                # V(s_{k+1}) = Q_target(s_{k+1}, π_current(s_{k+1}))
                obs_k1 = self.data['obs'][next_step_idx, env_idx]
                act_k1_current = actor.predict(obs_k1, deterministic=False)
                v_k1 = target_cost_critic(obs_k1, act_k1_current)[0]

                # TD error: δ_k = c_k + γ·V(s_{k+1}) − Q(s_k, a_k)
                delta_k = cost_k + gamma * v_k1 - q_k

                # Accumulate: G += γ^k · product_c · δ_k
                g_retrace = g_retrace + (gamma ** k) * product_c * delta_k

                # --- Update IS product for the *next* step ---
                # c_{k+1} = λ · min(1, π_current(a_{k+1}|s_{k+1}) / π_behavior(a_{k+1}|s_{k+1}))
                # We evaluate π_current log-prob of the *stored* action at s_{k+1}.
                if k + 1 < n_steps:
                    logp_stored_k1 = self.data['logp'][next_step_idx, env_idx]
                    act_k1_stored = self.data['act'][next_step_idx, env_idx]
                    # Call predict to set up the distribution at obs_{k+1}, then
                    # evaluate log_prob of the stored action (not the new sample).
                    actor.predict(obs_k1, deterministic=False)
                    logp_current_k1 = actor.log_prob(act_k1_stored)
                    is_ratio = torch.exp(logp_current_k1 - logp_stored_k1).clamp(max=1.0)
                    product_c = product_c * (retrace_lambda * is_ratio)

                # Episode boundary: zero product_c so future steps don't contribute
                product_c = product_c * (1.0 - done_k)

        return g_retrace

    def sample_batch_recency_weighted(
        self,
        decay: float,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch with exponential recency weighting (recent samples preferred).

        Each time step in the buffer is assigned weight ``exp(-decay * age)`` where
        ``age = 0`` for the most recently stored step and increases for older ones.
        Setting ``decay = 0`` recovers uniform sampling (identical to
        :meth:`sample_batch`).

        Args:
            decay (float): Exponential decay rate. Higher values concentrate
                sampling on more recent transitions.
            batch_size (int | None): Number of transitions per environment.
                Defaults to ``self._batch_size``.

        Returns:
            The sampled batch of data.
        """
        if batch_size is None:
            batch_size = self._batch_size

        # age[k] = how many steps ago buffer slot k was written (0 = most recent)
        all_time_idx = torch.arange(self._size, device=self._device)
        ages = (self._ptr - 1 - all_time_idx) % self._size
        weights = torch.exp(-decay * ages.float())
        weights = weights / weights.sum()

        idx = torch.multinomial(weights, batch_size * self._num_envs, replacement=True)
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(batch_size)
        return {key: value[idx, env_idx] for key, value in self.data.items()}
