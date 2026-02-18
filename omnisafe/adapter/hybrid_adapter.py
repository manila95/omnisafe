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
"""Hybrid Adapter for TRPOSACPID: supports both on-policy rollout (with off-policy buffer fill)
and off-policy rollout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.progress import track

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer, VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config

if TYPE_CHECKING:
    from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


class HybridAdapter(OffPolicyAdapter):
    """Hybrid adapter for TRPOSACPID algorithm.

    Extends OffPolicyAdapter to support:
    - on_policy_rollout: Same as OnPolicyAdapter.rollout but also stores transitions
      into an off-policy buffer for jump-starting Q-critic training.
    - rollout: Standard off-policy rollout (inherited from OffPolicyAdapter).

    This adapter is used during Phase 1 (on-policy TRPOPID) to both collect trajectories
    for TRPO updates and populate the replay buffer for parallel Q-critic training.
    """

    def set_off_policy_buffer(self, buffer: VectorOffPolicyBuffer) -> None:
        """Set the off-policy buffer to populate during on-policy rollout.

        Args:
            buffer: The VectorOffPolicyBuffer to store transitions into.
        """
        self._off_policy_buffer: VectorOffPolicyBuffer | None = buffer

    def on_policy_rollout(
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        on_policy_buffer: VectorOnPolicyBuffer,
        logger: Logger,
        off_policy_buffer: VectorOffPolicyBuffer | None = None,
    ) -> None:
        """Rollout using on-policy collection (like OnPolicyAdapter) and optionally
        store transitions into off-policy buffer.

        Args:
            steps_per_epoch: Number of steps per epoch.
            agent: Constraint actor-critic (on-policy, V-critics).
            on_policy_buffer: Buffer for on-policy data (trajectories).
            logger: Logger for metrics.
            off_policy_buffer: If provided, also store (obs, act, reward, cost, next_obs, done)
                into this buffer for off-policy Q-critic training.
        """
        buffer = off_policy_buffer if off_policy_buffer is not None else getattr(
            self, '_off_policy_buffer', None
        )

        self._reset_log()

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing on-policy rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            on_policy_buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            # Also store into off-policy buffer for Q-critic training
            if buffer is not None:
                real_next_obs = next_obs.clone()
                for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                    if done or time_out:
                        if 'final_observation' in info:
                            real_next_obs[idx] = info['final_observation'][idx]
                done_flag = torch.logical_and(
                    terminated,
                    torch.logical_xor(terminated, truncated),
                )
                buffer.store(
                    obs=obs,
                    act=act,
                    reward=reward,
                    cost=cost,
                    done=done_flag.float(),
                    next_obs=real_next_obs,
                )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        '\nWarning: trajectory cut off when rollout by epoch'
                        f' in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=obs.device)
                    last_value_c = torch.zeros(1, device=obs.device)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx : idx + 1])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx : idx + 1],
                            )
                        # Ensure 1D shape (batch of 1) for finish_path compatibility
                        last_value_r = last_value_r.view(-1)
                        last_value_c = last_value_c.view(-1)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    on_policy_buffer.finish_path(last_value_r, last_value_c, idx)
