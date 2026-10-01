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
"""Implementation of the Policy Gradient algorithm."""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.envs.core import make as make_env
from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze
from omnisafe.utils import distributed
from omnisafe.utils.eval_data_dump import log_eval_data_to_wandb, save_eval_data
from omnisafe.utils.state_snapshot import collect_on_policy_snapshots, enable_state_snapshots
from omnisafe.utils.value_eval import (
    estimate_true_value_same_state_mc,
    estimate_value_from_snapshots,
)
from omnisafe.utils.value_utils import pool_correlation_stats, sync_obs_normalizer


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class PolicyGradient(BaseAlgo):
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `PG <https://proceedings.neurips.cc/paper/1999/file64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    # --- value-evaluation state (algo_cfgs.eval_critic) -----------------------------------
    # Fixed same-layout probe seeds for the s0 study, built lazily on first use and then held
    # constant for the whole run so every eval epoch re-probes the same states.
    _mc_probe_seeds: list[int] | None = None
    # Set by learn() each epoch; read by _update's train-side diagnostics.
    _current_epoch: int = 0
    _pending_scatter_draw: tuple[int, torch.device] | None = None
    # Dedicated eval envs, kept separate from self._env so training's own vectorization
    # (train_cfgs.vector_env_nums) is independent of the studies' parallelism, and so the
    # studies' ObsNormalize can be a frozen snapshot rather than drifting with their rollouts.
    _mc_eval_env: Any = None
    _mc_eval_max_episode_steps: int | None = None
    _mc_intermediate_env: Any = None
    _mc_intermediate_max_episode_steps: int | None = None

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)

        # --- value-evaluation keys (algo_cfgs.eval_critic) --------------------------------
        # Deliberately just two quantities per stream per block: the estimation error and the
        # prediction/true correlation. Everything else the studies could report is a pure
        # function of the raw arrays dumped to eval_data/ and train_data/ each eval epoch, so it
        # is recomputable offline and does not need a column here. store() asserts on
        # unregistered keys, so this must mirror what the study code emits exactly.
        if getattr(self._cfgs.algo_cfgs, 'eval_critic', False):
            streams = ('r', 'c') if self._cfgs.algo_cfgs.use_cost else ('r',)
            # One logged block: the pooled set over s0 plus every intermediate position. The
            # per-study numbers are still computed and kept in the eval_data bundle, but pooling
            # is what answers "how accurate is the critic across the states we evaluate on" --
            # and it is not the per-category numbers averaged, so logging both was redundant.
            #
            # Correlation factors into two independent questions the end-to-end number cannot
            # separate: is the critic fitting its target (pred_target), and is that target a good
            # proxy for the truth (target_true, a property of the estimator, not the critic).
            for stream in ('r', 'c'):
                for k in (
                    'EstimationError', 'Correlation',
                    'Correlation_pred_target', 'Correlation_target_true',
                    'EstimationError_target_true',
                    # Ranking quality, invariant to scale and bias, reported next to the ceiling
                    # the noisy MC label makes attainable at this repeat count.
                    'AUROC', 'AUROC_ceiling',
                    # Rank correlation: no positive class to choose, and not dominated by the
                    # tail the way the Pearson one can be.
                    'Spearman', 'Spearman_pred_target', 'Spearman_target_true',
                ):
                    self._logger.register_key(f'ValueEval/{k}_{stream}')

            # The same two quantities on the training batch (see _log_train_critic_diagnostics):
            # prediction vs its own regression target, and vs the realised discounted return.
            for stream in streams:
                self._logger.register_key(f'Value/Train/EstimationError_{stream}')
                self._logger.register_key(f'Value/Train/Correlation_{stream}')
            self._logger.register_key('Value/Train/EstimationError_true_r')
            self._logger.register_key('Value/Train/Correlation_true_r')

    def _get_mc_value_study_env(self):
        """Dedicated env for the s0 study, built once and cached.

        Same wrapper recipe as training, but ``ObsNormalize(update_stats=False)``: the
        statistics are re-synced from the live training env before every call and must not
        drift as the probes run. Parallelism is its own knob, independent of
        ``train_cfgs.vector_env_nums``.
        """
        if self._mc_eval_env is not None:
            return self._mc_eval_env
        env_cfgs = {}
        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()
        n_envs = int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_vector_envs', 1))
        eval_env = make_env(self._env_id, num_envs=n_envs, device=self._device, **env_cfgs)
        if eval_env.need_time_limit_wrapper:
            eval_env = TimeLimit(eval_env, time_limit=eval_env.max_episode_steps, device=self._device)
        if eval_env.need_auto_reset_wrapper:
            eval_env = AutoReset(eval_env, device=self._device)
        if self._cfgs.algo_cfgs.obs_normalize:
            eval_env = ObsNormalize(eval_env, device=self._device, update_stats=False)
        eval_env = ActionScale(eval_env, low=-1.0, high=1.0, device=self._device)
        if n_envs == 1:
            eval_env = Unsqueeze(eval_env, device=self._device)
        self._mc_eval_env = eval_env
        probe_env = make_env(self._env_id, num_envs=1, device=self._device, **env_cfgs)
        self._mc_eval_max_episode_steps = probe_env.max_episode_steps
        probe_env.close()
        return self._mc_eval_env

    def _get_intermediate_state_env(self):
        """Dedicated env for the intermediate-state study, built once and cached.

        ``num_envs = intermediate_state_study_probes``: one collection rollout yields that
        many probes per position in a single pass. ``enable_state_snapshots()`` must run
        before construction so the vectorized env's forked workers inherit the patch.
        """
        if self._mc_intermediate_env is not None:
            return self._mc_intermediate_env
        enable_state_snapshots()
        env_cfgs = {}
        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()
        n_envs = int(getattr(self._cfgs.algo_cfgs, 'intermediate_state_study_probes', 20))
        env = make_env(self._env_id, num_envs=n_envs, device=self._device, **env_cfgs)
        if env.need_time_limit_wrapper:
            env = TimeLimit(env, time_limit=env.max_episode_steps, device=self._device)
        if env.need_auto_reset_wrapper:
            env = AutoReset(env, device=self._device)
        if self._cfgs.algo_cfgs.obs_normalize:
            env = ObsNormalize(env, device=self._device, update_stats=False)
        env = ActionScale(env, low=-1.0, high=1.0, device=self._device)
        if n_envs == 1:
            env = Unsqueeze(env, device=self._device)
        self._mc_intermediate_env = env
        probe_env = make_env(self._env_id, num_envs=1, device=self._device, **env_cfgs)
        self._mc_intermediate_max_episode_steps = probe_env.max_episode_steps
        probe_env.close()
        return self._mc_intermediate_env

    def _is_value_eval_epoch(self, epoch: int) -> bool:
        """Whether ``epoch`` is due for the value studies.

        ``early_eval_freq`` below ``early_eval_epochs``, ``value_eval_freq`` at or above it.
        The first evaluation is epoch 1, not 0: at epoch 0 the rollout comes from the
        freshly initialised policy and critics, so there is no trained update to look at.
        """
        eval_freq = getattr(self._cfgs.algo_cfgs, 'value_eval_freq', 50)
        early_eval_freq = getattr(self._cfgs.algo_cfgs, 'early_eval_freq', 5)
        early_eval_epochs = int(getattr(self._cfgs.algo_cfgs, 'early_eval_epochs', 100))
        effective_eval_freq = early_eval_freq if epoch < early_eval_epochs else eval_freq
        return epoch == 1 or (epoch > 0 and epoch % effective_eval_freq == 0)

    def _consume_scatter_rng(self) -> None:
        """Advance the RNG exactly as MICE's eval diagnostics do, for bit-parity.

        MICE draws ``torch.randperm(n)`` to subsample points for the scatter images its
        ``_log_critic_diagnostics`` logs. Those plots are not reproduced here, but the draw
        still has to happen: the global torch RNG is what the next epoch's rollout samples
        actions from, so a run that skips it diverges from MICE on the epoch *after* the first
        eval -- even though the eval itself is identical.

        Position matters as much as the draw. MICE takes it at the very end of ``_update``,
        after the critic loop's ``DataLoader(shuffle=True)`` has drawn its per-iteration seeds
        -- not alongside the diagnostics, which run before the update. So this is called from
        the end of each ``_update`` that MICE instruments (here, ``natural_pg``, ``focops``)
        rather than after ``self._update()`` returns in ``learn()``: CUP runs a second policy
        phase with its own shuffled ``DataLoader`` *after* ``super()._update()``, and a draw
        placed after the whole chain would land on the wrong side of it.

        Only fires on the eval epochs where MICE would have plotted; ``_pending_scatter_draw``
        is set by ``_log_train_critic_diagnostics`` after its ``eval_critic`` /
        ``_is_value_eval_epoch`` guards, so a run with eval off consumes nothing.
        """
        if self._pending_scatter_draw is None:
            return
        n, device = self._pending_scatter_draw
        self._pending_scatter_draw = None
        torch.randperm(n, device=device)

    def _run_eval_studies(self, epoch: int) -> None:
        """Run this epoch's value studies, if due, and persist the results.

        Called from ``learn()`` after the rollout and before ``_update()``, so the critic
        evaluated is the one that produced this epoch's advantages rather than one already
        fitted to them. Both studies feed one pooled ``ValueEval/`` block; the per-study
        breakdown and every raw array go to ``eval_data/``.
        """
        is_eval_epoch = self._is_value_eval_epoch(epoch)
        if is_eval_epoch:
            self._log_train_critic_diagnostics(self._buf.get(), epoch)
        eval_data_bundle: dict | None = {'epoch': epoch} if is_eval_epoch else None
        if getattr(self._cfgs.algo_cfgs, 'eval_critic', False) and is_eval_epoch:
            if self._mc_probe_seeds is None:
                n_probes = int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_probes', 100))
                seed_offset = int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_seed_offset', 100_000))
                self._mc_probe_seeds = list(range(seed_offset, seed_offset + n_probes))
            mc_env = self._get_mc_value_study_env()  # also sets _mc_eval_max_episode_steps
            mc_stats, mc_raw = estimate_true_value_same_state_mc(
                agent=self._actor_critic,
                env=mc_env,
                cfgs=self._cfgs,
                discount_r=self._cfgs.algo_cfgs.gamma,
                discount_c=getattr(self._cfgs.algo_cfgs, 'cost_gamma', self._cfgs.algo_cfgs.gamma),
                probe_seeds=self._mc_probe_seeds,
                mc_repeats=int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_repeats', 5)),
                epoch=epoch,
                sync_normalizer_from=self._env._env,
                max_episode_steps=self._mc_eval_max_episode_steps,
                return_raw=True,
                bootstrap_threshold=getattr(self._cfgs.algo_cfgs, 'mc_eval_bootstrap_threshold', None),
                tail_mode=getattr(self._cfgs.algo_cfgs, 'mc_eval_tail', None),
            )
            eval_data_bundle['mc_study'] = {'stats': mc_stats, 'raw': mc_raw}
        if getattr(self._cfgs.algo_cfgs, 'eval_critic', False) and is_eval_epoch:
            positions = list(
                getattr(
                    self._cfgs.algo_cfgs,
                    'intermediate_state_study_positions',
                    [100, 300, 500, 700, 900],
                ),
            )
            n_probes = int(getattr(self._cfgs.algo_cfgs, 'intermediate_state_study_probes', 20))
            repeats = int(getattr(self._cfgs.algo_cfgs, 'intermediate_state_study_repeats', 5))
            interm_env = self._get_intermediate_state_env()
            sync_obs_normalizer(interm_env, self._env._env)
            base_seed = 700_000 + epoch * n_probes
            collected = collect_on_policy_snapshots(
                self._actor_critic, interm_env, positions, base_seed=base_seed,
            )
            max_eps = self._mc_intermediate_max_episode_steps
            eval_data_bundle['intermediate_study'] = {}
            for pos in positions:
                pos_stats, pos_raw = estimate_value_from_snapshots(
                    agent=self._actor_critic,
                    env=interm_env,
                    cfgs=self._cfgs,
                    discount_r=self._cfgs.algo_cfgs.gamma,
                    discount_c=getattr(
                        self._cfgs.algo_cfgs, 'cost_gamma', self._cfgs.algo_cfgs.gamma,
                    ),
                    snapshots=collected[pos],
                    horizon=max_eps,
                    mc_repeats=repeats,
                    epoch=epoch,
                    return_raw=True,
                    bootstrap_threshold=getattr(self._cfgs.algo_cfgs, 'mc_eval_bootstrap_threshold', None),
                    tail_mode=getattr(self._cfgs.algo_cfgs, 'mc_eval_tail', None),
                )
                eval_data_bundle['intermediate_study'][pos] = {
                    'stats': pos_stats, 'raw': pos_raw,
                }
            pooled_sources = []
            if 'mc_study' in eval_data_bundle:
                pooled_sources.append(eval_data_bundle['mc_study']['raw'])
            pooled_sources.extend(
                pos_data['raw'] for pos_data in eval_data_bundle['intermediate_study'].values()
            )
            pooled_stats, pooled_raw = pool_correlation_stats(
                pooled_sources, prefix='ValueEval/',
            )
            self._logger.store(pooled_stats)
            eval_data_bundle['pooled'] = {'stats': pooled_stats, 'raw': pooled_raw}
        if is_eval_epoch:
            if len(eval_data_bundle) > 1:
                eval_data_path = save_eval_data(self._logger.log_dir, epoch, eval_data_bundle)
                log_eval_data_to_wandb(eval_data_path, epoch)
            self._logger.torch_save()
            checkpoint_path = os.path.join(self._logger.log_dir, 'torch_save', f'epoch-{epoch}.pt')
            if os.path.exists(checkpoint_path):
                log_eval_data_to_wandb(
                    checkpoint_path, epoch,
                    name_prefix='actor-snapshot', artifact_type='actor_snapshot',
                    description=(
                        f'Agent state dicts at epoch {epoch}, keyed by _what_to_save: '
                        f"'pi', 'reward_critic', 'cost_critic' and (when obs_normalize) "
                        f"'obs_normalizer'. This is the PRE-update critic -- the snapshot is "
                        f'taken before _update() runs, so it is exactly the critic an epoch-'
                        f'{epoch} evaluation scores. Enough on its own to re-run the value '
                        f'studies offline, and pairs with this epoch\'s eval-data artifact to '
                        f'recompute compute_gradient_alignment.'
                    ),
                )

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()
            # Read by _update via _log_train_critic_diagnostics, which needs to know which epoch
            # it is in order to share the eval cadence and name its dump.
            self._current_epoch = epoch

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            # Must run after the rollout (the studies score the critic that produced it) and
            # before _update() (so the critic evaluated is the one that computed this epoch's
            # advantages, not the one that has already been fitted to them).
            self._run_eval_studies(epoch)
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    @torch.no_grad()
    def _log_train_critic_diagnostics(self, data: dict, epoch: int) -> None:
        """Estimation error and correlation for the critic on the training batch.

        One forward pass over data already in memory, so no extra environment steps. Raw
        per-state arrays go to ``train_data/``.

        Args:
            data: The epoch batch from the buffer.
            epoch: For the dump filename.
        """
        if not getattr(self._cfgs.algo_cfgs, 'eval_critic', False):
            return
        if not self._is_value_eval_epoch(epoch):
            return

        obs = data['obs']
        # MICE subsamples here for its scatter plots; see _consume_scatter_rng.
        self._pending_scatter_draw = (obs.shape[0], obs.device)
        pred_r = self._actor_critic.reward_critic(obs)[0].flatten()
        streams = {'r': (pred_r, data['target_value_r'].flatten())}
        if self._cfgs.algo_cfgs.use_cost:
            pred_c = self._actor_critic.cost_critic(obs)[0].flatten()
            streams['c'] = (pred_c, data['target_value_c'].flatten())

        def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
            if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
                return float('nan')
            return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

        stats, raw = {}, {'epoch': epoch}
        for stream, (pred, target) in streams.items():
            stats[f'Value/Train/EstimationError_{stream}'] = (target - pred).mean().item()
            stats[f'Value/Train/Correlation_{stream}'] = _corr(pred, target)
            raw[stream] = {
                'pred': pred.detach().cpu().numpy(),
                'target': target.detach().cpu().numpy(),
            }
        if 'discounted_ret' in data:
            ret = data['discounted_ret'].flatten()
            stats['Value/Train/EstimationError_true_r'] = (ret - streams['r'][0]).mean().item()
            stats['Value/Train/Correlation_true_r'] = _corr(streams['r'][0], ret)
            raw['discounted_ret'] = ret.detach().cpu().numpy()

        self._logger.store(stats)
        train_data_path = save_eval_data(self._logger.log_dir, epoch, raw, subdir='train_data')
        log_eval_data_to_wandb(
            train_data_path, epoch,
            name_prefix='train-data', artifact_type='train_data',
            description=(
                f'Per-state critic predictions, regression targets and realised discounted '
                f'returns over epoch {epoch}\'s training batch.'
            ),
        )

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )
        self._consume_scatter_rng()

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} [
                \sum_{t=0}^T ( \frac{\pi^{'}_{\theta}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_{\theta}` is the policy network, :math:`\pi^{'}_{\theta}`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss
