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
"""Implementation of the SAM-FOCOPS algorithm."""

from __future__ import annotations

import torch
from rich.progress import track
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed


@registry.register
class SAMFOCOPS(PolicyGradient):
    """The Sharpness-Aware Minimization First Order Constrained Optimization in Policy Space (SAM-FOCOPS) algorithm.

    This algorithm combines FOCOPS with SAM to improve generalization by minimizing sharpness.
    
    References:
        - Title: First Order Constrained Optimization in Policy Space
        - Authors: Yiming Zhang, Quan Vuong, Keith W. Ross.
        - URL: `FOCOPS <https://arxiv.org/abs/2002.06506>`_
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_
    """

    _p_dist: Normal

    def _init(self) -> None:
        """Initialize the SAM-FOCOPS specific model.

        The SAM-FOCOPS algorithm uses a Lagrange multiplier to balance the cost and reward,
        and implements SAM for sharpness-aware optimization.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        
        # SAM-specific parameters
        self._sam_rho = getattr(self._cfgs.algo_cfgs, 'sam_rho', 0.05)  # perturbation radius
        self._sam_adaptive = getattr(self._cfgs.algo_cfgs, 'sam_adaptive', True)  # adaptive SAM
        self._sam_alpha = getattr(self._cfgs.algo_cfgs, 'sam_alpha', 0.7)  # momentum for adaptive SAM

    def _init_log(self) -> None:
        """Log the SAM-FOCOPS specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        | Metrics/SAMRho             | The SAM perturbation radius. |
        | Metrics/SAMSharpness       | The computed sharpness. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Metrics/SAMRho')



    def _apply_sam_perturbation(self, model, perturbation: dict) -> None:
        """Apply SAM perturbation to the model parameters.
        
        Args:
            model: The model to perturb
            perturbation: Dictionary of perturbations for each parameter
        """
        for name, param in model.named_parameters():
            if name in perturbation:
                param.data.add_(perturbation[name])

    def _remove_sam_perturbation(self, model, perturbation: dict) -> None:
        """Remove SAM perturbation from the model parameters.
        
        Args:
            model: The model to unperturb
            perturbation: Dictionary of perturbations for each parameter
        """
        for name, param in model.named_parameters():
            if name in perturbation:
                param.data.sub_(perturbation[name])

    def _update_with_sam(self, model, optimizer, loss_fn, *args) -> torch.Tensor:
        """Update model using SAM optimization.
        
        Args:
            model: The model to update
            optimizer: The optimizer
            loss_fn: Function that computes the loss
            *args: Arguments to pass to loss_fn
            
        Returns:
            The final loss value
        """
        # Step 1: Compute initial loss and gradients
        initial_loss = loss_fn(*args)
        initial_loss.backward()
        
        # Step 2: Store gradients and compute perturbation
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Compute gradient norm
        grad_norm = torch.norm(torch.stack([grad.norm() for grad in gradients.values()]))
        
        if grad_norm > 0:
            scale = self._sam_rho / (grad_norm + 1e-12)
            perturbation = {}
            for name, grad in gradients.items():
                perturbation[name] = grad * scale
        else:
            perturbation = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        
        # Step 3: Apply perturbation
        self._apply_sam_perturbation(model, perturbation)
        
        # Step 4: Compute loss at perturbed point and update
        optimizer.zero_grad()
        perturbed_loss = loss_fn(*args)
        perturbed_loss.backward()
        
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                model.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        
        distributed.avg_grads(model)
        optimizer.step()
        
        # Step 5: Remove perturbation
        self._remove_sam_perturbation(model, perturbation)
        
        return perturbed_loss

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute pi/actor loss.

        In SAM-FOCOPS, the loss is defined as:

        .. math::
            :nowrap:

            \begin{eqnarray}
                L = \nabla_{\theta} D_{K L} \left( \pi_{\theta}^{'} \| \pi_{\theta} \right)[s]
                - \frac{1}{\eta} \underset{a \sim \pi_{\theta}}{\mathbb{E}} \left[
                    \frac{\nabla_{\theta} \pi_{\theta} (a \mid s)}{\pi_{\theta}(a \mid s)}
                    \left( A^{R}_{\pi_{\theta}} (s, a) - \lambda A^C_{\pi_{\theta}} (s, a) \right)
                \right]
            \end{eqnarray}

        where :math:`\eta` is a hyperparameter, :math:`\lambda` is the Lagrange multiplier,
        :math:`A_{\pi_{\theta_k}}(s, a)` is the advantage function,
        :math:`A^C_{\pi_{\theta_k}}(s, a)` is the cost advantage function,
        :math:`\pi^*` is the optimal policy, and :math:`\pi_{\theta}` is the current policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)

        kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)
        loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) * (
            kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        ).type(torch.float32)
        loss = loss.mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

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

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        SAM-FOCOPS uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        return (adv_r - self._lagrange.lagrangian_multiplier * adv_c) / (
            1 + self._lagrange.lagrangian_multiplier
        )

    def _update_reward_critic_with_sam(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        """Update reward critic using SAM optimization."""
        def loss_fn(obs, target_value_r):
            return torch.nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)
        
        loss = self._update_with_sam(
            self._actor_critic.reward_critic,
            self._actor_critic.reward_critic_optimizer,
            loss_fn,
            obs,
            target_value_r
        )
        
        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic_with_sam(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        """Update cost critic using SAM optimization."""
        def loss_fn(obs, target_value_c):
            return torch.nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)
        
        loss = self._update_with_sam(
            self._actor_critic.cost_critic,
            self._actor_critic.cost_critic_optimizer,
            loss_fn,
            obs,
            target_value_c
        )
        
        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor_with_sam(self, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, adv_r: torch.Tensor, adv_c: torch.Tensor) -> None:
        """Update actor using SAM optimization."""
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        
        def loss_fn(obs, act, logp, adv):
            return self._loss_pi(obs, act, logp, adv)
        
        self._update_with_sam(
            self._actor_critic.actor,
            self._actor_critic.actor_optimizer,
            loss_fn,
            obs,
            act,
            logp,
            adv
        )

    def _update(self) -> None:
        r"""Update actor, critic, and Lagrange multiplier parameters using SAM.

        In SAM-FOCOPS, the Lagrange multiplier is updated as the naive lagrange multiplier update.
        Then in each iteration of the policy update, SAM-FOCOPS calculates current policy's
        distribution, which used to calculate the policy loss with SAM optimization.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)

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
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

        dataloader = DataLoader(
            dataset=TensorDataset(
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
                old_mean,
                old_std,
            ),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        final_steps = self._cfgs.algo_cfgs.update_iters
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating with SAM...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
                old_mean,
                old_std,
            ) in dataloader:
                # Update critics with SAM
                self._update_reward_critic_with_sam(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic_with_sam(obs, target_value_c)

                self._p_dist = Normal(old_mean, old_std)
                # Update actor with SAM
                self._update_actor_with_sam(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            self._logger.store({'Train/KL': kl.item()})
            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                final_steps = i + 1
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': final_steps,
                'Value/Adv': adv_r.mean().item(),
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
                'Metrics/SAMRho': self._sam_rho,
            },
        )
