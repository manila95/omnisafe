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
"""Implementation of the SAM-augmented on-policy CRPO algorithm."""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.primal.crpo import OnCRPO
from omnisafe.utils.config import Config


class SAMOptimizer:
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.
    
    This wrapper applies SAM to any existing optimizer to find flatter minima
    in the loss landscape for better generalization.
    
    References:
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_.
    """
    
    def __init__(self, base_optimizer, rho=0.05, adaptive=True, eps=1e-12):
        """Initialize SAM optimizer.
        
        Args:
            base_optimizer: The base optimizer to wrap with SAM.
            rho (float): The perturbation radius for SAM.
            adaptive (bool): Whether to use adaptive SAM.
            eps (float): Small constant for numerical stability.
        """
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.eps = eps
        self.state = {}
        
    def zero_grad(self):
        """Zero gradients for all parameters."""
        self.base_optimizer.zero_grad()
        
    def step(self, closure):
        """Perform a single optimization step with SAM.
        
        Args:
            closure: A closure that computes the loss.
            
        Returns:
            The computed loss.
        """
        # First forward pass
        loss = closure()
        if not torch.isfinite(loss).all():
            print(f"Warning: Non-finite loss detected: {loss}")
            return loss
            
        loss.backward()
        
        # Store original gradients
        original_grads = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    original_grads[p] = p.grad.clone()
        
        # Compute SAM perturbation
        self._compute_sam_perturbation()
        
        # Second forward pass with perturbed parameters
        self.base_optimizer.zero_grad()
        loss = closure()
        if not torch.isfinite(loss).all():
            print(f"Warning: Non-finite loss after perturbation: {loss}")
            # Restore parameters and return original loss
            self._restore_parameters()
            return loss
            
        loss.backward()
        
        # Restore original gradients for the actual update
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and p in original_grads:
                    p.grad = original_grads[p]
        
        # Perform the actual update
        self.base_optimizer.step()
        
        return loss
    
    def _compute_sam_perturbation(self):
        """Compute SAM perturbation for all parameters."""
        # Store original parameters for potential restoration
        self._original_params = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self._original_params[p] = p.data.clone()
        
        # Compute gradient norm
        grad_norm = 0.0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm += p.grad.norm(2) ** 2
        grad_norm = grad_norm ** 0.5
        
        # Check for numerical stability
        if not torch.isfinite(grad_norm) or grad_norm < self.eps:
            print(f"Warning: Invalid gradient norm: {grad_norm}")
            return
        
        # Compute perturbation scale
        scale = self.rho / (grad_norm + self.eps)
        
        # Apply perturbation with gradient clipping
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    perturbation = p.grad * scale
                    # Clip perturbation to prevent extreme values
                    perturbation = torch.clamp(perturbation, -self.rho, self.rho)
                    p.data.add_(perturbation)
    
    def _restore_parameters(self):
        """Restore parameters to their original state."""
        if hasattr(self, '_original_params'):
            for p, original_data in self._original_params.items():
                p.data = original_data
    
    def __getattr__(self, name):
        """Delegate attribute access to base optimizer."""
        return getattr(self.base_optimizer, name)


@registry.register
class SAMOnCRPO(OnCRPO):
    """The SAM-augmented on-policy CRPO algorithm.

    This algorithm combines CRPO with Sharpness Aware Minimization (SAM) to improve generalization
    by finding flatter minima in the loss landscape while maintaining safety constraints.

    References:
        - Title: CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee.
        - Authors: Tengyu Xu, Yingbin Liang, Guanghui Lan.
        - URL: `CRPO <https://arxiv.org/pdf/2011.05869.pdf>`_.
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_.
    """

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize an instance of :class:`SAMOnCRPO`."""
        super().__init__(env_id, cfgs)
        
        # SAM-specific parameters
        self._sam_rho = getattr(self._cfgs.algo_cfgs, 'sam_rho', 0.01)  # More conservative default
        self._sam_adaptive = getattr(self._cfgs.algo_cfgs, 'sam_adaptive', False)  # Disable adaptive by default
        self._sam_eps = getattr(self._cfgs.algo_cfgs, 'sam_eps', 1e-8)  # Larger epsilon for stability
        
        # Apply SAM to optimizers if they exist
        self._apply_sam_to_optimizers()

    def _init_log(self) -> None:
        """Log the SAM-CRPO specific information."""
        super()._init_log()
        self._logger.register_key('Misc/SAM_rho')
        self._logger.register_key('Misc/SAM_adaptive')
        self._logger.register_key('Loss/Loss_pi_sam')
        self._logger.register_key('Loss/Loss_reward_critic_sam')
        self._logger.register_key('Loss/Loss_cost_critic_sam')

    def _apply_sam_to_optimizers(self) -> None:
        """Apply SAM optimization to the algorithm's optimizers."""
        # Apply SAM to actor optimizer if it exists
        if hasattr(self._actor_critic, 'actor_optimizer'):
            original_optimizer = self._actor_critic.actor_optimizer
            self._actor_critic.actor_optimizer = SAMOptimizer(
                original_optimizer,
                rho=self._sam_rho,
                adaptive=self._sam_adaptive,
                eps=self._sam_eps
            )
        
        # Apply SAM to reward critic optimizer if it exists
        if hasattr(self._actor_critic, 'reward_critic_optimizer'):
            original_optimizer = self._actor_critic.reward_critic_optimizer
            self._actor_critic.reward_critic_optimizer = SAMOptimizer(
                original_optimizer,
                rho=self._sam_rho,
                adaptive=self._sam_adaptive,
                eps=self._sam_eps
            )
            
        # Apply SAM to cost critic optimizer if it exists
        if hasattr(self._actor_critic, 'cost_critic_optimizer'):
            original_optimizer = self._actor_critic.cost_critic_optimizer
            self._actor_critic.cost_critic_optimizer = SAMOptimizer(
                original_optimizer,
                rho=self._sam_rho,
                adaptive=self._sam_adaptive,
                eps=self._sam_eps
            )

    def _update_actor_with_sam(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update actor using SAM optimization.

        This method applies SAM to the actor update process, which helps find flatter minima
        in the loss landscape for better generalization.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        
        # Create closure for SAM optimizer
        def actor_loss_closure():
            return self._loss_pi(obs, act, logp, adv)
        
        # Use SAM optimizer if available, otherwise fall back to standard TRPO update
        if hasattr(self._actor_critic, 'actor_optimizer') and isinstance(
            self._actor_critic.actor_optimizer, SAMOptimizer
        ):
            self._actor_critic.actor_optimizer.zero_grad()
            loss = self._actor_critic.actor_optimizer.step(actor_loss_closure)
            if torch.isfinite(loss).all():
                self._logger.store({'Loss/Loss_pi_sam': loss.mean().item()})
            else:
                print(f"Warning: Non-finite actor loss: {loss}")
        else:
            # Fall back to standard TRPO update
            super()._update_actor(obs, act, logp, adv_r, adv_c)

    def _update_critic_with_sam(
        self,
        obs: torch.Tensor,
        target_value: torch.Tensor,
        critic_type: str = 'reward'
    ) -> None:
        """Update critic using SAM optimization.

        Args:
            obs (torch.Tensor): The observation tensor.
            target_value (torch.Tensor): The target value tensor.
            critic_type (str): Type of critic ('reward' or 'cost').
        """
        optimizer_name = f'{critic_type}_critic_optimizer'
        
        # Create closure for SAM optimizer
        def critic_loss_closure():
            critic = getattr(self._actor_critic, f'{critic_type}_critic')
            loss = nn.functional.mse_loss(critic(obs)[0], target_value)
            if hasattr(self._cfgs.algo_cfgs, 'use_critic_norm') and self._cfgs.algo_cfgs.use_critic_norm:
                for param in critic.parameters():
                    loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
            return loss
        
        # Use SAM optimizer if available
        if hasattr(self._actor_critic, optimizer_name) and isinstance(
            getattr(self._actor_critic, optimizer_name), SAMOptimizer
        ):
            optimizer = getattr(self._actor_critic, optimizer_name)
            optimizer.zero_grad()
            loss = optimizer.step(critic_loss_closure)
            if torch.isfinite(loss).all():
                self._logger.store({f'Loss/Loss_{critic_type}_critic_sam': loss.mean().item()})
            else:
                print(f"Warning: Non-finite {critic_type} critic loss: {loss}")
        else:
            # Fall back to standard critic update
            super()._update_critic(obs, target_value, critic_type)

    def _update(self) -> None:
        """Update actor and critic using SAM optimization."""
        data = self._buf.get()
        obs, act, logp, adv_r, adv_c, target_value_r, target_value_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['adv_r'],
            data['adv_c'],
            data['target_value_r'],
            data['target_value_c'],
        )
        
        # Update actor using SAM
        self._update_actor_with_sam(obs, act, logp, adv_r, adv_c)
        
        # Update critics using SAM
        self._update_critic_with_sam(obs, target_value_r, 'reward')
        if hasattr(self._actor_critic, 'cost_critic'):
            self._update_critic_with_sam(obs, target_value_c, 'cost')
        
        # Log SAM-specific metrics
        self._logger.store({
            'Misc/SAM_rho': self._sam_rho,
            'Misc/SAM_adaptive': float(self._sam_adaptive),
        })
    