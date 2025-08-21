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
"""Implementation of the CUP algorithm with SAM (Sharpness-Aware Minimization)."""

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.first_order.cup import CUP
from omnisafe.utils import distributed


class SAMOptimizer:
    """
    Sharpness Aware Minimization (SAM) optimizer wrapper.
    
    SAM finds parameters that lie in neighborhoods having uniformly low loss values.
    This is achieved by minimizing the maximum loss in a neighborhood around the current parameters.
    """
    
    def __init__(
        self,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12
    ):
        """
        Initialize SAM optimizer.
        
        Args:
            base_optimizer: The base optimizer (e.g., Adam, SGD)
            rho: The neighborhood size for SAM
            adaptive: Whether to use adaptive SAM (ASAM)
            eps: Small constant for numerical stability
        """
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.eps = eps
        
    def zero_grad(self):
        """Zero gradients for all parameters."""
        self.base_optimizer.zero_grad()
        
    def step(self, closure=None):
        """
        Perform SAM update step.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            The computed loss value
        """
        if closure is None:
            raise ValueError("SAM requires a closure function to compute loss")
            
        # First forward pass to compute gradients
        loss = closure()
        loss.backward()
        
        # Store original parameters
        original_params = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    original_params[p] = p.data.clone()
        
        # Compute SAM perturbation
        self._compute_sam_perturbation()
        
        # Zero gradients before second forward pass
        self.zero_grad()
        
        # Second forward pass with perturbed parameters
        loss = closure()
        loss.backward()
        
        # Restore original parameters
        for p, original_data in original_params.items():
            p.data = original_data
            
        # Update parameters using base optimizer
        self.base_optimizer.step()
        
        return loss
    
    def _compute_sam_perturbation(self):
        """Compute SAM perturbation for all parameters."""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if self.adaptive:
                        # Adaptive SAM: scale perturbation by parameter norm
                        param_norm = p.data.norm(2)
                        if param_norm > 0:
                            scale = self.rho / (param_norm + self.eps)
                        else:
                            scale = self.rho
                    else:
                        # Standard SAM: scale by gradient norm
                        grad_norm = p.grad.data.norm(2)
                        if grad_norm > 0:
                            scale = self.rho / (grad_norm + self.eps)
                        else:
                            scale = 0
                    
                    # Apply perturbation
                    p.data.add_(p.grad.data * scale)
    
    def __getattr__(self, name):
        """Delegate attribute access to base optimizer."""
        return getattr(self.base_optimizer, name)


@registry.register
class SAMCUP(CUP):
    """The Constrained Update Projection (CUP) Approach to Safe Policy Optimization with SAM.

    This algorithm extends CUP with Sharpness Aware Minimization (SAM) optimization,
    which helps find flatter minima in the loss landscape for better generalization
    while maintaining constraint satisfaction.

    References:
        - Title: Constrained Update Projection Approach to Safe Policy Optimization
        - Authors: Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li,
            Yaodong Yang, Gang Pan.
        - URL: `CUP <https://arxiv.org/abs/2209.07089>`_
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally apply SAM to optimizers.
        """
        super()._init()
        self._apply_sam_to_optimizers()

    def _apply_sam_to_optimizers(self) -> None:
        """Apply SAM optimization to the algorithm's optimizers."""
        # Get SAM configuration from algorithm configs
        sam_config = getattr(self._cfgs.algo_cfgs, 'sam_cfgs', {})
        
        # Apply SAM to actor optimizer
        if hasattr(self._actor_critic, 'actor_optimizer'):
            original_optimizer = self._actor_critic.actor_optimizer
            self._actor_critic.actor_optimizer = SAMOptimizer(
                original_optimizer,
                rho=sam_config.get('actor_rho', 0.04),
                adaptive=sam_config.get('adaptive', False),
                eps=sam_config.get('eps', 1e-12)
            )
            
        # Apply SAM to reward critic optimizer
        if hasattr(self._actor_critic, 'reward_critic_optimizer'):
            original_optimizer = self._actor_critic.reward_critic_optimizer
            self._actor_critic.reward_critic_optimizer = SAMOptimizer(
                original_optimizer,
                rho=sam_config.get('critic_rho', 0.05),
                adaptive=sam_config.get('adaptive', False),
                eps=sam_config.get('eps', 1e-12)
            )
            
        # Apply SAM to cost critic optimizer if it exists
        if hasattr(self._actor_critic, 'cost_critic_optimizer'):
            original_optimizer = self._actor_critic.cost_critic_optimizer
            self._actor_critic.cost_critic_optimizer = SAMOptimizer(
                original_optimizer,
                rho=sam_config.get('cost_critic_rho', 0.04),
                adaptive=sam_config.get('adaptive', False),
                eps=sam_config.get('eps', 1e-12)
            )

    def _init_log(self) -> None:
        """Log the SAM CUP specific information."""
        super()._init_log()
        # Add SAM-specific logging keys
        self._logger.register_key('Misc/SAM_rho')
        self._logger.register_key('Misc/SAM_adaptive')

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        """Update reward critic using SAM optimization."""
        # Create closure for SAM optimizer
        def critic_loss_closure():
            # Detach tensors to ensure fresh computation graph
            obs_detached = obs.detach()
            target_value_r_detached = target_value_r.detach()
            loss = torch.nn.functional.mse_loss(self._actor_critic.reward_critic(obs_detached)[0], target_value_r_detached)
            if self._cfgs.algo_cfgs.use_critic_norm:
                for param in self._actor_critic.reward_critic.parameters():
                    loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
            return loss
        
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = self._actor_critic.reward_critic_optimizer.step(critic_loss_closure)
        
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        
        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        """Update cost critic using SAM optimization."""
        # Create closure for SAM optimizer
        def cost_critic_loss_closure():
            # Detach tensors to ensure fresh computation graph
            obs_detached = obs.detach()
            target_value_c_detached = target_value_c.detach()
            loss = torch.nn.functional.mse_loss(self._actor_critic.cost_critic(obs_detached)[0], target_value_c_detached)
            if self._cfgs.algo_cfgs.use_critic_norm:
                for param in self._actor_critic.cost_critic.parameters():
                    loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
            return loss
        
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = self._actor_critic.cost_critic_optimizer.step(cost_critic_loss_closure)
        
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        
        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update actor using SAM optimization for reward advantage."""
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        
        # Create closure for SAM optimizer
        def actor_loss_closure():
            # Detach tensors to ensure fresh computation graph
            obs_detached = obs.detach()
            act_detached = act.detach()
            logp_detached = logp.detach()
            adv_detached = adv.detach()
            return self._loss_pi(obs_detached, act_detached, logp_detached, adv_detached)
        
        self._actor_critic.actor_optimizer.zero_grad()
        loss = self._actor_critic.actor_optimizer.step(actor_loss_closure)
        
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Compute surrogate advantage for actor update."""
        # This is a simple surrogate - you can modify this based on your needs
        return adv_r - self._lagrange.lagrangian_multiplier * adv_c

    def _update(self) -> None:
        r"""Update actor, critic, and Lagrange multiplier parameters using SAM.

        This overrides the parent's _update method to use SAM for the cost constraint optimization.
        """
        # Update Lagrange multiplier (inherited from CUP)
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._lagrange.update_lagrange_multiplier(Jc)

        # Call parent's _update for reward critic updates (skip CUP's cost constraint part)
        super(CUP, self)._update()

        # SAM-enhanced cost constraint optimization
        data = self._buf.get()
        obs, act, logp, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['adv_c'],
        )
        original_obs = obs
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

        from torch.utils.data import DataLoader, TensorDataset
        from torch.distributions import Normal
        from rich.progress import track

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, adv_c, old_mean, old_std),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        final_steps = self._cfgs.algo_cfgs.update_iters
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for obs, act, logp, adv_c, old_mean, old_std in dataloader:
                self._p_dist = Normal(old_mean, old_std)
                
                # Update actor using SAM for cost constraint
                def cost_loss_closure():
                    # Detach tensors to ensure fresh computation graph
                    obs_detached = obs.detach()
                    act_detached = act.detach()
                    logp_detached = logp.detach()
                    adv_c_detached = adv_c.detach()
                    return self._loss_pi_cost(obs_detached, act_detached, logp_detached, adv_c_detached)
                
                self._actor_critic.actor_optimizer.zero_grad()
                loss_cost = self._actor_critic.actor_optimizer.step(cost_loss_closure)
                
                # Apply gradient clipping if specified
                if self._cfgs.algo_cfgs.max_grad_norm is not None:
                    clip_grad_norm_(
                        self._actor_critic.actor.parameters(),
                        self._cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self._actor_critic.actor)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                final_steps = i + 1
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        # Log SAM-specific information
        sam_config = getattr(self._cfgs.algo_cfgs, 'sam_cfgs', {})
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.item(),
                'Train/SecondStepStopIter': final_steps,
                'Misc/SAM_rho': sam_config.get('actor_rho', 0.04),
                'Misc/SAM_adaptive': float(sam_config.get('adaptive', False)),
            },
        )
