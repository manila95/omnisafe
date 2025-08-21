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
"""Implementation of the PID-Lagrange version of the TRPO algorithm with SAM."""

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.algorithms.on_policy.pid_lagrange.trpo_pid import TRPOPID
from omnisafe.common.pid_lagrange import PIDLagrangian
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


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
        Perform SAM optimization step.
        
        Args:
            closure: Optional closure that computes the loss
            
        Returns:
            The computed loss value
        """
        if closure is None:
            return self.base_optimizer.step()
        
        # First forward pass
        loss = closure()
        loss.backward()
        
        # Store original parameters
        original_params = {}
        for group in self.base_optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    original_params[param] = param.data.clone()
        
        # Compute SAM perturbation
        for group in self.base_optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if self.adaptive:
                        # Adaptive SAM: scale perturbation by parameter norm
                        param_norm = param.data.norm(2)
                        if param_norm > 0:
                            scale = self.rho / (param_norm + self.eps)
                        else:
                            scale = self.rho
                    else:
                        # Standard SAM: scale by gradient norm
                        grad_norm = param.grad.data.norm(2)
                        if grad_norm > 0:
                            scale = self.rho / (grad_norm + self.eps)
                        else:
                            scale = 0
                    
                    # Apply perturbation
                    param.data.add_(param.grad.data * scale)
        
        # Second forward pass with perturbed parameters
        self.base_optimizer.zero_grad()
        loss = closure()
        loss.backward()
        
        # Restore original parameters
        for param, original_data in original_params.items():
            param.data = original_data
        
        # Perform the actual optimization step
        self.base_optimizer.step()
        
        return loss


@registry.register
class SAMTRPOPID(TRPOPID):
    """The SAM-augmented PID-Lagrange version of the TRPO algorithm.

    A combination of the PID-Lagrange method, Trust Region Policy Optimization algorithm,
    and Sharpness Aware Minimization (SAM) for improved generalization.
    
    References:
        - Title: Trust Region Policy Optimization
        - Authors: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel.
        - URL: `TRPO <https://arxiv.org/abs/1502.05477>`_
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_
    """

    def _init(self) -> None:
        """Initialize the SAMTRPOPID specific model.

        The SAMTRPOPID algorithm uses a PID-Lagrange multiplier to balance the cost and reward,
        and applies SAM optimization for improved generalization.
        """
        super()._init()
        self._lagrange: PIDLagrangian = PIDLagrangian(**self._cfgs.lagrange_cfgs)
        
        # Apply SAM to optimizers if they exist
        self._apply_sam_to_optimizers()

    def _apply_sam_to_optimizers(self) -> None:
        """Apply SAM optimization to all optimizers in the algorithm."""
        # Get SAM configuration from algorithm configs
        sam_config = getattr(self._cfgs.algo_cfgs, 'sam_cfgs', {})
        
        # Apply SAM to reward critic optimizer
        if hasattr(self._actor_critic, 'reward_critic_optimizer'):
            original_optimizer = self._actor_critic.reward_critic_optimizer
            self._actor_critic.reward_critic_optimizer = SAMOptimizer(
                original_optimizer,
                rho=sam_config.get('critic_rho', 0.05),
                adaptive=sam_config.get('adaptive', False),
                eps=sam_config.get('eps', 1e-12)
            )
            
        # Apply SAM to cost critic optimizer (if exists)
        if hasattr(self._actor_critic, 'cost_critic_optimizer'):
            original_optimizer = self._actor_critic.cost_critic_optimizer
            self._actor_critic.cost_critic_optimizer = SAMOptimizer(
                original_optimizer,
                rho=sam_config.get('cost_critic_rho', 0.05),
                adaptive=sam_config.get('adaptive', False),
                eps=sam_config.get('eps', 1e-12)
            )

    def _init_log(self) -> None:
        """Log the SAMTRPOPID specific information.

        +----------------------------+------------------------------+
        | Things to log              | Description                  |
        +============================+==============================+
        | Misc/SAM_rho               | The SAM neighborhood size.   |
        | Misc/SAM_adaptive          | Whether adaptive SAM is used.|
        +----------------------------+------------------------------+
        """
        super()._init_log()
        # Note: Metrics/LagrangeMultiplier is already registered by TRPOPID parent class
        self._logger.register_key('Misc/SAM_rho')
        self._logger.register_key('Misc/SAM_adaptive')

    def _compute_sam_perturbation(self, parameters, rho: float = 0.05, adaptive: bool = False) -> dict:
        """
        Compute SAM perturbation for parameters.
        
        This method applies SAM perturbation to parameters before computing gradients.
        It's used for the actor network which uses explicit optimization (TRPO).
        
        Args:
            parameters: Parameters to perturb
            rho: SAM neighborhood size
            adaptive: Whether to use adaptive SAM
            
        Returns:
            Dictionary of original parameters for restoration
        """
        # Store original parameters
        original_params = {}
        for param in parameters:
            if param.grad is not None:
                original_params[param] = param.data.clone()
        
        # Compute perturbation
        for param in parameters:
            if param.grad is not None:
                if adaptive:
                    # Adaptive SAM: scale perturbation by parameter norm
                    param_norm = param.data.norm(2)
                    if param_norm > 0:
                        scale = rho / (param_norm + 1e-12)
                    else:
                        scale = rho
                else:
                    # Standard SAM: scale by gradient norm
                    grad_norm = param.grad.data.norm(2)
                    if grad_norm > 0:
                        scale = rho / (grad_norm + 1e-12)
                    else:
                        scale = 0
                
                # Apply perturbation
                param.data.add_(param.grad.data * scale)
        
        return original_params

    def _restore_parameters(self, original_params: dict) -> None:
        """Restore parameters to their original values."""
        for param, original_data in original_params.items():
            param.data = original_data

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the PID-Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method, and apply SAM optimization for improved generalization.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = \mathbb{E}_{s_t \sim \rho_{\pi}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old} (a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the PID-Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update PID-Lagrange multiplier parameter
        self._lagrange.pid_update(Jc)
        
        # Get data from buffer
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
        
        # Update reward critic using SAM
        self._update_reward_critic(obs, target_value_r)
        
        # Update cost critic using SAM (if using cost)
        if self._cfgs.algo_cfgs.use_cost:
            self._update_cost_critic(obs, target_value_c)
        
        # Update actor using SAM-augmented TRPO
        self._update_actor(obs, act, logp, adv_r, adv_c)

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network using SAM-augmented TRPO with PID-Lagrange.

        This method combines SAM optimization with TRPO's trust region mechanism
        and TRPOPID's PID-Lagrange multiplier for constraint handling.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        
        # Get SAM configuration
        sam_config = getattr(self._cfgs.algo_cfgs, 'sam_cfgs', {})
        rho = sam_config.get('actor_rho', 0.05)
        adaptive = sam_config.get('adaptive', False)
        
        # First forward pass to compute gradients using TRPOPID's surrogate
        self._actor_critic.actor.zero_grad()
        adv = self._compute_adv_surrogate(adv_r, adv_c)  # Uses TRPOPID's method
        loss = self._loss_pi(obs, act, logp, adv)
        loss_before = distributed.dist_avg(loss)
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        # Apply SAM perturbation to actor parameters
        original_params = self._compute_sam_perturbation(
            self._actor_critic.actor.parameters(), rho, adaptive
        )

        # Second forward pass with perturbed parameters
        self._actor_critic.actor.zero_grad()
        loss = self._loss_pi(obs, act, logp, adv)
        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        # Restore original parameters
        self._restore_parameters(original_params)

        # Continue with TRPO's conjugate gradient optimization
        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        step_direction, accept_step = self._search_step_size(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv=adv,
            loss_before=loss_before,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/SAM_rho': rho,
                'Misc/SAM_adaptive': float(adaptive),
            },
        )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        """Update reward critic using SAM optimization."""
        # Create closure for SAM optimizer
        def critic_loss_closure():
            loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)
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
            loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)
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

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        SAMTRPOPID uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [A^{R}_{\pi_{\theta}}(s, a)
            - \lambda A^C_{\pi_{\theta}}(s, a)]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The ``advantage`` combined with ``reward_advantage`` and ``cost_advantage``.
        """
        penalty = self._lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
