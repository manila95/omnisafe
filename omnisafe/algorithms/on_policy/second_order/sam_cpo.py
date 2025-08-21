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
"""Implementation of the SAM-CPO algorithm."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.second_order.cpo import CPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class SAMCPO(CPO):
    """The SAM-augmented Constrained Policy Optimization (SAM-CPO) algorithm.

    This algorithm extends CPO with Sharpness Aware Minimization (SAM) to find flatter minima
    in the loss landscape, which typically leads to better generalization and robustness.

    References:
        - Title: Constrained Policy Optimization
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `CPO <https://arxiv.org/abs/1705.10528>`_
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_
    """

    def _init_log(self) -> None:
        super()._init_log()
        # SAM-specific logging
        self._logger.register_key('Misc/SAM_rho')
        self._logger.register_key('Misc/SAM_adaptive')
        self._logger.register_key('Misc/SAM_perturbation_norm')

    def _compute_sam_perturbation(
        self, 
        parameters, 
        rho: float = 0.05, 
        adaptive: bool = False
    ) -> dict:
        """
        Compute SAM perturbation for parameters.
        
        Args:
            parameters: Parameters to perturb
            rho: SAM neighborhood size
            adaptive: Whether to use adaptive SAM
            
        Returns:
            Dictionary mapping parameters to their original values
        """
        original_params = {}
        perturbation_norm = 0.0
        
        # Store original parameters and compute perturbation
        for param in parameters:
            if param.grad is not None:
                original_params[param] = param.data.clone()
                
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
                perturbation = param.grad.data * scale
                param.data.add_(perturbation)
                perturbation_norm += perturbation.norm(2).item()
        
        # Store perturbation norm for logging
        self._last_perturbation_norm = perturbation_norm
        
        return original_params

    def _restore_parameters(self, original_params: dict) -> None:
        """Restore parameters to their original values."""
        for param, original_data in original_params.items():
            param.data = original_data

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network using SAM-augmented CPO.

        This method extends CPO with SAM optimization by:
        1. Computing gradients with original parameters
        2. Applying SAM perturbation to parameters
        3. Computing gradients with perturbed parameters
        4. Using the perturbed gradients for the CPO optimization

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
        rho = sam_config.get('rho', 0.05)  # SAM neighborhood size
        adaptive = sam_config.get('adaptive', False)  # Whether to use adaptive SAM
        
        # First forward pass to compute initial gradients
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward_before = distributed.dist_avg(loss_reward)
        p_dist = self._actor_critic.actor(obs)

        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        # Apply SAM perturbation to actor parameters
        original_params = self._compute_sam_perturbation(
            self._actor_critic.actor.parameters(), rho, adaptive
        )

        # Second forward pass with perturbed parameters for reward loss
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        # Get reward gradients from perturbed parameters
        grads = -get_flat_gradients_from(self._actor_critic.actor)
        
        # Compute cost gradients from the SAME perturbed parameters
        self._actor_critic.actor.zero_grad()
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
        loss_cost_before = distributed.dist_avg(loss_cost)
        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        # Get cost gradients from the same perturbed parameters
        b_grads = get_flat_gradients_from(self._actor_critic.actor)

        # Restore original parameters
        self._restore_parameters(original_params)

        # Continue with CPO's conjugate gradient optimization
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit

        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=ep_costs,
            q=q,
            r=r,
            s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )

        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            loss = loss_reward + loss_cost

        # Compute perturbation norm for logging (compute it during the SAM perturbation)
        perturbation_norm = 0.0
        # We need to compute this during the SAM perturbation, so we'll store it
        if hasattr(self, '_last_perturbation_norm'):
            perturbation_norm = self._last_perturbation_norm
        else:
            perturbation_norm = 0.0

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
                'Misc/SAM_rho': rho,
                'Misc/SAM_adaptive': float(adaptive),
                'Misc/SAM_perturbation_norm': perturbation_norm,
            },
        )
