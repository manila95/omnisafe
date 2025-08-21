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
"""Implementation of the TRPO algorithm with SAM (Sharpness-Aware Minimization)."""

from __future__ import annotations

import torch
from torch.distributions import Distribution

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class SAMTRPO(TRPO):
    """The Trust Region Policy Optimization (TRPO) algorithm with SAM.

    This implementation augments TRPO with Sharpness-Aware Minimization (SAM) to improve
    generalization by finding and optimizing against worst-case perturbations.

    References:
        - Title: Trust Region Policy Optimization
        - Authors: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel.
        - URL: `TRPO <https://arxiv.org/abs/1502.05477>`_
        - Title: Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
        - URL: `SAM <https://arxiv.org/abs/2010.01412>`_
    """

    def _init_log(self) -> None:
        """Log the SAM-TRPO specific information.

        +---------------------+-----------------------------+
        | Things to log       | Description                 |
        +=====================+=============================+
        | Misc/SAM_rho        | The SAM perturbation radius.|
        +---------------------+-----------------------------+
        | Misc/SAM_perturb_norm| The norm of SAM perturbation.|
        +---------------------+-----------------------------+
        | Misc/SAM_loss_diff  | Difference between perturbed and original loss.|
        +---------------------+-----------------------------+
        """
        super()._init_log()
        self._logger.register_key('Misc/SAM_rho')
        self._logger.register_key('Misc/SAM_perturb_norm')
        self._logger.register_key('Misc/SAM_loss_diff')

    def _get_sam_perturbation(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        rho: float = 0.05,
    ) -> torch.Tensor:
        """Compute SAM perturbation by finding the worst-case direction.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage tensor.
            rho (float): The perturbation radius for SAM.

        Returns:
            torch.Tensor: The computed perturbation vector.
        """
        # Get current parameters
        theta = get_flat_params_from(self._actor_critic.actor)
        
        # Compute loss and gradients
        self._actor_critic.actor.zero_grad()
        loss = self._loss_pi(obs, act, logp, adv)
        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)
        
        # Get gradients
        grads = get_flat_gradients_from(self._actor_critic.actor)
        
        # Check if adaptive SAM is enabled
        sam_cfgs = getattr(self._cfgs.algo_cfgs, 'sam_cfgs', {})
        sam_adaptive = sam_cfgs.get('adaptive', False)
        
        if sam_adaptive:
            # Adaptive SAM: adjust rho based on gradient norm
            grad_norm = torch.norm(grads)
            if grad_norm > 0:
                # Scale rho by the gradient norm
                adaptive_rho = rho * grad_norm
                # Clip to min/max bounds
                rho_min = sam_cfgs.get('rho_min', 0.01)
                rho_max = sam_cfgs.get('rho_max', 0.1)
                adaptive_rho = torch.clamp(adaptive_rho, rho_min, rho_max)
                perturbation = adaptive_rho * grads / grad_norm
            else:
                perturbation = torch.zeros_like(grads)
        else:
            # Standard SAM: fixed rho
            grad_norm = torch.norm(grads)
            if grad_norm > 0:
                perturbation = rho * grads / grad_norm
            else:
                perturbation = torch.zeros_like(grads)
            
        return perturbation

    def _compute_sam_loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss at the perturbed parameters.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage tensor.
            perturbation (torch.Tensor): The perturbation to apply.

        Returns:
            torch.Tensor: The loss at perturbed parameters.
        """
        # Get current parameters
        theta_old = get_flat_params_from(self._actor_critic.actor)
        
        # Apply perturbation
        theta_perturbed = theta_old + perturbation
        set_param_values_to_model(self._actor_critic.actor, theta_perturbed)
        
        # Compute loss at perturbed parameters
        with torch.no_grad():
            loss_perturbed = self._loss_pi(obs, act, logp, adv)
            
        # Restore original parameters
        set_param_values_to_model(self._actor_critic.actor, theta_old)
        
        return loss_perturbed

    def _update_actor(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network using SAM-TRPO.

        This method extends TRPO's update by incorporating SAM:
        1. Computing SAM perturbation to find worst-case direction
        2. Computing gradients at the perturbed point
        3. Using TRPO's natural gradient update with SAM gradients

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        # Get SAM configuration
        sam_cfgs = getattr(self._cfgs.algo_cfgs, 'sam_cfgs', {})
        rho = sam_cfgs.get('actor_rho', 0.05)
        
        # Compute advantage
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        
        # Step 1: Compute SAM perturbation
        perturbation = self._get_sam_perturbation(obs, act, logp, adv, rho)
        
        # Step 2: Compute loss at perturbed parameters
        loss_perturbed = self._compute_sam_loss(obs, act, logp, adv, perturbation)
        
        # Step 3: Apply perturbation and compute gradients
        theta_old = get_flat_params_from(self._actor_critic.actor)
        theta_perturbed = theta_old + perturbation
        set_param_values_to_model(self._actor_critic.actor, theta_perturbed)
        
        # Step 4: Use TRPO's update logic with perturbed parameters
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        self._actor_critic.actor.zero_grad()
        loss = self._loss_pi(obs, act, logp, adv)
        loss_before = distributed.dist_avg(loss)
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        # Get gradients from perturbed point
        grads = -get_flat_gradients_from(self._actor_critic.actor)
        
        # Step 5: Use TRPO's natural gradient update
        from omnisafe.utils.math import conjugate_gradients
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        # Step 6: Use TRPO's line search
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

        # Step 7: Apply update
        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv)

        # Log SAM-specific metrics
        perturb_norm = torch.norm(perturbation).item()
        loss_diff = (loss_perturbed - loss_before).item()
        
        # Get adaptive rho if using adaptive SAM
        sam_adaptive = sam_cfgs.get('adaptive', False)
        if sam_adaptive:
            grad_norm = torch.norm(grads).item()
            adaptive_rho = min(max(rho * grad_norm, sam_cfgs.get('rho_min', 0.01)),
                             sam_cfgs.get('rho_max', 0.1))
            effective_rho = adaptive_rho
        else:
            effective_rho = rho
        
        # Store both TRPO and SAM metrics
        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/SAM_rho': effective_rho,
                'Misc/SAM_perturb_norm': perturb_norm,
                'Misc/SAM_loss_diff': loss_diff,
            },
        )
