import wandb
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable
import numpy as np
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)



def actor_sam_fn(sam_type, use_kl):
    if sam_type == "v1":
        if use_kl:
            return compute_sam_gradients_v1_kl
        else:
            return compute_sam_gradients_v1
    elif sam_type == "v2":
        if use_kl:
            return compute_sam_gradients_v2_kl
        else:
            return compute_sam_gradients_v2
    elif sam_type == "v3":
        if use_kl:
            return compute_sam_gradients_v3_kl
        else:
            return compute_sam_gradients_v3
    elif sam_type == "v4":
        return compute_sam_gradients_v4
    else:
        raise ValueError(f"Invalid SAM type: {sam_type}")




def compute_sam_gradients_v1(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """Compute Sharpness Aware Minimization gradients.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """
    # First compute the base loss and gradients
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = -(ratio * advantage_lag).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    grad_norm = torch.norm(grads)
    
    # Compute perturbation
    scale = rho / (grad_norm + 1e-12)
    perturbed_params = []
    for param in policy.actor.parameters():
        if param.grad is None:
            continue
        e_w = param.grad * scale.to(param)
        perturbed_params.append(e_w)
        param.data.add_(e_w)
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    for param, e_w in zip(policy.actor.parameters(), perturbed_params):
        if param.grad is None:
            continue
        param.data.sub_(e_w)
    
    return sam_grads, perturbed_params, None, None, None

def compute_sam_gradients_v2(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """Compute Sharpness Aware Minimization gradients.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """
    # First compute the base loss and gradients
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = (ratio * advantage_cost).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    grad_norm = torch.norm(grads)
    
    # Compute perturbation
    scale = rho / (grad_norm + 1e-12)
    perturbed_params = []
    for param in policy.actor.parameters():
        if param.grad is None:
            continue
        e_w = param.grad * scale.to(param)
        perturbed_params.append(e_w)
        param.data.add_(e_w)
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    for param, e_w in zip(policy.actor.parameters(), perturbed_params):
        if param.grad is None:
            continue
        param.data.sub_(e_w)
    
    return sam_grads, perturbed_params, None, None, None

def compute_sam_gradients_v3(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """Compute Sharpness Aware Minimization gradients.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """
    # First compute the base loss and gradients
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = -(ratio * advantage_reward).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    grad_norm = torch.norm(grads)
    
    # Compute perturbation
    scale = rho / (grad_norm + 1e-12)
    perturbed_params = []
    for param in policy.actor.parameters():
        if param.grad is None:
            continue
        e_w = param.grad * scale.to(param)
        perturbed_params.append(e_w)
        param.data.add_(e_w)
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    for param, e_w in zip(policy.actor.parameters(), perturbed_params):
        if param.grad is None:
            continue
        param.data.sub_(e_w)
    
    return sam_grads, perturbed_params, None, None, None


def compute_sam_gradients_critic(critic, data, target_values, rho=0.05, num_samples=10):
    """Compute Sharpness Aware Minimization gradients for critic.
    
    Args:
        critic: The critic network (reward or cost)
        data: Dictionary containing observations, risk values
        target_values: Target values for the critic
        rho: Perturbation radius for SAM
        
    Returns:
        sam_grads: Dictionary mapping parameter names to their SAM gradients
        perturbed_params: The perturbed parameters
    """
    # Store original parameters
    original_params = []
    for param in critic.parameters():
        if param.requires_grad:
            original_params.append(param.data.clone())
    
    # First compute the base loss and gradients
    critic.zero_grad()
    value_pred = critic(data["obs"], data["risk"])
    base_loss = nn.functional.mse_loss(value_pred, target_values)
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    
    # Get gradients and compute norm
    grad_norm = 0.0
    for param in critic.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    # Compute perturbation
    scale = rho / (grad_norm + 1e-12)
    perturbed_params = []
    for param in critic.parameters():
        if param.grad is None:
            continue
        e_w = param.grad * scale
        perturbed_params.append(e_w)
        param.data.add_(e_w)
    
    # Compute loss and gradients at perturbed point
    critic.zero_grad()
    value_pred = critic(data["obs"], data["risk"])
    perturbed_loss = nn.functional.mse_loss(value_pred, target_values)
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = {}
    for name, param in critic.named_parameters():
        if param.grad is not None:
            sam_grads[name] = param.grad.clone()
    
    # Restore original parameters
    for param, orig_param in zip(critic.parameters(), original_params):
        if param.requires_grad:
            param.data.copy_(orig_param)
    
    return sam_grads, perturbed_params, None, None, None




def compute_kl_constrained_perturbation(policy, data, step_direction, target_kl, max_search_steps=10):
    """Compute a perturbation that satisfies the KL constraint using line search.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        step_direction: The direction to perturb parameters
        target_kl: Target KL divergence constraint
        max_search_steps: Maximum number of line search steps
        
    Returns:
        step_frac: The step fraction that satisfies KL constraint
        final_kl: The final KL divergence achieved
        accepted_step: Whether a suitable step was found
    """
    theta_old = get_flat_params_from(policy.actor)
    
    # Store old distribution for KL computation
    with torch.no_grad():
        old_distribution = policy.actor(data["obs"], data["risk"])
    
    step_frac = 1.0
    final_kl = 0.0
    accepted_step = False
    
    for step in range(max_search_steps):
        # Compute perturbed parameters
        theta_perturbed = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_perturbed)
        
        # Compute KL divergence between old and perturbed policy
        with torch.no_grad():
            current_distribution = policy.actor(data["obs"], data["risk"])
            kl = torch.distributions.kl.kl_divergence(
                old_distribution, current_distribution
            ).mean().item()
        
        if kl <= target_kl:
            final_kl = kl
            accepted_step = True
            break
        else:
            step_frac *= 0.8
    else:
        # If no suitable step found, use a very small perturbation
        step_frac = 0.1
        theta_perturbed = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_perturbed)
        
        with torch.no_grad():
            current_distribution = policy.actor(data["obs"], data["risk"])
            final_kl = torch.distributions.kl.kl_divergence(
                old_distribution, current_distribution
            ).mean().item()
        accepted_step = True
    
    return step_frac, final_kl, accepted_step


def compute_natural_gradient_direction(fvp, policy, data, grads, target_kl, conjugate_gradient_iters=10):
    """Compute the natural gradient direction using conjugate gradients.
    
    Args:
        fvp: Fisher vector product function
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        grads: The policy gradients
        target_kl: Target KL divergence for step size computation
        
    Returns:
        step_direction: The natural gradient step direction
        x: The conjugate gradient solution
        xHx: The quadratic form x^T H x
    """
    x = conjugate_gradients(fvp, grads, conjugate_gradient_iters)
    assert torch.isfinite(x).all(), "x is not finite"
    xHx = torch.dot(x, fvp(x))
    assert xHx.item() >= 0, "xHx is negative"
    
    # Initial step size based on TRPO
    alpha = torch.sqrt(2 * target_kl / (xHx + 1e-8))
    step_direction = x * alpha
    
    return step_direction, x, xHx


def log_gradient_statistics(grads, step_direction, x):
    """Log gradient statistics for debugging and monitoring.
    
    Args:
        grads: The original policy gradients
        step_direction: The natural gradient step direction
        x: The conjugate gradient solution
        logger: Logger instance for logging
    """
    grad_norm = torch.norm(grads)
    
    # Compute cosine similarity between natural gradient and original gradient
    cos_sim = torch.nn.functional.cosine_similarity(x.view(1,-1), grads.view(1,-1))
    
    # Compute effective rho as the norm of the step direction
    effective_rho = torch.norm(step_direction).item()

    # Calculate scale by projecting step_direction onto original gradient direction
    grad_direction = grads / (grad_norm + 1e-12)  # Normalize gradient
    scale_along_grad = torch.dot(step_direction, grad_direction)

    return cos_sim, effective_rho, scale_along_grad


def compute_sam_gradients_v2_kl(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """Compute Sharpness Aware Minimization gradients with KL constraint.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        target_kl: Target KL divergence constraint
        max_search_steps: Maximum number of line search steps
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """
    print(data.keys())

    theta_old = get_flat_params_from(policy.actor)
    
    # First compute the base loss and gradients with respect to cost.
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = (ratio * advantage_cost).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    
    # Compute natural gradient direction
    step_direction, x, xHx = compute_natural_gradient_direction(fvp, policy, data, grads, target_kl, max_search_steps)

    # Log gradient statistics
    cos_sim, effective_rho, scale_along_grad = log_gradient_statistics(grads, step_direction, x)

    # Find KL-constrained perturbation in the direction of increasing cost.
    step_frac, final_kl, accepted_step = compute_kl_constrained_perturbation(
        policy, data, step_direction, target_kl, max_search_steps
    )
    
    # Store the final perturbation
    perturbed_params = step_frac * step_direction
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    set_param_values_to_model(policy.actor, theta_old)
    
    return sam_grads, perturbed_params, cos_sim, effective_rho, scale_along_grad

def compute_sam_gradients_v1_kl(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """Compute Sharpness Aware Minimization gradients with KL constraint.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        target_kl: Target KL divergence constraint
        max_search_steps: Maximum number of line search steps
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """

    theta_old = get_flat_params_from(policy.actor)
    
    # First compute the base loss and gradients with respect to cost.
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = -(ratio * advantage_lag).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    
    # Compute natural gradient direction
    step_direction, x, xHx = compute_natural_gradient_direction(fvp, policy, data, grads, target_kl, max_search_steps)

    # Log gradient statistics
    cos_sim, effective_rho, scale_along_grad = log_gradient_statistics(grads, step_direction, x)

    # Find KL-constrained perturbation in the direction of increasing cost.
    step_frac, final_kl, accepted_step = compute_kl_constrained_perturbation(
        policy, data, step_direction, target_kl, max_search_steps
    )
    
    # Store the final perturbation
    perturbed_params = step_frac * step_direction
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    set_param_values_to_model(policy.actor, theta_old)
    
    return sam_grads, perturbed_params, cos_sim, effective_rho, scale_along_grad

def compute_sam_gradients_v3_kl(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """Compute Sharpness Aware Minimization gradients with KL constraint.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        target_kl: Target KL divergence constraint
        max_search_steps: Maximum number of line search steps
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """

    theta_old = get_flat_params_from(policy.actor)
    
    # First compute the base loss and gradients with respect to cost.
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = -(ratio * advantage_reward).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    
    # Compute natural gradient direction
    step_direction, x, xHx = compute_natural_gradient_direction(fvp, policy, data, grads, target_kl, max_search_steps)

    # Log gradient statistics
    cos_sim, effective_rho, scale_along_grad = log_gradient_statistics(grads, step_direction, x)

    # Find KL-constrained perturbation in the direction of increasing cost.
    step_frac, final_kl, accepted_step = compute_kl_constrained_perturbation(
        policy, data, step_direction, target_kl, max_search_steps
    )
    
    # Store the final perturbation
    perturbed_params = step_frac * step_direction
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    set_param_values_to_model(policy.actor, theta_old)
    
    return sam_grads, perturbed_params, cos_sim, effective_rho, scale_along_grad


def compute_sam_gradients_critic_v4(critic, data, target_values, rho, num_samples=10):
    """
    SAM v4: Samples multiple perturbations uniformly from hypersphere and averages gradients
    
    Args:
        critic: Critic network
        data: Dictionary containing obs and risk
        target_values: Target values for critic
        rho: Size of perturbation
        num_samples: Number of perturbation samples
    """
    # Store original parameters
    original_params = []
    for param in critic.parameters():
        if param.requires_grad:
            original_params.append(param.data.clone())
    
    # Get base gradients and norm
    critic.zero_grad()
    value_pred = critic(data["obs"], data["risk"]) 
    base_loss = nn.functional.mse_loss(value_pred, target_values)
    base_loss.backward()
    
    # Calculate gradient norm
    grad_norm = 0.0
    for param in critic.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    # Store gradients from all perturbations
    all_grads = {}
    for name, param in critic.named_parameters():
        if param.grad is not None:
            all_grads[name] = []
    
    # Sample perturbations and compute gradients
    for _ in range(num_samples):
        # Sample random direction on unit sphere
        random_directions = {}
        for name, param in critic.named_parameters():
            if param.grad is not None:
                random_direction = torch.randn_like(param)
                random_direction = random_direction / random_direction.norm()
                random_directions[name] = random_direction
        
        # Apply perturbation
        for name, param in critic.named_parameters():
            if param.grad is not None:
                e_w = rho * random_directions[name]
                param.data.add_(e_w)
        
        # Get gradients at perturbed point
        critic.zero_grad()
        value_pred = critic(data["obs"], data["risk"])
        perturbed_loss = nn.functional.mse_loss(value_pred, target_values)
        perturbed_loss.backward()
        
        # Store gradients
        for name, param in critic.named_parameters():
            if param.grad is not None:
                all_grads[name].append(param.grad.clone())
        
        # Restore original parameters
        for param, orig_param in zip(critic.parameters(), original_params):
            if param.requires_grad:
                param.data.copy_(orig_param)
    
    # Average gradients across perturbations
    sam_grads = {}
    for name in all_grads:
        sam_grads[name] = torch.stack(all_grads[name]).mean(0)
    
    return sam_grads, None  # Return None for perturbed_params to match original interface


def compute_sam_gradients_v4(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, rho=0.05, target_kl=0.01, max_search_steps=10, num_samples=10):
    """
    SAM v4 for policy: Samples multiple perturbations uniformly from hypersphere and averages gradients
    
    Args:
        policy: Policy network
        data: Dictionary containing obs, risk, act, log_prob
        advantage_lag: Advantage values
        rho: Size of perturbation
        num_samples: Number of perturbation samples
    """
    # Store original parameters
    original_params = []
    for param in policy.actor.parameters():
        if param.requires_grad:
            original_params.append(param.data.clone())
    
    # Get base gradients and norm
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"], data["risk"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = -(ratio * advantage_lag).mean()
    base_loss.backward()
    
    # Calculate gradient norm
    grad_norm = 0.0
    for param in policy.actor.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    # Store gradients from all perturbations
    all_grads = []
    
    # Sample perturbations and compute gradients
    for _ in range(num_samples):
        # Sample random direction on unit sphere
        random_directions = {}
        for name, param in policy.actor.named_parameters():
            if param.grad is not None:
                random_direction = torch.randn_like(param)
                random_direction = random_direction / random_direction.norm()
                random_directions[name] = random_direction
        
        # Apply perturbation
        for name, param in policy.actor.named_parameters():
            if param.grad is not None:
                e_w = rho * random_directions[name]
                param.data.add_(e_w)
        
        # Get gradients at perturbed point
        policy.actor.zero_grad()
        temp_distribution = policy.actor(data["obs"], data["risk"])
        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        perturbed_loss = -(ratio * advantage_lag).mean()
        perturbed_loss.backward()
        
        # Store flat gradients
        flat_grads = get_flat_gradients_from(policy.actor)
        all_grads.append(flat_grads)
        
        # Restore original parameters
        for param, orig_param in zip(policy.actor.parameters(), original_params):
            if param.requires_grad:
                param.data.copy_(orig_param)
    
    # Average gradients across perturbations
    sam_grads = torch.stack(all_grads).mean(0)
    
    return sam_grads, None, None, None, None
