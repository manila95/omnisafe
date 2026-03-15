from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def estimate_true_value(
    actor_critic,
    adapter,
    logger,
    discount: float,
    eval_episodes: int = 10,
    step: int | None = None,
):
    """Estimates true Q-value by rolling out the policy from a sampled state until episode end,
    using the same wrapped environment and cost normalization that the critic was trained on.

    Args:
        actor_critic: ConstraintActorQCritic with actor, reward_critic, cost_critic.
        adapter: OffPolicyAdapter whose _eval_env is already wrapped with ObsNormalize,
                 CostLimitNormalize/CostNormalize, ActionScale, Unsqueeze — matching exactly
                 the observation and cost scale the critic was trained on.
        logger: OmniSafe Logger, used to log the scatter figure to tensorboard/wandb.
        discount: Discount factor gamma.
        eval_episodes: Number of episodes to evaluate.
        step: Global training step for logging alignment.
    """
    eval_env = adapter._eval_env
    assert eval_env is not None, 'adapter._eval_env is None — set need_evaluation=True in the env.'

    true_cvalues, true_rvalues = [], []
    estimate_cvalues, estimate_rvalues = [], []

    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()

        with torch.no_grad():
            action = actor_critic.actor.predict(obs, deterministic=True)
            # reward critic has 2 heads (SAC); use min consistent with actor loss
            qr1, qr2 = actor_critic.reward_critic(obs, action)
            estimate_rvalue = torch.min(qr1, qr2)
            estimate_cvalue = actor_critic.cost_critic(obs, action)[0]

        true_cvalue = torch.zeros_like(estimate_cvalue)
        true_rvalue = torch.zeros_like(estimate_rvalue)
        rollout_step = 0

        while True:
            with torch.no_grad():
                action = actor_critic.actor.predict(obs, deterministic=True)

            next_obs, r, c, terminated, truncated, _ = eval_env.step(action)

            true_cvalue += c * (discount ** rollout_step)
            true_rvalue += r * (discount ** rollout_step)
            rollout_step += 1
            obs = next_obs

            if terminated.any() or truncated.any():
                break

        true_cvalues.append(true_cvalue.squeeze().item())
        true_rvalues.append(true_rvalue.squeeze().item())
        estimate_cvalues.append(estimate_cvalue.squeeze().item())
        estimate_rvalues.append(estimate_rvalue.squeeze().item())

    c_error = torch.tensor(true_cvalues) - torch.tensor(estimate_cvalues)
    r_error = torch.tensor(true_rvalues) - torch.tensor(estimate_rvalues)

    true_c = torch.tensor(true_cvalues).mean()
    true_r = torch.tensor(true_rvalues).mean()
    estimate_c = torch.tensor(estimate_cvalues).mean()
    estimate_r = torch.tensor(estimate_rvalues).mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, xs, ys, xlabel, ylabel, title in (
        (axes[0], true_cvalues, estimate_cvalues, 'True C', 'Estimate C', 'Cost Value: True vs Estimate'),
        (axes[1], true_rvalues, estimate_rvalues, 'True R', 'Estimate R', 'Reward Value: True vs Estimate'),
    ):
        all_vals = xs + ys
        lo, hi = min(all_vals), max(all_vals)
        ax.scatter(xs, ys, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='y=x')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    logger.log_figure('Value/Scatter', fig, step=step)
    plt.close(fig)

    return c_error.mean(), true_c, estimate_c, r_error.mean(), true_r, estimate_r
