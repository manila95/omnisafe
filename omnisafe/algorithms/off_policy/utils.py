import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

def estimate_true_value(actor_critic,
                    adapter,
                    logger,
                    discount: float,
                    eval_episodes=100,
                    step=0):
        """Estimates true Q-value via launching given policy from sampled state until
        the end of an episode. """

        eval_env = adapter._eval_env
        
        true_cvalues, true_rvalues = [], []
        estimate_rvalues, estimate_cvalues = [], []
        for _ in range(eval_episodes):
            obs0, _ = eval_env.reset()

            _, estimate_rvalue, estimate_cvalue, _ = actor_critic.step(obs0, deterministic=False)

            obs = obs0

            true_cvalue = 0.0
            true_rvalue = 0.0
            t = 0
            while True:
                act, _, _, _ = actor_critic.step(obs, deterministic=False)
                next_obs, r, c, termniated, truncated, info = eval_env.step(act)
                true_cvalue += c * (discount ** t)
                true_rvalue += r * (discount ** t)
                t += 1
                obs = next_obs

                if termniated.squeeze().item() or truncated.squeeze().item():
                    break
            true_cvalues.append(true_cvalue)
            true_rvalues.append(true_rvalue)
            estimate_cvalues.append(estimate_cvalue)
            estimate_rvalues.append(estimate_rvalue)
            # print("Estimation took: ", step)

        c_error = torch.mean(torch.stack(true_cvalues) - torch.stack(estimate_cvalues))
        r_error = torch.mean(torch.stack(true_rvalues) - torch.stack(estimate_rvalues))

        true_c = torch.mean(torch.stack(true_cvalues))
        true_r = torch.mean(torch.stack(true_rvalues))
        estimate_c = torch.mean(torch.stack(estimate_cvalues))
        estimate_r = torch.mean(torch.stack(estimate_rvalues))
        # --- Scatter plot logging to wandb ---
        true_c_vals = torch.stack(true_cvalues).detach().cpu().numpy()
        estimate_c_vals = torch.stack(estimate_cvalues).detach().cpu().numpy()
        true_r_vals = torch.stack(true_rvalues).detach().cpu().numpy()
        estimate_r_vals = torch.stack(estimate_rvalues).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # C-values scatter
        axes[0].scatter(true_c_vals, estimate_c_vals, alpha=0.5, s=10, color="steelblue")
        min_c, max_c = min(true_c_vals.min(), estimate_c_vals.min()), max(true_c_vals.max(), estimate_c_vals.max())
        axes[0].plot([min_c, max_c], [min_c, max_c], "r--", linewidth=1, label="ideal")
        axes[0].set_xlabel("True C")
        axes[0].set_ylabel("Estimated C")
        axes[0].set_title("C-Values: True vs Estimated")
        axes[0].legend()

        # R-values scatter
        axes[1].scatter(true_r_vals, estimate_r_vals, alpha=0.5, s=10, color="darkorange")
        min_r, max_r = min(true_r_vals.min(), estimate_r_vals.min()), max(true_r_vals.max(), estimate_r_vals.max())
        axes[1].plot([min_r, max_r], [min_r, max_r], "r--", linewidth=1, label="ideal")
        axes[1].set_xlabel("True R")
        axes[1].set_ylabel("Estimated R")
        axes[1].set_title("R-Values: True vs Estimated")
        axes[1].legend()

        plt.tight_layout()
        if logger._use_wandb:
            wandb.log({"scatter/c_and_r_values": wandb.Image(fig)})
        plt.close(fig)
        return c_error, true_c, estimate_c, r_error, true_r, estimate_r


def estimate_true_qvalue(actor_critic,
                    adapter,
                    logger,
                    discount: float,
                    eval_episodes=100,
                    step=0):
        """Estimates true Q-value via launching given policy from sampled state until
        the end of an episode. For off-policy algorithms that use Q-functions."""

        eval_env = adapter._eval_env

        true_cvalues, true_rvalues = [], []
        estimate_rvalues, estimate_cvalues = [], []
        for _ in range(eval_episodes):
            obs0, _ = eval_env.reset()

            with torch.no_grad():
                init_act = actor_critic.step(obs0, deterministic=False)
                estimate_rvalue = torch.min(torch.stack(actor_critic.reward_critic(obs0, init_act)), dim=0).values.squeeze(-1)
                estimate_cvalue = actor_critic.cost_critic(obs0, init_act)[0].squeeze(-1)

            obs = obs0

            true_cvalue = 0.0
            true_rvalue = 0.0
            t = 0
            while True:
                with torch.no_grad():
                    act = actor_critic.step(obs, deterministic=False)
                next_obs, r, c, termniated, truncated, info = eval_env.step(act)
                true_cvalue += c * (discount ** t)
                true_rvalue += r * (discount ** t)
                t += 1
                obs = next_obs

                if termniated.squeeze().item() or truncated.squeeze().item():
                    break
            true_cvalues.append(true_cvalue)
            true_rvalues.append(true_rvalue)
            estimate_cvalues.append(estimate_cvalue)
            estimate_rvalues.append(estimate_rvalue)

        c_error = torch.mean(torch.stack(true_cvalues) - torch.stack(estimate_cvalues))
        r_error = torch.mean(torch.stack(true_rvalues) - torch.stack(estimate_rvalues))

        true_c = torch.mean(torch.stack(true_cvalues))
        true_r = torch.mean(torch.stack(true_rvalues))
        estimate_c = torch.mean(torch.stack(estimate_cvalues))
        estimate_r = torch.mean(torch.stack(estimate_rvalues))
        # --- Scatter plot logging to wandb ---
        true_c_vals = torch.stack(true_cvalues).detach().cpu().numpy()
        estimate_c_vals = torch.stack(estimate_cvalues).detach().cpu().numpy()
        true_r_vals = torch.stack(true_rvalues).detach().cpu().numpy()
        estimate_r_vals = torch.stack(estimate_rvalues).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # C-values scatter
        axes[0].scatter(true_c_vals, estimate_c_vals, alpha=0.5, s=10, color="steelblue")
        min_c, max_c = min(true_c_vals.min(), estimate_c_vals.min()), max(true_c_vals.max(), estimate_c_vals.max())
        axes[0].plot([min_c, max_c], [min_c, max_c], "r--", linewidth=1, label="ideal")
        axes[0].set_xlabel("True C")
        axes[0].set_ylabel("Estimated C")
        axes[0].set_title("C-Values: True vs Estimated")
        axes[0].legend()

        # R-values scatter
        axes[1].scatter(true_r_vals, estimate_r_vals, alpha=0.5, s=10, color="darkorange")
        min_r, max_r = min(true_r_vals.min(), estimate_r_vals.min()), max(true_r_vals.max(), estimate_r_vals.max())
        axes[1].plot([min_r, max_r], [min_r, max_r], "r--", linewidth=1, label="ideal")
        axes[1].set_xlabel("True R")
        axes[1].set_ylabel("Estimated R")
        axes[1].set_title("R-Values: True vs Estimated")
        axes[1].legend()

        plt.tight_layout()
        if logger._use_wandb:
            wandb.log({"scatter/c_and_r_values": wandb.Image(fig)}, step=step)
        plt.close(fig)
        return c_error, true_c, estimate_c, r_error, true_r, estimate_r


class RandomProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomProjection, self).__init__()
        self.linear_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_projection(x)