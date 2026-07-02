import os

import matplotlib.pyplot as plt
import numpy as np


def smooth_curve(scalars, weight=0.85):
    """Apply TensorBoard-style EMA smoothing to a 1D scalar sequence."""
    if len(scalars) == 0:
        return []

    smoothed = []
    last = scalars[0]
    for val in scalars:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_series_with_single_point_support(
    ax,
    x_values,
    y_values,
    color,
    label,
    linewidth=2,
):
    """Plot a line series and make single-point runs visible."""
    x_values = list(x_values)
    y_values = list(y_values)

    if len(x_values) == 0 or len(y_values) == 0:
        return

    if len(x_values) == 1:
        ax.plot(
            x_values,
            y_values,
            color=color,
            marker="o",
            markersize=8,
            linewidth=linewidth,
            label=label,
        )
        ax.annotate(
            f"{y_values[0]:.3f}",
            (x_values[0], y_values[0]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color=color,
        )
        ax.set_xlim(x_values[0] - 0.5, x_values[0] + 0.5)
        return

    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=4,
        label=label,
    )


def plot_queue_delay(history_Q_loc, history_Q_tx, save_dir):
    """Plot system queue backlog over micro time steps."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    steps = range(len(history_Q_loc))

    plt.fill_between(
        steps,
        0,
        history_Q_loc,
        color="#4A90E2",
        alpha=0.7,
        label="Local Comp. Queue (Q_loc)",
    )
    plt.fill_between(
        steps,
        history_Q_loc,
        np.array(history_Q_loc) + np.array(history_Q_tx),
        color="#F5A623",
        alpha=0.8,
        label="Transmission Queue (Q_tx)",
    )

    plt.plot(
        steps,
        np.array(history_Q_loc) + np.array(history_Q_tx),
        color="#D0021B",
        linewidth=1.5,
        label="Total Backlog",
    )

    plt.title("System Queue Backlog (Delay Equivalent) over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Micro Time Steps", fontsize=12)
    plt.ylabel("Queue Accumulation (Mbits)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "queue_delay.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Queue-delay plot saved to: {save_path}")


def plot_energy_breakdown(sum_E_comp, sum_E_trans, sum_E_flying, save_dir):
    """Plot the breakdown of total energy consumption."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))

    categories = ["Total Energy Consumption"]
    E_c = [sum_E_comp]
    E_t = [sum_E_trans]
    E_f = [sum_E_flying]
    bar_width = 0.5

    plt.bar(
        categories,
        E_f,
        bar_width,
        color="#D0021B",
        alpha=0.8,
        edgecolor="black",
        label=f"Flying ({sum_E_flying:.1f} J)",
    )
    plt.bar(
        categories,
        E_c,
        bar_width,
        bottom=E_f,
        color="#4A90E2",
        alpha=0.8,
        edgecolor="black",
        label=f"Computing ({sum_E_comp:.1f} J)",
    )
    plt.bar(
        categories,
        E_t,
        bar_width,
        bottom=np.array(E_f) + np.array(E_c),
        color="#50E3C2",
        alpha=0.9,
        edgecolor="black",
        label=f"Transmission ({sum_E_trans:.1f} J)",
    )

    total_max = sum_E_comp + sum_E_trans + sum_E_flying
    plt.ylim(0, total_max * 1.3)

    plt.text(
        0,
        E_f[0] / 2,
        f"{sum_E_flying:.1f} J",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    if E_c[0] > total_max * 0.05:
        plt.text(
            0,
            E_f[0] + E_c[0] / 2,
            f"{sum_E_comp:.1f} J",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    plt.title("UAV-MEC System: Energy Profile Breakdown", fontsize=14, fontweight="bold")
    plt.ylabel("Energy (Joules)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "energy_breakdown.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Energy-breakdown plot saved to: {save_path}")


def plot_learning_curve(macro_rewards, micro_rewards, save_dir):
    """Plot PPO macro reward and SAC micro reward across episodes."""
    os.makedirs(save_dir, exist_ok=True)

    episodes = range(len(macro_rewards))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = "tab:red"
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel("PPO Macro Reward", color=color, fontsize=12)
    macro_rewards_smooth = smooth_curve(macro_rewards)
    ax1.plot(episodes, macro_rewards_smooth, color=color, alpha=1.0, linewidth=2, label="Macro Reward")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("SAC Micro Reward", color=color, fontsize=12)
    micro_rewards_smooth = smooth_curve(micro_rewards)
    ax2.plot(episodes, micro_rewards_smooth, color=color, alpha=1.0, linewidth=2, label="Micro Reward")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("HRL Dual-Algorithm Learning Curve", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "learning_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close("all")
    print(f"Learning-curve plot saved to: {save_path}")


def plot_system_metrics(queue_history, energy_history, save_dir):
    """Plot queue backlog and total energy trends over training."""
    os.makedirs(save_dir, exist_ok=True)

    episodes = np.arange(1, len(queue_history) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    plot_series_with_single_point_support(
        ax1,
        episodes,
        queue_history,
        color="#F5A623",
        label="Avg Queue Backlog",
    )
    ax1.set_title("System Queue Delay Trend over Episodes", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Queue Backlog (Mbits)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    plot_series_with_single_point_support(
        ax2,
        episodes,
        energy_history,
        color="#50E3C2",
        label="Total Energy",
    )
    ax2.set_title("System Energy Consumption Trend over Episodes", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episodes", fontsize=11)
    ax2.set_ylabel("Energy (Joules)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "system_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"System-metrics plot saved to: {save_path}")


def plot_evaluation_metrics(
    eval_steps,
    eval_scores,
    eval_q_high,
    eval_q_low,
    eval_energy,
    eval_u2g,
    eval_cpu_util,
    eval_delay_high,
    eval_delay_low,
    eval_delay_mean_ms=None,
    eval_delay_weighted_ms=None,
    eval_delay_max_ms=None,
    save_dir=None,
):
    """Plot deterministic fixed-seed evaluation metrics over training."""
    if len(eval_steps) == 0:
        return
    if save_dir is None:
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()

    weighted_delay_series = (
        eval_delay_weighted_ms if eval_delay_weighted_ms is not None else eval_delay_low
    )
    max_delay_series = (
        eval_delay_max_ms if eval_delay_max_ms is not None else eval_delay_high
    )
    items = [
        ("Eval Score", eval_scores, "#D0021B"),
        ("High-Priority Queue Delay (ms)", eval_delay_high, "#D0021B"),
        ("Low-Priority Queue Delay (ms)", eval_delay_low, "#F5A623"),
        ("Weighted Queue Delay (ms)", weighted_delay_series, "#7B61FF"),
        ("Max UAV Queue Delay (ms)", max_delay_series, "#C0392B"),
        ("Avg Energy / Step", eval_energy, "#50E3C2"),
        ("Avg U2G Distance", eval_u2g, "#4A90E2"),
        ("Avg CPU Utilization", eval_cpu_util, "#FF7A00"),
    ]

    for ax, (title, values, color) in zip(axes, items):
        smooth_values = smooth_curve(values)
        plot_series_with_single_point_support(
            ax,
            eval_steps,
            smooth_values,
            color=color,
            label=title,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episodes", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.suptitle("Deterministic Fixed-Seed Evaluation Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(save_dir, "evaluation_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close("all")
    print(f"Evaluation metrics plot saved to: {save_path}")


def plot_sac_loss(actor_losses, critic_losses, save_dir):
    """Plot SAC actor and critic losses."""
    if len(actor_losses) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    steps = range(len(actor_losses))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = "tab:purple"
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Actor Loss", color=color, fontsize=12)
    actor_losses_smooth = smooth_curve(actor_losses)
    ax1.plot(steps, actor_losses_smooth, color=color, alpha=1.0, linewidth=2, label="Actor Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Critic Loss", color=color, fontsize=12)
    critic_losses_smooth = smooth_curve(critic_losses)
    ax2.plot(steps, critic_losses_smooth, color=color, alpha=1.0, linewidth=2, label="Critic Loss")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("SAC Micro-Layer Loss Convergence", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "sac_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close("all")
    print(f"SAC-loss plot saved to: {save_path}")


def plot_macro_diagnostics(
    macro_steps,
    move_distance,
    action_norm,
    hover_ratio,
    projected_u2g_distance,
    projected_u2u_rate,
    avg_upload_delay_penalty,
    position_cost,
    position_improvement,
    macro_reward,
    save_path,
):
    """Plot per-macro-step motion and utility diagnostics for one episode."""
    if len(macro_steps) == 0:
        return

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    panels = [
        ("Move Distance (m)", move_distance, "#4A90E2"),
        ("Projected Mean U2G Distance (m)", projected_u2g_distance, "#7B61FF"),
        ("Projected Mean U2U Rate (Mbps)", projected_u2u_rate, "#50E3C2"),
        ("Avg Upload Delay Penalty", avg_upload_delay_penalty, "#D0021B"),
        ("Position Cost", position_cost, "#F5A623"),
        ("Position Improvement", position_improvement, "#00A699"),
        ("Macro Reward", macro_reward, "#FF7A00"),
    ]

    axes[0].plot(
        macro_steps,
        move_distance,
        color="#4A90E2",
        marker="o",
        linewidth=2,
        label="Move Distance",
    )
    axes[0].set_title("Move Distance (m)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Macro Step", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(
        macro_steps,
        action_norm,
        color="#9B59B6",
        marker="o",
        linewidth=2,
        label="Action Norm",
    )
    axes[1].plot(
        macro_steps,
        hover_ratio,
        color="#34495E",
        marker="s",
        linewidth=2,
        label="Hover Ratio",
    )
    axes[1].set_title("Action Norm / Hover Ratio", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Macro Step", fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    for ax, (title, values, color) in zip(axes[2:], panels[1:]):
        ax.plot(
            macro_steps,
            values,
            color=color,
            marker="o",
            linewidth=2,
            label=title,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Macro Step", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.suptitle(
        "Per-Macro-Step UAV Motion Diagnostics",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close("all")
    print(f"Macro-diagnostics plot saved to: {save_path}")
