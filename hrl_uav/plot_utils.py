import os

import matplotlib.pyplot as plt


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
