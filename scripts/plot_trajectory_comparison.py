"""
论文轨迹对比图: 展示训练前后 UAV 站位变化
- 左图: 初始策略 (UAV 在锚点不动/随机移动)
- 右图: 收敛策略 (UAV 学到的最优站位)
- 用箭头标注 UAV 从锚点到最终位置的偏移方向

用法:
    python scripts/plot_trajectory_comparison.py --ckpt results/paper_batch/seed1/hrl/checkpoints/final.pt
    python scripts/plot_trajectory_comparison.py --results-dir results/central_zone125_55_35_cpu8_load09_11_p300
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not found")
    sys.exit(1)

# 设置中文字体
import matplotlib.font_manager as fm
_cn_font_set = False
for _fname in ["SimHei", "Microsoft YaHei", "STSong", "FangSong"]:
    if any(_fname.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_fname, "DejaVu Sans"]
        _cn_font_set = True
        break
if not _cn_font_set:
    # Windows 下直接用系统字体路径
    _sys_font = "C:/Windows/Fonts/simhei.ttf"
    if os.path.isfile(_sys_font):
        fm.fontManager.addfont(_sys_font)
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run_episode_and_get_positions(env, macro_policy, micro_policy, seed=1000, macro_steps=30, micro_steps=100):
    """Run one episode and record UAV positions at each macro step."""
    env.env.record_trajectory = True
    env.reset(seed=seed)

    positions_over_time = [env.env.uav_coords.copy()]
    local_obs, _ = env.get_macro_state()

    for _ in range(macro_steps):
        local_obs_batch = np.vstack(local_obs).astype(np.float32)
        macro_action = macro_policy(env, local_obs_batch)
        _, _, _ = env.macro_step(macro_action)

        for _ in range(micro_steps):
            if env.env.macro_fly_steps_remaining > 0:
                micro_action = np.zeros(2 * env.N * env.N, dtype=np.float32)
            else:
                micro_action = micro_policy(env)
            env.micro_step(micro_action)

        local_obs, _ = env.get_macro_state()
        _ = env.get_macro_reward()
        positions_over_time.append(env.env.uav_coords.copy())

    return {
        "positions": np.array(positions_over_time),
        "final_pos": env.env.uav_coords.copy(),
        "centroids": env.env.weighted_centroids.copy(),
        "anchors": env.env.anchor_points.copy(),
        "user_coords": env.env.user_coords.copy(),
        "user_allocations": env.env.user_allocations.copy(),
    }


def plot_single_panel(ax, data, title, show_trajectory=True):
    """Plot UAV positions, users, centroids on one axis."""
    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    N = data["anchors"].shape[0]

    # Draw users as small dots
    for i in range(N):
        mask = data["user_allocations"] == i
        ax.scatter(
            data["user_coords"][mask, 0],
            data["user_coords"][mask, 1],
            color=colors[i], alpha=0.3, s=15, marker="o",
            label=f"用户 (区域{i+1})" if i == 0 else None,
        )

    # Draw zone anchors
    for i in range(N):
        ax.scatter(
            data["anchors"][i, 0], data["anchors"][i, 1],
            color=colors[i], marker="x", s=100, linewidths=2, zorder=10,
        )

    # Draw weighted centroids
    for i in range(N):
        ax.scatter(
            data["centroids"][i, 0], data["centroids"][i, 1],
            color=colors[i], marker="D", s=60, edgecolors="black",
            linewidths=0.8, zorder=10,
            label=f"流量质心" if i == 0 else None,
        )

    # Draw UAV trajectory or just final position
    if show_trajectory and data["positions"].shape[0] > 1:
        for i in range(N):
            traj = data["positions"][:, i, :]
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=colors[i], linewidth=2.0, alpha=0.7,
                marker=".", markersize=3,
            )
            # Start marker
            ax.scatter(
                traj[0, 0], traj[0, 1],
                color=colors[i], marker="s", s=80, edgecolors="black",
                linewidths=1, zorder=15,
            )
            # End marker
            ax.scatter(
                traj[-1, 0], traj[-1, 1],
                color=colors[i], marker="*", s=200, edgecolors="black",
                linewidths=1, zorder=15,
                label=f"UAV-{i+1} 终点",
            )
    else:
        for i in range(N):
            ax.scatter(
                data["final_pos"][i, 0], data["final_pos"][i, 1],
                color=colors[i], marker="*", s=200, edgecolors="black",
                linewidths=1, zorder=15,
                label=f"UAV-{i+1}",
            )

    # Draw arrows from anchor to final position (shows learned offset)
    for i in range(N):
        dx = data["final_pos"][i, 0] - data["anchors"][i, 0]
        dy = data["final_pos"][i, 1] - data["anchors"][i, 1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 5:  # Only show arrow if meaningful movement
            ax.annotate(
                "", xy=data["final_pos"][i], xytext=data["anchors"][i],
                arrowprops=dict(
                    arrowstyle="->", color=colors[i],
                    lw=2.0, connectionstyle="arc3,rad=0.1"
                ),
                zorder=12,
            )

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_comparison(data_initial, data_trained, output_path):
    """Side-by-side comparison of initial vs trained trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    plot_single_panel(ax1, data_initial, "初始策略 (UAV 不动)", show_trajectory=False)
    plot_single_panel(ax2, data_trained, "收敛策略 (300 episodes)", show_trajectory=True)

    # Add legend
    ax2.legend(loc="upper right", fontsize=9)

    # Add annotation about the trade-off
    fig.text(
        0.5, 0.02,
        "UAV 最优站位 = 平衡上行时延(靠近用户) 与 U2U转发时延(靠近过载队友)",
        ha="center", fontsize=11, style="italic", color="#555555",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_from_npz(npz_path, output_path):
    """Plot from existing trajectory snapshot .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    uav_traj = data["uav_traj"]  # shape: (T, N, 2)
    centroid_traj = data["centroid_traj"]
    users_traj = data["users_traj"]
    user_alloc = data["user_allocations"]
    anchors = data["anchor_points"]

    N = uav_traj.shape[1]
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    # Users final positions
    for i in range(N):
        mask = user_alloc == i
        ax.scatter(
            users_traj[-1, mask, 0], users_traj[-1, mask, 1],
            color=colors[i], alpha=0.35, s=20, marker="o",
        )

    # UAV trajectories
    for i in range(N):
        ax.plot(
            uav_traj[:, i, 0], uav_traj[:, i, 1],
            color=colors[i], linewidth=2.5, alpha=0.8,
            label=f"UAV-{i+1}",
        )
        ax.scatter(uav_traj[0, i, 0], uav_traj[0, i, 1],
                   color=colors[i], marker="s", s=100, edgecolors="black", zorder=15)
        ax.scatter(uav_traj[-1, i, 0], uav_traj[-1, i, 1],
                   color=colors[i], marker="*", s=250, edgecolors="black", zorder=15)

    # Centroids
    for i in range(N):
        ax.scatter(centroid_traj[-1, i, 0], centroid_traj[-1, i, 1],
                   color=colors[i], marker="D", s=80, edgecolors="black", zorder=10)

    # Anchor points
    for i in range(N):
        ax.scatter(anchors[i, 0], anchors[i, 1],
                   color=colors[i], marker="x", s=120, linewidths=2.5, zorder=10)

    # Draw arrow from anchor to final position to highlight learned offset
    for i in range(N):
        dx = uav_traj[-1, i, 0] - anchors[i, 0]
        dy = uav_traj[-1, i, 1] - anchors[i, 1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 8:
            ax.annotate(
                "", xy=uav_traj[-1, i], xytext=anchors[i],
                arrowprops=dict(
                    arrowstyle="-|>", color=colors[i],
                    lw=2.5, mutation_scale=15,
                ),
                zorder=12,
            )

    # Annotations
    ax.scatter([], [], marker="s", color="gray", edgecolors="black", s=80, label="起点 (锚点)")
    ax.scatter([], [], marker="*", color="gray", edgecolors="black", s=180, label="终点 (学到的站位)")
    ax.scatter([], [], marker="D", color="gray", edgecolors="black", s=60, label="流量质心")
    ax.scatter([], [], marker="x", color="gray", s=100, linewidths=2, label="区域锚点")

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("UAV 轨迹优化结果 (zone=12.5/5.5/3.5 Mbps, CPU=8GHz)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory comparison for paper")
    parser.add_argument("--npz", type=str, default="",
                        help="Path to trajectory .npz file (simple single-panel plot)")
    parser.add_argument("--results-dir", type=str, default="",
                        help="Results directory with trajectory_snapshots/")
    parser.add_argument("--output", type=str, default="",
                        help="Output image path")
    args = parser.parse_args()

    if args.npz:
        output = args.output or os.path.join(
            PROJECT_ROOT, "results", "paper_figures", "trajectory_trained.png"
        )
        plot_from_npz(args.npz, output)
        return

    if args.results_dir:
        # Find the last trajectory npz
        traj_dir = os.path.join(args.results_dir, "trajectory_snapshots")
        npz_files = sorted([
            f for f in os.listdir(traj_dir) if f.endswith(".npz")
        ])
        if npz_files:
            last_npz = os.path.join(traj_dir, npz_files[-1])
            output = args.output or os.path.join(
                PROJECT_ROOT, "results", "paper_figures", "trajectory_trained.png"
            )
            plot_from_npz(last_npz, output)
        else:
            print(f"No .npz files found in {traj_dir}")
        return

    # Default: use existing results
    default_dir = os.path.join(
        PROJECT_ROOT, "results", "central_zone125_55_35_cpu8_load09_11_p300"
    )
    if os.path.isdir(default_dir):
        traj_dir = os.path.join(default_dir, "trajectory_snapshots")
        npz_files = sorted([
            f for f in os.listdir(traj_dir) if f.endswith(".npz")
        ])
        if npz_files:
            last_npz = os.path.join(traj_dir, npz_files[-1])
            output = args.output or os.path.join(
                PROJECT_ROOT, "results", "paper_figures", "trajectory_trained_cn.png"
            )
            plot_from_npz(last_npz, output)
            return

    print("No trajectory data found. Run training with --trajectory_plot_interval > 0 first.")


if __name__ == "__main__":
    main()
