import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle


def setup_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "results/robust_loadstate_prioritydemand_load08_12_p300/"
            "trajectory_snapshots/trajectory_final.npz"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/report_ready/uav_trajectory_cn.png",
    )
    args = parser.parse_args()

    setup_chinese_font()
    data = np.load(args.input)
    uav_traj = np.asarray(data["uav_traj"], dtype=np.float32)
    centroid_traj = np.asarray(data["centroid_traj"], dtype=np.float32)
    users_traj = np.asarray(data["users_traj"], dtype=np.float32)
    anchors = np.asarray(data["anchor_points"], dtype=np.float32)
    user_alloc = np.asarray(data["user_allocations"], dtype=np.int32)
    zone_radius = float(np.asarray(data["zone_radius"]))

    final_users = users_traj[-1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["区域1（高负载）", "区域2", "区域3"]

    fig, ax = plt.subplots(figsize=(9.6, 8.0), dpi=180)

    for idx, color in enumerate(colors):
        mask = user_alloc == idx
        ax.scatter(
            final_users[mask, 0],
            final_users[mask, 1],
            s=30,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            label=f"{labels[idx]}用户",
            zorder=3,
        )
        ax.add_patch(
            Circle(
                anchors[idx],
                zone_radius,
                fill=False,
                linestyle=":",
                linewidth=1.5,
                color=color,
                alpha=0.55,
            )
        )
        ax.plot(
            centroid_traj[:, idx, 0],
            centroid_traj[:, idx, 1],
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.75,
            label=f"{labels[idx]}负载质心",
        )
        ax.plot(
            uav_traj[:, idx, 0],
            uav_traj[:, idx, 1],
            linewidth=3.0,
            color=color,
            label=f"UAV-{idx + 1}轨迹",
            zorder=4,
        )
        ax.scatter(
            uav_traj[0, idx, 0],
            uav_traj[0, idx, 1],
            marker="s",
            s=70,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.scatter(
            uav_traj[-1, idx, 0],
            uav_traj[-1, idx, 1],
            marker="*",
            s=180,
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=6,
        )
        ax.text(
            uav_traj[-1, idx, 0] + 8,
            uav_traj[-1, idx, 1] + 8,
            f"UAV-{idx + 1}终点",
            color=color,
            fontsize=10,
            weight="bold",
        )

    ax.scatter([], [], marker="s", s=70, color="gray", edgecolor="black", label="UAV起点")
    ax.scatter([], [], marker="*", s=160, color="gray", edgecolor="black", label="UAV终点")

    ax.set_title("UAV轨迹与动态负载质心（阶段性结果）", fontsize=17, weight="bold", pad=12)
    ax.set_xlabel("X坐标（m）", fontsize=12)
    ax.set_ylabel("Y坐标（m）", fontsize=12)
    ax.set_xlim(250, 760)
    ax.set_ylim(280, 750)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.text(
        260,
        732,
        "说明：方块为起点，星号为终点；虚线为区域负载质心变化。",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="#cccccc"),
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
