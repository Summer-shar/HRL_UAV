"""Plot all ground-terminal and cooperative UAV trajectories on one axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = (
    ROOT
    / "result_archive"
    / "v35_reference_like_mobility_gate_final"
    / "traces"
    / "seed2000_cooperative.npz"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v35_reference_like_mobility_gate_final"
COLORS = ("#2878B5", "#F28E2B", "#2CA02C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all terminal and cooperative UAV trajectories together."
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2000)
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 10.5,
        }
    )


def main() -> None:
    args = parse_args()
    trace_path = args.trace if args.trace.is_absolute() else ROOT / args.trace
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    with np.load(trace_path) as archive:
        users = np.asarray(archive["users_trajectory_m"], dtype=np.float64)
        uavs = np.asarray(archive["uav_trajectory_m"], dtype=np.float64)
        centroids = np.asarray(
            archive["centroid_trajectory_m"], dtype=np.float64
        )
        allocations = np.asarray(archive["user_allocations"], dtype=np.int64)

    configure_plotting()
    fig, axis = plt.subplots(figsize=(9.4, 8.2))
    for zone, color in enumerate(COLORS):
        user_indices = np.flatnonzero(allocations == zone)
        for user_index in user_indices:
            trajectory = users[:, user_index]
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                linewidth=0.65,
                alpha=0.22,
                zorder=1,
            )
            axis.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                s=10,
                facecolor="white",
                edgecolor=color,
                linewidth=0.55,
                alpha=0.75,
                zorder=2,
            )
            axis.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                s=10,
                color=color,
                alpha=0.75,
                zorder=2,
            )

        axis.plot(
            centroids[:, zone, 0],
            centroids[:, zone, 1],
            color=color,
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            zorder=3,
        )
        axis.plot(
            uavs[:, zone, 0],
            uavs[:, zone, 1],
            color=color,
            linewidth=3.2,
            marker="o",
            markersize=3.2,
            markevery=2,
            zorder=5,
        )
        axis.scatter(
            uavs[0, zone, 0],
            uavs[0, zone, 1],
            marker="s",
            s=90,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=6,
        )
        axis.scatter(
            uavs[-1, zone, 0],
            uavs[-1, zone, 1],
            marker="*",
            s=210,
            facecolor=color,
            edgecolor="#222222",
            linewidth=1.0,
            zorder=7,
        )
        axis.annotate(
            f"UAV-{zone + 1}",
            xy=uavs[-1, zone],
            xytext=(7, 7),
            textcoords="offset points",
            color=color,
            fontsize=11,
            fontweight="bold",
            zorder=8,
        )

    all_x = np.concatenate((users[..., 0].ravel(), uavs[..., 0].ravel()))
    all_y = np.concatenate((users[..., 1].ravel(), uavs[..., 1].ravel()))
    margin = 25.0
    axis.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
    axis.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X 坐标 / m")
    axis.set_ylabel("Y 坐标 / m")
    axis.set_title(f"协作感知策略下的 UAV 与地面终端轨迹（seed={args.seed}）")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color="#888888", linewidth=0.8, alpha=0.55),
        Line2D([0], [0], color="#444444", linewidth=1.8, linestyle="--"),
        Line2D([0], [0], color="#444444", linewidth=3.2, marker="o", markersize=4),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="#222222",
            markersize=12,
        ),
    ]
    axis.legend(
        legend_handles,
        ("地面终端轨迹", "区域业务质心", "UAV 轨迹", "UAV 起点", "UAV 终点"),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.18)
    name = "06_协作UAV与地面终端轨迹_单图"
    fig.savefig(output / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(output / f"{name}.png")


if __name__ == "__main__":
    main()
