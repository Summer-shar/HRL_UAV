"""Plot why cooperative movement helps despite similar final UAV positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "result_archive" / "v35_reference_like_mobility_gate_final"
COLORS = {"centroid": "#F28E2B", "cooperative": "#2E7D6E"}
LABELS = {"centroid": "动态质心跟踪", "cooperative": "协作感知移动"}
HEAVY_BY_PHASE = np.asarray([0, 1, 2], dtype=np.int64)
SUPPORT_BY_PHASE = np.asarray([2, 0, 1], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot active U2U support distance and heavy-UAV queue."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--macro_period_s", type=float, default=10.0)
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )


def mean_ci(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    if matrix.shape[0] < 2:
        return mean, np.zeros_like(mean)
    half = student_t.ppf(0.975, matrix.shape[0] - 1) * np.std(
        matrix, axis=0, ddof=1
    ) / math.sqrt(matrix.shape[0])
    return mean, half


def extract_series(trace: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    phases = np.asarray(trace["trajectory_phases"], dtype=np.int64)
    uav = np.asarray(trace["uav_trajectory_m"], dtype=np.float64)
    queues = np.asarray(
        trace["queue_per_uav_trajectory_mbits"], dtype=np.float64
    )
    indices = np.arange(phases.size)
    heavy = HEAVY_BY_PHASE[phases]
    support = SUPPORT_BY_PHASE[phases]
    distance = np.linalg.norm(uav[indices, heavy] - uav[indices, support], axis=1)
    heavy_queue = queues[indices, heavy]
    return distance, heavy_queue


def load_traces(root: Path) -> tuple[list[int], dict[str, list[dict[str, np.ndarray]]]]:
    traces_root = root / "traces"
    seed_sets = []
    for policy in LABELS:
        seed_sets.append(
            {
                int(path.stem.split("_")[0].removeprefix("seed"))
                for path in traces_root.glob(f"seed*_{policy}.npz")
            }
        )
    seeds = sorted(set.intersection(*seed_sets))
    if not seeds:
        raise FileNotFoundError(f"No paired traces found under {traces_root}")
    traces = {policy: [] for policy in LABELS}
    for seed in seeds:
        for policy in LABELS:
            with np.load(traces_root / f"seed{seed}_{policy}.npz") as archive:
                traces[policy].append(
                    {key: archive[key].copy() for key in archive.files}
                )
    return seeds, traces


def main() -> None:
    args = parse_args()
    root = args.input if args.input.is_absolute() else ROOT / args.input
    seeds, traces = load_traces(root)
    series: dict[str, dict[str, np.ndarray]] = {}
    for policy, policy_traces in traces.items():
        distances = []
        queues = []
        for trace in policy_traces:
            distance, queue = extract_series(trace)
            distances.append(distance)
            queues.append(queue)
        series[policy] = {
            "distance": np.asarray(distances),
            "queue": np.asarray(queues),
        }

    summary = {
        "seeds": seeds,
        "centroid_active_link_distance_m": float(
            np.mean(series["centroid"]["distance"])
        ),
        "cooperative_active_link_distance_m": float(
            np.mean(series["cooperative"]["distance"])
        ),
        "centroid_heavy_queue_mbits": float(
            np.mean(series["centroid"]["queue"])
        ),
        "cooperative_heavy_queue_mbits": float(
            np.mean(series["cooperative"]["queue"])
        ),
    }
    summary["active_link_distance_reduction_m"] = (
        summary["centroid_active_link_distance_m"]
        - summary["cooperative_active_link_distance_m"]
    )
    summary["heavy_queue_reduction_mbits"] = (
        summary["centroid_heavy_queue_mbits"]
        - summary["cooperative_heavy_queue_mbits"]
    )
    (root / "cooperation_mechanism_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    configure_plotting()
    sample_count = series["centroid"]["distance"].shape[1]
    time_s = np.arange(sample_count, dtype=np.float64) * args.macro_period_s
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True)
    for policy in LABELS:
        for axis, field in zip(axes, ("distance", "queue")):
            mean, half = mean_ci(series[policy][field])
            axis.plot(
                time_s,
                mean,
                color=COLORS[policy],
                linewidth=2.5,
                label=LABELS[policy],
            )
            axis.fill_between(
                time_s,
                mean - half,
                mean + half,
                color=COLORS[policy],
                alpha=0.14,
                linewidth=0,
            )

    axes[0].set_title("(a) 当前重载区的 U2U 支援链路")
    axes[0].set_ylabel("重载 UAV 与支援 UAV 距离 / m")
    axes[1].set_title("(b) 当前重载 UAV 的队列")
    axes[1].set_ylabel("队列积压 / Mbit")
    for axis in axes:
        axis.set_xlabel("仿真时间 / s")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    axes[0].text(
        0.03,
        0.05,
        f"整段平均缩短 {summary['active_link_distance_reduction_m']:.1f} m",
        transform=axes[0].transAxes,
        color="#333333",
    )
    axes[1].text(
        0.03,
        0.05,
        f"整段平均降低 {summary['heavy_queue_reduction_mbits']:.3f} Mbit",
        transform=axes[1].transAxes,
        color="#333333",
    )
    fig.suptitle("协作移动的作用机制（20 个共同随机种子，均值与 95% CI）", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(root / "05_协作链路与重载队列机制.png", dpi=300, facecolor="white")
    fig.savefig(root / "05_协作链路与重载队列机制.pdf", facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
