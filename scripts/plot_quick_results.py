import argparse
import csv
import os
import re
import sys
from pathlib import Path

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

ROOT = Path(bootstrap())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_NAMES = {
    "sac": "eval_sac.csv",
    "micro_ppo": "eval_micro_ppo.csv",
    "no_coop": "eval_no_coop.csv",
    "random": "eval_random.csv",
    "sca": "eval_sca.csv",
}

DETAIL_NAMES = {
    "sac": "eval_sac_detail.npz",
    "micro_ppo": "eval_micro_ppo_detail.npz",
    "no_coop": "eval_no_coop_detail.npz",
    "random": "eval_random_detail.npz",
    "sca": "eval_sca_detail.npz",
}


LABELS = {
    "sac": "Proposed (HRL-SAC)",
    "micro_ppo": "PPO",
    "no_coop": "No Cooperation",
    "random": "Random",
    "sca": "SCA",
}


CPU_PATTERN = re.compile(r"(?:^|_)cpu(?P<cpu>\d+(?:\.\d+)?)")
TRAFFIC_PATTERN = re.compile(r"(?:^|_)ts(?P<traffic>\d+(?:\.\d+)?)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot quick preliminary figures from eval CSV files.",
    )
    parser.add_argument(
        "--result_dirs",
        nargs="*",
        default=None,
        help=(
            "Result subdirectories under ./results. "
            "If omitted, all result directories with eval CSV files are plotted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help=(
            "Optional directory for extra summary figures. "
            "If omitted, figures are saved only inside each result directory."
        ),
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Moving-average window for line smoothing.",
    )
    parser.add_argument(
        "--show_markers",
        action="store_true",
        help="Show point markers on smoothed training curves.",
    )
    parser.add_argument(
        "--convergence_window",
        type=int,
        default=3,
        help="Number of final eval points averaged as the converged value.",
    )
    return parser.parse_args()


def infer_algorithm(result_dir_name):
    name = result_dir_name.lower()
    if name.startswith("micro_ppo"):
        return "micro_ppo"
    if name.startswith("no_coop"):
        return "no_coop"
    if name.startswith("random"):
        return "random"
    if name.startswith("sca"):
        return "sca"
    if name.startswith("sac"):
        return "sac"
    return None


def parse_cpu_ghz(result_dir_name):
    match = CPU_PATTERN.search(result_dir_name.lower())
    if not match:
        return None
    return float(match.group("cpu"))


def parse_traffic_scale(result_dir_name):
    match = TRAFFIC_PATTERN.search(result_dir_name.lower())
    if not match:
        return None
    return float(match.group("traffic"))


def compact_float(value):
    return f"{float(value):g}"


def short_result_label(summary):
    cpu_ghz = summary.get("cpu_ghz")
    algorithm = summary.get("algorithm", "")
    if algorithm == "micro_ppo":
        return "PPO"
    if algorithm == "no_coop":
        return "No coop"
    label = LABELS.get(algorithm, algorithm)
    if cpu_ghz not in ("", None):
        return label
    return label


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, var ** 0.5


def smooth(values, window):
    values = [float(value) for value in values]
    window = max(int(window), 1)
    if window <= 1 or len(values) < 2:
        return values

    smoothed = []
    half = window // 2
    for idx in range(len(values)):
        left = max(0, idx - half)
        right = min(len(values), idx + half + 1)
        smoothed.append(sum(values[left:right]) / max(right - left, 1))
    return smoothed


def find_eval_csv(result_dir_name):
    algorithm = infer_algorithm(result_dir_name)
    if algorithm is None:
        return None, None

    csv_path = ROOT / "results" / result_dir_name / CSV_NAMES[algorithm]
    if not csv_path.is_file():
        return algorithm, None
    return algorithm, csv_path


def find_detail_npz(result_dir_name, algorithm=None):
    if algorithm is None:
        algorithm = infer_algorithm(result_dir_name)
    if algorithm is None:
        return None

    detail_name = DETAIL_NAMES.get(algorithm)
    if not detail_name:
        return None

    detail_path = ROOT / "results" / result_dir_name / detail_name
    return detail_path if detail_path.is_file() else None


def load_detail_npz(path):
    if path is None or not Path(path).is_file():
        return {}
    try:
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}
    except Exception:
        return {}


def discover_result_dirs():
    results_root = ROOT / "results"
    if not results_root.is_dir():
        return []

    discovered = []
    for path in sorted(results_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if path.name.startswith("quick_report"):
            continue
        _, csv_path = find_eval_csv(path.name)
        if csv_path is not None:
            discovered.append(path.name)
    return discovered


def plot_training_curves(
    rows,
    output_path,
    smooth_window,
    title,
    show_markers=False,
    train_rows=None,
):
    if not rows and not train_rows:
        return

    # [PLOT] 优先用逐轮训练日志(train_sac.csv)画**连续**训练曲线(1500点)。
    # 评估(eval_sac.csv)是每隔 eval_interval 轮才有一个点(稀疏、确定性策略),
    # 仅在没有逐轮训练日志的基线上作为回退。
    use_train = bool(train_rows) and any(
        to_float(row, "micro_reward_per_step") != 0.0 for row in train_rows
    )

    if use_train:
        ep = [to_float(row, "episode") for row in train_rows]

        def col(key):
            return [to_float(row, key) for row in train_rows]

        plots = [
            (ep, col("micro_reward_per_step"),
             "Training reward / micro step", "Reward"),
            (ep, col("micro_cost_raw"),
             "Training cost (J / micro step)", "Cost"),
            (ep, col("task_delay_avg_ms"),
             "Task completion delay", "Delay (ms)"),
            (ep, col("queue_backlog"),
             "Queue backlog", "Queue (Mbits)"),
            (ep, col("energy_per_step"),
             "Energy / micro step", "Energy"),
            (ep, col("task_completion_rate"),
             "Task completion rate", "Rate"),
        ]
    else:
        # 基线回退: 稀疏 eval 曲线
        episodes = [to_float(row, "episode") for row in rows]
        plots = [
            (episodes, [to_float(r, "score") for r in rows],
             "Evaluation score (-J, higher=better)", "Score"),
            (episodes, [to_float(r, "delay_weighted_ms") for r in rows],
             "Weighted queueing delay", "Delay (ms)"),
            (episodes,
             [to_float(r, "q_high") + to_float(r, "q_low") for r in rows],
             "Average queue backlog", "Queue (Mbits)"),
            (episodes, [to_float(r, "energy") for r in rows],
             "Average energy per step", "Energy"),
        ]

    n = len(plots)
    ncols = 3 if n > 4 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), dpi=150
    )
    axes = np.asarray(axes).reshape(-1)

    for axis, (plot_episodes, values, subplot_title, ylabel) in zip(axes, plots):
        smooth_values = smooth(values, smooth_window)
        axis.plot(
            plot_episodes,
            values,
            color="0.78",
            linewidth=1.0,
            alpha=0.9,
            label="raw",
        )
        line_kwargs = {
            "linewidth": 2.3,
            "label": f"smoothed (w={smooth_window})",
        }
        if show_markers:
            line_kwargs.update({"marker": "o", "markersize": 3.2})
        axis.plot(plot_episodes, smooth_values, **line_kwargs)
        axis.set_title(subplot_title)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.35)
        axis.legend(frameon=False, fontsize=8)

    # 隐藏多余的空子图
    for axis in axes[n:]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_kkt_cpu_curves(rows, output_path, smooth_window, show_markers=False):
    if not rows:
        return

    episodes = [to_float(row, "episode") for row in rows]
    cpu_util = [to_float(row, "cpu_util") for row in rows]
    cpu_high = [to_float(row, "cpu_high_util") for row in rows]
    cpu_low = [to_float(row, "cpu_low_util") for row in rows]
    cpu_sat = [to_float(row, "cpu_saturation") for row in rows]
    f_high = [to_float(row, "f_high_ghz") for row in rows]
    f_low = [to_float(row, "f_low_ghz") for row in rows]
    f_total = [to_float(row, "f_total_ghz") for row in rows]

    has_priority_alloc = any(value > 0 for value in cpu_high + cpu_low)
    has_frequency = any(value > 0 for value in f_total)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    line_kwargs = {"linewidth": 2.2}
    if show_markers:
        line_kwargs.update({"marker": "o", "markersize": 3.0})

    axes[0].plot(
        episodes,
        smooth(cpu_util, smooth_window),
        label="total util",
        **line_kwargs,
    )
    if has_priority_alloc:
        axes[0].plot(
            episodes,
            smooth(cpu_high, smooth_window),
            label="high-priority util",
            **line_kwargs,
        )
        axes[0].plot(
            episodes,
            smooth(cpu_low, smooth_window),
            label="low-priority util",
            **line_kwargs,
        )
    axes[0].plot(
        episodes,
        smooth(cpu_sat, smooth_window),
        label="saturation ratio",
        linestyle="--",
        linewidth=1.8,
    )
    axes[0].set_title("KKT CPU utilization")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Ratio")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=8)

    if has_frequency:
        axes[1].plot(
            episodes,
            smooth(f_total, smooth_window),
            label="total",
            **line_kwargs,
        )
        axes[1].plot(
            episodes,
            smooth(f_high, smooth_window),
            label="high-priority",
            **line_kwargs,
        )
        axes[1].plot(
            episodes,
            smooth(f_low, smooth_window),
            label="low-priority",
            **line_kwargs,
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Frequency columns are not available\nin this old eval CSV.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("KKT allocated CPU frequency")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("GHz")
    axes[1].grid(True, alpha=0.35)
    if has_frequency:
        axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("KKT CPU Allocation Diagnostics")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_final_comparison(final_rows, output_dir):
    if not final_rows:
        return

    labels = [item["label"] for item in final_rows]
    delay = [item["delay"] for item in final_rows]
    queue = [item["queue"] for item in final_rows]
    energy = [item["energy"] for item in final_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    metrics = [
        (delay, "Weighted delay", "Delay (ms)"),
        (queue, "Queue backlog", "Queue (Mbits)"),
        (energy, "Energy", "Energy"),
    ]

    for axis, (values, title, ylabel) in zip(axes, metrics):
        axis.bar(labels, values)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.35)
        axis.tick_params(axis="x", rotation=25)

    fig.suptitle("Preliminary Algorithm Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "preliminary_algorithm_comparison.png")
    plt.close(fig)


def plot_convergence_summary(summaries, output_path):
    if not summaries:
        return

    def positive_metric(item, keys):
        for key in keys:
            value = float(item.get(key, 0.0))
            if value > 0.0:
                return value
        return 0.0

    labels = [short_result_label(item) for item in summaries]
    delay_mean = [
        positive_metric(
            item,
            [
                "task_delay_avg_ms_mean",
                "task_delay_weighted_ms_mean",
                "delay_weighted_ms_mean",
            ],
        )
        for item in summaries
    ]
    delay_std = [
        positive_metric(
            item,
            [
                "task_delay_avg_ms_std",
                "task_delay_weighted_ms_std",
                "delay_weighted_ms_std",
            ],
        )
        for item in summaries
    ]
    queue_mean = [float(item["queue_mean"]) for item in summaries]
    queue_std = [float(item["queue_std"]) for item in summaries]
    energy_mean = [float(item["energy_mean"]) for item in summaries]
    energy_std = [float(item["energy_std"]) for item in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    metrics = [
        (delay_mean, delay_std, "Converged average E2E task delay", "Delay (ms)"),
        (queue_mean, queue_std, "Converged queue backlog", "Queue (Mbits)"),
        (energy_mean, energy_std, "Converged energy", "Energy"),
    ]

    for axis, (means, stds, title, ylabel) in zip(axes, metrics):
        axis.bar(labels, means, yerr=stds, capsize=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.35)
        axis.tick_params(axis="x", rotation=25)

    fig.suptitle("Convergence Summary")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cpu_sweep_summary(summaries, output_dir):
    cpu_rows = [
        item
        for item in summaries
        if item.get("cpu_ghz") not in ("", None)
    ]
    if not cpu_rows:
        return []

    grouped = {}
    for item in cpu_rows:
        traffic_scale = item.get("traffic_scale")
        grouped.setdefault(traffic_scale, []).append(item)

    saved_paths = []
    for traffic_scale, rows in sorted(
        grouped.items(),
        key=lambda pair: -1.0 if pair[0] in ("", None) else float(pair[0]),
    ):
        rows = sorted(rows, key=lambda item: float(item["cpu_ghz"]))
        by_algorithm = {}
        for item in rows:
            by_algorithm.setdefault(item["algorithm"], []).append(item)
        by_algorithm = {
            algorithm: sorted(items, key=lambda item: float(item["cpu_ghz"]))
            for algorithm, items in by_algorithm.items()
            if len(items) >= 2
        }
        if not by_algorithm:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=150)
        axes = axes.reshape(-1)
        metrics = [
            ("task_delay_avg_ms", "Average E2E task delay", "Delay (ms)"),
            ("task_delay_high_ms", "High-priority E2E delay", "Delay (ms)"),
            ("task_delay_low_ms", "Low-priority E2E delay", "Delay (ms)"),
            ("energy_total_episode", "Total energy", "Energy (J)"),
        ]

        for axis, (metric, title, ylabel) in zip(axes, metrics):
            for algorithm, items in sorted(by_algorithm.items()):
                x_values = [float(item["cpu_ghz"]) for item in items]
                means = [float(item.get(f"{metric}_mean", 0.0)) for item in items]
                stds = [float(item.get(f"{metric}_std", 0.0)) for item in items]
                axis.plot(
                    x_values,
                    means,
                    marker="o",
                    markersize=3.2,
                    linewidth=2.0,
                    label=LABELS.get(algorithm, algorithm),
                )
                axis.errorbar(
                    x_values,
                    means,
                    yerr=stds,
                    fmt="none",
                    ecolor="0.35",
                    capsize=3,
                    linewidth=0.9,
                    alpha=0.75,
                )
            axis.set_title(title)
            axis.set_xlabel("UAV CPU frequency (GHz)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.35)
            axis.legend(frameon=False, fontsize=8)

        title = "CPU Frequency Sweep"
        suffix = "unknown"
        if traffic_scale not in ("", None):
            title += f" (traffic scale = {compact_float(traffic_scale)})"
            suffix = f"ts{compact_float(traffic_scale)}"
        fig.suptitle(title)
        fig.tight_layout()

        output_path = output_dir / f"cpu_sweep_summary_{suffix}.png"
        fig.savefig(output_path)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def final_array(detail, key, shape, window=1):
    value = detail.get(key)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == len(shape):
        return value
    if value.shape[0] == 0:
        return np.zeros(shape, dtype=np.float32)
    window = max(int(window), 1)
    return np.mean(value[-window:], axis=0)


def plot_detail_figures(detail_records, output_dir, convergence_window=3):
    if not detail_records:
        return []

    saved_paths = []
    grouped = {}
    for record in detail_records:
        cpu = record.get("cpu_ghz")
        traffic = record.get("traffic_scale")
        if cpu in ("", None):
            continue
        grouped.setdefault((traffic, cpu), []).append(record)

    for (traffic_scale, cpu_ghz), records in sorted(
        grouped.items(),
        key=lambda pair: (
            -1.0 if pair[0][0] in ("", None) else float(pair[0][0]),
            float(pair[0][1]),
        ),
    ):
        records = [record for record in records if record.get("detail")]
        if not records:
            continue

        records = sorted(records, key=lambda item: item["label"])
        suffix = f"cpu{compact_float(cpu_ghz)}"
        if traffic_scale not in ("", None):
            suffix += f"_ts{compact_float(traffic_scale)}"

        labels = [record["label"] for record in records]
        num_uavs = max(
            final_array(
                record["detail"],
                "task_delay_source_uav_ms",
                (0,),
                convergence_window,
            ).size
            for record in records
        )
        if num_uavs <= 0:
            continue
        x = np.arange(num_uavs)
        width = min(0.78 / max(len(records), 1), 0.18)

        fig, axis = plt.subplots(figsize=(10.5, 4.6), dpi=150)
        valid_delay_records = []
        for record in records:
            completed = final_array(
                record["detail"],
                "task_completed_source_count",
                (num_uavs,),
                convergence_window,
            )
            if float(np.sum(completed)) > 0:
                valid_delay_records.append(record)
        width = min(0.78 / max(len(valid_delay_records), 1), 0.18)
        for idx, record in enumerate(valid_delay_records):
            values = final_array(
                record["detail"],
                "task_delay_source_uav_ms",
                (num_uavs,),
                convergence_window,
            )
            offset = (idx - (len(valid_delay_records) - 1) / 2.0) * width
            axis.bar(x + offset, values, width=width, label=record["label"])
        axis.set_title(
            f"Per-Region Task Delay ({compact_float(cpu_ghz)} GHz)"
        )
        axis.set_xlabel("Source region")
        axis.set_ylabel("Task delay (ms)")
        axis.set_xticks(x)
        axis.set_xticklabels([f"Region {idx + 1}" for idx in range(num_uavs)])
        axis.grid(True, axis="y", alpha=0.35)
        axis.legend(frameon=False, fontsize=8, ncols=2)
        fig.tight_layout()
        output_path = output_dir / f"region_task_delay_{suffix}.png"
        fig.savefig(output_path)
        plt.close(fig)
        saved_paths.append(output_path)

        has_priority_delay = any(
            "task_delay_source_high_ms" in record.get("detail", {})
            or "task_delay_source_low_ms" in record.get("detail", {})
            for record in records
        )
        if has_priority_delay:
            priority_specs = [
                ("high", "High-priority task delay"),
                ("low", "Low-priority task delay"),
            ]
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12.0, 4.6),
                dpi=150,
                squeeze=False,
            )
            any_priority_bar = False
            for axis, (priority, title) in zip(axes.reshape(-1), priority_specs):
                priority_records = []
                completed_key = f"task_completed_source_{priority}_count"
                delay_key = f"task_delay_source_{priority}_ms"
                for record in records:
                    completed = final_array(
                        record["detail"],
                        completed_key,
                        (num_uavs,),
                        convergence_window,
                    )
                    if float(np.sum(completed)) > 0:
                        priority_records.append(record)

                local_width = min(0.78 / max(len(priority_records), 1), 0.18)
                for idx, record in enumerate(priority_records):
                    completed = final_array(
                        record["detail"],
                        completed_key,
                        (num_uavs,),
                        convergence_window,
                    )
                    values = final_array(
                        record["detail"],
                        delay_key,
                        (num_uavs,),
                        convergence_window,
                    )
                    values = np.where(completed > 0, values, np.nan)
                    offset = (idx - (len(priority_records) - 1) / 2.0) * local_width
                    axis.bar(
                        x + offset,
                        values,
                        width=local_width,
                        label=record["label"],
                    )
                    any_priority_bar = True

                axis.set_title(title)
                axis.set_xlabel("Source region")
                axis.set_ylabel("Task delay (ms)")
                axis.set_xticks(x)
                axis.set_xticklabels([f"Region {idx + 1}" for idx in range(num_uavs)])
                axis.grid(True, axis="y", alpha=0.35)
                if priority_records:
                    axis.legend(frameon=False, fontsize=8)

            fig.suptitle(
                f"Per-Region Priority Task Delay ({compact_float(cpu_ghz)} GHz)"
            )
            fig.tight_layout()
            if any_priority_bar:
                output_path = output_dir / f"region_priority_task_delay_{suffix}.png"
                fig.savefig(output_path)
                saved_paths.append(output_path)
            plt.close(fig)

        fig, axes = plt.subplots(
            len(records),
            1,
            figsize=(10.5, max(3.0 * len(records), 3.8)),
            dpi=150,
            squeeze=False,
        )
        components = [
            ("energy_comp_episode_per_uav", "Compute"),
            ("energy_uplink_episode_per_uav", "Uplink"),
            ("energy_trans_episode_per_uav", "U2U"),
            ("energy_macro_episode_per_uav", "Flight/Hover"),
        ]
        for axis, record in zip(axes.reshape(-1), records):
            bottom = np.zeros(num_uavs, dtype=np.float32)
            for key, label in components:
                values = final_array(
                    record["detail"],
                    key,
                    (num_uavs,),
                    convergence_window,
                )
                axis.bar(x, values, bottom=bottom, label=label)
                bottom += values
            axis.set_title(record["label"])
            axis.set_ylabel("Energy (J)")
            axis.set_xticks(x)
            axis.set_xticklabels([f"UAV {idx + 1}" for idx in range(num_uavs)])
            axis.grid(True, axis="y", alpha=0.35)
        axes.reshape(-1)[0].legend(frameon=False, fontsize=8, ncols=4)
        fig.suptitle(f"Per-UAV Energy Breakdown ({compact_float(cpu_ghz)} GHz)")
        fig.tight_layout()
        output_path = output_dir / f"uav_energy_breakdown_{suffix}.png"
        fig.savefig(output_path)
        plt.close(fig)
        saved_paths.append(output_path)

        cols = min(3, len(records))
        rows = int(np.ceil(len(records) / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4.0 * cols, 3.8 * rows),
            dpi=150,
            squeeze=False,
        )
        matrices = [
            final_array(
                record["detail"],
                "coop_flow_mbits",
                (num_uavs, num_uavs),
                convergence_window,
            )
            for record in records
        ]
        vmax = max(float(np.max(matrix)) for matrix in matrices) if matrices else 1.0
        vmax = max(vmax, 1e-6)
        for axis, record, matrix in zip(axes.reshape(-1), records, matrices):
            image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
            axis.set_title(f"{record['label']} | Coop {record.get('coop_ratio', 0.0):.2f}")
            axis.set_xlabel("Target UAV")
            axis.set_ylabel("Source region")
            axis.set_xticks(x)
            axis.set_yticks(x)
            axis.set_xticklabels([idx + 1 for idx in range(num_uavs)])
            axis.set_yticklabels([idx + 1 for idx in range(num_uavs)])
            for row_idx in range(num_uavs):
                for col_idx in range(num_uavs):
                    axis.text(
                        col_idx,
                        row_idx,
                        f"{matrix[row_idx, col_idx]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if matrix[row_idx, col_idx] < 0.65 * vmax else "white",
                    )
        for axis in axes.reshape(-1)[len(records):]:
            axis.axis("off")
        fig.colorbar(image, ax=axes.reshape(-1).tolist(), shrink=0.85, label="Mbits")
        fig.suptitle(f"Cooperation Flow Matrix ({compact_float(cpu_ghz)} GHz)")
        fig.tight_layout()
        output_path = output_dir / f"coop_flow_heatmap_{suffix}.png"
        fig.savefig(output_path)
        plt.close(fig)
        saved_paths.append(output_path)

        split_records = [
            record
            for record in records
            if "split_ratios_high_mean" in record.get("detail", {})
            or "split_ratios_low_mean" in record.get("detail", {})
        ]
        if split_records:
            cols = min(3, len(split_records))
            rows = int(np.ceil(len(split_records) / cols))
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(4.0 * cols, 3.8 * rows),
                dpi=150,
                squeeze=False,
            )
            matrices = []
            for record in split_records:
                high = final_array(
                    record["detail"],
                    "split_ratios_high_mean",
                    (num_uavs, num_uavs),
                    convergence_window,
                )
                low = final_array(
                    record["detail"],
                    "split_ratios_low_mean",
                    (num_uavs, num_uavs),
                    convergence_window,
                )
                matrix = 0.5 * (high + low)
                row_sum = np.sum(matrix, axis=1, keepdims=True)
                matrix = np.divide(
                    matrix,
                    np.maximum(row_sum, 1e-12),
                    out=np.zeros_like(matrix),
                    where=row_sum > 0,
                )
                matrices.append(matrix)

            for axis, record, matrix in zip(axes.reshape(-1), split_records, matrices):
                image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
                axis.set_title(f"{record['label']} | Split")
                axis.set_xlabel("Target UAV")
                axis.set_ylabel("Source region")
                axis.set_xticks(x)
                axis.set_yticks(x)
                axis.set_xticklabels([idx + 1 for idx in range(num_uavs)])
                axis.set_yticklabels([idx + 1 for idx in range(num_uavs)])
                for row_idx in range(num_uavs):
                    for col_idx in range(num_uavs):
                        axis.text(
                            col_idx,
                            row_idx,
                            f"{matrix[row_idx, col_idx]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black" if matrix[row_idx, col_idx] < 0.65 else "white",
                        )
            for axis in axes.reshape(-1)[len(split_records):]:
                axis.axis("off")
            fig.colorbar(
                image,
                ax=axes.reshape(-1).tolist(),
                shrink=0.85,
                label="Average split ratio",
            )
            fig.suptitle(f"Average Split Ratio Matrix ({compact_float(cpu_ghz)} GHz)")
            fig.tight_layout()
            output_path = output_dir / f"split_ratio_heatmap_{suffix}.png"
            fig.savefig(output_path)
            plt.close(fig)
            saved_paths.append(output_path)

    return saved_paths


def build_convergence_summary(result_dir_name, algorithm, rows, window):
    window = max(int(window), 1)
    tail = rows[-window:]
    final = rows[-1]

    queue_values = [
        to_float(row, "q_high") + to_float(row, "q_low")
        for row in tail
    ]
    final_queue = to_float(final, "q_high") + to_float(final, "q_low")

    fields = {
        "score": [to_float(row, "score") for row in tail],
        "delay_weighted_ms": [
            to_float(row, "delay_weighted_ms") for row in tail
        ],
        "task_delay_avg_ms": [
            to_float(row, "task_delay_avg_ms") for row in tail
        ],
        "task_delay_weighted_ms": [
            to_float(row, "task_delay_weighted_ms") for row in tail
        ],
        "task_delay_high_ms": [
            to_float(row, "task_delay_high_ms") for row in tail
        ],
        "task_delay_low_ms": [
            to_float(row, "task_delay_low_ms") for row in tail
        ],
        "task_delay_sensor_ms": [
            to_float(row, "task_delay_sensor_ms") for row in tail
        ],
        "task_delay_controller_ms": [
            to_float(row, "task_delay_controller_ms") for row in tail
        ],
        "task_delay_vision_ms": [
            to_float(row, "task_delay_vision_ms") for row in tail
        ],
        "task_completion_rate": [
            to_float(row, "task_completion_rate") for row in tail
        ],
        "task_qos_violation_rate": [
            to_float(row, "task_qos_violation_rate") for row in tail
        ],
        "compute_service_high_mbits": [
            to_float(row, "compute_service_high_mbits") for row in tail
        ],
        "compute_service_low_mbits": [
            to_float(row, "compute_service_low_mbits") for row in tail
        ],
        "compute_service_sensor_mbits": [
            to_float(row, "compute_service_sensor_mbits") for row in tail
        ],
        "compute_service_controller_mbits": [
            to_float(row, "compute_service_controller_mbits") for row in tail
        ],
        "compute_service_vision_mbits": [
            to_float(row, "compute_service_vision_mbits") for row in tail
        ],
        "delay_high_ms": [to_float(row, "delay_high_ms") for row in tail],
        "delay_low_ms": [to_float(row, "delay_low_ms") for row in tail],
        "delay_max_ms": [to_float(row, "delay_max_ms") for row in tail],
        "queue": queue_values,
        "energy": [to_float(row, "energy") for row in tail],
        "energy_total_episode": [
            to_float(row, "energy_total_episode", to_float(row, "energy"))
            for row in tail
        ],
        "coop_ratio": [to_float(row, "coop_ratio") for row in tail],
        "cpu_util": [to_float(row, "cpu_util") for row in tail],
        "cpu_saturation": [
            to_float(row, "cpu_saturation") for row in tail
        ],
        "pressure_imbalance": [
            to_float(row, "pressure_imbalance") for row in tail
        ],
    }

    summary = {
        "result_dir": result_dir_name,
        "algorithm": algorithm,
        "label": LABELS.get(algorithm, algorithm),
        "cpu_ghz": parse_cpu_ghz(result_dir_name),
        "traffic_scale": parse_traffic_scale(result_dir_name),
        "num_eval_points": len(rows),
        "convergence_window": len(tail),
        "start_episode": to_float(tail[0], "episode"),
        "end_episode": to_float(tail[-1], "episode"),
        "final_score": to_float(final, "score"),
        "final_delay_weighted_ms": to_float(final, "delay_weighted_ms"),
        "final_task_delay_avg_ms": to_float(final, "task_delay_avg_ms"),
        "final_task_delay_weighted_ms": to_float(final, "task_delay_weighted_ms"),
        "final_task_delay_high_ms": to_float(final, "task_delay_high_ms"),
        "final_task_delay_low_ms": to_float(final, "task_delay_low_ms"),
        "final_task_delay_sensor_ms": to_float(final, "task_delay_sensor_ms"),
        "final_task_delay_controller_ms": to_float(final, "task_delay_controller_ms"),
        "final_task_delay_vision_ms": to_float(final, "task_delay_vision_ms"),
        "final_task_completed_count": to_float(final, "task_completed_count"),
        "final_task_arrived_count": to_float(final, "task_arrived_count"),
        "final_task_completion_rate": to_float(final, "task_completion_rate"),
        "final_task_qos_violation_rate": to_float(final, "task_qos_violation_rate"),
        "final_queue": final_queue,
        "final_energy": to_float(final, "energy"),
        "final_energy_total_episode": to_float(
            final,
            "energy_total_episode",
            to_float(final, "energy"),
        ),
        "final_coop_ratio": to_float(final, "coop_ratio"),
        "final_compute_service_high_mbits": to_float(final, "compute_service_high_mbits"),
        "final_compute_service_low_mbits": to_float(final, "compute_service_low_mbits"),
        "final_compute_service_sensor_mbits": to_float(final, "compute_service_sensor_mbits"),
        "final_compute_service_controller_mbits": to_float(final, "compute_service_controller_mbits"),
        "final_compute_service_vision_mbits": to_float(final, "compute_service_vision_mbits"),
        "final_pressure_imbalance": to_float(final, "pressure_imbalance"),
    }

    for key, values in fields.items():
        avg, std = mean_std(values)
        summary[f"{key}_mean"] = avg
        summary[f"{key}_std"] = std

    return summary


def save_summary_csv(path, summaries):
    if not summaries:
        return

    fieldnames = [
        "result_dir",
        "algorithm",
        "label",
        "cpu_ghz",
        "traffic_scale",
        "num_eval_points",
        "convergence_window",
        "start_episode",
        "end_episode",
        "final_score",
        "final_delay_weighted_ms",
        "final_task_delay_avg_ms",
        "final_task_delay_weighted_ms",
        "final_task_delay_high_ms",
        "final_task_delay_low_ms",
        "final_task_delay_sensor_ms",
        "final_task_delay_controller_ms",
        "final_task_delay_vision_ms",
        "final_task_completed_count",
        "final_task_arrived_count",
        "final_task_completion_rate",
        "final_task_qos_violation_rate",
        "final_queue",
        "final_energy",
        "final_energy_total_episode",
        "final_coop_ratio",
        "final_compute_service_high_mbits",
        "final_compute_service_low_mbits",
        "final_compute_service_sensor_mbits",
        "final_compute_service_controller_mbits",
        "final_compute_service_vision_mbits",
        "final_pressure_imbalance",
        "score_mean",
        "score_std",
        "delay_weighted_ms_mean",
        "delay_weighted_ms_std",
        "task_delay_avg_ms_mean",
        "task_delay_avg_ms_std",
        "task_delay_weighted_ms_mean",
        "task_delay_weighted_ms_std",
        "task_delay_high_ms_mean",
        "task_delay_high_ms_std",
        "task_delay_low_ms_mean",
        "task_delay_low_ms_std",
        "task_delay_sensor_ms_mean",
        "task_delay_sensor_ms_std",
        "task_delay_controller_ms_mean",
        "task_delay_controller_ms_std",
        "task_delay_vision_ms_mean",
        "task_delay_vision_ms_std",
        "task_completion_rate_mean",
        "task_completion_rate_std",
        "task_qos_violation_rate_mean",
        "task_qos_violation_rate_std",
        "compute_service_high_mbits_mean",
        "compute_service_high_mbits_std",
        "compute_service_low_mbits_mean",
        "compute_service_low_mbits_std",
        "compute_service_sensor_mbits_mean",
        "compute_service_sensor_mbits_std",
        "compute_service_controller_mbits_mean",
        "compute_service_controller_mbits_std",
        "compute_service_vision_mbits_mean",
        "compute_service_vision_mbits_std",
        "delay_high_ms_mean",
        "delay_high_ms_std",
        "delay_low_ms_mean",
        "delay_low_ms_std",
        "delay_max_ms_mean",
        "delay_max_ms_std",
        "queue_mean",
        "queue_std",
        "energy_mean",
        "energy_std",
        "energy_total_episode_mean",
        "energy_total_episode_std",
        "coop_ratio_mean",
        "coop_ratio_std",
        "cpu_util_mean",
        "cpu_util_std",
        "cpu_saturation_mean",
        "cpu_saturation_std",
        "pressure_imbalance_mean",
        "pressure_imbalance_std",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    result_dirs = args.result_dirs
    if result_dirs is None:
        result_dirs = discover_result_dirs()

    final_rows = []
    saved_paths = []
    convergence_summaries = []
    detail_records = []

    for result_dir_name in result_dirs:
        algorithm, csv_path = find_eval_csv(result_dir_name)
        if csv_path is None:
            continue

        rows = read_csv_rows(csv_path)
        if not rows:
            continue

        detail_path = find_detail_npz(result_dir_name, algorithm)
        detail = load_detail_npz(detail_path)
        result_dir = ROOT / "results" / result_dir_name
        convergence_summary = build_convergence_summary(
            result_dir_name,
            algorithm,
            rows,
            args.convergence_window,
        )
        save_summary_csv(
            result_dir / "convergence_summary.csv",
            [convergence_summary],
        )
        convergence_summaries.append(convergence_summary)
        detail_records.append(
            {
                "result_dir": result_dir_name,
                "algorithm": algorithm,
                "label": LABELS.get(algorithm, algorithm),
                "cpu_ghz": convergence_summary.get("cpu_ghz"),
                "traffic_scale": convergence_summary.get("traffic_scale"),
                "detail": detail,
                "detail_path": detail_path,
                "coop_ratio": convergence_summary.get("final_coop_ratio", 0.0),
            }
        )

        if algorithm == "sac":
            train_csv_path = result_dir / "train_sac.csv"
            train_rows = (
                read_csv_rows(train_csv_path)
                if train_csv_path.exists()
                else None
            )
            figure_path = result_dir / "proposed_training_curves.png"
            plot_training_curves(
                rows,
                figure_path,
                args.smooth_window,
                title="Proposed Algorithm Training Curves",
                show_markers=args.show_markers,
                train_rows=train_rows,
            )
            kkt_path = result_dir / "kkt_cpu_allocation.png"
            plot_kkt_cpu_curves(
                rows,
                kkt_path,
                args.smooth_window,
                show_markers=args.show_markers,
            )
            saved_paths.append(kkt_path)
        else:
            figure_path = result_dir / "training_curves.png"
            plot_training_curves(
                rows,
                figure_path,
                args.smooth_window,
                title=f"{LABELS.get(algorithm, algorithm)} Training Curves",
                show_markers=args.show_markers,
            )
        saved_paths.append(figure_path)

        final = rows[-1]
        final_rows.append(
            {
                "label": LABELS.get(algorithm, algorithm),
                "delay": to_float(final, "delay_weighted_ms"),
                "queue": to_float(final, "q_high") + to_float(final, "q_low"),
                "energy": to_float(final, "energy"),
            }
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_final_comparison(final_rows, output_dir)

    if not final_rows:
        print("No eval CSV files found. Run at least one experiment first.")
        sys.exit(1)

    summary_path = ROOT / "results" / "convergence_summary.csv"
    save_summary_csv(summary_path, convergence_summaries)
    summary_figure_path = ROOT / "results" / "convergence_summary.png"
    plot_convergence_summary(convergence_summaries, summary_figure_path)
    cpu_sweep_paths = plot_cpu_sweep_summary(
        convergence_summaries,
        ROOT / "results",
    )
    detail_paths = plot_detail_figures(
        detail_records,
        ROOT / "results",
    )

    print("Saved training figures:")
    for path in saved_paths:
        print(f"  {path}")
    print(f"Saved convergence summary -> {summary_path}")
    print(f"Saved convergence figure -> {summary_figure_path}")
    for path in cpu_sweep_paths:
        print(f"Saved CPU sweep figure -> {path}")
    for path in detail_paths:
        print(f"Saved detail figure -> {path}")
    if args.output_dir:
        print(f"Saved summary figures -> {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
