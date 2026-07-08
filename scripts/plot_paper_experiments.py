import argparse
import csv
from collections import defaultdict
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build paper-style experiment figures for the centralized HRL UAV runs.",
    )
    parser.add_argument(
        "--result_dir",
        default="results/robust_loadstate_prioritydemand_load08_12_p300",
        help="Result directory containing eval_central_macro.csv and detail npz.",
    )
    parser.add_argument(
        "--cross_csv",
        default="",
        help="CSV from evaluate_centralized_macro_loads.py. Defaults to result_dir/cross_cpu_load_eval_best.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory for generated figures. Defaults to result_dir/paper_figs.",
    )
    parser.add_argument(
        "--tail_window",
        type=int,
        default=3,
        help="Number of final eval points averaged for priority/region summaries.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Moving average window for convergence curves.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def compact(value):
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    if values.size <= 2 or window <= 1:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    half = window // 2
    for idx in range(values.size):
        left = max(0, idx - half)
        right = min(values.size, idx + half + 1)
        smoothed[idx] = float(np.mean(values[left:right]))
    return smoothed


def aggregate_cross_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(safe_float(row.get("cpu_ghz")), safe_float(row.get("load_scale")))].append(row)

    metrics = [
        "J",
        "task_delay_weighted_ms",
        "task_delay_high_ms",
        "task_delay_low_ms",
        "energy",
        "energy_total_episode",
        "cpu_saturation",
        "task_completion_rate",
        "task_qos_violation_rate",
        "coop_ratio",
        "u2g",
    ]
    output = []
    for (cpu_ghz, load_scale), group in sorted(grouped.items()):
        item = {"cpu_ghz": cpu_ghz, "load_scale": load_scale}
        for metric in metrics:
            values = np.asarray([safe_float(row.get(metric)) for row in group], dtype=float)
            item[metric] = float(np.mean(values)) if values.size else 0.0
            item[f"{metric}_std"] = float(np.std(values)) if values.size > 1 else 0.0
        output.append(item)
    return output


def metric_series(rows, x_key, metric):
    x_values = np.asarray([row[x_key] for row in rows], dtype=float)
    y_values = np.asarray([row.get(metric, 0.0) for row in rows], dtype=float)
    order = np.argsort(x_values)
    return x_values[order], y_values[order]


def set_axis_style(axis):
    axis.grid(True, alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_cpu_sweep(cross_rows, output_dir):
    saved = []
    loads = sorted({row["load_scale"] for row in cross_rows})
    metrics = [
        ("J", "Objective J", "lower is better"),
        ("task_delay_weighted_ms", "Weighted E2E Delay", "ms"),
        ("task_delay_high_ms", "High-priority Delay", "ms"),
        ("task_delay_low_ms", "Low-priority Delay", "ms"),
        ("energy", "Energy per Step", "J"),
        ("task_completion_rate", "Completion Rate", "ratio"),
    ]
    for load_scale in loads:
        rows = [row for row in cross_rows if abs(row["load_scale"] - load_scale) < 1e-9]
        if len(rows) < 2:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), dpi=160)
        axes = axes.reshape(-1)
        for axis, (metric, title, ylabel) in zip(axes, metrics):
            x_values, y_values = metric_series(rows, "cpu_ghz", metric)
            axis.plot(x_values, y_values, marker="o", linewidth=2.1, markersize=4.5)
            axis.set_title(title)
            axis.set_xlabel("UAV CPU Capacity (GHz)")
            axis.set_ylabel(ylabel)
            set_axis_style(axis)
        fig.suptitle(f"CPU Capacity Sweep at Load Scale {load_scale:g}")
        fig.tight_layout()
        path = output_dir / f"cpu_sweep_load_{compact(load_scale)}.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_load_sweep(cross_rows, output_dir):
    saved = []
    cpus = sorted({row["cpu_ghz"] for row in cross_rows})
    metrics = [
        ("J", "Objective J", "lower is better"),
        ("task_delay_weighted_ms", "Weighted E2E Delay", "ms"),
        ("task_delay_high_ms", "High-priority Delay", "ms"),
        ("task_delay_low_ms", "Low-priority Delay", "ms"),
        ("energy", "Energy per Step", "J"),
        ("task_qos_violation_rate", "QoS Violation Rate", "ratio"),
    ]
    for cpu_ghz in cpus:
        rows = [row for row in cross_rows if abs(row["cpu_ghz"] - cpu_ghz) < 1e-9]
        if len(rows) < 2:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), dpi=160)
        axes = axes.reshape(-1)
        for axis, (metric, title, ylabel) in zip(axes, metrics):
            x_values, y_values = metric_series(rows, "load_scale", metric)
            axis.plot(x_values, y_values, marker="o", linewidth=2.1, markersize=4.5)
            axis.set_title(title)
            axis.set_xlabel("Load Scale")
            axis.set_ylabel(ylabel)
            set_axis_style(axis)
        fig.suptitle(f"Load Sweep at CPU Capacity {cpu_ghz:g} GHz")
        fig.tight_layout()
        path = output_dir / f"load_sweep_cpu_{compact(cpu_ghz)}.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_cross_heatmaps(cross_rows, output_dir):
    saved = []
    cpus = sorted({row["cpu_ghz"] for row in cross_rows})
    loads = sorted({row["load_scale"] for row in cross_rows})
    if len(cpus) < 2 or len(loads) < 2:
        return saved

    metrics = [
        ("J", "Objective J"),
        ("task_delay_low_ms", "Low-priority Delay (ms)"),
        ("task_completion_rate", "Completion Rate"),
        ("energy", "Energy per Step"),
    ]
    lookup = {(row["cpu_ghz"], row["load_scale"]): row for row in cross_rows}
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.5), dpi=160)
    axes = axes.reshape(-1)
    last_image = None
    for axis, (metric, title) in zip(axes, metrics):
        matrix = np.asarray(
            [
                [lookup[(cpu, load)].get(metric, np.nan) for cpu in cpus]
                for load in loads
            ],
            dtype=float,
        )
        last_image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("CPU Capacity (GHz)")
        axis.set_ylabel("Load Scale")
        axis.set_xticks(range(len(cpus)))
        axis.set_xticklabels([f"{cpu:g}" for cpu in cpus])
        axis.set_yticks(range(len(loads)))
        axis.set_yticklabels([f"{load:g}" for load in loads])
        for row_idx, load in enumerate(loads):
            for col_idx, cpu in enumerate(cpus):
                value = matrix[row_idx, col_idx]
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > np.nanmean(matrix) else "black",
                )
        fig.colorbar(last_image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("CPU-Load Robustness Map")
    fig.tight_layout()
    path = output_dir / "cpu_load_heatmaps.png"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_convergence(eval_csv, output_dir, smooth_window):
    rows = read_csv_rows(eval_csv)
    if not rows:
        return []
    episodes = np.asarray([safe_float(row.get("episode")) for row in rows], dtype=float)
    metrics = [
        ("J", "Objective J", "lower is better"),
        ("task_delay_high_ms", "High-priority E2E Delay", "ms"),
        ("task_delay_low_ms", "Low-priority E2E Delay", "ms"),
        ("energy", "Energy per Step", "J"),
        ("coop_ratio", "Cooperation Ratio", "ratio"),
        ("task_completion_rate", "Completion Rate", "ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), dpi=160)
    axes = axes.reshape(-1)
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        values = np.asarray([safe_float(row.get(metric)) for row in rows], dtype=float)
        axis.plot(episodes, values, color="0.72", linewidth=1.1, label="eval")
        axis.plot(
            episodes,
            smooth(values, smooth_window),
            linewidth=2.2,
            label=f"smooth w={smooth_window}",
        )
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        set_axis_style(axis)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Centralized Macro Training Convergence")
    fig.tight_layout()
    path = output_dir / "convergence_central_macro.png"
    fig.savefig(path)
    plt.close(fig)

    j_values = np.asarray([safe_float(row.get("J")) for row in rows], dtype=float)
    score_values = -100.0 * j_values
    best_so_far = np.minimum.accumulate(j_values)
    best_idx = int(np.argmin(j_values))
    initial_j = float(j_values[0])
    best_j = float(j_values[best_idx])
    final_j = float(j_values[-1])
    improvement = (initial_j - best_j) / max(abs(initial_j), 1e-12) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), dpi=160)
    axes[0].plot(episodes, j_values, color="0.68", marker="o", label="eval J")
    axes[0].plot(
        episodes,
        smooth(j_values, smooth_window),
        linewidth=2.0,
        label=f"smooth w={smooth_window}",
    )
    axes[0].plot(episodes, best_so_far, linewidth=2.2, label="best-so-far J")
    axes[0].scatter(
        [episodes[best_idx]],
        [best_j],
        s=54,
        zorder=5,
        label=f"best ep {int(episodes[best_idx])}",
    )
    axes[0].set_title("Objective Convergence")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("J (lower is better)")
    axes[0].legend(frameon=False, fontsize=8)
    set_axis_style(axes[0])

    axes[1].plot(episodes, score_values, color="0.68", marker="o", label="eval score")
    axes[1].plot(
        episodes,
        smooth(score_values, smooth_window),
        linewidth=2.0,
        label=f"smooth w={smooth_window}",
    )
    axes[1].scatter(
        [episodes[best_idx]],
        [score_values[best_idx]],
        s=54,
        zorder=5,
        label=f"best ep {int(episodes[best_idx])}",
    )
    axes[1].set_title("Score Convergence")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score = -100J (higher is better)")
    axes[1].legend(frameon=False, fontsize=8)
    set_axis_style(axes[1])

    fig.suptitle(
        f"Best-so-far Objective: initial {initial_j:.4f}, "
        f"best {best_j:.4f}, final {final_j:.4f}, improvement {improvement:.2f}%"
    )
    fig.tight_layout()
    best_path = output_dir / "convergence_best_so_far.png"
    fig.savefig(best_path)
    plt.close(fig)

    summary_path = output_dir / "convergence_objective_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "initial_episode",
                "initial_J",
                "best_episode",
                "best_J",
                "final_episode",
                "final_J",
                "best_improvement_percent",
                "final_gap_to_best_percent",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "initial_episode": int(episodes[0]),
                "initial_J": initial_j,
                "best_episode": int(episodes[best_idx]),
                "best_J": best_j,
                "final_episode": int(episodes[-1]),
                "final_J": final_j,
                "best_improvement_percent": improvement,
                "final_gap_to_best_percent": (
                    (final_j - best_j) / max(abs(best_j), 1e-12) * 100.0
                ),
            }
        )

    return [path, best_path]


def tail_mean(data, key, tail_window):
    if key not in data:
        return None
    values = np.asarray(data[key], dtype=float)
    if values.size == 0:
        return None
    if values.ndim == 0:
        return values
    tail_window = max(int(tail_window), 1)
    return np.mean(values[-tail_window:], axis=0)


def completion_rate(completed, arrived):
    completed = np.asarray(completed, dtype=float)
    arrived = np.asarray(arrived, dtype=float)
    return np.divide(
        completed,
        np.maximum(arrived, 1e-12),
        out=np.zeros_like(completed, dtype=float),
        where=arrived > 0,
    )


def write_priority_region_summary(path, rows):
    fields = [
        "group",
        "region",
        "priority",
        "arrived",
        "completed",
        "completion_rate",
        "delay_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_priority_and_region(detail_npz, output_dir, tail_window):
    if not detail_npz.is_file():
        return []
    saved = []
    summary_rows = []
    with np.load(detail_npz) as data:
        high_arrived = tail_mean(data, "task_arrived_source_high_count", tail_window)
        high_completed = tail_mean(data, "task_completed_source_high_count", tail_window)
        low_arrived = tail_mean(data, "task_arrived_source_low_count", tail_window)
        low_completed = tail_mean(data, "task_completed_source_low_count", tail_window)
        high_delay = tail_mean(data, "task_delay_source_high_ms", tail_window)
        low_delay = tail_mean(data, "task_delay_source_low_ms", tail_window)
        total_arrived = tail_mean(data, "task_arrived_source_count", tail_window)
        total_completed = tail_mean(data, "task_completed_source_count", tail_window)
        total_delay = tail_mean(data, "task_delay_source_uav_ms", tail_window)
        coop_flow = tail_mean(data, "coop_flow_mbits", tail_window)

    if high_arrived is None or low_arrived is None:
        return saved

    high_rate = completion_rate(high_completed, high_arrived)
    low_rate = completion_rate(low_completed, low_arrived)
    total_rate = completion_rate(total_completed, total_arrived)
    regions = np.arange(len(high_rate))
    region_labels = [f"Zone {idx + 1}" for idx in regions]

    priority_arrived = np.asarray([np.sum(high_arrived), np.sum(low_arrived)], dtype=float)
    priority_completed = np.asarray([np.sum(high_completed), np.sum(low_completed)], dtype=float)
    priority_rate = completion_rate(priority_completed, priority_arrived)
    priority_delay = np.asarray(
        [
            np.average(high_delay, weights=np.maximum(high_completed, 1e-12)),
            np.average(low_delay, weights=np.maximum(low_completed, 1e-12)),
        ],
        dtype=float,
    )

    for idx, name in enumerate(["High", "Low"]):
        summary_rows.append(
            {
                "group": "priority_total",
                "region": "all",
                "priority": name.lower(),
                "arrived": priority_arrived[idx],
                "completed": priority_completed[idx],
                "completion_rate": priority_rate[idx],
                "delay_ms": priority_delay[idx],
            }
        )
    for idx, label in enumerate(region_labels):
        summary_rows.extend(
            [
                {
                    "group": "region",
                    "region": label,
                    "priority": "high",
                    "arrived": high_arrived[idx],
                    "completed": high_completed[idx],
                    "completion_rate": high_rate[idx],
                    "delay_ms": high_delay[idx],
                },
                {
                    "group": "region",
                    "region": label,
                    "priority": "low",
                    "arrived": low_arrived[idx],
                    "completed": low_completed[idx],
                    "completion_rate": low_rate[idx],
                    "delay_ms": low_delay[idx],
                },
                {
                    "group": "region",
                    "region": label,
                    "priority": "all",
                    "arrived": total_arrived[idx],
                    "completed": total_completed[idx],
                    "completion_rate": total_rate[idx],
                    "delay_ms": total_delay[idx],
                },
            ]
        )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=160)
    x = np.arange(2)
    axes[0].bar(x, priority_rate, width=0.58, color=["#1f77b4", "#ff7f0e"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["High", "Low"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Priority Completion Rate")
    axes[0].set_ylabel("ratio")
    set_axis_style(axes[0])

    axes[1].bar(x, priority_delay, width=0.58, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["High", "Low"])
    axes[1].set_title("Priority E2E Delay")
    axes[1].set_ylabel("ms")
    set_axis_style(axes[1])

    width = 0.34
    axes[2].bar(x - width / 2, priority_arrived, width=width, label="arrived")
    axes[2].bar(x + width / 2, priority_completed, width=width, label="completed")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["High", "Low"])
    axes[2].set_title("Priority Task Volume")
    axes[2].set_ylabel("tasks")
    axes[2].legend(frameon=False, fontsize=8)
    set_axis_style(axes[2])
    fig.suptitle(f"High/Low Priority Task Results (tail {tail_window} evals)")
    fig.tight_layout()
    path = output_dir / "priority_task_results.png"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), dpi=160)
    x = np.arange(len(regions))
    width = 0.26
    axes[0].bar(x - width, high_rate, width=width, label="high")
    axes[0].bar(x, low_rate, width=width, label="low")
    axes[0].bar(x + width, total_rate, width=width, label="all")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(region_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Completion Rate by Source Zone")
    axes[0].set_ylabel("ratio")
    axes[0].legend(frameon=False, fontsize=8)
    set_axis_style(axes[0])

    axes[1].bar(x - width / 2, high_delay, width=width, label="high")
    axes[1].bar(x + width / 2, low_delay, width=width, label="low")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(region_labels)
    axes[1].set_title("E2E Delay by Source Zone")
    axes[1].set_ylabel("ms")
    axes[1].legend(frameon=False, fontsize=8)
    set_axis_style(axes[1])

    axes[2].bar(x - width / 2, total_arrived, width=width, label="arrived")
    axes[2].bar(x + width / 2, total_completed, width=width, label="completed")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(region_labels)
    axes[2].set_title("Task Volume by Source Zone")
    axes[2].set_ylabel("tasks")
    axes[2].legend(frameon=False, fontsize=8)
    set_axis_style(axes[2])
    fig.suptitle(f"Regional Task Results (tail {tail_window} evals)")
    fig.tight_layout()
    path = output_dir / "region_task_results.png"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    if coop_flow is not None:
        fig, axis = plt.subplots(figsize=(5.8, 4.8), dpi=160)
        image = axis.imshow(coop_flow, cmap="Blues")
        axis.set_title("Cooperation Flow Matrix")
        axis.set_xlabel("Processing UAV")
        axis.set_ylabel("Source Zone")
        axis.set_xticks(regions)
        axis.set_yticks(regions)
        axis.set_xticklabels([f"UAV {idx + 1}" for idx in regions])
        axis.set_yticklabels(region_labels)
        max_value = max(float(np.max(coop_flow)), 1e-12)
        for row_idx in regions:
            for col_idx in regions:
                value = float(coop_flow[row_idx, col_idx])
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.62 * max_value else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Mbits")
        fig.tight_layout()
        path = output_dir / "cooperation_flow_matrix.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    write_priority_region_summary(output_dir / "priority_region_summary.csv", summary_rows)
    return saved


def write_cross_summary(path, cross_rows):
    fields = [
        "cpu_ghz",
        "load_scale",
        "J",
        "task_delay_weighted_ms",
        "task_delay_high_ms",
        "task_delay_low_ms",
        "energy",
        "energy_total_episode",
        "cpu_saturation",
        "task_completion_rate",
        "task_qos_violation_rate",
        "coop_ratio",
        "u2g",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in cross_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    args = parse_args()
    result_dir = resolve_path(args.result_dir)
    cross_csv = resolve_path(args.cross_csv) if args.cross_csv else result_dir / "cross_cpu_load_eval_best.csv"
    output_dir = resolve_path(args.output_dir) if args.output_dir else result_dir / "paper_figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if cross_csv.is_file():
        cross_rows = aggregate_cross_rows(read_csv_rows(cross_csv))
        write_cross_summary(output_dir / "cross_cpu_load_summary.csv", cross_rows)
        saved.extend(plot_cpu_sweep(cross_rows, output_dir))
        saved.extend(plot_load_sweep(cross_rows, output_dir))
        saved.extend(plot_cross_heatmaps(cross_rows, output_dir))
    else:
        print(f"Missing cross eval CSV: {cross_csv}")

    eval_csv = result_dir / "eval_central_macro.csv"
    if eval_csv.is_file():
        saved.extend(plot_convergence(eval_csv, output_dir, args.smooth_window))
    else:
        print(f"Missing convergence CSV: {eval_csv}")

    detail_npz = result_dir / "eval_central_macro_detail.npz"
    saved.extend(plot_priority_and_region(detail_npz, output_dir, args.tail_window))

    print(f"Saved experiment figures -> {output_dir}")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
