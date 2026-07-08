"""
论文训练曲线绘图: 多 seed 的 raw + rolling mean + std shadow
读取 paper_batch/seed*/method/ 下的 eval CSV, 汇总后画图

用法:
    python scripts/plot_paper_curves.py
    python scripts/plot_paper_curves.py --metric J
    python scripts/plot_paper_curves.py --output results/paper_figures/convergence.png
"""
import argparse
import glob
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
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("matplotlib not found, please install it.")
    sys.exit(1)


# ===== 方法配置 =====
METHODS = {
    "hrl": {
        "label": "HRL (MAPPO+SAC)",
        "color": "#E63946",
        "csv_pattern": "eval_*.csv",
        "linestyle": "-",
    },
    "fixed_pos": {
        "label": "Fixed-Pos + SAC",
        "color": "#457B9D",
        "csv_pattern": "eval_fixed_pos.csv",
        "linestyle": "--",
    },
    "no_coop": {
        "label": "MAPPO + No-Coop",
        "color": "#2A9D8F",
        "csv_pattern": "eval_no_coop.csv",
        "linestyle": "-.",
    },
    "random": {
        "label": "Random",
        "color": "#8D99AE",
        "csv_pattern": "eval_random.csv",
        "linestyle": ":",
    },
    "macro_only": {
        "label": "MAPPO-only (heuristic micro)",
        "color": "#F4A261",
        "csv_pattern": "eval_central_macro.csv",
        "linestyle": "--",
    },
}


def load_eval_csv(csv_path):
    """Load evaluation CSV and return dict of arrays keyed by column name."""
    import csv as csv_mod
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(np.nan)
    return {k: np.array(v) for k, v in data.items()}


def find_eval_csvs(batch_dir, method_key):
    """Find all eval CSVs for a method across seeds."""
    pattern = os.path.join(batch_dir, "seed*", method_key, "*.csv")
    # Also look for specific named patterns
    csv_pattern = METHODS[method_key]["csv_pattern"]
    specific_pattern = os.path.join(batch_dir, "seed*", method_key, csv_pattern)

    candidates = glob.glob(specific_pattern)
    if not candidates:
        # Try broader search
        candidates = glob.glob(os.path.join(batch_dir, "seed*", method_key, "eval_*.csv"))

    # For HRL, also check for train_episode_metrics_from_log.csv
    if method_key == "hrl" and not candidates:
        candidates = glob.glob(
            os.path.join(batch_dir, "seed*", method_key, "train_episode_metrics_from_log.csv")
        )

    return sorted(candidates)


def collect_method_data(batch_dir, method_key, metric):
    """Collect metric across all seeds for a method. Returns (episodes, values_per_seed)."""
    csvs = find_eval_csvs(batch_dir, method_key)
    if not csvs:
        return None, None

    all_data = []
    for csv_path in csvs:
        data = load_eval_csv(csv_path)
        if "episode" not in data or metric not in data:
            continue
        episodes = data["episode"]
        values = data[metric]
        all_data.append((episodes, values))

    if not all_data:
        return None, None

    # Align on common episode points
    ref_episodes = all_data[0][0]
    aligned_values = []
    for episodes, values in all_data:
        if len(episodes) == len(ref_episodes) and np.allclose(episodes, ref_episodes):
            aligned_values.append(values)
        else:
            # Interpolate to reference episodes
            interp = np.interp(ref_episodes, episodes, values, left=np.nan, right=np.nan)
            aligned_values.append(interp)

    return ref_episodes, np.array(aligned_values)


def plot_convergence(batch_dir, metric="J", output_path=None, title=None, ylabel=None):
    """Plot convergence curves with mean ± std across seeds."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    has_data = False
    for method_key, method_cfg in METHODS.items():
        episodes, values = collect_method_data(batch_dir, method_key, metric)
        if episodes is None or values is None or len(values) == 0:
            continue

        has_data = True
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        n_seeds = values.shape[0]

        ax.plot(
            episodes, mean,
            color=method_cfg["color"],
            linestyle=method_cfg["linestyle"],
            linewidth=2.0,
            label=f"{method_cfg['label']} ({n_seeds} seeds)",
        )
        ax.fill_between(
            episodes, mean - std, mean + std,
            alpha=0.15,
            color=method_cfg["color"],
        )

    if not has_data:
        print(f"No data found in {batch_dir}")
        plt.close(fig)
        return

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel(ylabel or metric, fontsize=12)
    ax.set_title(title or f"Training Convergence: {metric}", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        default_path = os.path.join(
            PROJECT_ROOT, "results", "paper_figures",
            f"convergence_{metric}.png"
        )
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        fig.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {default_path}")
    plt.close(fig)


def plot_bar_comparison(batch_dir, metrics=None, output_path=None):
    """Plot final performance bar chart comparing all methods."""
    if metrics is None:
        metrics = ["J", "task_delay_high_ms", "task_delay_low_ms", "task_completion_rate"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    method_names = []
    method_colors = []
    for method_key, method_cfg in METHODS.items():
        method_names.append(method_cfg["label"])
        method_colors.append(method_cfg["color"])

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        colors = []
        labels = []

        for method_key, method_cfg in METHODS.items():
            episodes, values = collect_method_data(batch_dir, method_key, metric)
            if episodes is None or values is None or len(values) == 0:
                continue
            # Take the last evaluation point as "final"
            final_values = values[:, -1]
            means.append(np.nanmean(final_values))
            stds.append(np.nanstd(final_values))
            colors.append(method_cfg["color"])
            labels.append(method_cfg["label"])

        if not means:
            continue

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "results", "paper_figures", "bar_comparison.png"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_raw_training_curve(csv_path, metric="J", window=20, output_path=None, title=None):
    """Plot a single run's raw per-episode training curve with rolling mean + std shadow."""
    data = load_eval_csv(csv_path)
    if "episode" not in data or metric not in data:
        print(f"Metric '{metric}' not found in {csv_path}")
        return

    episodes = data["episode"]
    values = data[metric]
    valid = ~np.isnan(values)
    episodes = episodes[valid]
    values = values[valid]

    if len(values) < window:
        print(f"Not enough data points ({len(values)}) for window={window}")
        return

    # Rolling statistics
    rolling_mean = np.convolve(values, np.ones(window) / window, mode="valid")
    rolling_episodes = episodes[window - 1:]

    # Rolling std
    rolling_std = np.array([
        np.std(values[max(0, i - window + 1):i + 1])
        for i in range(window - 1, len(values))
    ])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(episodes, values, alpha=0.3, color="#AAAAAA", linewidth=0.5, label="Raw")
    ax.plot(rolling_episodes, rolling_mean, color="#E63946", linewidth=2.0,
            label=f"Rolling Mean (w={window})")
    ax.fill_between(
        rolling_episodes,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.2, color="#E63946", label="±1 Std"
    )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title or f"Training Curve: {metric}", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "results", "paper_figures",
            f"raw_training_{metric}.png"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot paper experiment curves")
    parser.add_argument("--batch-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "paper_batch"),
                        help="Root directory of batch experiment results")
    parser.add_argument("--metric", type=str, default="J",
                        help="Metric to plot for convergence (J, task_delay_high_ms, etc.)")
    parser.add_argument("--output", type=str, default="",
                        help="Output file path for convergence plot")
    parser.add_argument("--raw-csv", type=str, default="",
                        help="Path to a single training CSV for raw curve plot")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling window size for raw curve plot")
    parser.add_argument("--bar", action="store_true",
                        help="Also produce a bar comparison chart")
    args = parser.parse_args()

    if args.raw_csv:
        plot_raw_training_curve(
            args.raw_csv,
            metric=args.metric,
            window=args.window,
            output_path=args.output or None,
        )
        return

    # Convergence plot
    plot_convergence(
        args.batch_dir,
        metric=args.metric,
        output_path=args.output or None,
        ylabel="System Cost $J$" if args.metric == "J" else args.metric,
    )

    # Also plot delay metrics
    for extra_metric in ["task_delay_high_ms", "task_delay_low_ms"]:
        if extra_metric != args.metric:
            plot_convergence(
                args.batch_dir,
                metric=extra_metric,
                ylabel=extra_metric.replace("_", " ").title(),
            )

    if args.bar:
        plot_bar_comparison(args.batch_dir)


if __name__ == "__main__":
    main()
