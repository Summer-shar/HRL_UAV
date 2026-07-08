"""
论文收敛曲线: 每轮真实训练 J + 平滑滑动平均 + 阴影标准差
画出来是一条平滑曲线覆盖在淡色散点上，不是锯齿折线。

用法:
    python scripts/plot_training_convergence.py --csv results/paper_batch/seed1/hrl/train_sac.csv
    python scripts/plot_training_convergence.py --batch results/paper_batch   # 多seed汇总
"""
import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
_cn_font_set = False
for _fname in ["SimHei", "Microsoft YaHei", "STSong", "FangSong"]:
    if any(_fname.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_fname, "DejaVu Sans"]
        _cn_font_set = True
        break
if not _cn_font_set:
    _sys_font = "C:/Windows/Fonts/simhei.ttf"
    if os.path.isfile(_sys_font):
        fm.fontManager.addfont(_sys_font)
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_train_csv(csv_path, metric="micro_cost_raw"):
    """Load per-episode training metric from train_sac.csv."""
    import csv as csv_mod
    episodes = []
    values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            if metric not in row:
                continue
            try:
                ep = int(row.get("episode", 0))
                val = float(row[metric])
                episodes.append(ep)
                values.append(val)
            except (ValueError, TypeError):
                continue
    return np.array(episodes), np.array(values)


def smooth_curve(values, window):
    """Compute rolling mean and rolling std with given window."""
    n = len(values)
    if n < window:
        window = max(n, 1)

    # Causal rolling (only look back, no future info)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        segment = values[start:i+1]
        mean[i] = np.mean(segment)
        std[i] = np.std(segment)

    return mean, std


def plot_single_run(csv_path, metric="micro_cost_raw", window=20, output_path=None,
                    title=None, ylabel=None, color="#E63946"):
    """
    画单次训练的收敛曲线:
    - 淡色散点: 每轮真实值
    - 粗实线: 滑动平均 (平滑曲线)
    - 阴影带: ±1 std
    """
    episodes, values = load_train_csv(csv_path, metric)
    if len(episodes) == 0:
        print(f"No data for metric '{metric}' in {csv_path}")
        return

    mean, std = smooth_curve(values, window)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Raw values as faint scatter
    ax.scatter(episodes, values, color=color, alpha=0.12, s=8, linewidths=0, zorder=1)

    # Smooth curve
    ax.plot(episodes, mean, color=color, linewidth=2.2, zorder=3, label=f"滑动平均 (w={window})")

    # Std shadow
    ax.fill_between(
        episodes,
        mean - std,
        mean + std,
        alpha=0.15, color=color, zorder=2,
    )

    ax.set_xlabel("训练轮次 (Episode)", fontsize=12)
    ax.set_ylabel(ylabel or "系统代价 J", fontsize=12)
    ax.set_title(title or "训练收敛曲线", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(episodes[0], episodes[-1])

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "results", "paper_figures", "training_convergence.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_multi_seed(batch_dir, method="hrl", metric="micro_cost_raw", window=20,
                    output_path=None, title=None, ylabel=None):
    """
    多 seed 汇总: 找到所有 seed 的 train_sac.csv, 对齐后画 mean±std
    - 淡色细线: 每个 seed 的滑动平均
    - 粗实线: 多 seed 的均值
    - 阴影: 多 seed 的 std (代表可重复性)
    """
    # Find all train CSVs
    patterns = [
        os.path.join(batch_dir, f"seed*/{method}/train_sac.csv"),
        os.path.join(batch_dir, f"seed*/{method}/train_episode_metrics_from_log.csv"),
    ]
    csv_files = []
    for pat in patterns:
        csv_files.extend(glob.glob(pat))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        print(f"No training CSVs found for method={method} in {batch_dir}")
        return

    # Load all seeds
    all_values = []
    ref_episodes = None
    for csv_path in csv_files:
        episodes, values = load_train_csv(csv_path, metric)
        if len(episodes) == 0:
            continue
        if ref_episodes is None:
            ref_episodes = episodes
        # Align to ref length
        min_len = min(len(ref_episodes), len(values))
        all_values.append(values[:min_len])

    if not all_values:
        print(f"No data found for metric={metric}")
        return

    min_len = min(len(v) for v in all_values)
    ref_episodes = ref_episodes[:min_len]
    all_values = np.array([v[:min_len] for v in all_values])  # (n_seeds, n_episodes)
    n_seeds = all_values.shape[0]

    # Smooth each seed
    smoothed = np.array([smooth_curve(v, window)[0] for v in all_values])

    # Multi-seed statistics
    multi_mean = np.nanmean(smoothed, axis=0)
    multi_std = np.nanstd(smoothed, axis=0)

    color_main = "#E63946"
    color_seeds = "#AAAAAA"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Individual seed curves (thin, faint)
    for i in range(n_seeds):
        ax.plot(ref_episodes, smoothed[i], color=color_seeds, linewidth=0.8, alpha=0.5,
                label=f"Seed {i+1}" if i == 0 else None)

    # Mean curve
    ax.plot(ref_episodes, multi_mean, color=color_main, linewidth=2.5, zorder=5,
            label=f"均值 ({n_seeds} seeds)")

    # Std shadow
    ax.fill_between(
        ref_episodes,
        multi_mean - multi_std,
        multi_mean + multi_std,
        alpha=0.2, color=color_main, zorder=4,
        label="±1σ",
    )

    ax.set_xlabel("训练轮次 (Episode)", fontsize=12)
    ax.set_ylabel(ylabel or "系统代价 J", fontsize=12)
    ax.set_title(title or f"训练收敛曲线 ({n_seeds} seeds)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(ref_episodes[0], ref_episodes[-1])

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "results", "paper_figures",
                                   f"training_convergence_{method}_{n_seeds}seeds.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_multi_method_comparison(batch_dir, methods=None, metric="micro_cost_raw",
                                  window=20, output_path=None, title=None, ylabel=None):
    """
    多方法对比: 每种方法的多 seed 均值曲线画在同一张图上
    """
    if methods is None:
        methods = {
            "hrl": {"label": "HRL (MAPPO+SAC)", "color": "#E63946"},
            "fixed_pos": {"label": "Fixed-Pos+SAC", "color": "#457B9D"},
            "no_coop": {"label": "MAPPO+No-Coop", "color": "#2A9D8F"},
            "random": {"label": "Random", "color": "#8D99AE"},
            "macro_only": {"label": "MAPPO-only", "color": "#F4A261"},
        }

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    has_data = False

    for method_key, cfg in methods.items():
        # Find CSVs
        patterns = [
            os.path.join(batch_dir, f"seed*/{method_key}/train_sac.csv"),
            os.path.join(batch_dir, f"seed*/{method_key}/train_episode_metrics_from_log.csv"),
            os.path.join(batch_dir, f"seed*/{method_key}/eval_*.csv"),
        ]
        csv_files = []
        for pat in patterns:
            csv_files.extend(glob.glob(pat))
        csv_files = sorted(set(csv_files))

        if not csv_files:
            continue

        all_values = []
        ref_episodes = None
        for csv_path in csv_files:
            episodes, values = load_train_csv(csv_path, metric)
            if len(episodes) == 0:
                continue
            if ref_episodes is None:
                ref_episodes = episodes
            min_len = min(len(ref_episodes), len(values))
            all_values.append(values[:min_len])

        if not all_values:
            continue

        min_len = min(len(v) for v in all_values)
        ref_episodes = ref_episodes[:min_len]
        all_values_arr = np.array([v[:min_len] for v in all_values])
        n_seeds = all_values_arr.shape[0]

        smoothed = np.array([smooth_curve(v, window)[0] for v in all_values_arr])
        multi_mean = np.nanmean(smoothed, axis=0)
        multi_std = np.nanstd(smoothed, axis=0)

        ax.plot(ref_episodes, multi_mean, color=cfg["color"], linewidth=2.2,
                label=f"{cfg['label']} ({n_seeds}s)", zorder=5)
        ax.fill_between(ref_episodes, multi_mean - multi_std, multi_mean + multi_std,
                        alpha=0.12, color=cfg["color"], zorder=4)
        has_data = True

    if not has_data:
        print("No data found for any method")
        plt.close(fig)
        return

    ax.set_xlabel("训练轮次 (Episode)", fontsize=12)
    ax.set_ylabel(ylabel or "系统代价 J", fontsize=12)
    ax.set_title(title or "多方法训练收敛对比", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "results", "paper_figures",
                                   "training_convergence_comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot smooth training convergence curves")
    parser.add_argument("--csv", type=str, default="",
                        help="Single train_sac.csv path (single-run plot)")
    parser.add_argument("--batch", type=str, default="",
                        help="Batch results directory (multi-seed plot)")
    parser.add_argument("--method", type=str, default="hrl",
                        help="Method to plot when using --batch (single method multi-seed)")
    parser.add_argument("--compare", action="store_true",
                        help="Plot all methods on one figure")
    parser.add_argument("--metric", type=str, default="micro_cost_raw",
                        help="Column name in CSV (micro_cost_raw = per-step J)")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling average window size")
    parser.add_argument("--output", type=str, default="",
                        help="Output image path")
    args = parser.parse_args()

    if args.csv:
        plot_single_run(
            args.csv, metric=args.metric, window=args.window,
            output_path=args.output or None,
        )
    elif args.batch and args.compare:
        plot_multi_method_comparison(
            args.batch, metric=args.metric, window=args.window,
            output_path=args.output or None,
        )
    elif args.batch:
        plot_multi_seed(
            args.batch, method=args.method, metric=args.metric, window=args.window,
            output_path=args.output or None,
        )
    else:
        # Default: try to find existing train CSV
        default_csv = os.path.join(
            PROJECT_ROOT, "results", "auto_prioritydemand_base_best300", "train_sac.csv"
        )
        if os.path.isfile(default_csv):
            plot_single_run(default_csv, metric=args.metric, window=args.window)
        else:
            print("Usage:")
            print("  python plot_training_convergence.py --csv path/to/train_sac.csv")
            print("  python plot_training_convergence.py --batch results/paper_batch --compare")


if __name__ == "__main__":
    main()
