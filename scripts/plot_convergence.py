import argparse
import csv
import os
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def configure_chinese_font():
    candidates = ("Microsoft YaHei", "SimHei", "STSong")
    installed = {font.name for font in fm.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training log: {path}")
    return rows


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    width = min(max(int(window), 1), values.size)
    kernel = np.ones(width, dtype=np.float64) / width
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = np.full(width - 1, np.nan, dtype=np.float64)
    return np.concatenate([prefix, smoothed])


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def plot_series(axis, episodes, values, window, color, label):
    axis.plot(
        episodes,
        values,
        color=color,
        alpha=0.22,
        linewidth=0.8,
        label=f"{label}（逐轮真实值）",
    )
    axis.plot(
        episodes,
        moving_average(values, window),
        color=color,
        linewidth=2.0,
        label=f"{label}（{window}轮滑动平均）",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["smoke", "diagnostic", "pilot", "formal"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--results-version", default="joint_hrl_v1")
    parser.add_argument(
        "--micro-results-version",
        default="",
        help="Results version containing the reused SAC pretraining log",
    )
    args = parser.parse_args()

    joint_root = os.path.join(
        PROJECT_ROOT,
        "results",
        args.results_version,
        args.stage,
        f"seed{args.seed}",
    )
    micro_root = os.path.join(
        PROJECT_ROOT,
        "results",
        args.micro_results_version or args.results_version,
        args.stage,
        f"seed{args.seed}",
    )
    micro_path = os.path.join(
        micro_root,
        "micro_pretrain",
        "train_centralized.csv",
    )
    joint_path = os.path.join(
        joint_root,
        "joint_td3_sac",
        "train_td3.csv",
    )
    if not os.path.isfile(micro_path):
        raise FileNotFoundError(micro_path)
    if not os.path.isfile(joint_path):
        raise FileNotFoundError(joint_path)

    micro_rows = read_csv(micro_path)
    joint_rows = read_csv(joint_path)
    micro_episodes = numeric(micro_rows, "episode")
    joint_episodes = numeric(joint_rows, "episode")
    service_decisions = np.maximum(
        numeric(micro_rows, "service_resource_decisions"),
        1.0,
    )
    macro_periods = service_decisions / 18.0
    micro_steps_per_macro = np.maximum(
        numeric(micro_rows, "macro_micro_steps_max"),
        1.0,
    )
    micro_steps = np.maximum(
        macro_periods * micro_steps_per_macro,
        1.0,
    )
    macro_steps = max(
        float(joint_rows[0]["service_resource_decisions"]) / 18.0,
        1.0,
    )
    micro_returns = numeric(micro_rows, "micro_reward") / micro_steps
    macro_returns = numeric(joint_rows, "macro_actor_reward") / macro_steps

    configure_chinese_font()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    plot_series(
        axes[0],
        micro_episodes,
        micro_returns,
        args.window,
        "#2878B5",
        "微观回报",
    )
    axes[0].set_title("(a) SAC微观层预训练")
    axes[0].set_xlabel("预训练轮次")
    axes[0].set_ylabel("每0.1 s积分步真实训练回报")

    plot_series(
        axes[1],
        joint_episodes,
        macro_returns,
        args.window,
        "#C82423",
        "宏观回报",
    )
    update_counts = numeric(joint_rows, "td3_updates")
    first_update = np.flatnonzero(update_counts > 0)
    if first_update.size:
        warmup_episode = joint_episodes[first_update[0]]
        axes[1].axvline(
            warmup_episode,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"联合更新开始（第{int(warmup_episode)}轮）",
        )
    axes[1].set_title("(b) TD3-SAC联合训练")
    axes[1].set_xlabel("联合训练轮次")
    axes[1].set_ylabel("每宏观周期真实训练回报")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle("双时间尺度分层强化学习训练过程")
    figure.tight_layout()
    output_dir = os.path.join(joint_root, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_convergence.png")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        os.path.join(output_dir, "training_convergence.pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
