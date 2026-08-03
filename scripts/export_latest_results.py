from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "results"
    / "tcom_dynamic_u2u_sparse_v23_target_noise"
    / "pilot"
    / "seed1"
    / "joint_td3_sac"
)
AUDIT_JSON = RUN_DIR / "position_reference_audit" / "position_reference_audit.json"
EVAL_CSV = RUN_DIR / "eval_central_td3.csv"
CONFIG_JSON = ROOT / "configs" / "tcom_dynamic_u2u_sparse_v23_target_noise.json"
OUTPUT_DIR = ROOT / "latest_results"


def configure_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "STSong"]
    installed = {font.name.lower(): font.name for font in fm.fontManager.ttflist}
    for candidate in candidates:
        if candidate.lower() in installed:
            plt.rcParams["font.sans-serif"] = [installed[candidate.lower()], "DejaVu Sans"]
            break
    else:
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if font_path.is_file():
            fm.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def load_eval_rows() -> list[dict[str, float]]:
    with EVAL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = [
        "episode",
        "J",
        "task_delay_weighted_ms",
        "average_power_w",
        "average_power_ucb95_w",
        "u2g",
        "coop_ratio",
    ]
    return [{field: float(row[field]) for field in fields} for row in rows]


def annotate_endpoints(ax: plt.Axes, x: np.ndarray, y: np.ndarray, unit: str) -> None:
    ax.annotate(
        f"{y[0]:.1f}{unit}",
        (x[0], y[0]),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate(
        f"{y[-1]:.1f}{unit}",
        (x[-1], y[-1]),
        xytext=(-5, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )


def plot_macro_training_diagnostic(rows: list[dict[str, float]]) -> None:
    episodes = np.asarray([row["episode"] for row in rows])
    delay = np.asarray([row["task_delay_weighted_ms"] for row in rows])
    u2g = np.asarray([row["u2g"] for row in rows])
    power_ucb = np.asarray([row["average_power_ucb95_w"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    axes[0].plot(episodes, delay, color="#1f77b4", marker="o", markersize=3.5, linewidth=1.8)
    axes[0].set_title("(a) 加权端到端时延")
    axes[0].set_ylabel("时延 (ms，越低越好)")
    annotate_endpoints(axes[0], episodes, delay, "")

    axes[1].plot(episodes, u2g, color="#d95f02", marker="o", markersize=3.5, linewidth=1.8)
    axes[1].set_title("(b) 平均 U2G 距离")
    axes[1].set_ylabel("距离 (m)")
    annotate_endpoints(axes[1], episodes, u2g, "")

    axes[2].plot(
        episodes,
        power_ucb,
        color="#2ca02c",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="平均功率 UCB95",
    )
    axes[2].axhline(840.0, color="#b2182b", linestyle="--", linewidth=1.5, label="功率预算 840 W")
    axes[2].set_title("(c) 功率约束")
    axes[2].set_ylabel("系统平均功率 (W)")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")

    for ax in axes:
        ax.set_xlabel("宏观训练轮次")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("v23 宏观 TD3 训练诊断（seed=1，400 轮）", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "01_macro_training_diagnostic.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "01_macro_training_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_position_reference_comparison(audit: dict) -> None:
    keys = ["fixed_anchors", "traffic_centroid", "uplink_reference", "joint_search"]
    labels = ["固定锚点", "动态质心", "上行参考点", "联合搜索"]
    colors = ["#777777", "#1f77b4", "#2ca02c", "#d95f02"]
    validation = audit["validation"]

    delays = np.asarray([validation[key]["task_delay_weighted_ms"] for key in keys])
    delay_std = np.asarray([validation[key]["task_delay_weighted_ms_std"] for key in keys])
    counts = np.asarray([validation[key]["sample_count"] for key in keys])
    delay_ci95 = 1.96 * delay_std / np.sqrt(counts)
    powers = np.asarray([validation[key]["average_power_w"] for key in keys])
    power_ucb = np.asarray([validation[key]["average_power_ucb95_w"] for key in keys])

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)

    axes[0].bar(x, delays, color=colors, width=0.62, alpha=0.9)
    axes[0].errorbar(x, delays, yerr=delay_ci95, fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    axes[0].set_ylabel("加权端到端时延 (ms)")
    axes[0].set_title("(a) 时延对比（均值及 95% CI）")
    axes[0].set_ylim(min(delays) - 14.0, max(delays) + 18.0)
    for xi, value in zip(x, delays):
        axes[0].text(xi, value + 4.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, powers, color=colors, width=0.62, alpha=0.9, label="样本均值")
    axes[1].scatter(x, power_ucb, marker="D", color="#111111", s=30, zorder=3, label="UCB95")
    axes[1].axhline(840.0, color="#b2182b", linestyle="--", linewidth=1.5, label="功率预算")
    axes[1].set_ylabel("系统平均功率 (W)")
    axes[1].set_title("(b) 功率可行性")
    axes[1].set_ylim(min(powers) - 8.0, 845.0)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("位置参考策略对比（20 个共同随机种子）", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "02_position_reference_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "02_position_reference_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_trajectory_figures() -> None:
    source_dir = RUN_DIR / "trajectory_snapshots"
    mapping = {
        "trajectory_best.png": "03_trajectory_best.png",
        "trajectory_ep_0100.png": "04_trajectory_ep0100.png",
        "trajectory_ep_0200.png": "05_trajectory_ep0200.png",
        "trajectory_ep_0300.png": "06_trajectory_ep0300.png",
        "trajectory_ep_0400.png": "07_trajectory_ep0400.png",
        "trajectory_final.png": "08_trajectory_final.png",
    }
    for source_name, output_name in mapping.items():
        shutil.copy2(source_dir / source_name, OUTPUT_DIR / output_name)


def copy_source_data() -> None:
    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EVAL_CSV, data_dir / "eval_central_td3.csv")
    shutil.copy2(AUDIT_JSON, data_dir / "position_reference_audit.json")
    shutil.copy2(CONFIG_JSON, data_dir / "tcom_dynamic_u2u_sparse_v23_target_noise.json")


def main() -> None:
    required = [RUN_DIR, AUDIT_JSON, EVAL_CSV, CONFIG_JSON]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    rows = load_eval_rows()
    with AUDIT_JSON.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    plot_macro_training_diagnostic(rows)
    plot_position_reference_comparison(audit)
    copy_trajectory_figures()
    copy_source_data()
    print(f"Exported latest figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
