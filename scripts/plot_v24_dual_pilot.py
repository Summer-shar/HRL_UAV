import argparse
import csv
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())
ON_TRAIN = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/train_td3.csv"
)
OFF_TRAIN = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_off/pilot/seed1/"
    "joint_td3_sac/train_td3.csv"
)
AUDIT_RAW = "result_archive/v24_system_dual_pilot_trajectory_audit/per_seed_raw.csv"
DEFAULT_OUTPUT = (
    "result_archive/v24_system_dual_pilot_trajectory_audit/"
    "dual_pilot_mechanism"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-on-train", default=ON_TRAIN)
    parser.add_argument("--dual-off-train", default=OFF_TRAIN)
    parser.add_argument("--audit-raw", default=AUDIT_RAW)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_fonts():
    names = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "STSong"):
        if candidate in names:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_train(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "episode": np.asarray([int(row["episode"]) for row in rows]),
        "power": np.asarray([float(row["average_power_w"]) for row in rows]),
        "smoothed_power": np.asarray(
            [float(row["dual_update_power_w"]) for row in rows]
        ),
        "dual": np.asarray([float(row["energy_dual_after"]) for row in rows]),
    }


def read_audit(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {"dual_on": {}, "dual_off": {}}
    for row in rows:
        grouped[row["arm"]][int(row["seed"])] = row
    if set(grouped["dual_on"]) != set(grouped["dual_off"]):
        raise RuntimeError("Dual-on/off seed sets do not match.")
    return grouped


def paired_panel(ax, grouped, metric, ylabel, panel_title, colors):
    seeds = sorted(grouped["dual_on"])
    on = np.asarray(
        [float(grouped["dual_on"][seed][metric]) for seed in seeds]
    )
    off = np.asarray(
        [float(grouped["dual_off"][seed][metric]) for seed in seeds]
    )
    for left, right in zip(off, on):
        ax.plot((0, 1), (left, right), color="#B8B8B8", linewidth=0.8, alpha=0.65)
    ax.scatter(np.zeros_like(off), off, s=26, color=colors["off"], zorder=3)
    ax.scatter(np.ones_like(on), on, s=26, color=colors["on"], zorder=3)
    ax.scatter(
        (0, 1),
        (np.mean(off), np.mean(on)),
        s=95,
        marker="D",
        edgecolor="black",
        linewidth=0.8,
        color=(colors["off"], colors["on"]),
        zorder=4,
    )
    ax.set_xticks((0, 1), ("关闭对偶", "开启对偶"))
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)


def main():
    args = parse_args()
    configure_fonts()
    colors = {"on": "#C44E52", "off": "#4C72B0"}
    on = read_train(project_path(args.dual_on_train))
    off = read_train(project_path(args.dual_off_train))
    grouped = read_audit(project_path(args.audit_raw))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(off["episode"], off["power"], color=colors["off"], alpha=0.18, linewidth=0.8)
    ax.plot(on["episode"], on["power"], color=colors["on"], alpha=0.18, linewidth=0.8)
    ax.plot(
        off["episode"],
        off["smoothed_power"],
        color=colors["off"],
        linewidth=1.8,
        label="关闭对偶：20轮均值",
    )
    ax.plot(
        on["episode"],
        on["smoothed_power"],
        color=colors["on"],
        linewidth=1.8,
        label="开启对偶：20轮均值",
    )
    ax.axhline(840.0, color="#222222", linestyle="--", linewidth=1.1, label="840 W预算")
    ax.axvline(50.5, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_xlabel("联合训练轮次")
    ax.set_ylabel("系统平均功率（W）")
    ax.set_title("(a) 功率跟踪", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(alpha=0.22)

    ax = axes[0, 1]
    ax.plot(off["episode"], off["dual"], color=colors["off"], linewidth=1.6, label="关闭对偶")
    ax.plot(on["episode"], on["dual"], color=colors["on"], linewidth=1.8, label="开启对偶")
    ax.axvline(50.5, color="#777777", linestyle=":", linewidth=1.0, label="结束warm-up")
    ax.set_xlabel("联合训练轮次")
    ax.set_ylabel("系统功率乘子 λ")
    ax.set_title("(b) 乘子响应", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.22)

    paired_panel(
        axes[1, 0],
        grouped,
        "mean_centroid_distance_m",
        "平均距动态业务质心（m）",
        "(c) 质心偏离（20个共同种子）",
        colors,
    )
    paired_panel(
        axes[1, 1],
        grouped,
        "path_total_m",
        "三机累计路径（m）",
        "(d) 累计机动（20个共同种子）",
        colors,
    )

    fig.suptitle("系统功率对偶机制与轨迹行为审计", fontsize=15, fontweight="bold")
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output.with_suffix('.png')}")
    print(f"Saved {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
