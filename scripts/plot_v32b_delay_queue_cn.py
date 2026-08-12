import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import make_episode_load_switch_schedule
from run_load_switch_gate import run_gate_episode
from search_position_oracle import load_checkpoint_components
from train_centralized_td3 import td3_macro_policy


POLICIES = ("TD3", "P0", "P1_prime")
COLORS = {"TD3": "#2878B5", "P0": "#777777", "P1_prime": "#E17C05"}
LABELS = {
    "TD3": "分层 TD3-SAC",
    "P0": "业务质心跟踪 P0",
    "P1_prime": "因果支援规则 P1′",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Chinese v32b end-to-end delay and queue figures."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/tcom_dynamic_load_switch_v32b_formal/formal/seed1/"
            "joint_td3_sac/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--selection",
        default="result_archive/p1_prime_prescreen/selection.json",
    )
    parser.add_argument(
        "--archive-dir",
        default="result_archive/v32b_matched_reference/seed1",
    )
    parser.add_argument("--seed", type=int, default=1012)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def confidence_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    half_width = float(
        stats.t.ppf(0.975, values.size - 1) * stats.sem(values)
    )
    return mean, mean - half_width, mean + half_width


def plot_delay(rows, output_dir):
    configure_chinese_font()
    fig, axis = plt.subplots(figsize=(8.1, 5.4), constrained_layout=True)
    rng = np.random.default_rng(20260812)
    positions = np.arange(len(POLICIES), dtype=np.float64)

    for index, policy in enumerate(POLICIES):
        values = np.asarray(
            [
                float(row["task_delay_weighted_ms"])
                for row in rows
                if row["policy"] == policy
            ],
            dtype=np.float64,
        )
        mean, ci_low, ci_high = confidence_interval(values)
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        axis.scatter(
            np.full(values.size, positions[index]) + jitter,
            values,
            s=25,
            color=COLORS[policy],
            alpha=0.36,
            edgecolors="none",
            zorder=2,
        )
        axis.errorbar(
            positions[index],
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="s",
            markersize=9,
            color=COLORS[policy],
            markeredgecolor="black",
            markeredgewidth=0.7,
            capsize=7,
            linewidth=2.0,
            zorder=4,
        )
        note = f"均值 {mean:.1f} ms"
        if policy == "P1_prime":
            note += "\n功率约束不满足"
        axis.annotate(
            note,
            (positions[index], ci_high),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS[policy],
            fontweight="bold" if policy != "P0" else "normal",
        )

    axis.set_xticks(positions, [LABELS[name] for name in POLICIES])
    axis.set_ylabel("优先级加权平均端到端时延 / ms", fontsize=11)
    axis.set_title(
        "20 个共同随机种子上的平均端到端时延",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    axis.text(
        0.01,
        0.97,
        "越低越好；方块为均值，误差棒为 95% 置信区间，浅色点为单个评估种子",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    axis.grid(True, axis="y", linestyle=":", alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    png = output_dir / "average_end_to_end_delay_cn.png"
    pdf = output_dir / "average_end_to_end_delay_cn.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def replay_queue_traces(checkpoint, selection_path, seed, device):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    offset_m = float(selection["selected"]["offset_m"])
    residual_cap_m = float(selection["selected"]["residual_cap_per_macro_m"])
    (
        _checkpoint_path,
        checkpoint_args,
        env,
        actor,
        micro_policy,
        _device,
    ) = load_checkpoint_components(checkpoint, device)
    learned_policy = td3_macro_policy(actor)
    schedule = make_episode_load_switch_schedule(
        checkpoint_args,
        int(seed),
        int(env.N),
    )
    if schedule is None:
        raise RuntimeError("The checkpoint did not produce a load-switch schedule.")

    traces = {}
    hashes = {}
    for policy in POLICIES:
        if policy == "TD3":
            policy_name = "TD3"
            macro_policy = learned_policy
            slew_cap = None
        elif policy == "P0":
            policy_name = "P0"
            macro_policy = None
            slew_cap = None
        else:
            policy_name = "P1"
            macro_policy = None
            slew_cap = residual_cap_m
        result, trace = run_gate_episode(
            env,
            checkpoint_args,
            micro_policy,
            policy_name,
            int(seed),
            macro_steps=int(checkpoint_args.macro_steps),
            offset_m=offset_m,
            residual_slew_cap_m=slew_cap,
            load_switch_schedule=schedule,
            macro_policy=macro_policy,
        )
        traces[policy] = trace
        hashes[policy] = result["external_trace_hash"]
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"CRN trace mismatch: {hashes}")
    return traces, schedule, float(env.env.macro_dt)


def tail_slope(time_s, values):
    start = max(0, int(np.ceil(values.size * 2.0 / 3.0)) - 1)
    if values.size - start < 2:
        return 0.0
    return float(np.polyfit(time_s[start:], values[start:], 1)[0])


def write_queue_csv(path, traces, schedule, macro_dt):
    fieldnames = (
        "policy",
        "time_s",
        "phase_index",
        "queue_high_mbits",
        "queue_low_mbits",
        "queue_total_mbits",
        "queue_uav1_mbits",
        "queue_uav2_mbits",
        "queue_uav3_mbits",
    )
    rows = []
    for policy in POLICIES:
        high = traces[policy]["queue_high_trajectory_mbits"]
        low = traces[policy]["queue_low_trajectory_mbits"]
        per_uav = traces[policy]["queue_per_uav_trajectory_mbits"]
        for index in range(high.size):
            macro_index = min(max(index - 1, 0), schedule.macro_steps - 1)
            rows.append(
                {
                    "policy": policy,
                    "time_s": index * macro_dt,
                    "phase_index": int(schedule.phase_index(macro_index)),
                    "queue_high_mbits": float(high[index]),
                    "queue_low_mbits": float(low[index]),
                    "queue_total_mbits": float(high[index] + low[index]),
                    "queue_uav1_mbits": float(per_uav[index, 0]),
                    "queue_uav2_mbits": float(per_uav[index, 1]),
                    "queue_uav3_mbits": float(per_uav[index, 2]),
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_queue(traces, schedule, macro_dt, seed, output_dir):
    configure_chinese_font()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )
    time_s = np.arange(
        traces["TD3"]["queue_high_trajectory_mbits"].size,
        dtype=np.float64,
    ) * macro_dt

    for policy in POLICIES:
        high = np.asarray(
            traces[policy]["queue_high_trajectory_mbits"], dtype=np.float64
        )
        low = np.asarray(
            traces[policy]["queue_low_trajectory_mbits"], dtype=np.float64
        )
        total = high + low
        slope = tail_slope(time_s, total)
        axes[0].plot(
            time_s,
            total,
            color=COLORS[policy],
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label=(
                f"{LABELS[policy]}：平均 {np.mean(total):.1f} Mbit，"
                f"末值 {total[-1]:.1f} Mbit，尾段斜率 {slope:+.3f} Mbit/s"
            ),
        )

    td3_high = np.asarray(
        traces["TD3"]["queue_high_trajectory_mbits"], dtype=np.float64
    )
    td3_low = np.asarray(
        traces["TD3"]["queue_low_trajectory_mbits"], dtype=np.float64
    )
    axes[1].plot(
        time_s,
        td3_high,
        color="#C84630",
        linewidth=2.1,
        marker="o",
        markersize=3.0,
        label="高优先级队列",
    )
    axes[1].plot(
        time_s,
        td3_low,
        color="#4C78A8",
        linewidth=2.1,
        marker="s",
        markersize=3.0,
        label="低优先级队列",
    )

    for axis in axes:
        for switch_step in schedule.switch_steps:
            axis.axvline(
                float(switch_step) * macro_dt,
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
            )
        axis.grid(True, linestyle=":", alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("系统总积压 / Mbit")
    axes[0].set_title(
        f"代表性 episode 的队列积压变化（评估种子 {seed}）",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    axes[0].legend(loc="upper left", fontsize=8.8, frameon=True)
    axes[1].set_ylabel("TD3 队列积压 / Mbit")
    axes[1].set_xlabel("仿真时间 / s")
    axes[1].set_title("TD3-SAC 的高、低优先级队列分解", fontsize=11.5)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].text(
        0.99,
        0.04,
        "竖虚线表示负载阶段切换",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color="#555555",
    )
    png = output_dir / "queue_backlog_cn.png"
    pdf = output_dir / "queue_backlog_cn.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main():
    args = parse_args()
    checkpoint = project_path(args.checkpoint).resolve()
    selection = project_path(args.selection).resolve()
    output_dir = project_path(args.archive_dir).resolve()
    rows = read_rows(output_dir / "per_seed_raw.csv")
    delay_paths = plot_delay(rows, output_dir)
    traces, schedule, macro_dt = replay_queue_traces(
        checkpoint,
        selection,
        args.seed,
        args.device,
    )
    queue_csv = output_dir / "queue_backlog_seed1012.csv"
    write_queue_csv(queue_csv, traces, schedule, macro_dt)
    queue_paths = plot_queue(
        traces,
        schedule,
        macro_dt,
        args.seed,
        output_dir,
    )
    for path in (*delay_paths, *queue_paths, queue_csv):
        print(path)


if __name__ == "__main__":
    main()
