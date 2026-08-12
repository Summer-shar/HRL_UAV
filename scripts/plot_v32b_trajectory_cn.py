import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import make_episode_load_switch_schedule
from run_load_switch_gate import run_gate_episode
from search_position_oracle import load_checkpoint_components


POLICIES = ("TD3", "P0", "P1_prime")
COLORS = ("#2878B5", "#E17C05", "#2E9B58")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a readable Chinese trajectory figure for v32b."
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


def load_policy_rows(path, seed):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        row["policy"]: row for row in rows if int(row["seed"]) == int(seed)
    }
    missing = set(POLICIES) - set(selected)
    if missing:
        raise RuntimeError(f"Missing policy rows for seed {seed}: {sorted(missing)}")
    return selected


def reproduce_user_trajectories(checkpoint, selection_path, seed, device):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    residual_cap_m = float(selection["selected"]["residual_cap_per_macro_m"])
    offset_m = float(selection["selected"]["offset_m"])
    (
        _checkpoint_path,
        checkpoint_args,
        env,
        _actor,
        micro_policy,
        _device,
    ) = load_checkpoint_components(checkpoint, device)
    base_env = env.env
    base_env.record_trajectory = True
    schedule = make_episode_load_switch_schedule(
        checkpoint_args,
        int(seed),
        int(env.N),
    )
    if schedule is None:
        raise RuntimeError("The checkpoint did not produce a load-switch schedule.")
    run_gate_episode(
        env,
        checkpoint_args,
        micro_policy,
        "P0",
        int(seed),
        macro_steps=int(checkpoint_args.macro_steps),
        offset_m=offset_m,
        residual_slew_cap_m=residual_cap_m,
        load_switch_schedule=schedule,
        macro_policy=None,
    )
    users = np.asarray(base_env.users_trajectory, dtype=np.float64)
    if users.ndim != 3 or users.shape[0] < 2:
        raise RuntimeError("Ground-user trajectories were not recorded during replay.")
    return users, np.asarray(base_env.user_allocations, dtype=np.int64), schedule


def load_traces(trace_dir, seed):
    traces = {}
    for policy in POLICIES:
        path = trace_dir / f"{policy}_seed{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as payload:
            traces[policy] = {
                "uav": np.asarray(payload["uav_trajectory_m"], dtype=np.float64),
                "centroid": np.asarray(
                    payload["centroid_trajectory_m"], dtype=np.float64
                ),
            }
    return traces


def policy_caption(policy, row):
    names = {
        "TD3": "分层 TD3-SAC",
        "P0": "业务质心跟踪 P0",
        "P1_prime": "因果支援规则 P1′",
    }
    delay = float(row["task_delay_weighted_ms"])
    power = float(row["average_power_w"])
    feasibility = "满足功率约束" if power <= 840.0 else "功率超限"
    return f"{names[policy]}\n时延 {delay:.1f} ms，功率 {power:.1f} W（{feasibility}）"


def draw_figure(users, allocations, schedule, traces, rows, seed, output_dir):
    configure_chinese_font()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 5.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    all_points = [users.reshape(-1, 2)]
    for trace in traces.values():
        all_points.extend((trace["uav"].reshape(-1, 2), trace["centroid"].reshape(-1, 2)))
    all_points = np.concatenate(all_points, axis=0)
    lower = np.min(all_points, axis=0) - 22.0
    upper = np.max(all_points, axis=0) + 22.0
    center = 0.5 * (lower + upper)
    span = max(float(upper[0] - lower[0]), float(upper[1] - lower[1]))
    x_limits = (center[0] - 0.5 * span, center[0] + 0.5 * span)
    y_limits = (center[1] - 0.5 * span, center[1] + 0.5 * span)

    switch_steps = [int(step) for step in schedule.switch_steps if int(step) < 30]
    for axis, policy in zip(axes, POLICIES):
        for zone_index, color in enumerate(COLORS):
            zone_users = np.where(allocations == zone_index)[0]
            for user_index in zone_users:
                axis.plot(
                    users[:, user_index, 0],
                    users[:, user_index, 1],
                    color=color,
                    linewidth=0.75,
                    linestyle=":",
                    alpha=0.30,
                    zorder=1,
                )
                axis.scatter(
                    users[-1, user_index, 0],
                    users[-1, user_index, 1],
                    color=color,
                    s=7,
                    alpha=0.40,
                    zorder=2,
                )

            centroid = traces[policy]["centroid"][:, zone_index]
            uav = traces[policy]["uav"][:, zone_index]
            axis.plot(
                centroid[:, 0],
                centroid[:, 1],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.90,
                zorder=3,
            )
            axis.plot(
                uav[:, 0],
                uav[:, 1],
                color=color,
                linewidth=2.6,
                zorder=5,
            )
            axis.scatter(
                uav[0, 0],
                uav[0, 1],
                marker="s",
                s=42,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=7,
            )
            axis.scatter(
                uav[-1, 0],
                uav[-1, 1],
                marker="*",
                s=125,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=7,
            )
            for switch_step in switch_steps:
                point_index = min(switch_step, uav.shape[0] - 1)
                axis.scatter(
                    uav[point_index, 0],
                    uav[point_index, 1],
                    marker="D",
                    s=27,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.1,
                    zorder=8,
                )

        axis.set_title(policy_caption(policy, rows[policy]), fontsize=11.5, pad=9)
        axis.set_xlabel("横坐标 / m", fontsize=10.5)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
    axes[0].set_ylabel("纵坐标 / m", fontsize=10.5)

    legend_handles = [
        Line2D([0], [0], color="#777777", linestyle=":", linewidth=1.2, label="地面用户轨迹"),
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.6, label="业务质心轨迹"),
        Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.6, label="无人机轨迹"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="black", markersize=6, label="无人机起点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#777777", markeredgecolor="black", markersize=10, label="无人机终点"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#777777", markersize=6, label="负载切换时位置"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=6,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        f"负载切换场景下的地面用户、业务质心与无人机轨迹（代表性评估种子 {seed}）",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "颜色对应三个服务区域；单个种子仅用于解释轨迹行为，统计结论以 20 个共同随机种子为准。",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.84, bottom=0.16, wspace=0.15)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "matched_reference_trajectories_cn.png"
    pdf_path = output_dir / "matched_reference_trajectories_cn.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    checkpoint = project_path(args.checkpoint).resolve()
    selection = project_path(args.selection).resolve()
    archive_dir = project_path(args.archive_dir).resolve()
    rows = load_policy_rows(archive_dir / "per_seed_raw.csv", args.seed)
    traces = load_traces(archive_dir / "traces", args.seed)
    users, allocations, schedule = reproduce_user_trajectories(
        checkpoint,
        selection,
        args.seed,
        args.device,
    )
    png_path, pdf_path = draw_figure(
        users,
        allocations,
        schedule,
        traces,
        rows,
        args.seed,
        archive_dir,
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
