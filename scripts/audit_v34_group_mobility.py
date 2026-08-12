"""Audit the v34 RPGM-inspired ground-terminal mobility without RL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "tcom_v34_cooperative_mobility.json"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v34_group_mobility_audit"
COLORS = ("#2878B5", "#F28E2B", "#2CA02C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit curved group-correlated terminal mobility."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1,7,42")
    parser.add_argument("--duration_s", type=float, default=300.0)
    return parser.parse_args()


def build_env(config: dict):
    from hrl_uav.env import UAV_MEC_Env

    return UAV_MEC_Env(
        num_uavs=int(config["num_uavs"]),
        num_users=int(config["num_users"]),
        resource_frame_dt=float(config["resource_frame_s"]),
        users_static=bool(config["users_static"]),
        gm_mean_speed_mps=float(config["gm_mean_speed_mps"]),
        gm_mean_heading_rad=float(config["gm_mean_heading_rad"]),
        gm_heading_mode=str(config["gm_heading_mode"]),
        gm_speed_memory=float(config["gm_speed_memory"]),
        gm_heading_memory=float(config["gm_heading_memory"]),
        gm_speed_noise_std=float(config["gm_speed_noise_std"]),
        gm_heading_noise_std=float(config["gm_heading_noise_std"]),
        user_zone_radius=float(config["user_zone_radius_m"]),
        rpgm_group_speed_mps=float(config["rpgm_group_speed_mps"]),
        rpgm_group_speed_memory=float(config["rpgm_group_speed_memory"]),
        rpgm_group_speed_noise_std=float(
            config["rpgm_group_speed_noise_std"]
        ),
        rpgm_group_turn_std_rad_sqrt_s=float(
            config["rpgm_group_turn_std_rad_sqrt_s"]
        ),
        rpgm_local_velocity_memory=float(
            config["rpgm_local_velocity_memory"]
        ),
        rpgm_local_velocity_std_mps=float(
            config["rpgm_local_velocity_std_mps"]
        ),
        rpgm_local_restore_rate_s2=float(
            config["rpgm_local_restore_rate_s2"]
        ),
        rpgm_local_speed_cap_mps=float(
            config["rpgm_local_speed_cap_mps"]
        ),
    )


def run_seed(config: dict, seed: int, duration_s: float) -> tuple[dict, dict]:
    env = build_env(config)
    env.reset(seed=int(seed))
    frames = int(round(float(duration_s) / float(env.resource_frame_dt)))
    users = [env.user_coords.copy()]
    centers = [env.mobility_group_centers.copy()]
    speeds = [env.user_speeds.copy()]

    for _ in range(frames):
        env._update_gauss_markov_mobility()
        for _ in range(env.resource_frame_steps):
            env._integrate_user_positions()
        users.append(env.user_coords.copy())
        centers.append(env.mobility_group_centers.copy())
        speeds.append(env.user_speeds.copy())

    users = np.asarray(users, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    user_steps = np.linalg.norm(np.diff(users, axis=0), axis=2)
    user_path = np.sum(user_steps, axis=0)
    user_net = np.linalg.norm(users[-1] - users[0], axis=1)
    straightness = user_net / np.maximum(user_path, 1e-12)
    group_steps = np.linalg.norm(np.diff(centers, axis=0), axis=2)
    group_path = np.sum(group_steps, axis=0)
    group_net = np.linalg.norm(centers[-1] - centers[0], axis=1)

    dispersion = np.empty((users.shape[0], env.N), dtype=np.float64)
    for time_index in range(users.shape[0]):
        for zone in range(env.N):
            mask = env.user_allocations == zone
            offsets = users[time_index, mask] - centers[time_index, zone]
            dispersion[time_index, zone] = float(
                np.sqrt(np.mean(np.sum(offsets**2, axis=1)))
            )

    edge_hits = np.sum((users <= 1e-9) | (users >= 1000.0 - 1e-9))
    result = {
        "seed": int(seed),
        "duration_s": float(duration_s),
        "mean_user_speed_mps": float(np.mean(speeds)),
        "p95_user_speed_mps": float(np.percentile(speeds, 95)),
        "mean_user_path_m": float(np.mean(user_path)),
        "mean_user_net_displacement_m": float(np.mean(user_net)),
        "mean_straightness": float(np.mean(straightness)),
        "p10_straightness": float(np.percentile(straightness, 10)),
        "p90_straightness": float(np.percentile(straightness, 90)),
        "mean_group_path_m": float(np.mean(group_path)),
        "mean_group_net_displacement_m": float(np.mean(group_net)),
        "mean_zone_rms_dispersion_m": float(np.mean(dispersion)),
        "final_zone_rms_dispersion_m": float(np.mean(dispersion[-1])),
        "map_edge_hit_fraction": float(edge_hits / users.size),
    }
    trace = {
        "users": users,
        "group_centers": centers,
        "user_allocations": env.user_allocations.copy(),
        "speeds": speeds,
        "straightness": straightness,
        "dispersion": dispersion,
    }
    return result, trace


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
        }
    )


def plot_trace(
    trace: dict, output: Path, seed: int, version_label: str = "v34"
) -> None:
    configure_plotting()
    users = trace["users"]
    centers = trace["group_centers"]
    allocations = trace["user_allocations"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    for zone, (axis, color) in enumerate(zip(axes, COLORS)):
        mask = np.flatnonzero(allocations == zone)
        for user_index in mask:
            axis.plot(
                users[:, user_index, 0],
                users[:, user_index, 1],
                color=color,
                alpha=0.45,
                linewidth=0.9,
            )
            axis.scatter(
                users[0, user_index, 0],
                users[0, user_index, 1],
                s=13,
                facecolors="white",
                edgecolors=color,
                linewidths=0.7,
            )
            axis.scatter(
                users[-1, user_index, 0],
                users[-1, user_index, 1],
                s=13,
                color=color,
                alpha=0.8,
            )
        axis.plot(
            centers[:, zone, 0],
            centers[:, zone, 1],
            color="#202020",
            linestyle="--",
            linewidth=2.2,
            label="移动热点中心",
        )
        axis.scatter(
            centers[0, zone, 0],
            centers[0, zone, 1],
            marker="s",
            s=45,
            color="#202020",
            zorder=5,
        )
        axis.scatter(
            centers[-1, zone, 0],
            centers[-1, zone, 1],
            marker="*",
            s=120,
            color="#202020",
            zorder=5,
        )
        axis.set_title(f"区域 {zone + 1}")
        axis.set_xlabel("X 坐标 / m")
        axis.set_ylabel("Y 坐标 / m")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"群组相关地面终端移动轨迹（seed={seed}, 300 s）",
        fontsize=15,
        fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color="#777777", linewidth=0.9, alpha=0.55, label="地面终端轨迹"),
        Line2D([0], [0], color="#202020", linestyle="--", linewidth=2.0, label="移动热点中心"),
        Line2D([0], [0], marker="o", color="#777777", markerfacecolor="white", linewidth=0, label="终端起点"),
        Line2D([0], [0], marker="o", color="#777777", markerfacecolor="#777777", linewidth=0, label="终端终点"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "细线为独立终端轨迹；虚线为区域移动热点；空心圆/实心圆分别为终端起点/终点",
        ha="center",
        color="#444444",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.21, wspace=0.22)
    figure_name = f"{version_label}_群组移动轨迹"
    fig.savefig(output / f"{figure_name}.png", dpi=300, facecolor="white")
    fig.savefig(output / f"{figure_name}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    rows = []
    representative = None
    for seed in seeds:
        result, trace = run_seed(config, seed, args.duration_s)
        rows.append(result)
        np.savez_compressed(output / f"trajectory_seed{seed}.npz", **trace)
        if representative is None:
            representative = (seed, trace)
        print(
            f"seed={seed}: speed={result['mean_user_speed_mps']:.3f} m/s, "
            f"straightness={result['mean_straightness']:.3f}, "
            f"group displacement={result['mean_group_net_displacement_m']:.1f} m, "
            f"dispersion={result['mean_zone_rms_dispersion_m']:.1f} m"
        )

    with (output / "per_seed_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key not in {"seed"}
    }
    checks = {
        "curved_not_ballistic": aggregate["mean_straightness"] < 0.9,
        "hotspot_moves_materially": (
            aggregate["mean_group_net_displacement_m"] > 30.0
        ),
        "plausible_mean_speed": (
            0.25 <= aggregate["mean_user_speed_mps"] <= 0.8
        ),
        "users_remain_spatially_distributed": (
            20.0 <= aggregate["mean_zone_rms_dispersion_m"] <= 90.0
        ),
        "no_boundary_pileup": aggregate["map_edge_hit_fraction"] < 0.001,
    }
    report = {
        "model": "RPGM-inspired moving reference plus local correlated motion",
        "config": str(config_path),
        "seeds": seeds,
        "aggregate": aggregate,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if representative is not None:
        version_label = str(config.get("results_version", "v34")).split("_")[1]
        plot_trace(
            representative[1],
            output,
            representative[0],
            version_label=version_label,
        )
    print(f"verdict={'PASS' if report['pass'] else 'FAIL'}; output={output}")


if __name__ == "__main__":
    main()
