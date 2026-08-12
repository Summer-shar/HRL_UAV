"""Zero-training sanity check for the v33 environment definition."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrl_uav.hrl_wrappers import HRL_Wrapper
from scripts.compare_sca import solve_sca_micro_action


DEFAULT_CONFIG = ROOT / "configs" / "tcom_v33_soft_energy_random_mobility.json"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v33_environment_sanity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check mobility diversity and queue feasibility without RL training."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1,7,42")
    parser.add_argument("--macro_steps", type=int, default=30)
    parser.add_argument("--load_scale", type=float, default=None)
    parser.add_argument(
        "--micro_policy",
        choices=["sca", "capacity"],
        default="sca",
    )
    return parser.parse_args()


def queue_totals(base_env) -> tuple[float, float]:
    high = float(np.sum(base_env.Q_loc_high) + np.sum(base_env.Q_tx_high))
    low = float(np.sum(base_env.Q_loc_low) + np.sum(base_env.Q_tx_low))
    return high, low


def circular_resultant(angles: np.ndarray) -> float:
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * values))))


def capacity_matched_micro_action(env: HRL_Wrapper) -> np.ndarray:
    """Route only the analytically overloaded share to spare processors."""
    base = env.env
    offered = np.asarray(base.zone_target_load_mbps, dtype=np.float64) * float(
        base.load_scale
    )
    service_fraction = max(
        (float(base.macro_dt) - float(base.deployment_dt))
        / float(base.macro_dt),
        0.0,
    )
    capacity = (
        base.F_max_array.astype(np.float64)
        / max(float(base.C) * 1e6, 1e-12)
        * service_fraction
    )
    spare = np.maximum(capacity - offered, 0.0)
    routing = np.eye(base.N, dtype=np.float64)
    for source in np.flatnonzero(offered > capacity + 1e-12):
        excess = float(offered[source] - capacity[source])
        destinations = np.flatnonzero(spare > 1e-12)
        available = float(np.sum(spare[destinations]))
        if available <= 0.0:
            continue
        routed = min(excess, available)
        routing[source, :] = 0.0
        routing[source, source] = max(float(offered[source] - routed), 0.0) / max(
            float(offered[source]),
            1e-12,
        )
        for target in destinations:
            assigned = routed * float(spare[target]) / available
            routing[source, target] = assigned / max(float(offered[source]), 1e-12)
            spare[target] = max(float(spare[target]) - assigned, 0.0)
    return np.stack([routing, routing], axis=0).reshape(-1).astype(np.float32)


def build_env(config: dict, record_trajectory: bool) -> HRL_Wrapper:
    env = HRL_Wrapper(
        num_uavs=int(config["num_uavs"]),
        num_users=int(config["num_users"]),
        record_trajectory=record_trajectory,
        traffic_profile="normal",
        traffic_scale=0.5,
        arrival_dt=float(config["resource_frame_s"]),
        resource_frame_dt=float(config["resource_frame_s"]),
        deployment_dt=float(config["deployment_s"]),
        deployment_transient_energy_mode="kinetic_lower_bound",
        uav_mass_kg=2.038735983690112,
        uplink_bandwidth_mhz=20.0,
        u2u_bandwidth_mhz=3.0,
        micro_diag_bias=0.0,
        micro_action_scale=1.0,
        micro_action_encoding=str(config["micro_action_encoding"]),
        micro_reward_scale=float(config["micro_reward_scale"]),
        micro_state_mode="pressure",
        u2u_state_representation=str(config["u2u_state_representation"]),
        u2u_goal_representation=str(config["u2u_goal_representation"]),
        u2u_contention_aware_pressure=bool(
            config["u2u_contention_aware_pressure"]
        ),
        u2u_prev_active_state=bool(config["u2u_prev_active_state"]),
        cpu_alloc_mode=str(config["cpu_alloc_mode"]),
        cpu_power_cap_w=0.0,
        load_scale=float(config["load_scale"]),
        zone_loads=config["zone_loads_mbps"],
        macro_flow_state=True,
        macro_load_state=True,
        macro_capacity_state=True,
        macro_comm_weight=0.0,
        macro_action_mode=str(config["macro_action_mode"]),
        macro_waypoint_radius_m=float(config["macro_waypoint_radius_m"]),
        macro_waypoint_tolerance_m=float(config["macro_waypoint_tolerance_m"]),
        macro_reward_mode="strict_global",
        strict_hierarchy_validation=True,
        pause_service_during_flight=True,
        arrivals_during_deployment=True,
        objective_mode=str(config["objective_mode"]),
        power_budget_w=0.0,
        system_dual_state=False,
        users_static=bool(config["users_static"]),
        gm_mean_speed_mps=float(config["gm_mean_speed_mps"]),
        gm_mean_heading_rad=float(config["gm_mean_heading_rad"]),
        gm_heading_mode=str(config["gm_heading_mode"]),
        gm_speed_memory=float(config["gm_speed_memory"]),
        gm_heading_memory=float(config["gm_heading_memory"]),
        gm_speed_noise_std=float(config["gm_speed_noise_std"]),
        gm_heading_noise_std=float(config["gm_heading_noise_std"]),
        user_zone_radius=float(config["user_zone_radius_m"]),
        delay_objective_weight=float(config["delay_objective_weight"]),
        energy_objective_weight=float(config["energy_objective_weight"]),
    )
    env.env.F_max_array[:] = float(config["cpu_ghz"]) * 1e9
    env.max_macro_displacement = float(config["max_deployment_distance_m"])
    env.sca_max_iter = 3
    env.sca_tol = 1e-3
    env.sca_solver = "CLARABEL"
    env.sca_delay_weight = float(config["delay_objective_weight"])
    env.sca_energy_weight = float(config["energy_objective_weight"])
    env.sca_proximal_weight = 0.1
    return env


def run_seed(
    config: dict,
    seed: int,
    macro_steps: int,
    output: Path,
    micro_policy: str,
) -> dict:
    env = build_env(config, record_trajectory=True)
    env.reset(seed=seed)
    base = env.env
    initial_users = base.user_coords.copy()
    initial_heading_resultant = circular_resultant(base.user_mean_headings)
    phases = [np.asarray(row, dtype=np.float64) for row in config["load_switch_phases_mbps"]]

    queue_rows = []
    total_energy_j = 0.0
    for macro_index in range(macro_steps):
        phase_index = (macro_index // 10) % len(phases)
        if macro_index % 10 == 0:
            base.set_zone_target_loads(phases[phase_index])

        _, _, macro_info = env.macro_step(
            np.zeros((base.N, 2), dtype=np.float32)
        )
        total_energy_j += float(macro_info.get("E_macro_total", 0.0))
        for _ in range(base.resource_frames_per_macro):
            paused = bool(
                base.pause_service_during_flight
                and base.macro_service_pause_steps_remaining > 0
            )
            if paused:
                action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
            else:
                env.prepare_resource_frame()
                if micro_policy == "sca":
                    action = solve_sca_micro_action(env, deterministic=True)
                else:
                    action = capacity_matched_micro_action(env)
            frame = env.step_resource_frame(action)
            total_energy_j += sum(
                float(info.get("total_energy", 0.0))
                for info in frame["infos"]
            )
        env.get_macro_reward()
        high, low = queue_totals(base)
        queue_rows.append(
            {
                "macro_step": macro_index + 1,
                "time_s": (macro_index + 1) * float(base.macro_dt),
                "phase_index": phase_index,
                "queue_high_mbit": high,
                "queue_low_mbit": low,
                "queue_total_mbit": high + low,
            }
        )

    final_users = base.user_coords.copy()
    displacements = final_users - initial_users
    displacement_norms = np.linalg.norm(displacements, axis=1)
    moving = displacement_norms > 1e-6
    displacement_angles = np.arctan2(
        displacements[moving, 1],
        displacements[moving, 0],
    )
    tail = np.asarray(
        [row["queue_total_mbit"] for row in queue_rows[-10:]],
        dtype=np.float64,
    )
    tail_slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
    queue_values = np.asarray(
        [row["queue_total_mbit"] for row in queue_rows],
        dtype=np.float64,
    )
    cycle_boundaries = queue_values[29::30]
    cycle_growth = (
        float(np.polyfit(np.arange(cycle_boundaries.size), cycle_boundaries, 1)[0])
        if cycle_boundaries.size >= 2
        else float("nan")
    )
    task_stats = base.get_task_delay_stats()
    result = {
        "seed": seed,
        "micro_policy": micro_policy,
        "task_delay_weighted_ms": float(task_stats["task_delay_weighted_ms"]),
        "task_arrived_count": int(task_stats["task_arrived_count"]),
        "task_completed_count": int(task_stats["task_completed_count"]),
        "terminal_queue_mbit": float(queue_rows[-1]["queue_total_mbit"]),
        "tail_queue_slope_mbit_per_macro": tail_slope,
        "cycle_boundary_count": int(cycle_boundaries.size),
        "cycle_boundary_growth_mbit_per_cycle": cycle_growth,
        "average_power_w": total_energy_j / (macro_steps * float(base.macro_dt)),
        "initial_mean_heading_resultant": initial_heading_resultant,
        "net_displacement_direction_resultant": circular_resultant(
            displacement_angles
        ),
        "mean_user_displacement_m": float(np.mean(displacement_norms)),
    }

    seed_dir = output / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "queue_trace.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(queue_rows[0]))
        writer.writeheader()
        writer.writerows(queue_rows)
    np.savez_compressed(
        seed_dir / "trajectory.npz",
        uav=np.asarray(base.uav_trajectory, dtype=np.float32),
        users=np.asarray(base.users_trajectory, dtype=np.float32),
        centroids=np.asarray(base.centroid_trajectory, dtype=np.float32),
        user_allocations=base.user_allocations.astype(np.int32),
        queue_total_mbit=np.asarray(
            [row["queue_total_mbit"] for row in queue_rows],
            dtype=np.float32,
        ),
    )
    return result


def plot_representative(output: Path, seed: int, micro_policy: str) -> None:
    data = np.load(output / f"seed{seed}" / "trajectory.npz")
    users = data["users"]
    uav = data["uav"]
    centroids = data["centroids"]
    allocations = data["user_allocations"]
    queue = data["queue_total_mbit"]
    colors = ("#2878B5", "#F28522", "#2A9D55")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    stride = max(users.shape[0] // 120, 1)
    for user_index in range(users.shape[1]):
        color = colors[int(allocations[user_index])]
        axes[0].plot(
            users[::stride, user_index, 0],
            users[::stride, user_index, 1],
            color=color,
            linewidth=0.55,
            alpha=0.42,
        )
        axes[0].scatter(
            users[0, user_index, 0],
            users[0, user_index, 1],
            s=5,
            color=color,
            alpha=0.8,
        )
    for uav_index, color in enumerate(colors):
        axes[0].plot(
            uav[:, uav_index, 0],
            uav[:, uav_index, 1],
            color=color,
            linewidth=2.0,
        )
        axes[0].plot(
            centroids[:, uav_index, 0],
            centroids[:, uav_index, 1],
            color=color,
            linestyle="--",
            linewidth=1.2,
        )
    axes[0].set_title(f"Independent user mobility (seed {seed})")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_xlim(0, 1000)
    axes[0].set_ylim(0, 1000)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.2)

    times = np.arange(1, queue.size + 1) * 10.0
    axes[1].plot(times, queue, color="#202020", linewidth=2.0)
    axes[1].axvline(100.0, color="#999999", linestyle="--", linewidth=1.0)
    axes[1].axvline(200.0, color="#999999", linestyle="--", linewidth=1.0)
    policy_label = "SCA" if micro_policy == "sca" else "capacity-matched routing"
    axes[1].set_title(f"System queue under centroid + {policy_label}")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("backlog (Mbit)")
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "v33_environment_sanity.png", dpi=220)
    fig.savefig(output / "v33_environment_sanity.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.load_scale is not None:
        config["load_scale"] = float(args.load_scale)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        run_seed(config, seed, args.macro_steps, args.output, args.micro_policy)
        for seed in seeds
    ]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_representative(args.output, seeds[0], args.micro_policy)
    for result in results:
        print(
            "seed={seed} delay={task_delay_weighted_ms:.1f}ms "
            "queue={terminal_queue_mbit:.2f}Mbit "
            "tail_slope={tail_queue_slope_mbit_per_macro:.3f}Mbit/macro "
            "cycle_growth={cycle_boundary_growth_mbit_per_cycle:.3f}Mbit/cycle "
            "power={average_power_w:.1f}W heading_R={initial_mean_heading_resultant:.3f} "
            "displacement_R={net_displacement_direction_resultant:.3f}".format(
                **result
            )
        )


if __name__ == "__main__":
    main()
