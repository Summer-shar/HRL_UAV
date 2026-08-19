import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import t as student_t

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = Path(bootstrap())

from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import resolve_project_path
from compare_common import torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SACActor
import hrl_uav.agents.sac_agent as sac_agent_mod
from hrl_uav.hrl_wrappers import cooperation_aligned_basis
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import project_unit_disk_np
from train_centralized_macro import sac_micro_policy
from train_centralized_td3 import MacroActor
from train_centralized_td3 import td3_macro_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Approximate the best static three-UAV placement with CEM while "
            "replaying the checkpoint's frozen micro policy."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite_fraction", type=float, default=0.25)
    parser.add_argument("--search_episodes", type=int, default=3)
    parser.add_argument("--validation_episodes", type=int, default=20)
    parser.add_argument("--eval_base_seed", type=int, default=1000)
    parser.add_argument("--search_seed", type=int, default=20260727)
    parser.add_argument("--initial_std_m", type=float, default=80.0)
    parser.add_argument("--minimum_std_m", type=float, default=5.0)
    parser.add_argument("--max_offset_m", type=float, default=200.0)
    parser.add_argument("--map_min_m", type=float, default=0.0)
    parser.add_argument("--map_max_m", type=float, default=1000.0)
    parser.add_argument("--reference_trajectory", default="")
    parser.add_argument("--reuse_search_json", default="")
    parser.add_argument("--medium_offset_m", type=float, default=60.0)
    parser.add_argument("--light_offset_m", type=float, default=120.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def project_targets(
    targets,
    anchors,
    max_offset_m,
    map_min_m=0.0,
    map_max_m=1000.0,
):
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(targets.shape)
    offsets = targets - anchors
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    scale = np.minimum(
        1.0,
        float(max_offset_m) / np.maximum(norms, 1e-12),
    )
    projected = anchors + offsets * scale
    return np.clip(projected, float(map_min_m), float(map_max_m))


def target_delta_action(
    current_positions,
    target_positions,
    max_displacement_m,
    tolerance_m=0.5,
):
    current = np.asarray(current_positions, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64).reshape(current.shape)
    displacement = targets - current
    distances = np.linalg.norm(displacement, axis=1, keepdims=True)
    move_distance = np.minimum(
        distances,
        max(float(max_displacement_m), 1e-8),
    )
    direction = displacement / np.maximum(distances, 1e-12)
    physical_action = direction * move_distance
    physical_action[distances[:, 0] <= float(tolerance_m)] = 0.0
    normalized_action = physical_action / max(
        float(max_displacement_m),
        1e-8,
    )
    return project_unit_disk_np(normalized_action.astype(np.float32))


def target_waypoint_action(
    target_positions,
    waypoint_origins,
    waypoint_radius_m,
):
    targets = np.asarray(target_positions, dtype=np.float64)
    origins = np.asarray(waypoint_origins, dtype=np.float64).reshape(
        targets.shape
    )
    normalized_action = (
        (targets - origins) / max(float(waypoint_radius_m), 1e-8)
    )
    return project_unit_disk_np(normalized_action.astype(np.float32))


def target_centroid_waypoint_action(
    target_positions,
    centroids,
    waypoint_radius_m,
):
    targets = np.asarray(target_positions, dtype=np.float64)
    centroid_array = np.asarray(centroids, dtype=np.float64).reshape(
        targets.shape
    )
    normalized_action = (
        (targets - centroid_array) / max(float(waypoint_radius_m), 1e-8)
    )
    return project_unit_disk_np(normalized_action.astype(np.float32))


def target_centroid_relative_action(env, targets, centroids):
    """Encode a world-frame centroid residual for the configured action mode."""
    world_action = (
        np.asarray(targets, dtype=np.float64)
        - np.asarray(centroids, dtype=np.float64)
    ) / max(float(env.macro_waypoint_radius_m), 1e-8)
    if env.macro_action_mode == "cooperation_waypoint":
        base_env = getattr(env, "env", env)
        basis, _ = cooperation_aligned_basis(
            centroids,
            base_env.get_raw_state()["uav_coords"],
            base_env.zone_expected_offered_load_mbps,
        )
        world_action = np.einsum(
            "nij,ni->nj",
            basis,
            world_action,
        )
    return project_unit_disk_np(world_action.astype(np.float32))


def uplink_reference_objective_and_gradient(
    position,
    user_coords,
    weighted_load_mbps,
    altitude_m,
    bandwidth_per_user_hz,
    p_gt_power,
    beta_0,
    noise_power,
):
    position = np.asarray(position, dtype=np.float64).reshape(2)
    users = np.asarray(user_coords, dtype=np.float64).reshape(-1, 2)
    loads_mbps = np.asarray(
        weighted_load_mbps,
        dtype=np.float64,
    ).reshape(-1)
    if users.shape[0] != loads_mbps.size:
        raise ValueError("Each user must have one weighted uplink load.")
    if bandwidth_per_user_hz <= 0.0 or noise_power <= 0.0:
        raise ValueError("Bandwidth and noise power must be positive.")

    horizontal = position[None, :] - users
    distance_sq = (
        np.sum(horizontal**2, axis=1) + float(altitude_m) ** 2
    )
    snr_scale = float(p_gt_power) * float(beta_0) / float(noise_power)
    rates_bps = float(bandwidth_per_user_hz) * np.log2(
        1.0 + snr_scale / np.maximum(distance_sq, 1e-12)
    )
    rates_bps = np.maximum(rates_bps, 1e-12)
    loads_bps = np.maximum(loads_mbps, 0.0) * 1e6
    objective = float(np.sum(loads_bps / rates_bps))

    rate_gradient = (
        -2.0
        * float(bandwidth_per_user_hz)
        * snr_scale
        * horizontal
        / (
            np.log(2.0)
            * distance_sq[:, None]
            * (distance_sq + snr_scale)[:, None]
        )
    )
    gradient = np.sum(
        (-loads_bps / rates_bps**2)[:, None] * rate_gradient,
        axis=0,
    )
    return objective, gradient


def solve_uplink_reference_position(
    user_coords,
    weighted_load_mbps,
    initial_position,
    altitude_m,
    bandwidth_per_user_hz,
    p_gt_power,
    beta_0,
    noise_power,
    map_min_m=0.0,
    map_max_m=1000.0,
    max_iterations=50,
):
    users = np.asarray(user_coords, dtype=np.float64).reshape(-1, 2)
    loads = np.asarray(weighted_load_mbps, dtype=np.float64).reshape(-1)
    initial = np.clip(
        np.asarray(initial_position, dtype=np.float64).reshape(2),
        float(map_min_m),
        float(map_max_m),
    )
    if users.shape[0] != loads.size:
        raise ValueError("Each user must have one weighted uplink load.")
    if users.shape[0] == 0 or float(np.sum(np.maximum(loads, 0.0))) <= 0.0:
        return initial, {
            "success": True,
            "status": 0,
            "iterations": 0,
            "initial_objective": 0.0,
            "final_objective": 0.0,
            "message": "No positive offered load.",
        }

    def objective_with_gradient(position):
        return uplink_reference_objective_and_gradient(
            position,
            users,
            loads,
            altitude_m=altitude_m,
            bandwidth_per_user_hz=bandwidth_per_user_hz,
            p_gt_power=p_gt_power,
            beta_0=beta_0,
            noise_power=noise_power,
        )

    initial_objective, _ = objective_with_gradient(initial)
    result = minimize(
        objective_with_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (float(map_min_m), float(map_max_m)),
            (float(map_min_m), float(map_max_m)),
        ],
        options={
            "maxiter": max(int(max_iterations), 1),
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )
    candidate = np.clip(
        np.asarray(result.x, dtype=np.float64),
        float(map_min_m),
        float(map_max_m),
    )
    final_objective, _ = objective_with_gradient(candidate)
    if not np.all(np.isfinite(candidate)) or (
        final_objective > initial_objective + 1e-10
    ):
        candidate = initial
        final_objective = initial_objective

    return candidate, {
        "success": bool(result.success),
        "status": int(result.status),
        "iterations": int(result.nit),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "message": str(result.message),
    }


def expected_priority_weighted_uplink_loads(
    env,
    priority_high_weight=None,
    priority_low_weight=None,
):
    base_env = getattr(env, "env", env)
    high_weight = (
        float(base_env.w_D_high)
        if priority_high_weight is None
        else float(priority_high_weight)
    )
    low_weight = (
        float(base_env.w_D_low)
        if priority_low_weight is None
        else float(priority_low_weight)
    )
    return (
        high_weight
        * np.asarray(base_env.user_lambda_high, dtype=np.float64)
        * np.asarray(base_env.user_task_size_high_mbits, dtype=np.float64)
        + low_weight
        * np.asarray(base_env.user_lambda_low, dtype=np.float64)
        * np.asarray(base_env.user_task_size_low_mbits, dtype=np.float64)
    )


def solve_uplink_reference_positions(
    env,
    initial_positions=None,
    priority_high_weight=None,
    priority_low_weight=None,
    map_min_m=0.0,
    map_max_m=1000.0,
    max_iterations=50,
):
    base_env = getattr(env, "env", env)
    if initial_positions is None:
        initial_positions = base_env.weighted_centroids
    initial_positions = np.asarray(
        initial_positions,
        dtype=np.float64,
    ).reshape(base_env.N, 2)
    user_loads = expected_priority_weighted_uplink_loads(
        base_env,
        priority_high_weight=priority_high_weight,
        priority_low_weight=priority_low_weight,
    )
    bandwidth_per_user_hz = float(base_env.B_gt_total) / max(
        int(base_env.M),
        1,
    )
    targets = np.zeros((base_env.N, 2), dtype=np.float64)
    diagnostics = []
    for uav_index in range(base_env.N):
        mask = np.asarray(base_env.user_allocations) == uav_index
        target, diagnostic = solve_uplink_reference_position(
            np.asarray(base_env.user_coords, dtype=np.float64)[mask],
            user_loads[mask],
            initial_positions[uav_index],
            altitude_m=base_env.H,
            bandwidth_per_user_hz=bandwidth_per_user_hz,
            p_gt_power=base_env.p_gt,
            beta_0=base_env.beta_0,
            noise_power=base_env.noise_power,
            map_min_m=map_min_m,
            map_max_m=map_max_m,
            max_iterations=max_iterations,
        )
        targets[uav_index] = target
        diagnostics.append(
            {
                "uav_index": int(uav_index),
                **diagnostic,
            }
        )
    return targets, diagnostics


def uplink_reference_macro_ema_beta(env):
    base_env = getattr(env, "env", env)
    eligible_steps = int(base_env.micro_steps_per_macro)
    if not bool(getattr(base_env, "arrivals_during_deployment", False)):
        eligible_steps -= int(base_env.deployment_steps)
    arrival_updates = max(
        eligible_steps // max(int(base_env.arrival_interval_steps), 1),
        1,
    )
    return float(base_env.centroid_ema_beta) ** arrival_updates


def make_uplink_reference_policy(
    offsets=None,
    directed_moves=None,
    priority_high_weight=None,
    priority_low_weight=None,
    ema_beta=None,
    map_min_m=0.0,
    map_max_m=1000.0,
    trace=None,
):
    fixed_offsets = None
    if offsets is not None:
        fixed_offsets = np.asarray(offsets, dtype=np.float64).copy()
    directed_moves = tuple(directed_moves or ())
    state = {
        "smoothed_targets": None,
        "last_global_micro_step": None,
    }

    def _policy(env, _local_obs_batch, _eval_mode):
        base_env = getattr(env, "env", env)
        current_step = int(getattr(base_env, "global_micro_step", 0))
        previous_step = state["last_global_micro_step"]
        if previous_step is not None and current_step <= previous_step:
            state["smoothed_targets"] = None

        raw_targets, diagnostics = solve_uplink_reference_positions(
            base_env,
            initial_positions=state["smoothed_targets"],
            priority_high_weight=priority_high_weight,
            priority_low_weight=priority_low_weight,
            map_min_m=map_min_m,
            map_max_m=map_max_m,
        )
        beta = (
            uplink_reference_macro_ema_beta(base_env)
            if ema_beta is None
            else float(ema_beta)
        )
        if not 0.0 <= beta < 1.0:
            raise ValueError("ema_beta must lie in [0, 1).")
        if state["smoothed_targets"] is None:
            smoothed_targets = raw_targets
        else:
            smoothed_targets = (
                beta * state["smoothed_targets"]
                + (1.0 - beta) * raw_targets
            )
        state["smoothed_targets"] = smoothed_targets.copy()
        state["last_global_micro_step"] = current_step

        targets = smoothed_targets.copy()
        if fixed_offsets is not None:
            targets += fixed_offsets.reshape(targets.shape)
        for source_index, target_index, distance_m in directed_moves:
            direction = (
                smoothed_targets[int(target_index)]
                - smoothed_targets[int(source_index)]
            )
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                targets[int(source_index)] += (
                    direction / norm * max(float(distance_m), 0.0)
                )
        targets = np.clip(
            targets,
            float(map_min_m),
            float(map_max_m),
        )

        if trace is not None:
            trace.append(
                {
                    "global_micro_step": current_step,
                    "ema_beta": beta,
                    "raw_targets_m": raw_targets.copy(),
                    "smoothed_targets_m": smoothed_targets.copy(),
                    "commanded_targets_m": targets.copy(),
                    "solver_diagnostics": diagnostics,
                }
            )

        if env.macro_action_mode == "waypoint":
            return target_waypoint_action(
                targets,
                env.macro_waypoint_origins,
                env.macro_waypoint_radius_m,
            )
        raw_state = base_env.get_raw_state()
        if env.macro_action_mode in (
            "centroid_waypoint",
            "cooperation_waypoint",
        ):
            return target_centroid_relative_action(
                env, targets, raw_state["weighted_centroids"]
            )
        return target_delta_action(
            raw_state["uav_coords"],
            targets,
            env.max_macro_displacement,
            tolerance_m=0.5,
        )

    return _policy


def checkpoint_rank(metrics):
    scope = str(metrics.get("power_constraint_scope", "per_uav"))
    violation_key = (
        "power_ucb95_violation_w"
        if scope == "system"
        else "max_battery_power_ucb95_violation_w"
    )
    violation = max(float(metrics.get(violation_key, 0.0)), 0.0)
    delay = float(metrics["delay_objective"])
    return (int(violation > 1e-8), violation, delay)


def cem_search(
    evaluate_target,
    anchors,
    population,
    iterations,
    elite_fraction,
    initial_std_m,
    minimum_std_m,
    max_offset_m,
    map_min_m,
    map_max_m,
    seed,
    progress_callback=None,
):
    anchors = np.asarray(anchors, dtype=np.float64)
    population = max(int(population), 4)
    iterations = max(int(iterations), 1)
    elite_count = max(2, int(np.ceil(population * float(elite_fraction))))
    elite_count = min(elite_count, population)
    rng = np.random.default_rng(int(seed))
    mean_offset = np.zeros_like(anchors)
    std_offset = np.full_like(anchors, float(initial_std_m))
    best_record = None
    history = []

    for iteration in range(iterations):
        sampled_offsets = rng.normal(
            mean_offset,
            std_offset,
            size=(population, *anchors.shape),
        )
        sampled_offsets[0] = mean_offset
        records = []
        for candidate_index, offsets in enumerate(sampled_offsets):
            targets = project_targets(
                anchors + offsets,
                anchors,
                max_offset_m=max_offset_m,
                map_min_m=map_min_m,
                map_max_m=map_max_m,
            )
            metrics = evaluate_target(targets)
            record = {
                "iteration": int(iteration + 1),
                "candidate": int(candidate_index + 1),
                "targets_m": targets.copy(),
                "metrics": metrics,
                "rank": checkpoint_rank(metrics),
            }
            records.append(record)
            if best_record is None or record["rank"] < best_record["rank"]:
                best_record = record

        records.sort(key=lambda item: item["rank"])
        elite_offsets = np.stack(
            [record["targets_m"] - anchors for record in records[:elite_count]],
            axis=0,
        )
        mean_offset = np.mean(elite_offsets, axis=0)
        std_offset = np.maximum(
            np.std(elite_offsets, axis=0),
            float(minimum_std_m),
        )
        iteration_summary = {
            "iteration": int(iteration + 1),
            "best_targets_m": records[0]["targets_m"].copy(),
            "best_metrics": records[0]["metrics"],
            "global_best_targets_m": best_record["targets_m"].copy(),
            "global_best_metrics": best_record["metrics"],
            "mean_offset_m": mean_offset.copy(),
            "std_offset_m": std_offset.copy(),
        }
        history.append(iteration_summary)
        if progress_callback is not None:
            progress_callback(iteration_summary)

    return best_record, history


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def load_checkpoint_components(checkpoint_path, requested_device):
    checkpoint_path = resolve_project_path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    if "macro_actor" not in checkpoint or "args" not in checkpoint:
        raise KeyError("Checkpoint must contain macro_actor and args")
    if str(checkpoint.get("micro_solver", "")) != "sac":
        raise ValueError("Position search currently requires a SAC micro policy")

    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sac_agent_mod.device = device

    env = make_env_from_args(checkpoint_args, record_trajectory=False)
    configure_macro_objective(
        env,
        checkpoint_args,
        checkpoint_args.macro_shaping_final,
    )
    saved_duals = getattr(
        checkpoint_args,
        "energy_lagrange_multipliers_current",
        None,
    )
    if saved_duals is not None:
        env.set_energy_lagrange_multipliers(saved_duals)

    _, global_state = env.reset(seed=checkpoint_args.seed)[0]
    actor = MacroActor(
        int(np.asarray(global_state).size),
        int(checkpoint_args.num_agents) * 2,
        hidden_dim=checkpoint_args.td3_hidden_dim,
    ).to(device)
    actor.load_state_dict(checkpoint["macro_actor"])
    actor.eval()

    micro_actor = SACActor(
        int(env.sac_obs_dim),
        int(env.micro_action_space.shape[0]),
    )
    micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    micro_actor.action_net.eval()
    return (
        checkpoint_path,
        checkpoint_args,
        env,
        actor,
        sac_micro_policy(micro_actor),
        device,
    )


def make_target_policy(target_positions):
    targets = np.asarray(target_positions, dtype=np.float64).copy()

    def _policy(env, _local_obs_batch, _eval_mode):
        if env.macro_action_mode == "waypoint":
            return target_waypoint_action(
                targets,
                env.macro_waypoint_origins,
                env.macro_waypoint_radius_m,
            )
        raw_state = env.env.get_raw_state()
        if env.macro_action_mode in (
            "centroid_waypoint",
            "cooperation_waypoint",
        ):
            return target_centroid_relative_action(
                env, targets, raw_state["weighted_centroids"]
            )
        return target_delta_action(
            raw_state["uav_coords"],
            targets,
            env.max_macro_displacement,
            tolerance_m=0.5,
        )

    return _policy


def centroid_offset_targets(
    centroids,
    offsets,
    map_min_m=0.0,
    map_max_m=1000.0,
):
    centroids = np.asarray(centroids, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64).reshape(centroids.shape)
    return np.clip(
        centroids + offsets,
        float(map_min_m),
        float(map_max_m),
    )


def capacity_ordered_targets(
    centroids,
    medium_offset_m=60.0,
    light_offset_m=120.0,
    map_min_m=0.0,
    map_max_m=1000.0,
):
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.shape[0] < 3:
        raise ValueError("Capacity-ordered placement requires at least 3 UAVs.")

    targets = centroids.copy()
    for index, distance_m in (
        (1, medium_offset_m),
        (2, light_offset_m),
    ):
        toward_heavy = centroids[0] - centroids[index]
        norm = float(np.linalg.norm(toward_heavy))
        if norm > 1e-12:
            targets[index] += (
                toward_heavy / norm * max(float(distance_m), 0.0)
            )
    return np.clip(
        targets,
        float(map_min_m),
        float(map_max_m),
    )


def make_centroid_offset_policy(
    offsets,
    map_min_m=0.0,
    map_max_m=1000.0,
):
    offsets = np.asarray(offsets, dtype=np.float64).copy()

    def _policy(env, _local_obs_batch, _eval_mode):
        raw_state = env.env.get_raw_state()
        targets = centroid_offset_targets(
            raw_state["weighted_centroids"],
            offsets,
            map_min_m=map_min_m,
            map_max_m=map_max_m,
        )
        if env.macro_action_mode == "waypoint":
            return target_waypoint_action(
                targets,
                env.macro_waypoint_origins,
                env.macro_waypoint_radius_m,
            )
        if env.macro_action_mode in (
            "centroid_waypoint",
            "cooperation_waypoint",
        ):
            return target_centroid_relative_action(
                env, targets, raw_state["weighted_centroids"]
            )
        return target_delta_action(
            raw_state["uav_coords"],
            targets,
            env.max_macro_displacement,
            tolerance_m=0.5,
        )

    return _policy


def make_capacity_ordered_policy(
    medium_offset_m=60.0,
    light_offset_m=120.0,
    map_min_m=0.0,
    map_max_m=1000.0,
):
    def _policy(env, _local_obs_batch, _eval_mode):
        raw_state = env.env.get_raw_state()
        targets = capacity_ordered_targets(
            raw_state["weighted_centroids"],
            medium_offset_m=medium_offset_m,
            light_offset_m=light_offset_m,
            map_min_m=map_min_m,
            map_max_m=map_max_m,
        )
        if env.macro_action_mode == "waypoint":
            return target_waypoint_action(
                targets,
                env.macro_waypoint_origins,
                env.macro_waypoint_radius_m,
            )
        if env.macro_action_mode in (
            "centroid_waypoint",
            "cooperation_waypoint",
        ):
            return target_centroid_relative_action(
                env, targets, raw_state["weighted_centroids"]
            )
        return target_delta_action(
            raw_state["uav_coords"],
            targets,
            env.max_macro_displacement,
            tolerance_m=0.5,
        )

    return _policy


def make_uav1_centroid_override_policy(learned_policy):
    def _policy(env, local_obs_batch, eval_mode):
        learned_action = np.asarray(
            learned_policy(env, local_obs_batch, eval_mode),
            dtype=np.float32,
        ).copy()
        raw_state = env.env.get_raw_state()
        centroid_targets = np.asarray(
            raw_state["weighted_centroids"],
            dtype=np.float64,
        )
        if env.macro_action_mode == "waypoint":
            centroid_action = target_waypoint_action(
                centroid_targets,
                env.macro_waypoint_origins,
                env.macro_waypoint_radius_m,
            )
        elif env.macro_action_mode in (
            "centroid_waypoint",
            "cooperation_waypoint",
        ):
            centroid_action = target_centroid_relative_action(
                env, centroid_targets, centroid_targets
            )
        else:
            centroid_action = target_delta_action(
                raw_state["uav_coords"],
                centroid_targets,
                env.max_macro_displacement,
                tolerance_m=0.5,
            )
        learned_action[0] = centroid_action[0]
        return learned_action

    return _policy


def load_reference_geometry(checkpoint_path, reference_trajectory=""):
    if reference_trajectory:
        trajectory_path = Path(resolve_project_path(reference_trajectory))
    else:
        trajectory_path = (
            Path(checkpoint_path).resolve().parents[1]
            / "trajectory_snapshots"
            / "trajectory_final.npz"
        )
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Reference trajectory not found: {trajectory_path}"
        )

    with np.load(trajectory_path) as payload:
        uav_traj = np.asarray(payload["uav_traj"], dtype=np.float64)
        centroid_traj = np.asarray(
            payload["centroid_traj"],
            dtype=np.float64,
        )
    if uav_traj.ndim != 3 or centroid_traj.shape != uav_traj.shape:
        raise ValueError(
            "Reference trajectory must contain matching "
            "[time, UAV, xy] arrays."
        )
    return {
        "path": str(trajectory_path),
        "endpoints_m": uav_traj[-1].copy(),
        "centroids_m": centroid_traj[-1].copy(),
        "offsets_m": (uav_traj[-1] - centroid_traj[-1]).copy(),
    }


PAIRED_METRIC_KEYS = (
    "task_delay_weighted_ms",
    "delay_objective",
    "average_power_w",
    "power_budget_w",
    "average_battery_power_per_uav_w",
    "uav_power_budgets_w",
    "u2g",
    "coop_ratio",
    "coop_flow_mbits",
    "tx_service_matrix_mbits",
    "task_delay_source_uav_ms",
)


def compact_metrics(metrics):
    return {
        key: np.asarray(metrics[key]).copy()
        if isinstance(metrics[key], np.ndarray)
        else float(metrics[key])
        for key in PAIRED_METRIC_KEYS
    }


def summarize_paired_records(records, power_constraint_scope="per_uav"):
    summary = {}
    sample_count = len(records)
    if sample_count == 0:
        return summary

    for key in PAIRED_METRIC_KEYS:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        mean = np.mean(values, axis=0)
        std = (
            np.std(values, axis=0, ddof=1)
            if sample_count > 1
            else np.zeros_like(mean)
        )
        summary[key] = mean
        summary[f"{key}_std"] = std

    if str(power_constraint_scope) == "system":
        powers = np.asarray(
            [record["average_power_w"] for record in records],
            dtype=np.float64,
        )
        budget = float(records[0]["power_budget_w"])
        if sample_count > 1:
            sem = np.std(powers, ddof=1) / np.sqrt(sample_count)
            ucb95 = float(
                np.mean(powers)
                + student_t.ppf(0.95, sample_count - 1) * sem
            )
        else:
            ucb95 = float("inf")
        summary["average_power_ucb95_w"] = ucb95
        summary["power_ucb95_violation_w"] = max(ucb95 - budget, 0.0)
        summary["energy_feasible_ucb95"] = float(ucb95 <= budget)
    else:
        power_per_uav = np.asarray(
            [record["average_battery_power_per_uav_w"] for record in records],
            dtype=np.float64,
        )
        budgets = np.asarray(
            records[0]["uav_power_budgets_w"],
            dtype=np.float64,
        )
        if sample_count > 1:
            sem = np.std(power_per_uav, axis=0, ddof=1) / np.sqrt(sample_count)
            ucb95 = np.mean(power_per_uav, axis=0) + float(
                student_t.ppf(0.95, sample_count - 1)
            ) * sem
        else:
            ucb95 = np.full_like(budgets, np.inf)
        summary["battery_power_ucb95_per_uav_w"] = ucb95
        summary["energy_feasible_ucb95"] = float(np.all(ucb95 <= budgets))
    summary["sample_count"] = int(sample_count)
    return summary


def paired_difference(candidate, reference, key):
    candidate_values = np.asarray(
        [record[key] for record in candidate],
        dtype=np.float64,
    )
    reference_values = np.asarray(
        [record[key] for record in reference],
        dtype=np.float64,
    )
    differences = candidate_values - reference_values
    count = int(differences.shape[0])
    mean = np.mean(differences, axis=0)
    if count > 1:
        sem = np.std(differences, axis=0, ddof=1) / np.sqrt(count)
        half_width = float(student_t.ppf(0.975, count - 1)) * sem
    else:
        half_width = np.full_like(mean, np.inf)
    return {
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "sample_count": count,
    }


def evaluate_paired_policies(
    env,
    checkpoint_args,
    policies,
    micro_policy,
    episodes,
    eval_base_seed,
    reference_name="learned_policy",
):
    if reference_name not in policies:
        raise KeyError(
            f"reference_name '{reference_name}' is not present in policies."
        )
    raw = {name: [] for name in policies}
    for eval_index in range(int(episodes)):
        seed = int(eval_base_seed) + eval_index
        for name, policy in policies.items():
            eval_args = argparse.Namespace(**vars(checkpoint_args))
            eval_args.eval_episodes = 1
            eval_args.eval_base_seed = seed
            metrics = evaluate_policy(
                env,
                eval_args,
                policy,
                micro_policy,
            )
            raw[name].append(compact_metrics(metrics))

    aggregate = {
        name: summarize_paired_records(
            records,
            power_constraint_scope=getattr(
                checkpoint_args,
                "power_constraint_scope",
                "system",
            ),
        )
        for name, records in raw.items()
    }
    differences = {
        name: {
            "task_delay_weighted_ms": paired_difference(
                records,
                raw[reference_name],
                "task_delay_weighted_ms",
            ),
            "delay_objective": paired_difference(
                records,
                raw[reference_name],
                "delay_objective",
            ),
            "average_power_w": paired_difference(
                records,
                raw[reference_name],
                "average_power_w",
            ),
        }
        for name, records in raw.items()
        if name != reference_name
    }
    return raw, aggregate, differences


def evaluate_with_common_random_numbers(
    env,
    checkpoint_args,
    macro_policy,
    micro_policy,
    episodes,
    eval_base_seed,
):
    eval_args = argparse.Namespace(**vars(checkpoint_args))
    eval_args.eval_episodes = int(episodes)
    eval_args.eval_base_seed = int(eval_base_seed)
    return evaluate_policy(
        env,
        eval_args,
        macro_policy,
        micro_policy,
    )


def main():
    args = parse_args()
    (
        checkpoint_path,
        checkpoint_args,
        env,
        actor,
        micro_policy,
        device,
    ) = load_checkpoint_components(args.checkpoint, args.device)
    anchors = np.asarray(env.env.anchor_points, dtype=np.float64)

    def evaluate_target(targets):
        return evaluate_with_common_random_numbers(
            env,
            checkpoint_args,
            make_target_policy(targets),
            micro_policy,
            episodes=args.search_episodes,
            eval_base_seed=args.eval_base_seed,
        )

    def report_progress(summary):
        metrics = summary["global_best_metrics"]
        print(
            f"[CEM {summary['iteration']:02d}/{args.iterations:02d}] "
            f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
            f"objective={metrics['delay_objective']:.6f} "
            f"violation={checkpoint_rank(metrics)[1]:.3f}W"
        )
        print(
            "  targets="
            + np.array2string(
                summary["global_best_targets_m"],
                precision=1,
                suppress_small=True,
            )
        )

    if args.reuse_search_json:
        reuse_path = Path(resolve_project_path(args.reuse_search_json))
        with open(reuse_path, "r", encoding="utf-8") as fp:
            reuse_payload = json.load(fp)
        best_record = {
            "targets_m": np.asarray(
                reuse_payload["best_targets_m"],
                dtype=np.float64,
            ),
            "metrics": {},
        }
        history = reuse_payload.get("search_history", [])
        print(f"[REUSE] Position search loaded from {reuse_path}")
    else:
        best_record, history = cem_search(
            evaluate_target=evaluate_target,
            anchors=anchors,
            population=args.population,
            iterations=args.iterations,
            elite_fraction=args.elite_fraction,
            initial_std_m=args.initial_std_m,
            minimum_std_m=args.minimum_std_m,
            max_offset_m=args.max_offset_m,
            map_min_m=args.map_min_m,
            map_max_m=args.map_max_m,
            seed=args.search_seed,
            progress_callback=report_progress,
        )

    reference_geometry = load_reference_geometry(
        checkpoint_path,
        args.reference_trajectory,
    )
    reference_offsets = np.asarray(
        reference_geometry["offsets_m"],
        dtype=np.float64,
    )
    if reference_offsets.shape != anchors.shape:
        raise ValueError(
            "Reference trajectory UAV count does not match the checkpoint."
        )
    learned_policy = td3_macro_policy(actor)
    policies = {
        "learned_policy": learned_policy,
        "learned_with_uav1_centroid": (
            make_uav1_centroid_override_policy(learned_policy)
        ),
        "capacity_ordered_placement": make_capacity_ordered_policy(
            medium_offset_m=args.medium_offset_m,
            light_offset_m=args.light_offset_m,
            map_min_m=args.map_min_m,
            map_max_m=args.map_max_m,
        ),
        "anchor_placement": make_target_policy(anchors),
        "cem_placement": make_target_policy(best_record["targets_m"]),
    }
    paired_raw, paired_aggregate, paired_differences = (
        evaluate_paired_policies(
            env,
            checkpoint_args,
            policies,
            micro_policy,
            episodes=args.validation_episodes,
            eval_base_seed=args.eval_base_seed,
        )
    )
    search_validation = evaluate_with_common_random_numbers(
        env,
        checkpoint_args,
        make_target_policy(best_record["targets_m"]),
        micro_policy,
        episodes=args.validation_episodes,
        eval_base_seed=args.eval_base_seed,
    )

    output_dir = args.output_dir
    if not output_dir:
        output_dir = (
            Path(checkpoint_path).resolve().parents[1]
            / "position_oracle"
        )
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "position_oracle.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "search_protocol": {
            "population": args.population,
            "iterations": args.iterations,
            "elite_fraction": args.elite_fraction,
            "search_episodes": args.search_episodes,
            "validation_episodes": args.validation_episodes,
            "eval_base_seed": args.eval_base_seed,
            "search_seed": args.search_seed,
            "initial_std_m": args.initial_std_m,
            "minimum_std_m": args.minimum_std_m,
            "max_offset_m": args.max_offset_m,
        },
        "anchors_m": anchors,
        "best_targets_m": best_record["targets_m"],
        "reference_geometry": reference_geometry,
        "counterfactual_protocol": {
            "medium_offset_m": args.medium_offset_m,
            "light_offset_m": args.light_offset_m,
            "uav1_override": (
                "Replace only UAV-1 learned macro action with a direct "
                "move toward its current weighted centroid."
            ),
        },
        "validation": paired_aggregate,
        "paired_differences_vs_learned": paired_differences,
        "paired_raw": paired_raw,
        "search_validation_crosscheck": search_validation,
        "placement_targets": {
            "anchor_placement": anchors,
            "capacity_ordered_reference_centroids": (
                capacity_ordered_targets(
                    reference_geometry["centroids_m"],
                    medium_offset_m=args.medium_offset_m,
                    light_offset_m=args.light_offset_m,
                    map_min_m=args.map_min_m,
                    map_max_m=args.map_max_m,
                )
            ),
            "cem_placement": best_record["targets_m"],
        },
        "search_history": history,
    }
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(_json_ready(payload), fp, indent=2, ensure_ascii=False)

    print(f"[DONE] Position search saved to {output_path}")
    for label, metrics in paired_aggregate.items():
        difference = paired_differences.get(label, {}).get(
            "task_delay_weighted_ms",
            None,
        )
        difference_text = ""
        if difference is not None:
            difference_text = (
                f" delta={float(difference['mean']):+.2f}ms "
                f"CI=[{float(difference['ci95_low']):+.2f},"
                f"{float(difference['ci95_high']):+.2f}]"
            )
        print(
            f"{label}: task_delay={metrics['task_delay_weighted_ms']:.2f}ms "
            f"objective={metrics['delay_objective']:.6f} "
            f"u2g={metrics['u2g']:.2f}m "
            f"feasible={int(metrics['energy_feasible_ucb95'])}"
            f"{difference_text}"
        )


if __name__ == "__main__":
    main()
