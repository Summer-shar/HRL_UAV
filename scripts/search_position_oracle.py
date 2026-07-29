import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
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
        if env.macro_action_mode == "centroid_waypoint":
            return target_centroid_waypoint_action(
                targets,
                raw_state["weighted_centroids"],
                env.macro_waypoint_radius_m,
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
        if env.macro_action_mode == "centroid_waypoint":
            return target_centroid_waypoint_action(
                targets,
                raw_state["weighted_centroids"],
                env.macro_waypoint_radius_m,
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
        if env.macro_action_mode == "centroid_waypoint":
            return target_centroid_waypoint_action(
                targets,
                raw_state["weighted_centroids"],
                env.macro_waypoint_radius_m,
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
        elif env.macro_action_mode == "centroid_waypoint":
            centroid_action = target_centroid_waypoint_action(
                centroid_targets,
                centroid_targets,
                env.macro_waypoint_radius_m,
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
):
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
    reference_name = "learned_policy"
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
