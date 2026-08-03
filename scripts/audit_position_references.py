import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = Path(bootstrap())

from search_position_oracle import _json_ready
from search_position_oracle import cem_search
from search_position_oracle import checkpoint_rank
from search_position_oracle import evaluate_paired_policies
from search_position_oracle import evaluate_with_common_random_numbers
from search_position_oracle import load_checkpoint_components
from search_position_oracle import make_centroid_offset_policy
from search_position_oracle import make_target_policy
from search_position_oracle import make_uplink_reference_policy
from search_position_oracle import solve_uplink_reference_positions
from search_position_oracle import uplink_reference_macro_ema_beta


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit anchor, traffic-centroid, U2G-optimal, and approximate "
            "joint UAV positions with common-random-number replay."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--search-episodes", type=int, default=3)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--search-base-seed", type=int, default=64000)
    parser.add_argument("--validation-base-seed", type=int, default=65000)
    parser.add_argument("--search-seed", type=int, default=20260730)
    parser.add_argument("--initial-std-m", type=float, default=35.0)
    parser.add_argument("--minimum-std-m", type=float, default=5.0)
    parser.add_argument("--max-residual-m", type=float, default=100.0)
    parser.add_argument("--directed-distances-m", default="20,40,60")
    parser.add_argument("--priority-high-weight", type=float, default=10.0)
    parser.add_argument("--priority-low-weight", type=float, default=1.0)
    parser.add_argument("--ema-beta", type=float, default=None)
    parser.add_argument("--practical-margin-ms", type=float, default=5.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def parse_distances(raw_value):
    distances = sorted(
        {
            float(value.strip())
            for value in str(raw_value).split(",")
            if value.strip()
        }
    )
    if any(distance < 0.0 for distance in distances):
        raise ValueError("Directed distances must be non-negative.")
    return distances


def compact_audit_metrics(metrics):
    scalar_keys = (
        "J",
        "delay_objective",
        "task_delay_weighted_ms",
        "task_delay_high_ms",
        "task_delay_low_ms",
        "average_power_w",
        "average_power_ucb95_w",
        "power_budget_w",
        "power_ucb95_violation_w",
        "energy_feasible_ucb95",
        "u2g",
        "coop_ratio",
    )
    array_keys = (
        "coop_flow_mbits",
        "tx_service_matrix_mbits",
        "task_delay_source_uav_ms",
    )
    compact = {
        key: float(metrics.get(key, np.nan))
        for key in scalar_keys
    }
    compact["power_constraint_scope"] = str(
        metrics.get("power_constraint_scope", "system")
    )
    for key in array_keys:
        if key in metrics:
            compact[key] = np.asarray(metrics[key], dtype=np.float64)
    return compact


def write_json_atomic(path, payload):
    temporary_path = path.with_suffix(".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def scalar(value):
    return float(np.asarray(value, dtype=np.float64))


def classify_centroid_vs_uplink(difference, margin_ms):
    mean = scalar(difference["mean"])
    low = scalar(difference["ci95_low"])
    high = scalar(difference["ci95_high"])
    margin = float(margin_ms)
    if low >= -margin and high <= margin:
        branch = "traffic_centroid_equivalent_to_uplink_reference"
    elif mean > margin and low > 0.0:
        branch = "uplink_reference_materially_better"
    elif mean < -margin and high < 0.0:
        branch = "traffic_centroid_materially_better"
    else:
        branch = "inconclusive"
    return {
        "difference_definition": (
            "traffic_centroid_delay_minus_uplink_reference_delay"
        ),
        "mean_ms": mean,
        "ci95_low_ms": low,
        "ci95_high_ms": high,
        "equivalence_margin_ms": margin,
        "branch": branch,
    }


def classify_joint_gain(difference, margin_ms):
    difference_mean = scalar(difference["mean"])
    difference_low = scalar(difference["ci95_low"])
    difference_high = scalar(difference["ci95_high"])
    gain_mean = -difference_mean
    gain_low = -difference_high
    gain_high = -difference_low
    margin = float(margin_ms)
    if gain_mean >= margin and gain_low > 0.0:
        branch = "joint_position_has_material_cooperation_gain"
    elif gain_high < margin:
        branch = "no_material_joint_gain_detected"
    else:
        branch = "inconclusive"
    return {
        "gain_definition": (
            "uplink_reference_delay_minus_joint_search_delay"
        ),
        "mean_ms": gain_mean,
        "ci95_low_ms": gain_low,
        "ci95_high_ms": gain_high,
        "material_gain_threshold_ms": margin,
        "statistically_positive": bool(gain_low > 0.0),
        "mean_exceeds_threshold": bool(gain_mean >= margin),
        "branch": branch,
    }


def main():
    cli_args = parse_args()
    (
        checkpoint_path,
        checkpoint_args,
        env,
        _actor,
        micro_policy,
        device,
    ) = load_checkpoint_components(cli_args.checkpoint, cli_args.device)
    num_uavs = int(checkpoint_args.num_agents)
    zero_offsets = np.zeros((num_uavs, 2), dtype=np.float64)
    directed_distances = parse_distances(
        cli_args.directed_distances_m
    )
    ema_beta = (
        uplink_reference_macro_ema_beta(env)
        if cli_args.ema_beta is None
        else float(cli_args.ema_beta)
    )

    output_dir = (
        Path(cli_args.output_dir)
        if cli_args.output_dir
        else Path(checkpoint_path).resolve().parents[1]
        / "position_reference_audit"
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "search_progress.json"
    output_path = output_dir / "position_reference_audit.json"

    def evaluate_policy_once(policy):
        metrics = evaluate_with_common_random_numbers(
            env,
            checkpoint_args,
            policy,
            micro_policy,
            episodes=cli_args.search_episodes,
            eval_base_seed=cli_args.search_base_seed,
        )
        return compact_audit_metrics(metrics)

    search_progress = {
        "checkpoint": str(checkpoint_path),
        "search_protocol": {
            "population": int(cli_args.population),
            "iterations": int(cli_args.iterations),
            "elite_fraction": float(cli_args.elite_fraction),
            "search_episodes": int(cli_args.search_episodes),
            "search_base_seed": int(cli_args.search_base_seed),
            "search_seed": int(cli_args.search_seed),
            "initial_std_m": float(cli_args.initial_std_m),
            "minimum_std_m": float(cli_args.minimum_std_m),
            "max_residual_m": float(cli_args.max_residual_m),
            "directed_distances_m": directed_distances,
        },
        "cem_history": [],
        "candidate_records": [],
    }

    def evaluate_residual(offsets):
        return evaluate_policy_once(
            make_uplink_reference_policy(
                offsets=np.asarray(offsets, dtype=np.float64),
                priority_high_weight=cli_args.priority_high_weight,
                priority_low_weight=cli_args.priority_low_weight,
                ema_beta=ema_beta,
            )
        )

    def report_cem_progress(summary):
        search_progress["cem_history"].append(summary)
        write_json_atomic(progress_path, search_progress)
        metrics = summary["global_best_metrics"]
        print(
            f"[CEM {summary['iteration']:02d}/{cli_args.iterations:02d}] "
            f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
            f"objective={metrics['delay_objective']:.6f} "
            f"violation={checkpoint_rank(metrics)[1]:.3f}W"
        )
        print(
            "  residuals="
            + np.array2string(
                np.asarray(summary["global_best_targets_m"]),
                precision=1,
                suppress_small=True,
            )
        )

    cem_best, cem_history = cem_search(
        evaluate_target=evaluate_residual,
        anchors=zero_offsets,
        population=cli_args.population,
        iterations=cli_args.iterations,
        elite_fraction=cli_args.elite_fraction,
        initial_std_m=cli_args.initial_std_m,
        minimum_std_m=cli_args.minimum_std_m,
        max_offset_m=cli_args.max_residual_m,
        map_min_m=-cli_args.max_residual_m,
        map_max_m=cli_args.max_residual_m,
        seed=cli_args.search_seed,
        progress_callback=report_cem_progress,
    )
    candidate_records = [
        {
            "candidate_id": "cem_best",
            "policy_type": "constant_residual",
            "offsets_m": np.asarray(
                cem_best["targets_m"],
                dtype=np.float64,
            ),
            "metrics": cem_best["metrics"],
        }
    ]

    uplink_metrics = evaluate_residual(zero_offsets)
    candidate_records.append(
        {
            "candidate_id": "uplink_reference",
            "policy_type": "constant_residual",
            "offsets_m": zero_offsets.copy(),
            "metrics": uplink_metrics,
        }
    )
    print(
        "[candidate uplink_reference] "
        f"delay={uplink_metrics['task_delay_weighted_ms']:.2f}ms"
    )

    if num_uavs >= 3:
        for distance_m in directed_distances:
            directed_move = (2, 0, float(distance_m))
            metrics = evaluate_policy_once(
                make_uplink_reference_policy(
                    directed_moves=[directed_move],
                    priority_high_weight=cli_args.priority_high_weight,
                    priority_low_weight=cli_args.priority_low_weight,
                    ema_beta=ema_beta,
                )
            )
            record = {
                "candidate_id": (
                    f"uav3_toward_uav1_{int(round(distance_m)):03d}m"
                ),
                "policy_type": "directed_dynamic_residual",
                "directed_moves": [directed_move],
                "metrics": metrics,
            }
            candidate_records.append(record)
            print(
                f"[candidate {record['candidate_id']}] "
                f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
                f"violation={checkpoint_rank(metrics)[1]:.3f}W"
            )

    candidate_records.sort(key=lambda record: checkpoint_rank(record["metrics"]))
    selected = candidate_records[0]
    search_progress["cem_history"] = cem_history
    search_progress["candidate_records"] = candidate_records
    search_progress["selected_candidate"] = selected
    write_json_atomic(progress_path, search_progress)

    if selected["policy_type"] == "constant_residual":
        joint_policy = make_uplink_reference_policy(
            offsets=np.asarray(selected["offsets_m"], dtype=np.float64),
            priority_high_weight=cli_args.priority_high_weight,
            priority_low_weight=cli_args.priority_low_weight,
            ema_beta=ema_beta,
        )
    else:
        joint_policy = make_uplink_reference_policy(
            directed_moves=selected["directed_moves"],
            priority_high_weight=cli_args.priority_high_weight,
            priority_low_weight=cli_args.priority_low_weight,
            ema_beta=ema_beta,
        )

    policies = {
        "fixed_anchors": make_target_policy(
            np.asarray(env.env.anchor_points, dtype=np.float64)
        ),
        "traffic_centroid": make_centroid_offset_policy(zero_offsets),
        "uplink_reference": make_uplink_reference_policy(
            offsets=zero_offsets,
            priority_high_weight=cli_args.priority_high_weight,
            priority_low_weight=cli_args.priority_low_weight,
            ema_beta=ema_beta,
        ),
        "joint_search": joint_policy,
    }
    paired_raw, paired_aggregate, paired_differences = (
        evaluate_paired_policies(
            env,
            checkpoint_args,
            policies,
            micro_policy,
            episodes=cli_args.validation_episodes,
            eval_base_seed=cli_args.validation_base_seed,
            reference_name="uplink_reference",
        )
    )

    env.reset(seed=cli_args.validation_base_seed)
    initial_uplink_targets, initial_solver_diagnostics = (
        solve_uplink_reference_positions(
            env,
            initial_positions=env.env.weighted_centroids,
            priority_high_weight=cli_args.priority_high_weight,
            priority_low_weight=cli_args.priority_low_weight,
        )
    )
    centroid_difference = paired_differences["traffic_centroid"][
        "task_delay_weighted_ms"
    ]
    joint_difference = paired_differences["joint_search"][
        "task_delay_weighted_ms"
    ]
    decisions = {
        "centroid_vs_uplink_reference": classify_centroid_vs_uplink(
            centroid_difference,
            cli_args.practical_margin_ms,
        ),
        "joint_gain_over_uplink_reference": classify_joint_gain(
            joint_difference,
            cli_args.practical_margin_ms,
        ),
    }
    payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "reference_definition": {
            "name": "priority_weighted_uplink_time_reference",
            "objective": (
                "sum_k (w_high*lambda_high*b_high + "
                "w_low*lambda_low*b_low) / R_k(q)"
            ),
            "priority_high_weight": float(cli_args.priority_high_weight),
            "priority_low_weight": float(cli_args.priority_low_weight),
            "causal_inputs": (
                "current user coordinates and configured arrival statistics"
            ),
            "ema_beta_per_macro": float(ema_beta),
            "ema_source": (
                "automatic equivalent of the existing traffic-centroid EMA"
                if cli_args.ema_beta is None
                else "explicit command-line value"
            ),
        },
        "validation_protocol": {
            "positions": [
                "fixed_anchors",
                "traffic_centroid",
                "uplink_reference",
                "joint_search",
            ],
            "validation_episodes": int(cli_args.validation_episodes),
            "validation_base_seed": int(cli_args.validation_base_seed),
            "common_random_numbers": True,
            "practical_margin_ms": float(cli_args.practical_margin_ms),
            "frozen_micro_policy": True,
            "training_or_scene_modified": False,
        },
        "initial_geometry": {
            "anchors_m": np.asarray(
                env.env.anchor_points,
                dtype=np.float64,
            ),
            "traffic_centroids_m": np.asarray(
                env.env.weighted_centroids,
                dtype=np.float64,
            ),
            "uplink_reference_m": initial_uplink_targets,
            "solver_diagnostics": initial_solver_diagnostics,
        },
        "search": search_progress,
        "validation": paired_aggregate,
        "paired_differences_vs_uplink_reference": paired_differences,
        "paired_raw": paired_raw,
        "decisions": decisions,
    }
    write_json_atomic(output_path, payload)

    print(f"[DONE] {output_path}")
    print(f"[SELECTED] {selected['candidate_id']}")
    for name, metrics in paired_aggregate.items():
        print(
            f"{name}: delay="
            f"{scalar(metrics['task_delay_weighted_ms']):.2f}ms "
            f"objective={scalar(metrics['delay_objective']):.6f} "
            f"power={scalar(metrics['average_power_w']):.2f}W "
            f"u2g={scalar(metrics['u2g']):.2f}m "
            f"feasible={int(scalar(metrics['energy_feasible_ucb95']))}"
        )
    print(
        "[DECISION centroid-vs-uplink] "
        + decisions["centroid_vs_uplink_reference"]["branch"]
    )
    print(
        "[DECISION joint-gain] "
        + decisions["joint_gain_over_uplink_reference"]["branch"]
    )


if __name__ == "__main__":
    main()
