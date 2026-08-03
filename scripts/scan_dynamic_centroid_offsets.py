import argparse
import itertools
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
from search_position_oracle import checkpoint_rank
from search_position_oracle import evaluate_paired_policies
from search_position_oracle import evaluate_with_common_random_numbers
from search_position_oracle import load_checkpoint_components
from search_position_oracle import make_centroid_offset_policy
from train_centralized_td3 import td3_macro_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan deterministic offsets around the time-varying traffic "
            "centroids while replaying a frozen SAC policy."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--radii-m", default="0,20,40,60,80")
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--search-episodes", type=int, default=3)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--eval-base-seed", type=int, default=63000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_radii(raw_value):
    radii = sorted(
        {
            float(value.strip())
            for value in str(raw_value).split(",")
            if value.strip()
        }
    )
    if not radii or radii[0] < 0.0:
        raise ValueError("Radii must be a non-empty list of non-negative values.")
    return radii


def offset_candidates(radii, directions):
    candidates = [
        {
            "candidate_id": "r000_a000",
            "radius_m": 0.0,
            "angle_deg": 0.0,
            "offset_m": np.zeros(2, dtype=np.float64),
        }
    ]
    for radius in radii:
        if radius <= 0.0:
            continue
        for direction_index in range(int(directions)):
            angle_rad = 2.0 * np.pi * direction_index / int(directions)
            angle_deg = 360.0 * direction_index / int(directions)
            candidates.append(
                {
                    "candidate_id": (
                        f"r{int(round(radius)):03d}_"
                        f"a{int(round(angle_deg)):03d}"
                    ),
                    "radius_m": float(radius),
                    "angle_deg": float(angle_deg),
                    "offset_m": np.asarray(
                        [
                            radius * np.cos(angle_rad),
                            radius * np.sin(angle_rad),
                        ],
                        dtype=np.float64,
                    ),
                }
            )
    return candidates


def compact_scan_metrics(metrics):
    keys = (
        "J",
        "delay_objective",
        "task_delay_weighted_ms",
        "task_delay_high_ms",
        "task_delay_low_ms",
        "average_power_w",
        "average_power_ucb95_w",
        "power_ucb95_violation_w",
        "energy_feasible_ucb95",
        "u2g",
        "coop_ratio",
    )
    return {
        key: float(metrics.get(key, np.nan))
        for key in keys
    }


def scan_rank(record):
    return checkpoint_rank(record["metrics"])


def write_progress(path, payload):
    temporary_path = path.with_suffix(".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def evaluate_offsets(
    env,
    checkpoint_args,
    micro_policy,
    offsets,
    episodes,
    eval_base_seed,
):
    metrics = evaluate_with_common_random_numbers(
        env,
        checkpoint_args,
        make_centroid_offset_policy(offsets),
        micro_policy,
        episodes=episodes,
        eval_base_seed=eval_base_seed,
    )
    return compact_scan_metrics(metrics)


def main():
    cli_args = parse_args()
    (
        checkpoint_path,
        checkpoint_args,
        env,
        actor,
        micro_policy,
        device,
    ) = load_checkpoint_components(cli_args.checkpoint, cli_args.device)
    num_uavs = int(checkpoint_args.num_agents)
    radii = parse_radii(cli_args.radii_m)
    candidates = offset_candidates(radii, cli_args.directions)
    output_dir = (
        Path(cli_args.output_dir)
        if cli_args.output_dir
        else Path(checkpoint_path).resolve().parents[1]
        / "dynamic_centroid_scan"
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "scan_progress.json"
    final_path = output_dir / "dynamic_centroid_scan.json"

    progress = {
        "checkpoint": str(checkpoint_path),
        "search_protocol": {
            "radii_m": radii,
            "directions": int(cli_args.directions),
            "search_episodes": int(cli_args.search_episodes),
            "validation_episodes": int(cli_args.validation_episodes),
            "eval_base_seed": int(cli_args.eval_base_seed),
            "top_k": int(cli_args.top_k),
            "other_uavs": (
                "deterministic zero-offset tracking of their own dynamic "
                "centroids under the same 20 m/period flight limit"
            ),
        },
        "single_uav_scan": {},
        "joint_scan": {},
    }
    if cli_args.resume and progress_path.is_file():
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        print(f"[RESUME] {progress_path}")

    for uav_index in range(num_uavs):
        uav_key = f"uav{uav_index + 1}"
        existing = {
            record["candidate_id"]: record
            for record in progress["single_uav_scan"].get(uav_key, [])
        }
        records = list(existing.values())
        for candidate_index, candidate in enumerate(candidates):
            if candidate["candidate_id"] in existing:
                continue
            offsets = np.zeros((num_uavs, 2), dtype=np.float64)
            offsets[uav_index] = candidate["offset_m"]
            metrics = evaluate_offsets(
                env,
                checkpoint_args,
                micro_policy,
                offsets,
                cli_args.search_episodes,
                cli_args.eval_base_seed,
            )
            record = {
                "candidate_id": candidate["candidate_id"],
                "radius_m": candidate["radius_m"],
                "angle_deg": candidate["angle_deg"],
                "offset_m": candidate["offset_m"],
                "metrics": metrics,
            }
            records.append(record)
            progress["single_uav_scan"][uav_key] = records
            write_progress(progress_path, progress)
            rank = scan_rank(record)
            print(
                f"[{uav_key} {candidate_index + 1:02d}/"
                f"{len(candidates):02d}] "
                f"offset={np.array2string(candidate['offset_m'], precision=1)} "
                f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
                f"violation={rank[1]:.3f}W"
            )
        records.sort(key=scan_rank)
        progress["single_uav_scan"][uav_key] = records
        write_progress(progress_path, progress)

    top_candidates = []
    for uav_index in range(num_uavs):
        records = progress["single_uav_scan"][f"uav{uav_index + 1}"]
        top_candidates.append(records[: max(int(cli_args.top_k), 1)])

    existing_joint = {
        record["candidate_id"]: record
        for record in progress.get("joint_scan", {}).get("records", [])
    }
    joint_records = list(existing_joint.values())
    combinations = list(itertools.product(*top_candidates))
    for combination_index, combination in enumerate(combinations):
        candidate_id = "__".join(
            record["candidate_id"] for record in combination
        )
        if candidate_id in existing_joint:
            continue
        offsets = np.stack(
            [
                np.asarray(record["offset_m"], dtype=np.float64)
                for record in combination
            ],
            axis=0,
        )
        metrics = evaluate_offsets(
            env,
            checkpoint_args,
            micro_policy,
            offsets,
            cli_args.search_episodes,
            cli_args.eval_base_seed,
        )
        record = {
            "candidate_id": candidate_id,
            "offsets_m": offsets,
            "metrics": metrics,
        }
        joint_records.append(record)
        progress["joint_scan"] = {"records": joint_records}
        write_progress(progress_path, progress)
        rank = scan_rank(record)
        print(
            f"[joint {combination_index + 1:02d}/"
            f"{len(combinations):02d}] "
            f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
            f"violation={rank[1]:.3f}W"
        )
    joint_records.sort(key=scan_rank)
    progress["joint_scan"] = {"records": joint_records}
    write_progress(progress_path, progress)

    independent_offsets = np.stack(
        [
            np.asarray(records[0]["offset_m"], dtype=np.float64)
            for records in top_candidates
        ],
        axis=0,
    )
    joint_best_offsets = np.asarray(
        joint_records[0]["offsets_m"],
        dtype=np.float64,
    )
    policies = {
        "learned_policy": td3_macro_policy(actor),
        "centroid_tracking": make_centroid_offset_policy(
            np.zeros((num_uavs, 2), dtype=np.float64)
        ),
        "independent_best_offsets": make_centroid_offset_policy(
            independent_offsets
        ),
        "joint_best_offsets": make_centroid_offset_policy(
            joint_best_offsets
        ),
    }
    paired_raw, paired_aggregate, paired_differences = (
        evaluate_paired_policies(
            env,
            checkpoint_args,
            policies,
            micro_policy,
            episodes=cli_args.validation_episodes,
            eval_base_seed=(
                int(cli_args.eval_base_seed)
                + int(cli_args.search_episodes)
                + 1000
            ),
        )
    )
    payload = {
        **progress,
        "device": str(device),
        "candidate_count_per_uav": len(candidates),
        "independent_best_offsets_m": independent_offsets,
        "joint_best_offsets_m": joint_best_offsets,
        "validation": paired_aggregate,
        "paired_differences_vs_learned": paired_differences,
        "paired_raw": paired_raw,
    }
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
    print(f"[DONE] {final_path}")
    for name, metrics in paired_aggregate.items():
        print(
            f"{name}: delay={float(metrics['task_delay_weighted_ms']):.2f}ms "
            f"objective={float(metrics['delay_objective']):.6f} "
            f"power={float(metrics['average_power_w']):.2f}W "
            f"u2g={float(metrics['u2g']):.2f}m "
            f"feasible={int(metrics['energy_feasible_ucb95'])}"
        )


if __name__ == "__main__":
    main()
