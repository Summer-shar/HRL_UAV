import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from audit_dual_pilot_trajectories import replay_checkpoint
from audit_dual_pilot_trajectories import write_csv
from compare_common import torch_load_checkpoint
import hrl_uav.agents.sac_agent as sac_agent_mod


DEFAULT_COVERAGE_CHECKPOINT = (
    "results/tcom_dynamic_u2u_sparse_v25_dual_neutral_coverage/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_COVERAGE_TRAIN_LOG = (
    "results/tcom_dynamic_u2u_sparse_v25_dual_neutral_coverage/pilot/seed1/"
    "joint_td3_sac/train_td3.csv"
)
DEFAULT_CONTROL_ROWS = (
    "result_archive/v24_system_dual_pilot_trajectory_audit/per_seed_raw.csv"
)
DEFAULT_CENTROID_ROWS = (
    "result_archive/v24_centroid_reference_audit/per_seed_raw.csv"
)
DEFAULT_OUTPUT = "result_archive/v25_neutral_coverage_pilot_audit"

METRICS = (
    "J",
    "task_delay_weighted_ms",
    "average_power_w",
    "propulsion_energy_j",
    "compute_energy_j",
    "mean_centroid_distance_m",
    "p95_centroid_distance_m",
    "path_total_m",
    "net_total_m",
)

DELAY_THRESHOLD_MS = 763.076310
CENTROID_DISTANCE_THRESHOLD_M = 14.732044
POWER_BUDGET_W = 840.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pre-registered v25 joint-neutral coverage pilot against "
            "the v24 dual-on control and same-architecture centroid reference."
        )
    )
    parser.add_argument(
        "--coverage-checkpoint", default=DEFAULT_COVERAGE_CHECKPOINT
    )
    parser.add_argument("--coverage-train-log", default=DEFAULT_COVERAGE_TRAIN_LOG)
    parser.add_argument("--control-rows", default=DEFAULT_CONTROL_ROWS)
    parser.add_argument("--centroid-rows", default=DEFAULT_CENTROID_ROWS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_arm_rows(path, arm, seeds):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        source = [
            row
            for row in csv.DictReader(handle)
            if row["arm"] == arm and int(row["seed"]) in seeds
        ]
    rows = []
    for row in source:
        parsed = {"arm": arm, "seed": int(row["seed"])}
        for key, value in row.items():
            if key in ("arm", "seed", "centroid_trace_hash", "users_trace_hash"):
                continue
            parsed[key] = float(value)
        parsed["centroid_trace_hash"] = row["centroid_trace_hash"]
        parsed["users_trace_hash"] = row["users_trace_hash"]
        rows.append(parsed)
    if {row["seed"] for row in rows} != set(seeds):
        raise RuntimeError(f"{path} is missing requested {arm} seeds.")
    return sorted(rows, key=lambda row: row["seed"])


def paired_statistics(left_rows, right_rows, left_label, right_label):
    left = {row["seed"]: row for row in left_rows}
    right = {row["seed"]: row for row in right_rows}
    if set(left) != set(right):
        raise RuntimeError("Paired seed sets do not match.")

    per_seed = []
    summary = {}
    for seed in sorted(left):
        left_row = left[seed]
        right_row = right[seed]
        trace_match = (
            left_row["centroid_trace_hash"]
            == right_row["centroid_trace_hash"]
            and left_row["users_trace_hash"] == right_row["users_trace_hash"]
        )
        if not trace_match:
            raise RuntimeError(f"External trajectory mismatch for seed {seed}.")
        row = {"seed": seed, "external_trajectory_match": True}
        for metric in METRICS:
            row[f"{metric}_{left_label}"] = left_row[metric]
            row[f"{metric}_{right_label}"] = right_row[metric]
            row[f"{metric}_{left_label}_minus_{right_label}"] = (
                left_row[metric] - right_row[metric]
            )
        per_seed.append(row)

    for metric in METRICS:
        left_values = np.asarray(
            [left[seed][metric] for seed in sorted(left)], dtype=np.float64
        )
        right_values = np.asarray(
            [right[seed][metric] for seed in sorted(right)], dtype=np.float64
        )
        deltas = left_values - right_values
        n = int(deltas.size)
        sem = float(stats.sem(deltas)) if n > 1 else 0.0
        critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        mean_delta = float(np.mean(deltas))
        summary[metric] = {
            "n": n,
            f"{left_label}_mean": float(np.mean(left_values)),
            f"{right_label}_mean": float(np.mean(right_values)),
            f"{left_label}_minus_{right_label}_mean": mean_delta,
            f"{left_label}_minus_{right_label}_ci95_low": (
                mean_delta - critical * sem
            ),
            f"{left_label}_minus_{right_label}_ci95_high": (
                mean_delta + critical * sem
            ),
            "paired_p_value": float(
                stats.ttest_rel(left_values, right_values).pvalue
            )
            if n > 1
            else None,
        }
    return per_seed, summary


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return float(np.mean(values))
    return float(
        np.mean(values)
        + stats.t.ppf(0.95, values.size - 1) * stats.sem(values)
    )


def read_coverage_diagnostics(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Coverage training log is empty.")
    final = rows[-1]
    return {
        "episodes": len(rows),
        "neutral_episode_count": int(
            sum(int(float(row["td3_neutral_episode"])) for row in rows)
        ),
        "neutral_episode_fraction": float(
            final["td3_neutral_episode_fraction_cumulative"]
        ),
        "replay_all_uavs_le_0p10_fraction": float(
            final["td3_replay_all_uavs_le_0p10_fraction"]
        ),
        "replay_all_uavs_le_0p25_fraction": float(
            final["td3_replay_all_uavs_le_0p25_fraction"]
        ),
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    device = torch.device(args.device)
    sac_agent_mod.device = device
    checkpoint = project_path(args.coverage_checkpoint)
    payload = torch_load_checkpoint(str(checkpoint))
    base_seed = int(payload["args"]["eval_base_seed"])
    seeds = list(range(base_seed, base_seed + int(args.episodes)))

    control_rows = read_arm_rows(
        project_path(args.control_rows), "dual_on", seeds
    )
    centroid_rows = read_arm_rows(
        project_path(args.centroid_rows), "centroid_reference", seeds
    )
    coverage_rows = replay_checkpoint(
        "neutral_coverage", str(checkpoint), seeds, device
    )
    paired_rows, versus_control = paired_statistics(
        coverage_rows, control_rows, "coverage", "control"
    )
    _, versus_centroid = paired_statistics(
        coverage_rows, centroid_rows, "coverage", "centroid"
    )

    delay_control = versus_control["task_delay_weighted_ms"]["control_mean"]
    delay_coverage = versus_control["task_delay_weighted_ms"]["coverage_mean"]
    delay_centroid = versus_centroid["task_delay_weighted_ms"]["centroid_mean"]
    distance_control = versus_control["mean_centroid_distance_m"]["control_mean"]
    distance_coverage = versus_control["mean_centroid_distance_m"]["coverage_mean"]
    distance_centroid = versus_centroid["mean_centroid_distance_m"]["centroid_mean"]
    delay_recovery = (delay_control - delay_coverage) / (
        delay_control - delay_centroid
    )
    distance_recovery = (distance_control - distance_coverage) / (
        distance_control - distance_centroid
    )
    power_ucb95 = one_sided_ucb95(
        [row["average_power_w"] for row in coverage_rows]
    )
    delay_ci_high = versus_control["task_delay_weighted_ms"][
        "coverage_minus_control_ci95_high"
    ]
    distance_ci_high = versus_control["mean_centroid_distance_m"][
        "coverage_minus_control_ci95_high"
    ]
    coverage_diagnostics = read_coverage_diagnostics(
        project_path(args.coverage_train_log)
    )

    criteria = {
        "delay_recovery_at_least_80_percent": bool(
            delay_coverage <= DELAY_THRESHOLD_MS
        ),
        "delay_paired_ci_below_zero": bool(delay_ci_high < 0.0),
        "centroid_recovery_at_least_80_percent": bool(
            distance_coverage <= CENTROID_DISTANCE_THRESHOLD_M
        ),
        "centroid_paired_ci_below_zero": bool(distance_ci_high < 0.0),
        "power_ucb95_within_budget": bool(power_ucb95 <= POWER_BUDGET_W),
        "coverage_diagnostics_present": bool(
            coverage_diagnostics["neutral_episode_count"] > 0
            and coverage_diagnostics["replay_all_uavs_le_0p10_fraction"] > 0.0
        ),
    }
    verdict = "PASS" if all(criteria.values()) else "FAIL"

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_raw.csv", coverage_rows)
    write_csv(output_dir / "paired_deltas.csv", paired_rows)
    result = {
        "status": "PRE_REGISTERED_PILOT_VERDICT",
        "verdict": verdict,
        "checkpoint": str(checkpoint),
        "eval_seeds": seeds,
        "external_trajectory_match_all": True,
        "fixed_thresholds": {
            "delay_ms_max": DELAY_THRESHOLD_MS,
            "mean_centroid_distance_m_max": CENTROID_DISTANCE_THRESHOLD_M,
            "power_ucb95_w_max": POWER_BUDGET_W,
        },
        "observed": {
            "delay_ms": delay_coverage,
            "delay_recovery_fraction": float(delay_recovery),
            "mean_centroid_distance_m": distance_coverage,
            "centroid_distance_recovery_fraction": float(distance_recovery),
            "average_power_w": float(
                np.mean([row["average_power_w"] for row in coverage_rows])
            ),
            "power_ucb95_w": power_ucb95,
        },
        "criteria": criteria,
        "coverage_diagnostics": coverage_diagnostics,
        "versus_v24_control": versus_control,
        "versus_centroid_reference": versus_centroid,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("\nv25 neutral coverage verdict")
    print(f"  verdict: {verdict}")
    print(
        f"  delay: {delay_coverage:.6f} ms, recovery={delay_recovery:.3%}, "
        f"threshold<={DELAY_THRESHOLD_MS:.6f}"
    )
    print(
        f"  centroid: {distance_coverage:.6f} m, "
        f"recovery={distance_recovery:.3%}, "
        f"threshold<={CENTROID_DISTANCE_THRESHOLD_M:.6f}"
    )
    print(
        f"  power: mean={result['observed']['average_power_w']:.6f} W, "
        f"UCB95={power_ucb95:.6f} W"
    )
    print(f"  criteria: {criteria}")
    print(f"Saved audit -> {output_dir}")


if __name__ == "__main__":
    main()
