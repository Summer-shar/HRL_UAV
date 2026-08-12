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
from compare_common import resolve_project_path
from compare_common import torch_load_checkpoint
import hrl_uav.agents.sac_agent as sac_agent_mod


DEFAULT_CHECKPOINT = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_LEARNED_ROWS = (
    "result_archive/v24_system_dual_pilot_trajectory_audit/per_seed_raw.csv"
)
DEFAULT_OUTPUT = "result_archive/v24_centroid_reference_audit"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a zero-residual dynamic-centroid macro reference with "
            "the v24 dual-on checkpoint's SAC policy and frozen lambda."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--learned-rows", default=DEFAULT_LEARNED_ROWS)
    parser.add_argument("--learned-arm", default="dual_on")
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


def read_learned_rows(path, seeds, learned_arm="dual_on"):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["arm"] == learned_arm and int(row["seed"]) in seeds
        ]
    converted = []
    for row in rows:
        parsed = {"arm": row["arm"], "seed": int(row["seed"])}
        for key, value in row.items():
            if key in ("arm", "seed", "centroid_trace_hash", "users_trace_hash"):
                continue
            parsed[key] = float(value)
        parsed["centroid_trace_hash"] = row["centroid_trace_hash"]
        parsed["users_trace_hash"] = row["users_trace_hash"]
        converted.append(parsed)
    if {row["seed"] for row in converted} != set(seeds):
        raise RuntimeError("The learned-policy audit is missing requested seeds.")
    return sorted(converted, key=lambda row: row["seed"])


def summarize(learned_rows, centroid_rows):
    learned = {row["seed"]: row for row in learned_rows}
    centroid = {row["seed"]: row for row in centroid_rows}
    if set(learned) != set(centroid):
        raise RuntimeError("Learned and centroid seed sets do not match.")
    metrics = (
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
    paired_rows = []
    summary = {}
    for seed in sorted(learned):
        left = learned[seed]
        right = centroid[seed]
        trace_match = (
            left["centroid_trace_hash"] == right["centroid_trace_hash"]
            and left["users_trace_hash"] == right["users_trace_hash"]
        )
        if not trace_match:
            raise RuntimeError(f"External trajectory mismatch for seed {seed}.")
        row = {"seed": seed, "external_trajectory_match": True}
        for metric in metrics:
            row[f"{metric}_learned"] = left[metric]
            row[f"{metric}_centroid"] = right[metric]
            row[f"{metric}_learned_minus_centroid"] = (
                left[metric] - right[metric]
            )
        paired_rows.append(row)

    for metric in metrics:
        learned_values = np.asarray(
            [learned[seed][metric] for seed in sorted(learned)],
            dtype=np.float64,
        )
        centroid_values = np.asarray(
            [centroid[seed][metric] for seed in sorted(centroid)],
            dtype=np.float64,
        )
        deltas = learned_values - centroid_values
        n = int(deltas.size)
        sem = float(stats.sem(deltas)) if n > 1 else 0.0
        critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        mean_delta = float(np.mean(deltas))
        summary[metric] = {
            "n": n,
            "learned_mean": float(np.mean(learned_values)),
            "centroid_mean": float(np.mean(centroid_values)),
            "learned_minus_centroid_mean": mean_delta,
            "learned_minus_centroid_ci95_low": mean_delta - critical * sem,
            "learned_minus_centroid_ci95_high": mean_delta + critical * sem,
            "paired_p_value": float(
                stats.ttest_rel(learned_values, centroid_values).pvalue
            )
            if n > 1
            else None,
        }
    return paired_rows, summary


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    device = torch.device(args.device)
    sac_agent_mod.device = device
    checkpoint = project_path(args.checkpoint)
    payload = torch_load_checkpoint(str(checkpoint))
    checkpoint_args = payload["args"]
    base_seed = int(checkpoint_args["eval_base_seed"])
    seeds = list(range(base_seed, base_seed + int(args.episodes)))
    learned_rows = read_learned_rows(
        project_path(args.learned_rows),
        seeds,
        learned_arm=args.learned_arm,
    )
    centroid_rows = replay_checkpoint(
        "centroid_reference",
        str(checkpoint),
        seeds,
        device,
        macro_policy_kind="centroid_zero",
    )
    paired_rows, summary = summarize(learned_rows, centroid_rows)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_raw.csv", learned_rows + centroid_rows)
    write_csv(output_dir / "paired_deltas.csv", paired_rows)
    result = {
        "status": "READ_ONLY_REFERENCE",
        "checkpoint": str(checkpoint),
        "frozen_system_lambda": float(
            checkpoint_args["energy_lagrange_multiplier_current"]
        ),
        "eval_seeds": seeds,
        "external_trajectory_match_all": True,
        "macro_reference": "zero residual to current dynamic business centroid",
        "metrics": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("\nPaired learned - centroid summary")
    for metric, values in summary.items():
        print(
            f"  {metric}: {values['learned_minus_centroid_mean']:.6f} "
            f"[{values['learned_minus_centroid_ci95_low']:.6f}, "
            f"{values['learned_minus_centroid_ci95_high']:.6f}], "
            f"p={values['paired_p_value']:.6g}"
        )
    print(f"Saved reference audit -> {output_dir}")


if __name__ == "__main__":
    main()
