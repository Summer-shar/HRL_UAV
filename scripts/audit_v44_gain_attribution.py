"""Attribute the v44 gain to macro motion, U2U cooperation, and interaction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_joint_hrl_v1_matched import (  # noqa: E402
    load_policies,
    make_eval_env,
    mean_ci95,
)
from compare_common import evaluate_policy  # noqa: E402
from run_v34_cooperative_mobility_gate import V34MacroPolicy  # noqa: E402
from search_position_oracle import make_centroid_offset_policy  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v44_joint_cooperation_finetune"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v44_gain_attribution"
MACRO_POLICIES = ("fixed", "centroid", "learned")
ROUTING_MODES = ("cooperative", "local_only")


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_macro_policy(name: str, learned_policy):
    if name == "fixed":
        return V34MacroPolicy("fixed", 0.0, 0.0)
    if name == "centroid":
        return make_centroid_offset_policy(np.zeros((3, 2)))
    if name == "learned":
        return learned_policy
    raise ValueError(name)


def scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64))


def vector_columns(prefix: str, value) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return {f"{prefix}_uav{index + 1}": float(item) for index, item in enumerate(array)}


def evaluate_arm(
    checkpoint_args,
    macro_policy,
    micro_policy,
    seed: int,
    force_local: bool,
) -> dict:
    env = make_eval_env(checkpoint_args, record_trajectory=False)
    env.force_local_routing = bool(force_local)
    eval_args = argparse.Namespace(**vars(checkpoint_args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(env, eval_args, macro_policy, micro_policy)
    flow = np.asarray(metrics["coop_flow_mbits"], dtype=np.float64)
    cross_flow = float(np.sum(flow) - np.trace(flow))
    total_flow = float(np.sum(flow))
    result = {
        "task_delay_weighted_ms": scalar(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": scalar(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": scalar(metrics["task_delay_low_ms"]),
        "task_delay_sensor_ms": scalar(metrics["task_delay_sensor_ms"]),
        "task_delay_controller_ms": scalar(metrics["task_delay_controller_ms"]),
        "task_delay_vision_ms": scalar(metrics["task_delay_vision_ms"]),
        "J": scalar(metrics["J"]),
        "queue_mbits": scalar(metrics["q_high"]) + scalar(metrics["q_low"]),
        "queue_high_mbits": scalar(metrics["q_high"]),
        "queue_low_mbits": scalar(metrics["q_low"]),
        "average_power_w": scalar(metrics["average_power_w"]),
        "u2g_distance_m": scalar(metrics["u2g"]),
        "cpu_util": scalar(metrics["cpu_util"]),
        "f_total_ghz": scalar(metrics["f_total_ghz"]),
        "pressure_imbalance": scalar(metrics["pressure_imbalance"]),
        "coop_cross_flow_mbits": cross_flow,
        "coop_total_flow_mbits": total_flow,
        "coop_ratio_recomputed": cross_flow / max(total_flow, 1e-12),
        "compute_service_high_mbits": scalar(metrics["compute_service_high_mbits"]),
        "compute_service_low_mbits": scalar(metrics["compute_service_low_mbits"]),
        "task_completion_rate": scalar(metrics["task_completion_rate"]),
        "propulsion_energy_j": scalar(metrics["energy_propulsion_episode"]),
        "compute_energy_j": scalar(metrics["energy_compute_episode"]),
        "u2u_energy_j": scalar(metrics["energy_u2u_episode"]),
        **vector_columns("queue", metrics["q_total_per_uav"]),
        **vector_columns("source_delay_ms", metrics["task_delay_source_uav_ms"]),
        **vector_columns("compute_service_mbits", metrics["compute_service_per_uav_mbits"]),
        **vector_columns("arrival_mbits", metrics["arrival_total_per_uav"]),
    }
    return result


def summarize(rows: list[dict], seeds: list[int]) -> dict:
    numeric_fields = [
        key
        for key in rows[0]
        if key not in {"seed", "macro_policy", "routing_mode"}
    ]
    by_arm = {
        (macro, routing): {
            int(row["seed"]): row
            for row in rows
            if row["macro_policy"] == macro and row["routing_mode"] == routing
        }
        for macro in MACRO_POLICIES
        for routing in ROUTING_MODES
    }
    means = {
        f"{macro}_{routing}": {
            field: float(
                np.mean([float(by_arm[(macro, routing)][seed][field]) for seed in seeds])
            )
            for field in numeric_fields
        }
        for macro in MACRO_POLICIES
        for routing in ROUTING_MODES
    }

    paired = {}
    for macro in MACRO_POLICIES:
        for field in numeric_fields:
            values = [
                float(by_arm[(macro, "local_only")][seed][field])
                - float(by_arm[(macro, "cooperative")][seed][field])
                for seed in seeds
            ]
            paired[f"cooperation_gain_{macro}_{field}"] = mean_ci95(values)
    for routing in ROUTING_MODES:
        for baseline in ("fixed", "centroid"):
            for field in numeric_fields:
                values = [
                    float(by_arm[(baseline, routing)][seed][field])
                    - float(by_arm[("learned", routing)][seed][field])
                    for seed in seeds
                ]
                paired[f"learned_macro_gain_vs_{baseline}_{routing}_{field}"] = mean_ci95(values)

    delay_interaction_vs_centroid = []
    delay_interaction_vs_fixed = []
    for seed in seeds:
        learned_coop_gain = (
            float(by_arm[("learned", "local_only")][seed]["task_delay_weighted_ms"])
            - float(by_arm[("learned", "cooperative")][seed]["task_delay_weighted_ms"])
        )
        centroid_coop_gain = (
            float(by_arm[("centroid", "local_only")][seed]["task_delay_weighted_ms"])
            - float(by_arm[("centroid", "cooperative")][seed]["task_delay_weighted_ms"])
        )
        fixed_coop_gain = (
            float(by_arm[("fixed", "local_only")][seed]["task_delay_weighted_ms"])
            - float(by_arm[("fixed", "cooperative")][seed]["task_delay_weighted_ms"])
        )
        delay_interaction_vs_centroid.append(learned_coop_gain - centroid_coop_gain)
        delay_interaction_vs_fixed.append(learned_coop_gain - fixed_coop_gain)
    return {
        "arm_means": means,
        "paired": paired,
        "delay_interaction_learned_vs_centroid": mean_ci95(delay_interaction_vs_centroid),
        "delay_interaction_learned_vs_fixed": mean_ci95(delay_interaction_vs_fixed),
        "interpretation_note": (
            "The learned macro policy remains state-dependent in local-only runs; "
            "therefore this is a policy-level factorial ablation, not an exact "
            "same-physical-trajectory replay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_args, learned_policy, micro_policy = load_policies(
        args.checkpoint.resolve(), args.device
    )
    rows: list[dict] = []
    for seed in seeds:
        for macro_name in MACRO_POLICIES:
            for routing_mode in ROUTING_MODES:
                result = evaluate_arm(
                    checkpoint_args,
                    make_macro_policy(macro_name, learned_policy),
                    micro_policy,
                    seed,
                    force_local=routing_mode == "local_only",
                )
                row = {
                    "seed": seed,
                    "macro_policy": macro_name,
                    "routing_mode": routing_mode,
                    **result,
                }
                rows.append(row)
                print(
                    f"seed={seed} macro={macro_name} routing={routing_mode} "
                    f"delay={result['task_delay_weighted_ms']:.2f} ms "
                    f"queue={result['queue_mbits']:.2f} Mbit",
                    flush=True,
                )
    write_csv(output / "per_seed_raw.csv", rows)
    report = summarize(rows, seeds)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
