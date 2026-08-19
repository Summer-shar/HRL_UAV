"""Build the frozen v39 centroid-state probe set and reference anchor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrl_uav.load_switching import build_load_switch_schedule
from scripts.run_load_switch_gate import run_gate_episode
from scripts.run_v34_cooperative_mobility_gate import (
    V34MacroPolicy,
    load_frozen_micro_policy,
    make_gate_env,
    parse_seed_list,
)
from scripts.search_position_oracle import target_centroid_waypoint_action


DEFAULT_CONFIG = ROOT / "configs" / "tcom_v39_calibrated_soft_objective.json"
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v39_calibrated_soft_objective"
    / "formal"
    / "seed1"
    / "micro_pretrain"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_GATE = ROOT / "result_archive" / "v39_calibrated_objective_gate"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v39_formal_probe_reference"
DEFAULT_SEEDS = "7000-7019"
PROBE_MACRO_INDICES = (0, 5, 10, 15, 20, 25)
COOPERATIVE_OFFSET_M = 60.0
MOVEMENT_TRIGGER_M = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gate-output", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def state_digest(state: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(state, dtype=np.float32))
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def cooperative_probe_action(
    expected_loads_mbps: np.ndarray,
    uav_coords_m: np.ndarray,
    centroids_m: np.ndarray,
    waypoint_radius_m: float,
    offset_m: float = COOPERATIVE_OFFSET_M,
    movement_trigger_m: float = MOVEMENT_TRIGGER_M,
) -> tuple[np.ndarray, int, int]:
    loads = np.asarray(expected_loads_mbps, dtype=np.float64)
    uavs = np.asarray(uav_coords_m, dtype=np.float64)
    centroids = np.asarray(centroids_m, dtype=np.float64)
    heavy = int(np.argmax(loads))
    candidates = [index for index in range(loads.size) if index != heavy]
    support = min(candidates, key=lambda index: (float(loads[index]), index))
    direction = uavs[heavy] - centroids[support]
    norm = float(np.linalg.norm(direction))
    residual = np.zeros_like(centroids)
    if norm > 1e-12:
        residual[support] = direction / norm * float(offset_m)
    targets = np.clip(centroids + residual, 0.0, 1000.0)
    distances = np.linalg.norm(targets - uavs, axis=1)
    targets = np.where(
        (distances > float(movement_trigger_m))[:, None], targets, uavs
    )
    action = target_centroid_waypoint_action(
        targets, centroids, float(waypoint_radius_m)
    )
    return np.asarray(action, dtype=np.float32), heavy, support


def build_anchor_summary(gate_output: Path) -> dict:
    report_path = gate_output / "imbalanced_load" / "gate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    centroid = report["policy_means"]["centroid"]
    return {
        "evidence_role": "V39_MATCHED_CENTROID_REFERENCE",
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "evaluation_seeds": report["seeds"],
        "task_delay_weighted_ms": centroid["task_delay_weighted_ms"],
        "mean_queue_mbits": centroid["mean_queue_mbits"],
        "terminal_queue_mbits": centroid["terminal_queue_mbits"],
        "average_power_w": centroid["average_power_w"],
        "soft_objective_j": centroid["soft_objective_j"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    gate_output = resolve_path(args.gate_output)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = parse_seed_list(args.seeds)
    checkpoint_args, micro_policy, device = load_frozen_micro_policy(
        checkpoint_path, config, args.device
    )
    phases = np.asarray(config["load_switch_phases_mbps"], dtype=np.float64)
    macro_steps = int(config["macro_steps"])

    states = []
    centroid_actions = []
    cooperative_actions = []
    rows = []
    schedule_rows = []
    external_hashes = {}

    for seed_index, seed in enumerate(seeds, start=1):
        schedule = build_load_switch_schedule(
            mode=config["load_switch_mode"],
            phases_mbps=phases,
            macro_steps=macro_steps,
            episode_seed=seed,
            dwell_min=int(config["load_switch_dwell_min"]),
            dwell_max=int(config["load_switch_dwell_max"]),
        )
        env = make_gate_env(checkpoint_args)
        centroid_policy = V34MacroPolicy(
            "centroid", COOPERATIVE_OFFSET_M, MOVEMENT_TRIGGER_M
        )
        result, trace = run_gate_episode(
            env=env,
            checkpoint_args=checkpoint_args,
            micro_policy=micro_policy,
            policy_name="centroid_probe_reference",
            seed=seed,
            macro_steps=macro_steps,
            load_switch_schedule=schedule,
            macro_policy=centroid_policy,
            macro_metadata_provider=centroid_policy.metadata,
            record_macro_states=True,
        )
        records = {int(row["macro_index"]): row for row in trace["macro_records"]}
        missing = sorted(set(PROBE_MACRO_INDICES) - set(records))
        if missing:
            raise RuntimeError(f"seed={seed} missing probe steps: {missing}")
        external_hashes[str(seed)] = result["external_trace_hash"]
        schedule_rows.append(
            {
                "seed": int(seed),
                "switch_steps": list(schedule.switch_steps),
                "dwell_steps": list(schedule.dwell_steps),
                "external_trace_hash": result["external_trace_hash"],
            }
        )
        for macro_index in PROBE_MACRO_INDICES:
            record = records[macro_index]
            state = np.asarray(record["global_state"], dtype=np.float32)
            centroid_action = np.asarray(record["macro_action"], dtype=np.float32)
            cooperative_action, heavy, support = cooperative_probe_action(
                record["expected_loads_mbps"],
                record["uav_before_m"],
                record["centroids_before_m"],
                float(checkpoint_args.macro_waypoint_radius_m),
            )
            probe_id = f"seed{seed}:macro{macro_index:02d}"
            states.append(state)
            centroid_actions.append(centroid_action.reshape(-1))
            cooperative_actions.append(cooperative_action.reshape(-1))
            rows.append(
                {
                    "probe_id": probe_id,
                    "seed": int(seed),
                    "macro_index": int(macro_index),
                    "phase_index": int(record["phase_index"]),
                    "switch_flag": int(record["switch_flag"]),
                    "heavy_uav": heavy,
                    "support_uav": support,
                    "state_sha256": state_digest(state),
                    "state_dim": int(state.size),
                    "centroid_action_norm": float(np.linalg.norm(centroid_action)),
                    "cooperative_action_norm": float(
                        np.linalg.norm(cooperative_action)
                    ),
                }
            )
        print(f"[{seed_index}/{len(seeds)}] seed={seed} probes={len(PROBE_MACRO_INDICES)}")

    state_matrix = np.stack(states).astype(np.float32)
    centroid_action_matrix = np.stack(centroid_actions).astype(np.float32)
    cooperative_action_matrix = np.stack(cooperative_actions).astype(np.float32)
    np.savez_compressed(
        output / "fixed_probe_set.npz",
        probe_id=np.asarray([row["probe_id"] for row in rows]),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int32),
        macro_index=np.asarray([row["macro_index"] for row in rows], dtype=np.int16),
        phase_index=np.asarray([row["phase_index"] for row in rows], dtype=np.int16),
        switch_flag=np.asarray([row["switch_flag"] for row in rows], dtype=np.int8),
        heavy_uav=np.asarray([row["heavy_uav"] for row in rows], dtype=np.int8),
        support_uav=np.asarray([row["support_uav"] for row in rows], dtype=np.int8),
        state=state_matrix,
        action_centroid=centroid_action_matrix,
        action_cooperative=cooperative_action_matrix,
    )
    write_csv(output / "probe_index.csv", rows)
    anchor = build_anchor_summary(gate_output)
    (output / "centroid_reference.json").write_text(
        json.dumps(anchor, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "FROZEN_V39_FORMAL_PROBE_REFERENCE",
        "scientific_scope": (
            "Checkpoint monitoring only. The episode-level gate establishes the "
            "policy-level preference; it is not a per-state ground-truth label."
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "micro_checkpoint": str(checkpoint_path),
        "micro_checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "probe_episode_seeds": seeds,
        "probe_macro_indices": list(PROBE_MACRO_INDICES),
        "probe_count": len(rows),
        "state_dim": int(state_matrix.shape[1]),
        "macro_action_dim": int(centroid_action_matrix.shape[1]),
        "reference_policy": "causal dynamic centroid tracking",
        "compared_action": "60 m causal cooperative residual",
        "continuation_policy": "not encoded; checkpoint-specific audit must define it",
        "schedules": schedule_rows,
        "external_trace_hashes": external_hashes,
        "centroid_reference": anchor,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "status", "probe_count", "state_dim", "macro_action_dim"
    )}, indent=2))


if __name__ == "__main__":
    main()
