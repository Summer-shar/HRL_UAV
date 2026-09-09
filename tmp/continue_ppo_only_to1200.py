"""Continue the existing joint run to two fixed, user-requested budgets."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import write_json
from hrl_uav.qualification import means, run_mainline
from hrl_uav.resumption import load_source


SOURCE = ROOT / "results/ppo_only_unlock_seed1"
OUT = ROOT / "results/ppo_only_unlock_continue800_to1200_seed1"
TARGETS = (1000, 1200)


def now():
    return datetime.now(timezone.utc).isoformat()


def training_config(base, global_window):
    result = copy.deepcopy(base)
    result["training"]["joint_windows"] = global_window - result["training"]["pretrain_windows"]
    return result


def comparison_row(global_window, report, reference):
    ref = {row["seed"]: row for row in reference["final"]}
    paired = [{"seed": row["seed"], **{
        key: row[key] - ref[row["seed"]][key] for key in
        ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "J", "queue_total_mbit")
    }} for row in report["final"]]
    return {
        "global_window": global_window,
        "pretrain_windows": 200,
        "joint_windows": global_window - 200,
        "mean": report["mean"],
        "same_sac_centroid_mean": means(report["matched_centroid"]),
        "uniform_mean": report["uniform_mean"],
        "paired_minus800": paired,
        "updates_total": report["updates_total"],
        "minutes_this_segment": report["minutes"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    initial = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    base = manifest["config"]
    assert base["training"]["pretrain_windows"] == 200
    assert base["training"]["joint_windows"] == 600
    assert base["training"]["sac_stage_policy"] == "preserve"
    assert base["qualification"]["joint_evaluation_mode"] == "final_only"
    assert base["training"]["ppo_std_decay_windows"] == 200
    checkpoint = SOURCE / "final.pt"
    saved, metadata = load_source(checkpoint, training_config(base, TARGETS[0]))
    assert metadata["stage"] == "joint" and metadata["start_window"] == 600
    assert metadata["rng_mode"] == "restored_global_rng"
    assert saved["config"] == base
    del saved
    if args.check_only:
        print(json.dumps({"status": "READY", "source": metadata, "targets_global_windows": TARGETS}, indent=2), flush=True)
        return

    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    origin_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    plan = {
        "started_at": now(), "source_checkpoint": str(checkpoint),
        "source_sha256": origin_hash, "global_targets": TARGETS,
        "changed": "Only total joint-training window budget; no network, state, action, reward, bandwidth, optimizer, replay or exploration schedule changes.",
        "method": "Continue the same PPO and direct SAC; no teacher or routing fallback.",
        "selection": "Evaluate both fixed endpoints, continue from 1000 final to 1200 regardless of performance. Do not choose best or stop on favorable development results.",
        "evaluation": "Same three reused development seeds, 900 seconds each; full method, same-SAC centroid, uniform diagnostic at both endpoints. One training seed only.",
        "boundary_scope": "Use existing completed-rollout resumption. The 1000-window endpoint ends a shortened sampling stream and the next segment resets the environment. This is not claimed bitwise equivalent to an uninterrupted, preplanned 1200-window run.",
        "baseline800": comparison_row(800, initial, initial),
        "segments": [],
    }
    write_json(OUT / "plan.json", plan)
    state = {"status": "RUNNING", "started_at": plan["started_at"], "targets": TARGETS}
    try:
        for target in TARGETS:
            output = OUT / f"at{target}"
            config = training_config(base, target)
            state.update(current_target=target, segment_output=str(output), source_checkpoint=str(checkpoint), updated_at=now())
            write_json(OUT / "status.json", state)
            print(f"[segment] target_total={target}, pretrain=200, joint_total={target-200}, source={checkpoint}", flush=True)
            report = run_mainline(config, output, resume_from=checkpoint)
            if report["status"] != "JOINT_PILOT_COMPLETE_NOT_FORMAL":
                raise RuntimeError(f"Segment did not finish normally: {report['status']}")
            row = comparison_row(target, report, initial)
            plan["segments"].append(row)
            write_json(OUT / "comparison.json", {"baseline800": plan["baseline800"], "segments": plan["segments"]})
            print(f"[endpoint {target}] " + json.dumps(row["mean"]), flush=True)
            checkpoint = output / "final.pt"
        state.update(status="COMPLETE", completed_at=now(), minutes=(time.perf_counter()-started)/60)
        state["original_checkpoint_unchanged"] = hashlib.sha256((SOURCE / "final.pt").read_bytes()).hexdigest() == origin_hash
        assert state["original_checkpoint_unchanged"]
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), failed_at=now(), traceback=traceback.format_exc())
        raise
    finally:
        write_json(OUT / "status.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
