"""Paired 1200-to-1400 continuation with one versus two Critic fits."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import write_json
from hrl_uav.qualification import means, run_mainline
from hrl_uav.resumption import load_source, state_equal


SOURCE = ROOT/"results/ppo_only_unlock_continue800_to1200_seed1/at1200/final.pt"
OUTPUT = ROOT/"results/critic_fit_repeat_1200to1400_seed1"
KEY = "sac_critic_steps_per_update"
ARMS = (("control", 1), ("two_fits", 2))


def now():
    return datetime.now(timezone.utc).isoformat()


def arm_config(base, fits):
    config = copy.deepcopy(base)
    config["training"]["joint_windows"] += 200
    config["training"][KEY] = fits
    return config


def metric_deltas(rows, reference):
    by_seed = {r["seed"]: r for r in reference}
    keys = ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "J",
            "queue_total_mbit", "mean_queue_compute_mbit", "mean_queue_transfer_mbit",
            "task_completion_rate", "late_queue_slope_mbit_s", "power_w")
    return [{"seed": row["seed"], **{key: row[key]-by_seed[row["seed"]][key] for key in keys}}
            for row in rows]


def summarize(report, origin):
    return {"mean": report["mean"], "same_sac_centroid_mean": means(report["matched_centroid"]),
            "uniform_mean": report["uniform_mean"], "updates_total": report["updates_total"],
            "stage_audits": report["stage_audits"], "minutes": report["minutes"],
            "paired_minus1200": metric_deltas(report["final"], origin["final"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    origin = json.loads((SOURCE.parent/"report.json").read_text(encoding="utf-8"))
    base = json.loads((SOURCE.parent/"manifest.json").read_text(encoding="utf-8"))["config"]
    assert base["training"]["pretrain_windows"] == 200
    assert base["training"]["joint_windows"] == 1000
    assert base["training"].get(KEY, 1) == 1
    assert base["training"]["sac_stage_policy"] == "preserve"
    assert base["training"]["ppo_std_decay_windows"] == 200
    assert base["qualification"]["joint_evaluation_mode"] == "final_only"
    configs = {name: arm_config(base, fits) for name, fits in ARMS}
    normalized = copy.deepcopy(configs["two_fits"])
    normalized["training"][KEY] = 1
    assert normalized == configs["control"]
    preflight = {}
    for name, fits in ARMS:
        saved, metadata = load_source(SOURCE, configs[name], allowed_training_overrides=(KEY,))
        assert metadata["start_window"] == 1000 and metadata["stage"] == "joint"
        assert metadata["rng_mode"] == "restored_global_rng"
        assert saved["config"] == base
        expected = {} if fits == 1 else {KEY: {"from": 1, "to": 2}}
        assert metadata["authorized_training_overrides"] == expected
        preflight[name] = metadata
        del saved
    if args.check_only:
        print(json.dumps({"status": "READY", "global_target": 1400, "arms": preflight}, indent=2))
        return
    OUTPUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    protocol = {
        "started_at": now(), "source": str(SOURCE), "source_sha256": source_hash,
        "global_start": 1200, "global_end": 1400, "added_windows_per_arm": 200,
        "only_treatment_difference": KEY, "arms": {n: f for n, f in ARMS},
        "configs": configs, "preflight": preflight,
        "fit_semantics": "One batch, one fixed target, one/two consecutive Q optimizer steps. No extra RNG samples, actor/entropy/target/PPO update events.",
        "evaluation": "Final only: full method, same-SAC centroid, uniform diagnostic; same 3 reused development seeds, 900 s; one training initialization.",
        "selection": "Both arms run to fixed final budget regardless of intermediate performance; retain both. No default promotion based on a favorable development endpoint.",
        "protocol_doc": "docs/critic_fit_repeat_protocol.md",
    }
    write_json(OUTPUT/"protocol.json", protocol)
    status = {"status": "RUNNING", "started_at": protocol["started_at"], "completed_arms": []}
    reports = {}
    try:
        for name, fits in ARMS:
            status.update(current_arm=name, updated_at=now())
            write_json(OUTPUT/"status.json", status)
            print(f"[arm {name}] source=1200 target=1400 fits={fits}", flush=True)
            report = run_mainline(configs[name], OUTPUT/name, resume_from=SOURCE,
                                  resume_training_overrides=(KEY,))
            if report["status"] != "JOINT_PILOT_COMPLETE_NOT_FORMAL":
                raise RuntimeError(f"Incomplete arm {name}: {report['status']}")
            assert report["source_hashes_unchanged"]
            assert all(report["resume"]["exact_restore_checks"].values())
            stage = report["stage_audits"][-1]
            assert stage["trained_windows_this_run"] == 200
            assert stage["critic_gradient_steps"] == fits * stage["critic_updates"]
            assert stage["target_updates"] == stage["critic_updates"]
            assert stage["actor_updates"] > 0 and stage["ppo_updates"] > 0
            reports[name] = report
            status["completed_arms"].append(name)
            write_json(OUTPUT/"comparison_partial.json", {n: summarize(r, origin) for n, r in reports.items()})
            print(f"[arm {name} complete] " + json.dumps(report["mean"]), flush=True)
        control, treatment = reports["control"], reports["two_fits"]
        controls = {}
        for key in ("critic_updates", "actor_updates", "target_updates", "ppo_updates", "replay_at_stage_entry", "replay_at_stage_exit"):
            controls[key] = control["stage_audits"][-1][key] == treatment["stage_audits"][-1][key]
        controls["same_optimizer_rates_at_entry"] = control["resume"]["optimizer_learning_rates"] == treatment["resume"]["optimizer_learning_rates"]
        for filename, key in (("manifest.json", "training_scene_seeds"),):
            left = json.loads((OUTPUT/"control"/filename).read_text(encoding="utf-8"))
            right = json.loads((OUTPUT/"two_fits"/filename).read_text(encoding="utf-8"))
            controls[key] = left[key] == right[key]
        controls["source_unchanged"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
        assert all(controls.values()), controls
        left = torch.load(OUTPUT/"control/final.pt", map_location="cpu", weights_only=False)
        right = torch.load(OUTPUT/"two_fits/final.pt", map_location="cpu", weights_only=False)
        rng_equal = {"global": state_equal(left["run_state"]["rng"], right["run_state"]["rng"]),
                     "replay_sampler": state_equal(left["replay"]["rng_state"], right["replay"]["rng_state"])}
        assert all(rng_equal.values()), rng_equal
        crn = {}
        for seed in base["test_seeds"]:
            with np.load(OUTPUT/"control"/f"final_{seed}_trajectory.npz") as a, np.load(OUTPUT/"two_fits"/f"final_{seed}_trajectory.npz") as b:
                crn[str(seed)] = bool(np.array_equal(a["users"], b["users"]))
        assert all(crn.values())
        del left, right
        pairs = metric_deltas(treatment["final"], control["final"])
        result = {
            "protocol": protocol, "baseline1200": summarize(origin, origin),
            "arms": {n: summarize(r, origin) for n, r in reports.items()},
            "treatment_minus_control": pairs,
            "mean_treatment_minus_control": {k: float(np.mean([row[k] for row in pairs])) for k in pairs[0] if k != "seed"},
            "same_sac_centroid_treatment_minus_control": metric_deltas(treatment["matched_centroid"], control["matched_centroid"]),
            "cadence_and_restore_checks": controls, "final_rng_equal": rng_equal,
            "paired_users_equal": crn,
            "scope": "One training initialization, three reused development evaluation scenes. An experiment on continued training, not a new from-scratch multi-seed performance claim.",
            "minutes": (time.perf_counter()-started)/60,
        }
        write_json(OUTPUT/"comparison.json", result)
        status.update(status="COMPLETE", completed_at=now(), minutes=result["minutes"])
        print(json.dumps(result["mean_treatment_minus_control"], indent=2), flush=True)
    except BaseException as error:
        status.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(), failed_at=now())
        raise
    finally:
        write_json(OUTPUT/"status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
