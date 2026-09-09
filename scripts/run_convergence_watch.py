"""Bounded continuation; retain boundary snapshots and evaluate after training."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import write_json
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import means, metered_evaluate, run_mainline, verify_crn
from hrl_uav.resumption import capture_rng, load_source, restore_agent, save_boundary_checkpoint, state_equal
from scripts.run_critic_fit_repeat import metric_deltas


SOURCE = ROOT / "results/two_fits_continue1400to1600_seed1/at1600/final.pt"
OUTPUT = ROOT / "results/convergence_watch1600to1800_seed1"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot_windows(pretrain, start, end, stream, cadence):
    next_checkpoint = (start // cadence + 1) * cadence
    records = []
    for offset in range(start, end, stream):
        completed = min(offset + stream, end)
        if completed == end or completed >= next_checkpoint:
            records.append(pretrain + completed)
            next_checkpoint = (completed // cadence + 1) * cadence
    return records


def continuation_config(base, checkpoint_every=None):
    config = copy.deepcopy(base)
    config["training"]["joint_windows"] += 200
    if checkpoint_every is not None:
        if type(checkpoint_every) is not int or checkpoint_every <= 0:
            raise ValueError("Checkpoint cadence must be a positive integer")
        config["qualification"]["checkpoint_every_windows"] = checkpoint_every
    return config


class RetainBoundaries:
    """Copy already-saved checkpoints without splitting rollouts or updating agents."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.records = []

    def __call__(self, agent, config, path, completed, next_seed):
        save_boundary_checkpoint(agent, config, path, completed, next_seed)
        if agent.stage != "joint" or Path(path).name != "joint_latest.pt":
            return
        total = config["training"]["pretrain_windows"] + completed
        self.folder.mkdir(exist_ok=True)
        target = self.folder / f"global_{total}.pt"
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(path, target)
        self.records.append({"global_window": total, "joint_window": completed,
                             "checkpoint": str(target), "sha256": digest(target)})
        write_json(self.folder / "index.json", self.records)
        print(f"[snapshot] global={total} retained; no evaluation/update inserted", flush=True)


def paired_readings(config, agent, seeds, macros, output, label):
    """Evaluate the deployed full policy and its matched centroid intervention."""
    before = copy.deepcopy(agent.checkpoint(config))
    rng = capture_rng()
    full = metered_evaluate(config, agent, seeds, macros, output, label, macro_enabled=True)
    centroid = metered_evaluate(config, agent, seeds, macros, output, label + "_centroid")
    checks = {"agent_replay_and_optimizers_unchanged": state_equal(before, agent.checkpoint(config)),
              "global_rng_unchanged": state_equal(rng, capture_rng())}
    crn = verify_crn(output, seeds, label, label + "_centroid")
    checks["exogenous_streams_paired"] = all(r["users"] and r["arrivals"] for r in crn.values())
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"full": full, "centroid": centroid, "mean": means(full), "centroid_mean": means(centroid),
            "macro_deltas": metric_deltas(full, centroid), "checks": checks, "crn": crn}


def trend_summary(points):
    """Descriptive correlated-checkpoint trends, never an IID significance test."""
    result = {}
    for group, subset in (("all_points", points), ("last_three", points[-3:])):
        x = np.asarray([p["global_window"] for p in subset], dtype=float)
        result[group] = {}
        for policy in ("mean", "centroid_mean"):
            result[group][policy] = {}
            for key in ("task_delay_weighted_ms", "task_delay_low_ms", "J"):
                y = np.asarray([p[policy][key] for p in subset])
                result[group][policy][key] = {
                    "first": float(y[0]), "last": float(y[-1]), "last_minus_first": float(y[-1]-y[0]),
                    "range": float(np.ptp(y)), "relative_range": float(np.ptp(y)/max(abs(y.mean()), 1e-12)),
                    "slope_per_100_windows": float(np.polyfit(x, y, 1)[0]*100),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-every", type=int, help="Optional monitoring override; otherwise preserve the source cadence")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Output must be new; archived results are never overwritten")
    origin = json.loads((source.parent / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    base = manifest["config"]
    if base["training"].get("micro_algorithm", "sac") != "sac":
        raise ValueError("This runner continues only PPO + Direct SAC")
    current_core = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "hrl_uav").rglob("*.py")}
    frozen_core = {k: v for k, v in manifest["source_hashes"].items() if Path(k).parts[0] == "hrl_uav"}
    if frozen_core != current_core:
        raise ValueError("Core learning/environment code changed since the source run")
    if origin["status"] != "JOINT_PILOT_COMPLETE_NOT_FORMAL":
        raise ValueError("Source training and final evaluation must be complete")
    config = continuation_config(base, args.checkpoint_every)
    cadence = config["qualification"]["checkpoint_every_windows"]
    start_joint = base["training"]["joint_windows"]
    start_global = base["training"]["pretrain_windows"] + start_joint
    end_global = start_global + 200
    assert base["training"]["pretrain_windows"] == 200 and start_joint >= 1400 and start_joint % 200 == 0
    assert base["training"]["sac_critic_steps_per_update"] == 2
    assert base["qualification"]["joint_evaluation_mode"] == "final_only"
    saved, metadata = load_source(source, config)
    assert metadata["start_window"] == start_joint and not metadata["authorized_training_overrides"]
    assert saved["config"] == base
    del saved
    if args.check_only:
        print(json.dumps({"status": "READY", "source": metadata,
                          "global_start": start_global, "global_target": end_global, "snapshot_cadence": cadence,
                          "core_code_unchanged": True}, indent=2))
        return
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    status = {"status": "RUNNING", "stage": "training", "started_at": now()}
    protocol = {
        "source": str(source), "source_sha256": digest(source), "config": config,
        "start_global": start_global, "end_global": end_global, "additional_windows": 200,
        "snapshot_cadence": f"Every {cadence} joint windows, at the first complete sampling-stream boundary; log actual indices",
        "expected_global_snapshots": snapshot_windows(200, start_joint, start_joint + 200,
                                                       config["training"]["stream_windows"], cadence),
        "validation_seeds": config["validation_seeds"], "validation_macros": 90,
        "evaluation": "After training: source and all retained checkpoints; deterministic full policy and matched same-SAC centroid. Preserve all source evaluation seeds and final-only monitoring.",
        "read_only_checks": "Exact agent/optimizer/replay and RNG preservation; paired users and arrivals; immutable source/snapshot hashes",
        "analysis": "All points and last three points: raw values, ranges, slopes and paired macro deltas. No post-hoc convergence PASS threshold, IID checkpoint CI, early stopping or best selection.",
        "training_runs": 1, "continued_algorithm": "sac", "tqc_continued": False,
        "core_code_unchanged": True,
        "scope": "One training initialization; reused development scenes; no global optimality, formal significance or infinite-horizon queue stability claim",
        "stop": f"Fixed {end_global} endpoint. No further training or method changes authorized by this runner.",
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "config.json", config)
    write_json(output / "status.json", status)
    recorder = RetainBoundaries(output / "checkpoints")
    try:
        with patch("hrl_uav.qualification.save_boundary_checkpoint", side_effect=recorder):
            report = run_mainline(config, output / f"at{end_global}", resume_from=source)
        assert report["status"] == "JOINT_PILOT_COMPLETE_NOT_FORMAL"
        assert all(report["resume"]["exact_restore_checks"].values())
        assert report["source_hashes_unchanged"]
        audit = report["stage_audits"][-1]
        assert audit["trained_windows_this_run"] == 200
        assert audit["critic_updates"] == 5400 and audit["critic_gradient_steps"] == 10800
        assert audit["actor_updates"] == 2700 and audit["target_updates"] == 5400 and audit["ppo_updates"] == 20
        assert [r["global_window"] for r in recorder.records] == protocol["expected_global_snapshots"]
        references = [{"global_window": start_global, "joint_window": start_joint, "checkpoint": str(source),
                       "sha256": protocol["source_sha256"]}] + recorder.records
        points = []
        validation = output / "validation"
        validation.mkdir()
        status.update(stage="checkpoint_replay", training_minutes=report["minutes"])
        write_json(output / "status.json", status)
        for record in references:
            status.update(current_global_window=record["global_window"])
            write_json(output / "status.json", status)
            print(f"[replay] global={record['global_window']}, fixed validation scenes", flush=True)
            state = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
            local = state["config"]
            mdp = QueueMDP(local)
            mdp.reset(local["training"]["seed"], local["training"]["window_macros"])
            agent = DirectHRL.from_mdp(mdp, local)
            restored = restore_agent(agent, state, {"replay_counter": state["replay"]["memory_counter"]})
            del state, mdp
            point = {**record, **paired_readings(local, agent, config["validation_seeds"], 90,
                                                validation, f"global_{record['global_window']}")}
            point["restore_checks"] = restored["exact_restore_checks"]
            assert digest(record["checkpoint"]) == record["sha256"]
            points.append(point)
            write_json(output / "validation_points.json", points)
            print(f"[replay complete] global={record['global_window']} " + json.dumps(point["mean"]), flush=True)
            del agent
        assert digest(source) == protocol["source_sha256"]
        final_delta = metric_deltas(report["final"], origin["final"])
        result = {"protocol": protocol, "validation_points": points, "descriptive_trends": trend_summary(points),
                  "final_mean": report["mean"], "same_sac_centroid_mean": means(report["matched_centroid"]),
                  "uniform_mean": report["uniform_mean"], f"final_minus{start_global}": final_delta,
                  f"centroid_minus{start_global}": metric_deltas(report["matched_centroid"], origin["matched_centroid"]),
                  "final_macro_deltas": metric_deltas(report["final"], report["matched_centroid"]),
                  "training_audit": audit, "source_unchanged": True,
                  "minutes": (time.perf_counter()-started)/60}
        write_json(output / "comparison.json", result)
        status.update(stage="plotting")
        write_json(output / "status.json", status)
        from scripts.plot_convergence_watch import build_figures
        build_figures(output)
        old, new = origin["mean"], report["mean"]
        lines = [f"# PPO + Direct SAC: {start_global} -> {end_global}", "",
                 "Only SAC continued; TQC was not trained. Both PPO and SAC continued updating, with no learning-rule or environment changes.", "",
                 "| Endpoint | Weighted delay / ms | High / ms | Low / ms | J | Completion | Terminal queue / Mbit |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for label, values in ((str(start_global), old), (str(end_global), new),
                              (f"{end_global} same SAC + centroid", result["same_sac_centroid_mean"]),
                              ("Centroid + uniform (non-RL diagnostic)", result["uniform_mean"])):
            lines.append(f"| {label} | {values['task_delay_weighted_ms']:.3f} | {values['task_delay_high_ms']:.3f} | {values['task_delay_low_ms']:.3f} | {values['J']:.6f} | {values['task_completion_rate']:.6f} | {values['queue_total_mbit']:.3f} |")
        lines.extend(["", "One training seed; reused evaluation scenes. Finite-horizon queue behavior is not a proof of long-run stability. No best-checkpoint selection or automatic further extension.", "",
                      "![Fixed evaluation](figures/01_fixed_validation.png)",
                      "![Training](figures/02_training_trace.png)",
                      "![Queues](figures/03_queue_and_unfinished.png)"])
        (output / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        status.update(status="COMPLETE", stage="complete", minutes=result["minutes"], completed_at=now())
    except BaseException as error:
        status.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
