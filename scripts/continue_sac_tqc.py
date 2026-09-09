"""Continue both matched SAC/TQC endpoints without changing their learning rules."""

import argparse
import copy
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import isolated_rng, write_json
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import means, run_mainline
from hrl_uav.resumption import load_source, restore_agent, save_boundary_checkpoint
from scripts.compare_sac_tqc import (
    ARMS, crn_check, evaluate_snapshots, now, paired_rows, pair_summary, read_json,
)
from scripts.run_convergence_watch import digest, snapshot_windows


def extended_config(saved, additional):
    if type(additional) is not int or additional <= 0:
        raise ValueError("Additional joint windows must be a positive integer")
    config = copy.deepcopy(saved)
    config["training"]["joint_windows"] += additional
    return config


def scheduled_updates(updates):
    # PPO's KL stop changes inner gradient counts, not scheduled update events.
    return {k: updates[k] for k in
            ("sac_critic", "sac_critic_gradients", "sac_target", "sac_actor", "ppo")}


def prepare(source, additional):
    source = Path(source).resolve()
    status = read_json(source / "status.json")
    origin = read_json(source / "comparison.json")
    if status["status"] != "COMPLETE" or set(status["completed_arms"]) != set(ARMS):
        raise ValueError("Both source arms must have completed training and evaluation")
    configs, audits, source_files = {}, {}, {}
    current_core = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "hrl_uav").rglob("*.py")}
    with isolated_rng():
        for name in ARMS:
            folder = source / name
            manifest = read_json(folder / "manifest.json")
            base = manifest["config"]
            if (base["training"].get("micro_algorithm", "sac") != name
                    or base != origin["protocol"]["configs"][name]):
                raise ValueError("Source algorithm/configuration does not match its comparison")
            frozen_core = {k: v for k, v in manifest["source_hashes"].items()
                           if Path(k).parts[0] == "hrl_uav"}
            if frozen_core != current_core:
                raise ValueError("Core learning/environment sources changed since the source run")
            config = extended_config(base, additional)
            if config["qualification"]["joint_evaluation_mode"] != "final_only":
                raise ValueError("This continuation keeps final-only monitoring")
            checkpoint = folder / "final.pt"
            saved, metadata = load_source(checkpoint, config)
            if (metadata["stage"] != "joint" or metadata["rng_mode"] != "restored_global_rng"
                    or metadata["start_window"] != base["training"]["joint_windows"]
                    or metadata["authorized_training_overrides"]):
                raise ValueError("Expected an unchanged, complete joint-stage endpoint")
            env = QueueMDP(config)
            env.reset(config["training"]["seed"], config["training"]["window_macros"])
            agent = DirectHRL.from_mdp(env, config)
            audits[name] = restore_agent(agent, saved, metadata)
            configs[name] = config
            for filename in ("final.pt", "manifest.json", "report.json"):
                path = folder / filename
                source_files[str(path)] = digest(path)
            del saved, env, agent
    source_files[str(source / "comparison.json")] = digest(source / "comparison.json")
    if scheduled_updates(audits["sac"]["updates_at_entry"]) != scheduled_updates(audits["tqc"]["updates_at_entry"]):
        raise ValueError("Source update clocks differ between arms")
    if audits["sac"]["next_training_scene_seed"] != audits["tqc"]["next_training_scene_seed"]:
        raise ValueError("Source continuation scene schedules differ")
    control = copy.deepcopy(configs["tqc"])
    control["training"]["micro_algorithm"] = "sac"
    for key in ("tqc_quantiles", "tqc_drop_per_critic"):
        control["training"].pop(key)
    if control != configs["sac"]:
        raise ValueError("Non-algorithm configuration difference between arms")
    cfg = configs["sac"]["training"]
    start = audits["sac"]["start_window"]
    continuation = {
        "source_run": str(source), "source_files": source_files, "restore_audits": audits,
        "additional_windows_per_arm": additional,
        "start_global": cfg["pretrain_windows"] + start,
        "end_global": cfg["pretrain_windows"] + cfg["joint_windows"],
        "expected_snapshot_windows": snapshot_windows(
            cfg["pretrain_windows"], start, cfg["joint_windows"], cfg["stream_windows"],
            configs["sac"]["qualification"]["checkpoint_every_windows"],
        ),
    }
    return configs, origin, continuation


def verify_source_files(records):
    if any(digest(path) != expected for path, expected in records.items()):
        raise AssertionError("An archived source file changed during continuation")


def run_continuation(source, output, additional=200):
    torch.set_num_threads(1)
    configs, origin, continuation = prepare(source, additional)
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {
        "started_at": now(), "smoke": origin["protocol"]["smoke"],
        "configs": configs, "training_seed_count": 1, "training_runs": 2,
        "continuation": continuation,
        "history_runs": origin["protocol"].get("history_runs", []) + [str(source)],
        "selection": "Continue each own final only, equal additional budget, no best selection or automatic extension.",
        "preservation": "Restore both layers, targets, temperature, all optimizers, replay, clocks, pending PPO samples and RNG; change only joint_windows.",
        "evaluation": "Same 33000/33001 development scenes and same 34500-34504 endpoint scenes; these are now reused, not untouched tests. Read checkpoints after training only.",
        "scope": (f"One training initialization. This extension was authorized after inspecting "
                  f"{continuation['start_global']}-window results; exploratory, not a retrospectively "
                  f"preregistered {continuation['end_global']}-window experiment."),
        "stop": f"Stop both arms at global window {continuation['end_global']} regardless of whether delay is below 600 ms.",
    }
    write_json(output / "protocol.json", protocol)
    status = {"status": "RUNNING", "started_at": protocol["started_at"], "completed_arms": []}
    write_json(output / "status.json", status)
    reports, curves = {}, {}
    try:
        for name in ARMS:
            verify_source_files(continuation["source_files"])
            status.update(current_arm=name, phase="training", updated_at=now())
            write_json(output / "status.json", status)
            cfg, directory = configs[name], output / name
            write_json(output / f"{name}_config.json", cfg)
            entries = []

            def save_snapshot(agent, config, path, completed, next_seed):
                save_boundary_checkpoint(agent, config, path, completed, next_seed)
                if agent.stage != "joint" or Path(path).name != "joint_latest.pt":
                    return
                folder = directory / "curve_snapshots"
                folder.mkdir(exist_ok=True)
                total = config["training"]["pretrain_windows"] + completed
                target = folder / f"global_{total}.pt"
                if target.exists():
                    raise FileExistsError(target)
                torch.save(agent.checkpoint(config, include_replay=False), target)
                entries.append({"global_window": total, "path": str(target.relative_to(directory)),
                                "sha256": digest(target)})
                write_json(folder / "index.json", entries)

            checkpoint = source / name / "final.pt"
            print(f"[arm {name}] continue own final: global {continuation['start_global']} -> {continuation['end_global']}", flush=True)
            with patch("hrl_uav.qualification.save_boundary_checkpoint", side_effect=save_snapshot):
                report = run_mainline(cfg, directory, resume_from=checkpoint)
            if (report["status"] != "JOINT_PILOT_COMPLETE_NOT_FORMAL"
                    or not report["source_hashes_unchanged"]
                    or not all(report["resume"]["exact_restore_checks"].values())):
                raise AssertionError("Continuation failed integrity checks")
            audit = report["stage_audits"][-1]
            old_counter = continuation["restore_audits"][name]["replay_counter"]
            expected_frames = additional * cfg["training"]["window_macros"] * 18
            if (len(report["stage_audits"]) != 1 or audit["stage"] != "joint"
                    or audit["trained_windows_this_run"] != additional
                    or audit["replay_at_stage_entry"] != old_counter
                    or audit["replay_at_stage_exit"] - old_counter != expected_frames
                    or audit["critic_gradient_steps"] != 2 * audit["critic_updates"]
                    or audit["actor_updates"] <= 0 or audit["ppo_updates"] <= 0):
                raise AssertionError("Continuation changed budgets or learning clocks")
            if [e["global_window"] for e in entries] != continuation["expected_snapshot_windows"]:
                raise AssertionError("Missing retained continuation checkpoint")
            reports[name] = report
            status.update(phase="posthoc_evaluation", updated_at=now())
            write_json(output / "status.json", status)
            new_points = evaluate_snapshots(cfg, directory, entries)
            history = copy.deepcopy(origin["curves"][name])
            for point in history:
                point.setdefault("source_run", str(source))
            curves[name] = history + new_points
            status["completed_arms"].append(name)
            write_json(output / "comparison_partial.json", {
                n: {"mean": r["mean"], "minutes": r["minutes"]} for n, r in reports.items()
            })
            write_json(output / "status.json", status)
        verify_source_files(continuation["source_files"])
        manifests = {n: read_json(output / n / "manifest.json") for n in ARMS}
        checks = {
            "update_counts_equal": reports["sac"]["updates_total"] == reports["tqc"]["updates_total"],
            "training_scenes_equal": manifests["sac"]["training_scene_seeds"] == manifests["tqc"]["training_scene_seeds"],
            "code_hashes_equal": manifests["sac"]["source_hashes"] == manifests["tqc"]["source_hashes"],
            "own_final_only": all(Path(manifests[n]["old_checkpoint"]) == source / n / "final.pt" for n in ARMS),
            "no_teacher": all(not m["teacher"] for m in manifests.values()),
            "source_files_unchanged": True,
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        pairs = paired_rows(reports["tqc"]["final"], reports["sac"]["final"])
        result = {
            "protocol": protocol, "status": "SMOKE_ONLY" if protocol["smoke"] else "SINGLE_TRAINING_SEED_CONTINUATION_COMPLETE",
            "arms": {n: {"mean": r["mean"], "matched_centroid_mean": means(r["matched_centroid"]),
                         "final": r["final"], "matched_centroid": r["matched_centroid"],
                         "updates": r["updates_total"], "minutes": r["minutes"],
                         "stage_audits": r["stage_audits"]} for n, r in reports.items()},
            "tqc_minus_sac": pairs, "paired_summary": pair_summary(pairs),
            "matched_centroid_tqc_minus_sac": pair_summary(paired_rows(reports["tqc"]["matched_centroid"], reports["sac"]["matched_centroid"])),
            "macro_effect": {n: pair_summary(paired_rows(r["final"], r["matched_centroid"])) for n, r in reports.items()},
            "endpoint_change": {n: pair_summary(paired_rows(r["final"], origin["arms"][n]["final"])) for n, r in reports.items()},
            "curves": curves, "checks": checks,
            "crn": crn_check(output / "sac", output / "tqc", configs["sac"]["test_seeds"]),
            "minutes": (time.perf_counter() - started) / 60,
        }
        write_json(output / "comparison.json", result)
        from scripts.plot_sac_tqc import build_figures
        build_figures(output)
        status.update(status="COMPLETE", phase="complete", completed_at=now(), minutes=result["minutes"])
    except BaseException as error:
        status.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(), failed_at=now())
        raise
    finally:
        write_json(output / "status.json", status)
    print(result["paired_summary"]["task_delay_weighted_ms"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-windows", type=int, default=200)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new; archived results are never overwritten")
    if args.check_only:
        torch.set_num_threads(1)
        _, _, audit = prepare(args.source, args.additional_windows)
        import json
        print(json.dumps({"status": "READY", **audit}, ensure_ascii=False, indent=2))
        return
    run_continuation(args.source, args.output, args.additional_windows)


if __name__ == "__main__":
    main()
