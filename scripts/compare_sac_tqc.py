"""Fresh paired PPO+SAC versus PPO+TQC pilot; never reuse learned weights."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import append_csv, isolated_rng, smoke_config, write_json
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import means, metered_evaluate, run_mainline
from hrl_uav.resumption import capture_rng, save_boundary_checkpoint, state_equal


OUTPUT = ROOT / "results/sac_tqc_fresh1000_seed1"
ARMS = ("sac", "tqc")
METRICS = ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "J",
           "task_completion_rate", "queue_total_mbit", "mean_queue_compute_mbit",
           "mean_queue_transfer_mbit", "late_queue_slope_mbit_s", "power_w")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configurations(smoke=False):
    base = read_json(ROOT / "configs/hrl_ppo_only_unlock.json")
    base["training"].update(pretrain_windows=200, joint_windows=800, sac_critic_steps_per_update=2)
    # These are monitoring/budget settings, identical in both fresh runs.
    base["qualification"].update(validation_every_windows=60, checkpoint_every_windows=100,
                                 joint_evaluation_mode="final_only", max_stage_seconds=28800,
                                 test_seeds=list(range(34500, 34505)))
    base["test_seeds"] = base["qualification"]["test_seeds"]
    if smoke:
        base = smoke_config(base)
        base["qualification"].update(validation_every_windows=1, checkpoint_every_windows=1,
                                     tail_checkpoints=1, validation_macros=6,
                                     validation_seeds=[33000], test_seeds=[34500])
        base["validation_seeds"], base["test_seeds"] = [33000], [34500]
    configs = {name: copy.deepcopy(base) for name in ARMS}
    for name in ARMS:
        configs[name]["training"]["micro_algorithm"] = name
    configs["tqc"]["training"].update(tqc_quantiles=25, tqc_drop_per_critic=2)
    control = copy.deepcopy(configs["tqc"])
    for key in ("tqc_quantiles", "tqc_drop_per_critic"):
        control["training"].pop(key)
    control["training"]["micro_algorithm"] = "sac"
    if control != configs["sac"]:
        raise AssertionError("Non-algorithm configuration difference")
    return configs


def network_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initial_audit(configs):
    states, rngs, checks = {}, {}, {}
    with isolated_rng():
        for name, config in configs.items():
            seed = config["training"]["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            env = QueueMDP(config)
            env.reset(seed, config["training"]["window_macros"])
            agent = DirectHRL.from_mdp(env, config)
            state = copy.deepcopy(agent.checkpoint(config, include_replay=False))
            states[name], rngs[name] = state, capture_rng()
            checks[name] = {
                "sac_actor_sha256": network_hash(state["sac_actor"]),
                "macro_actor_sha256": network_hash(state["macro_actor"]),
                "macro_critic_sha256": network_hash(state["macro_critic"]),
                "micro_critic_parameters": sum(p.numel() for p in agent.sac_critic.critic_v.parameters()),
                "replay_empty": agent.replay.memory_counter == 0,
                "optimizer_states_empty": all(not o["state"] for o in state["optimizers"]),
                "optimizer_learning_rates": [[g["lr"] for g in o["param_groups"]] for o in state["optimizers"]],
            }
        equal = {key: state_equal(states["sac"][key], states["tqc"][key])
                 for key in ("sac_actor", "macro_actor", "macro_critic", "log_alpha", "optimizers", "updates")}
        equal["global_rng_after_initialization"] = state_equal(rngs["sac"], rngs["tqc"])
        if not all(equal.values()) or not all(c["replay_empty"] and c["optimizer_states_empty"] for c in checks.values()):
            raise AssertionError({"initialization": checks, "equality": equal})
    return {"arms": checks, "matched": equal,
            "critic_scope": "Output dimensions differ; initialized critic weights/initial Q values are not asserted equal. No trained weights are loaded."}


def paired_rows(left, right):
    lookup = {int(row["seed"]): row for row in right}
    if len(lookup) != len(right) or set(lookup) != {int(row["seed"]) for row in left} or len(left) != len(right):
        raise ValueError("Incomplete or duplicate CRN pairs")
    return [{"seed": int(row["seed"]), **{key: row[key] - lookup[int(row["seed"])][key] for key in METRICS}}
            for row in left]


def pair_summary(rows):
    result = {}
    for key in METRICS:
        values = np.asarray([r[key] for r in rows], dtype=float)
        mean = float(values.mean())
        # This interval is conditional on the TWO fixed trained policies.
        if len(values) == 5:
            half = 2.7764451051977987 * values.std(ddof=1) / np.sqrt(5)
            ci = [float(mean-half), float(mean+half)]
        else:
            ci = None
        result[key] = {"mean": mean, "paired_t95": ci,
                       "negative_pairs": int((values < 0).sum()), "positive_pairs": int((values > 0).sum())}
    return result


def crn_check(left_dir, right_dir, seeds):
    checks = {}
    for seed in seeds:
        with np.load(left_dir / f"final_{seed}_trajectory.npz") as a, np.load(right_dir / f"final_{seed}_trajectory.npz") as b:
            users_equal = bool(np.array_equal(a["users"], b["users"]))
        arrivals = []
        for directory in (left_dir, right_dir):
            with (directory / f"final_{seed}_queues.csv").open(encoding="utf-8-sig") as handle:
                arrivals.append([float(r["task_arrived_count"]) for r in csv.DictReader(handle)])
        checks[str(seed)] = {"users": users_equal, "arrival_counts": arrivals[0] == arrivals[1]}
    if not all(all(c.values()) for c in checks.values()):
        raise AssertionError(f"Evaluation CRN mismatch: {checks}")
    return checks


def evaluate_snapshots(config, directory, entries):
    points = []
    output = directory / "posthoc_validation"
    output.mkdir()
    with isolated_rng():
        for entry in entries:
            path = directory / entry["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise AssertionError("Snapshot changed before read-only evaluation")
            state = torch.load(path, map_location="cpu", weights_only=False)
            env = QueueMDP(config)
            env.reset(config["training"]["seed"], config["training"]["window_macros"])
            agent = DirectHRL.from_mdp(env, config)
            for key, network in (("sac_actor", agent.sac_actor.action_net), ("macro_actor", agent.macro_actor),
                                 ("sac_critic", agent.sac_critic.critic_v), ("sac_target", agent.sac_critic.target_critic_v),
                                 ("macro_critic", agent.macro_critic)):
                network.load_state_dict(state[key])
            with torch.no_grad():
                agent.entropy.log_alpha.copy_(state["log_alpha"].to(agent.entropy.log_alpha.device))
            before = copy.deepcopy(agent.checkpoint(config))
            rng = capture_rng()
            rows = metered_evaluate(config, agent, config["qualification"]["validation_seeds"],
                                    config["qualification"]["validation_macros"], output,
                                    f"global_{entry['global_window']}", macro_enabled=True)
            if not state_equal(before, agent.checkpoint(config)) or not state_equal(rng, capture_rng()):
                raise AssertionError("Posthoc evaluation changed network/replay/optimizer/RNG")
            append_csv(output / "readings.csv", [{"global_window": entry["global_window"], **r} for r in rows])
            points.append({**entry, "mean": means(rows), "rows": rows})
            print(f"[posthoc {directory.name} {entry['global_window']}] {means(rows)}", flush=True)
    write_json(output / "points.json", points)
    return points


def run_comparison(output, smoke=False):
    configs = configurations(smoke)
    torch.set_num_threads(1)
    audit = initial_audit(configs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {
        "started_at": now(), "smoke": smoke, "training_seed_count": 1,
        "training_runs": 2, "configs": configs, "initialization": audit,
        "comparison": "Fresh PPO+direct SAC vs PPO+direct TQC. No teacher, anchors, action gates, old replay, old actor or optimizer states.",
        "treatment": "Two 25-quantile critics; pool 50 next quantiles, drop largest 4; mean quantile-Huber loss; actor uses untruncated current mean.",
        "shared_schedule": "200 micro-only + 800 joint windows; two Critic fits on each fixed batch/target per event from the beginning in BOTH arms. Same actor/entropy/target/PPO clocks and learning rates.",
        "selection": "Final weights only. No best-checkpoint selection, no performance-based early stopping or automatic extension.",
        "scope": "Single matched training initialization. Five new evaluation seeds do not equal five training replications. 1000 windows is a pilot budget, not a convergence guarantee.",
        "metrics": "Primary: finite-horizon J and weighted completed-task delay; report high/low delay, unfinished tasks, queues, completion, power, actual wall time, and matched macro effect.",
        "interval_scope": "Paired t95 across 5 evaluation scenes, conditional on the two trained checkpoints; exploratory, not training robustness or multiplicity-adjusted proof.",
        "historical_scope": "Historical 1800/2000 endpoints and their reused seeds are not matched controls. This control is independently retrained, with two Critic fits from window 1.",
        "references": ["https://proceedings.mlr.press/v119/kuznetsov20a.html",
                       "https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/tqc/tqc.html"],
    }
    write_json(output / "protocol.json", protocol)
    status = {"status": "RUNNING", "started_at": protocol["started_at"], "completed_arms": []}
    write_json(output / "status.json", status)
    reports, curves = {}, {}
    try:
        for name in ARMS:
            status.update(current_arm=name, phase="training", updated_at=now())
            write_json(output / "status.json", status)
            cfg = configs[name]
            write_json(output / f"{name}_config.json", cfg)
            directory = output / name
            entries = []
            desired = ([cfg["training"]["joint_windows"]] if smoke else [200, 400, 600, 700, 800])

            def save_with_snapshot(agent, config, path, completed, next_seed):
                save_boundary_checkpoint(agent, config, path, completed, next_seed)
                if agent.stage != "joint" or Path(path).name != "joint_latest.pt" or not desired or completed < desired[0]:
                    return
                while desired and completed >= desired[0]:
                    desired.pop(0)
                snapshots = directory / "curve_snapshots"
                snapshots.mkdir(exist_ok=True)
                global_window = cfg["training"]["pretrain_windows"] + completed
                dest = snapshots / f"global_{global_window}.pt"
                torch.save(agent.checkpoint(config, include_replay=False), dest)
                entries.append({"global_window": global_window, "path": str(dest.relative_to(directory)),
                                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
                write_json(snapshots / "index.json", entries)

            print(f"[arm {name}] fresh start, pretrain={cfg['training']['pretrain_windows']}, joint={cfg['training']['joint_windows']}", flush=True)
            with patch("hrl_uav.qualification.save_boundary_checkpoint", side_effect=save_with_snapshot):
                report = run_mainline(cfg, directory)
            if report["status"] != "JOINT_PILOT_COMPLETE_NOT_FORMAL" or not report["source_hashes_unchanged"]:
                raise RuntimeError(f"Arm did not complete a valid budget: {name}: {report['status']}")
            if desired:
                raise AssertionError(f"Missing curve snapshots: {desired}")
            for stage in report["stage_audits"]:
                if stage["trained_windows_this_run"] != cfg["training"][f"{stage['stage']}_windows"]:
                    raise AssertionError("Incomplete training stage")
                if stage["actor_updates"] <= 0 or stage["critic_gradient_steps"] != 2 * stage["critic_updates"]:
                    raise AssertionError("Incorrect micro learning cadence")
            if report["stage_audits"][-1]["ppo_updates"] <= 0:
                raise AssertionError("PPO did not train")
            reports[name] = report
            status.update(phase="posthoc_evaluation", updated_at=now())
            write_json(output / "status.json", status)
            curves[name] = evaluate_snapshots(cfg, directory, entries)
            status["completed_arms"].append(name)
            write_json(output / "comparison_partial.json", {n: {"mean": r["mean"], "minutes": r["minutes"]} for n, r in reports.items()})
            write_json(output / "status.json", status)
        a, b = reports["sac"], reports["tqc"]
        checks = {"update_counts_equal": a["updates_total"] == b["updates_total"]}
        manifests = {n: read_json(output / n / "manifest.json") for n in ARMS}
        checks["training_scenes_equal"] = manifests["sac"]["training_scene_seeds"] == manifests["tqc"]["training_scene_seeds"]
        checks["code_hashes_equal"] = manifests["sac"]["source_hashes"] == manifests["tqc"]["source_hashes"]
        checks["no_old_checkpoint"] = all(m["old_checkpoint"] is None and not m["teacher"] for m in manifests.values())
        if not all(checks.values()):
            raise AssertionError(checks)
        pairs = paired_rows(b["final"], a["final"])
        result = {
            "protocol": protocol, "arms": {n: {"mean": r["mean"], "matched_centroid_mean": means(r["matched_centroid"]),
                "final": r["final"], "matched_centroid": r["matched_centroid"], "updates": r["updates_total"],
                "minutes": r["minutes"], "stage_audits": r["stage_audits"]} for n, r in reports.items()},
            "tqc_minus_sac": pairs, "paired_summary": pair_summary(pairs),
            "matched_centroid_tqc_minus_sac": pair_summary(paired_rows(b["matched_centroid"], a["matched_centroid"])),
            "macro_effect": {n: pair_summary(paired_rows(r["final"], r["matched_centroid"])) for n, r in reports.items()},
            "curves": curves, "checks": checks,
            "crn": crn_check(output / "sac", output / "tqc", configs["sac"]["test_seeds"]),
            "minutes": (time.perf_counter()-started)/60,
            "status": "SMOKE_ONLY" if smoke else "SINGLE_TRAINING_SEED_PILOT_COMPLETE",
        }
        write_json(output / "comparison.json", result)
        from plot_sac_tqc import build_figures
        build_figures(output)
        status.update(status="COMPLETE", phase="complete", completed_at=now(), minutes=result["minutes"])
    except BaseException as error:
        status.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(), failed_at=now())
        raise
    finally:
        write_json(output / "status.json", status)
    print(json.dumps(result["paired_summary"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output.resolve() == OUTPUT.resolve():
        parser.error("Smoke requires a separate --output directory")
    if args.check_only:
        torch.set_num_threads(1)
        print(json.dumps({"status": "READY", "configs": configurations(args.smoke),
                          "initialization": initial_audit(configurations(args.smoke))}, indent=2))
        return
    run_comparison(args.output, args.smoke)


if __name__ == "__main__":
    main()
