"""Fixed 120-window orchestration around the unchanged d3debca residual controller."""

import copy
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .bandwidth import summarize_meter
from .experiment import append_csv, isolated_rng, parameter_delta, rollout, snapshot, write_json
from .mdp import QueueMDP
from .ppo_frozen import (
    DEFAULT_SOURCE, SOURCE_DIGESTS, STAGE_TRANSITION, assert_sac_frozen, load_json, sac_learning_state,
)
from .ppo_residual_b import (
    SEMANTICS, ResidualAgent, ResidualFrozen, ResidualJournal, ResidualMeter,
    frozen_guards, prepare_residual_source,
)
from .priority_grid import file_hash
from .resumption import capture_rng, restore_agent, save_boundary_checkpoint, state_equal
from .scene_prescreen import ROOT, assert_crn_equal, source_hashes
from .target_tracking_check import TrackingPolicy, numeric_rows
from .trajectory_check import crn_records
from .wireless_check import ENERGY, PRIORITY_MEASURES
from .wireless_pilot import route_statistics


ACCEPTED_HEAD = "d3debca82de7c1e6bf3574ff458c1769683c114a"
OBSERVATIONS = (0, 30, 60, 90, 120)
VALIDATION = (33600, 33601)
DEVELOPMENT = (33610, 33611, 33612)
OLD = ROOT/"results/ppo_frozen_sac_pilot120_seed101"
B_RUN = ROOT/"results/progressive_centroid_dev_recheck"
ACCEPTED_FILES = ("hrl_uav/ppo_residual_b.py", "scripts/run_ppo_residual_b.py", "tests/test_ppo_residual_b.py")


def accepted_source():
    smoke = load_json(ROOT/"results/ppo_residual_b_smoke_seed101/manifest.json")
    if smoke["actual_head"] != ACCEPTED_HEAD or smoke["status"] != "RESIDUAL_B_SMOKE_ONLY_COMPLETE":
        raise ValueError("Missing accepted residual smoke provenance")
    for name, digest in smoke["source_hashes"].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError(f"Accepted implementation/config changed: {name}")
    return {k: smoke["source_hashes"][k] for k in ACCEPTED_FILES}


class PilotJournal(ResidualJournal):
    def before(self, agent, gae_arguments, gae_result):
        super().before(agent, gae_arguments, gae_result)
        path = self.directory/f"update{agent.ppo_updates+1:03d}.npz"
        with np.load(path) as z:
            data = {k: z[k].copy() for k in z.files}
        if not all(np.isfinite(v).all() for v in data.values()):
            raise FloatingPointError("Nonfinite actual PPO sample journal")
        n = len(data["state"])
        data.update(update_id=np.full(n, agent.ppo_updates+1, dtype=int),
                    training_initialization_seed=np.full(n, agent.cfg["seed"], dtype=int))
        np.savez_compressed(path, **data)


class FullBatchFrozen(ResidualFrozen):
    def update_ppo(self, progress, force=False):
        if force:
            raise AssertionError("Formal pilot forbids forced PPO updates")
        count = len(self._agent.macro_rollout)
        if count > self._agent.cfg["ppo_update_macros"]:
            raise AssertionError("PPO batch exceeded its fixed native threshold")
        before = self.ppo_updates
        result = super().update_ppo(progress, force=False)
        if self.ppo_updates != before:
            if count != 300 or self.ppo_update_events[-1]["forced"]:
                raise AssertionError("Formal pilot requires exactly 300 unforced transitions per update")
            if not all(np.isfinite(v) for v in self.last_ppo_stats.values()):
                raise FloatingPointError("Nonfinite PPO update statistics")
        return result

    def checkpoint(self, config, include_replay=True):
        state = super().checkpoint(config, include_replay)
        state["experiment"].update(smoke_only=False, fixed_budget_windows=120,
                                   scope="development pilot, final window120 only",
                                   resume_boundary="Audit snapshot; no interruption-resume entry is exposed")
        return state


def assert_budget(proxy, journal, end_window):
    transitions = end_window*30
    if (proxy.consumed_macro_transitions+len(proxy.macro_rollout) != transitions
            or proxy.ppo_updates != transitions//300 or len(proxy.macro_rollout) != transitions % 300
            or journal.cursor != proxy.consumed_macro_transitions or len(journal.records) != transitions
            or journal.pending is not None
            or any(e["transitions"] != 300 or e["forced"] for e in proxy.ppo_update_events)):
        raise AssertionError("Full-batch PPO/journal conservation failed")
    if end_window == 120 and (proxy.ppo_updates != 12 or proxy.macro_rollout or journal.cursor != 3600):
        raise AssertionError("Expected exactly 12 complete unforced PPO batches")


def reference_rows(cfg, source_saved):
    """Reuse audited A/B/absolute-PPO development records, never reserved data."""
    from scripts.run_progressive_centroid_recheck import FIELDS
    bm, pm = load_json(B_RUN/"manifest.json"), load_json(OLD/"manifest.json")
    if pm["config"] != cfg or pm["source_checkpoint_sha256"] != SOURCE_DIGESTS["checkpoint"]:
        raise ValueError("Original PPO configuration/source mismatch")
    if bm["status"] != "COMPLETE" or bm["source_digests"] != SOURCE_DIGESTS:
        raise ValueError("B development source mismatch")
    if any(bm["config"][k] != cfg[k] for k in cfg if k != "training"):
        raise ValueError("B physical/MDP settings mismatch")
    inputs = {}

    def remember(path):
        path = Path(path)
        inputs[str(path.relative_to(ROOT))] = file_hash(path)

    for path in (B_RUN/"manifest.json", B_RUN/"per_seed.csv", OLD/"manifest.json", OLD/"report.json"):
        remember(path)
    for name, digest in bm["reused_file_hashes"].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError(f"Archived reference file changed: {name}")
    old_final = Path(pm["final_checkpoint"])
    if file_hash(old_final) != pm["final_checkpoint_sha256"]:
        raise ValueError("Original PPO final checkpoint hash mismatch")
    saved = torch.load(old_final, map_location="cpu", weights_only=False)
    for key in ("sac_actor", "sac_critic", "sac_target", "log_alpha", "observations"):
        if not state_equal(saved[key], source_saved[key]):
            raise ValueError(f"Historical frozen SAC differs: {key}")
    if not state_equal(saved["optimizers"][:3], source_saved["optimizers"][:3]):
        raise ValueError("Historical SAC optimizer mismatch")
    remember(old_final)
    rows = numeric_rows(B_RUN/"per_seed.csv")
    if {(r["policy"], int(r["seed"])) for r in rows} != {(p, s) for p in ("A", "B", "PPO120") for s in VALIDATION+DEVELOPMENT} or len(rows) != 15:
        raise ValueError("Expected exactly 15 existing development references")
    for r in rows:
        r["seed"] = int(r["seed"])
        if r["policy"] == "B":
            directory = Path(r["source_path"])
            r.update(crn_path=str(directory/"crn.csv"), trajectory_path=str(directory/"trajectory.npz"))
        else:
            split = "validation" if r["seed"] in VALIDATION else "development_final"
            controller = "centroid" if r["policy"] == "A" else "ppo"
            prefix = OLD/f"{split}_{controller}_w120_{r['seed']}"
            r.update(crn_path=str(prefix)+"_crn.csv", trajectory_path=str(prefix)+"_trajectory.npz")
        for k in ("crn_path", "trajectory_path"):
            remember(r[k])
        if not all(np.isfinite(r[k]) for k in FIELDS):
            raise ValueError("Incomplete historical common metrics")
    return rows, inputs


def evaluate_full(cfg, agent, output, seed, window, *, residual=True):
    if type(seed) is not int or seed not in VALIDATION+DEVELOPMENT:
        raise ValueError("Only the five authorized development scenes can be evaluated")
    if window not in OBSERVATIONS or (seed in DEVELOPMENT and window != 120):
        raise ValueError("Scene/window outside the fixed protocol")
    before, rng = copy.deepcopy(agent.checkpoint(cfg)), capture_rng()
    label = "Residual_PPO" if residual else "B"
    directory = Path(output)/"evaluations"/f"window{window}"/label/str(seed)
    directory.mkdir(parents=True, exist_ok=False)
    policy = agent if residual else TrackingPolicy(agent, dict(name="B", offsets_m=[[0, 0]]*3, tracking_coefficient=.5))

    def factory(config):
        meter = ResidualMeter(config, residual=residual)
        if not residual:
            policy.meter = meter
        return meter

    with isolated_rng(), frozen_guards(agent, evaluation=True), patch("hrl_uav.experiment.QueueMDP", factory):
        summary, rows, trajectory, routes, meter = rollout(cfg, policy, seed, 90,
            macro_enabled=True, train=False, complete_tail=False, record=True)
    if (meter.reset_calls != 1 or summary["time_s"] != 900 or len(meter.controls) != 90
            or summary["boundary_transitions"] or len(routes) != 90*18
            or any(r["tail_only"] or not r["execution_complete"] for r in meter.controls)):
        raise AssertionError("900s evaluation protocol changed")
    summary.update(summarize_meter(meter.readings, 900))
    summary.update({k: meter.readings[-1][k] for k in PRIORITY_MEASURES})
    summary.update({k: sum(r[k] for r in meter.readings) for k in ENERGY})
    summary.update(J_holding=meter.objective_holding/meter.integration_steps,
                   J_energy=meter.objective_energy/meter.integration_steps)
    np.testing.assert_allclose(summary["J"], summary["J_holding"]+summary["J_energy"], atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(summary["power_w"]*900, sum(summary[k] for k in ENERGY), atol=1e-6, rtol=1e-12)
    tail = rows[len(rows)*2//3:]
    uav, users, centroids, variable = map(np.asarray, zip(*trajectory))
    raw, route = map(np.asarray, zip(*routes))
    controls = {k: np.asarray([r[k] for r in meter.controls]) for k in (
        "state", "current_uav", "current_centroid", "current_load", "B_target", "B_feasible_target", "a_B",
        "sampled_delta", "pre_projection", "a_exec", "projected", "actual_displacement", "B_projection")}
    summary.update(late_queue_slope_mbit_s=float(np.polyfit([r["time_s"] for r in tail], [r["queue_total_mbit"] for r in tail], 1)[0]),
        mean_queue_total_mbit=sum(summary[f"mean_queue_{q}_mbit"] for q in ("upload", "compute", "transfer")),
        mean_uav_centroid_distance_m=float(np.linalg.norm(uav-centroids, axis=-1).mean()),
        fleet_distance_m=sum(r["distance_m"] for r in meter.deployments),
        residual_magnitude_mean=float(np.linalg.norm(controls["sampled_delta"], axis=1).mean()),
        residual_magnitude_max=float(np.linalg.norm(controls["sampled_delta"], axis=1).max()),
        projection_uav_fraction=float(controls["projected"].mean()),
        projection_macro_fraction=float(controls["projected"].any(axis=1).mean()))
    for p in ("high", "low"):
        if summary[f"task_arrived_{p}_count"] != summary[f"task_completed_{p}_count"]+summary[f"task_unfinished_{p}_count"]:
            raise AssertionError("Parent-task count conservation failed")
    if not all(np.isfinite(v) for v in summary.values() if isinstance(v, (int, float, np.number))):
        raise FloatingPointError("Nonfinite evaluation output")
    np.savez_compressed(directory/"trajectory.npz", uav=uav, users=users, centroid=centroids,
        macro_action=controls["a_exec"], policy_variable=variable, raw_action=raw, executed_route=route)
    np.savez_compressed(directory/"controls.npz", **controls)
    for name, records in (("queues", rows), ("frames", meter.readings), ("deployments", meter.deployments),
                          ("routes", meter.route_readings), ("links", meter.link_readings),
                          ("uavs", meter.uav_readings), ("crn", crn_records(meter))):
        append_csv(directory/f"{name}.csv", records)
    route_rows = route_statistics(meter.route_readings, label, seed, window, "development")
    append_csv(directory/"route_summary.csv", route_rows)
    result = dict(policy=label, seed=seed, window=window, evaluation_macros=90, reused=False,
        source_path=str(directory), crn_path=str(directory/"crn.csv"), trajectory_path=str(directory/"trajectory.npz"),
        **summary)
    write_json(directory/"metrics.json", result)
    if not state_equal(before, agent.checkpoint(cfg)) or not state_equal(rng, capture_rng()):
        raise AssertionError("Evaluation changed learning state, replay or RNG")
    return result


def paired_checks(reference, current, *, identical=False):
    from scripts.run_progressive_centroid_recheck import FIELDS, compare_processes
    check = compare_processes(reference, current)
    if identical:
        for key in FIELDS:
            np.testing.assert_allclose(current[key], reference[key], atol=1e-9, rtol=0, err_msg=key)
        with np.load(reference["trajectory_path"]) as a, np.load(current["trajectory_path"]) as b:
            for key in ("uav", "raw_action", "executed_route"):
                np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=0, err_msg=key)
    return check


def zero_check(reference, current):
    paired_checks(reference, current, identical=True)
    a_dir, b_dir = Path(reference["source_path"]), Path(current["source_path"])
    with np.load(a_dir/"controls.npz") as a, np.load(b_dir/"controls.npz") as b:
        np.testing.assert_array_equal(b["sampled_delta"], 0)
        for k in ("B_target", "B_feasible_target", "a_B", "a_exec", "actual_displacement"):
            np.testing.assert_allclose(a[k], b[k], atol=1e-9, rtol=0, err_msg=k)
    keys = ("J", "J_holding", "J_energy", "power_w", *ENERGY)
    for k in keys:
        np.testing.assert_allclose(reference[k], current[k], atol=1e-9, rtol=0, err_msg=k)
    from .wireless_check import read_csv
    assert_crn_equal(read_csv(a_dir/"crn.csv"), read_csv(b_dir/"crn.csv"))
    return dict(seed=current["seed"], duration_s=900, window0_equal_B=True,
                max_metric_error=max(abs(reference[k]-current[k]) for k in keys),
                full_physical_step_CRN_equal=True, trajectories_and_routes_equal=True)


def run_pilot(output):
    accepted = accepted_source()
    cfg, saved, provenance = prepare_residual_source()
    references, reused_hashes = reference_rows(cfg, saved)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"checkpoints").mkdir()
    torch.set_num_threads(1)
    model = QueueMDP(cfg)
    model.reset(72000, 30)
    agent = ResidualAgent.from_mdp(model, cfg)
    audit = restore_agent(agent, saved, dict(checkpoint=str(DEFAULT_SOURCE),
        checkpoint_sha256=SOURCE_DIGESTS["checkpoint"], transition=STAGE_TRANSITION,
        source_stage="pretrain", source_sac_window=2000, target_stage="B_residual_ppo_only",
        completed_ppo_windows=0, replay_counter=int(saved["replay"]["memory_counter"])))
    del saved
    frozen = sac_learning_state(agent, cfg)
    agent.set_stage("joint")
    assert_sac_frozen(agent, cfg, frozen)
    if (agent.ppo_updates or agent.ppo_gradient_steps or agent.macro_rollout
            or agent.ppo_actor_optimizer.state or agent.ppo_critic_optimizer.state
            or torch.count_nonzero(agent.macro_actor.out.weight).item()
            or torch.count_nonzero(agent.macro_actor.out.bias).item()):
        raise AssertionError("Formal pilot requires the untouched source PPO initialization")
    np.testing.assert_allclose(agent.macro_actor.log_std.detach().exp().cpu(), 1/3, atol=1e-7, rtol=0)
    proxy = FullBatchFrozen(agent, SOURCE_DIGESTS["checkpoint"])
    journal = PilotJournal(output/"ppo_samples")
    agent.ppo_journal = journal
    initial_actor, initial_critic = snapshot(agent.macro_actor), snapshot(agent.macro_critic)
    initial_replay = agent.replay.memory_counter
    hashes = source_hashes()
    manifest = dict(status="PREFLIGHT_PASSED", accepted_head=ACCEPTED_HEAD,
        actual_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        worktree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        config=cfg, observation_spec=agent.observation_spec,
        accepted_residual_hashes=accepted, source_hashes=hashes,
        source_checkpoint=str(DEFAULT_SOURCE), source_checkpoint_sha256=SOURCE_DIGESTS["checkpoint"],
        provenance=provenance, restore_audit=audit, stage_transition=STAGE_TRANSITION,
        target_stage="B_residual_ppo_only", probability_semantics=SEMANTICS,
        training_initialization_seed=101, training_scene_seeds=list(range(72000, 72040)),
        validation_seeds=list(VALIDATION), development_seeds=list(DEVELOPMENT), observations=list(OBSERVATIONS),
        total_windows=120, total_macro_transitions=3600, expected_full_batches=12,
        checkpoint_selection="final window120 only", forced_updates_allowed=False,
        sac_learning=False, sac_training_action_mode="stochastic", sac_evaluation_action_mode="deterministic",
        replay_collection=True, replay_sampling=False, smoke_resume=False, resume_supported=False,
        reused_file_hashes=reused_hashes, reserved_scenes_executed=[],
        scope="one training initialization, reused development scenes, no inference beyond this protocol")
    write_json(output/"manifest.json", manifest)
    write_json(output/"reference_records.json", references)
    started = time.perf_counter()
    observations, crn_checks, zero_checks = [], [], []
    for seed in VALIDATION:
        b = evaluate_full(cfg, agent, output, seed, 0, residual=False)
        reference = next(r for r in references if r["policy"] == "B" and r["seed"] == seed)
        crn_checks.append(paired_checks(reference, b, identical=True))
        r = evaluate_full(cfg, agent, output, seed, 0)
        zero_checks.append(zero_check(b, r))
        observations.append(r)
    write_json(output/"window0_checks.json", zero_checks)
    save_boundary_checkpoint(proxy, cfg, output/"checkpoints/window0.pt", 0, 72000)
    print("Window0: both 900s development pairs equal B; starting 120-window pilot", flush=True)

    for stream, offset in enumerate(range(0, 120, 3)):
        before_replay, before_blocked = agent.replay.memory_counter, proxy.blocked_update_calls
        scene = 72000+stream

        def factory(config):
            return ResidualMeter(config, agent=agent, journal=journal, global_offset=offset*30)

        with frozen_guards(agent), patch("hrl_uav.experiment.QueueMDP", factory):
            summary, rows, _, _, meter = rollout(cfg, proxy, scene, 90,
                macro_enabled=True, train=True, complete_tail=True, window_offset=offset,
                progress_offset=offset/120, record=False)
        end = offset+3
        if (agent.replay.memory_counter-before_replay != 1620 or summary["boundary_transitions"] != 90
                or summary["final_pending_dropped"] or not np.isclose(summary["boundary_discount_min"], .99**3)
                or proxy.blocked_update_calls-before_blocked != 81 or len(meter.controls) != 91
                or not meter.controls[-1]["tail_only"] or meter.reset_calls != 1):
            raise AssertionError("Training replay/boundary protocol violation")
        assert_budget(proxy, journal, end)
        assert_sac_frozen(agent, cfg, frozen)
        append_csv(output/"training.csv", rows)
        append_csv(output/"boundary_audit.csv", [dict(end_window=end, scene=scene,
            replay_added=agent.replay.memory_counter-before_replay, boundary_transitions=90,
            boundary_discount=summary["boundary_discount_min"], pending_dropped=0,
            blocked_sac_updates=81, ppo_updates=agent.ppo_updates,
            consumed_macros=proxy.consumed_macro_transitions, pending_macros=len(agent.macro_rollout),
            summary_time_s=summary["time_s"], environment_time_s=meter.env._current_time())])
        save_boundary_checkpoint(proxy, cfg, output/"boundary_latest.pt", end, scene+1)
        if end in OBSERVATIONS:
            save_boundary_checkpoint(proxy, cfg, output/f"checkpoints/window{end}.pt", end, scene+1)
            for seed in VALIDATION:
                r = evaluate_full(cfg, agent, output, seed, end)
                reference = next(x for x in references if x["policy"] == "B" and x["seed"] == seed)
                crn_checks.append(paired_checks(reference, r))
                observations.append(r)
            print(f"Observation {end}: J={np.mean([r['J'] for r in observations if r['window']==end]):.9f}, PPO batches={agent.ppo_updates}", flush=True)
        write_json(output/"observations.json", observations)
        write_json(output/"progress.json", dict(status="TRAINING", completed_windows=end, target_windows=120,
            ppo_updates=agent.ppo_updates, macro_transitions=journal.cursor,
            elapsed_seconds=time.perf_counter()-started))
        manifest.update(status="TRAINING", completed_windows=end)
        write_json(output/"manifest.json", manifest)

    assert_budget(proxy, journal, 120)
    save_boundary_checkpoint(proxy, cfg, output/"final.pt", 120, 72040)
    final = [r for r in observations if r["window"] == 120]
    for seed in DEVELOPMENT:
        final.append(evaluate_full(cfg, agent, output, seed, 120))
    for r in final:
        for policy in ("A", "B", "PPO120"):
            reference = next(x for x in references if x["policy"] == policy and x["seed"] == r["seed"])
            crn_checks.append(paired_checks(reference, r))
    checks = assert_sac_frozen(agent, cfg, frozen)
    if agent.replay.memory_counter-initial_replay != 64800:
        raise AssertionError("Final replay conservation failed")
    if any(file_hash(ROOT/p) != h for p, h in hashes.items()) or any(file_hash(ROOT/p) != h for p, h in reused_hashes.items()):
        raise AssertionError("Sources or reused records changed during pilot")
    for key, path in (("checkpoint", DEFAULT_SOURCE), ("manifest.json", DEFAULT_SOURCE.parent.parent/"manifest.json"),
                      ("report.json", DEFAULT_SOURCE.parent.parent/"report.json")):
        if file_hash(path) != SOURCE_DIGESTS[key]:
            raise AssertionError("Source SAC provenance changed during pilot")
    report = dict(status="RESIDUAL_PPO120_DEVELOPMENT_COMPLETE", ppo_windows=120,
        ppo_updates=agent.ppo_updates, ppo_actor_gradient_steps=agent.ppo_gradient_steps,
        ppo_critic_gradient_steps=sum(int(r["critic_epochs"]) for r in numeric_rows(output/"ppo_samples/updates.csv")),
        update_events=proxy.ppo_update_events, macro_transitions_consumed=journal.cursor,
        macro_transitions_pending=len(agent.macro_rollout), sac_frozen_checks=checks,
        sac_updates_at_entry=frozen["sac_updates"], sac_updates_at_exit=sac_learning_state(agent, cfg)["sac_updates"],
        sac_update_increment=0, replay_added=agent.replay.memory_counter-initial_replay,
        blocked_sac_update_calls=proxy.blocked_update_calls,
        macro_actor_max_delta=parameter_delta(initial_actor, agent.macro_actor, ("log_std",)),
        macro_critic_max_delta=parameter_delta(initial_critic, agent.macro_critic),
        observations=observations, final_development=final, source_unchanged=True,
        reserved_scenes_executed=[], forced_updates=0, resume_supported=False)
    write_json(output/"pairing_checks.json", crn_checks)
    write_json(output/"report.json", report)
    manifest.update(status=report["status"], elapsed_seconds=time.perf_counter()-started,
        final_checkpoint=str(output/"final.pt"), final_checkpoint_sha256=file_hash(output/"final.pt"),
        source_unchanged=True)
    write_json(output/"manifest.json", manifest)
    write_json(output/"progress.json", dict(status=report["status"], completed_windows=120, ppo_updates=12))
    return report
