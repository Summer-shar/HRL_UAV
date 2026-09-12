"""Stage 3A: fixed Predictive-centered, one-intervention mission branches."""

import copy
import gzip
import json
import pickle
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .bandwidth import summarize_meter
from .experiment import append_csv, isolated_rng, write_json
from .local_counterfactual import actions, physical_fields, semantic_hash, objective_values
from .mdp import holding_energy_cost
from .mission_micro_pilot import verify_previous
from .mission_pilot import RUN_FILES, MissionPilotMeter, ObservedPolicy, crn_records
from .mission_provenance import PHYSICS_BASE_COMMIT, assert_committed_sources, physics_provenance
from .mission_smoke import ROOT, load_spec, load_source, mission_config
from .priority_grid import file_hash
from .resumption import capture_rng, restore_rng, state_equal
from .scene_prescreen import assert_crn_equal, role_summary
from .trajectory_check import ReadOnlySAC, no_learning
from .wireless_check import ENERGY, read_csv
from .wireless_pilot import route_statistics

SCENES = (33600, 33601)
TIMES = tuple(range(100, 801, 100))
HORIZON = 6
C_ATOL, R_ATOL, POSITION_ATOL = 1e-8, 1e-10, 1e-8
OUTPUT = ROOT / "results/mission_predictive_stage3a"
NEW_FILES = ("hrl_uav/mission_opportunity.py", "hrl_uav/mission_opportunity_analysis.py",
             "scripts/run_mission_opportunity.py", "tests/test_mission_opportunity.py",
             "docs/mission_predictive_stage3a.md")
DEPENDENCIES = ("hrl_uav/local_counterfactual.py", "hrl_uav/mission_micro_pilot.py",
                "scripts/analyze_local_counterfactual.py")
HOOKS = {"_enqueue_compute_chunk", "_register_arrived_parent_task", "_integrate_user_positions",
         "record_trajectory_snapshot", "_update_gauss_markov_mobility"}
DIGESTS = ("arrivals_digest", "motion_digest", "users_digest", "centroid_digest", "innovation_digest")


def protocol():
    return dict(scenes=list(SCENES), branch_times_s=list(TIMES), actions=actions(), epsilon=.05,
                branch_states=16, branches=208, macros_per_branch=HORIZON,
                branch_physical_steps=600, total_branch_macros=1248, future_replicates=1,
                baseline="unchanged Predictive Centroid, causal 10s lead",
                branch_instant="after advance_macro_clock, before next macro observation/action",
                intervention="first macro only; subsequent five use current branch Predictive",
                mapping="raw clip -> per-UAV unit disk ->20m -> map -> joint safety -> continuous motion",
                replay="each scene to860s, same90-macro schedule as prior900s run; no reset at300/600s",
                primary="sum of600 original physical-step costs, not completed-only delay",
                training_return="native frame reward; sum(gamma^i*r_i)/20; macro gamma=gamma^20; no tail V",
                denominators="queues:120 frame ends; J:C60/600; power:full energy/60s; rates:actual bits/sender seconds",
                completion="branch completions/(initial unfinished + branch arrivals), not completions/arrivals",
                c_atol=C_ATOL, return_atol=R_ATOL, position_atol_m=POSITION_ATOL,
                training=False, train=False, complete_tail=False, replay_writes=0, replay_samples=0,
                excluded_scenes=[33610, 33611, 33612, *range(33700, 33710)],
                limits="single future per state; retrospective oracle selection; descriptive development evidence only")


def validate_branch(seed, start, action, horizon=HORIZON):
    if type(seed) is not int or seed not in SCENES or start not in TIMES or horizon != HORIZON:
        raise ValueError("Only the sixteen declared development states and six-macro horizon are allowed")
    if action not in actions():
        raise ValueError("Only the thirteen fixed direct-action perturbations are allowed")


def physical_state(meter, fields):
    overrides = {k for k, v in vars(meter.env).items() if callable(v)}
    if overrides != HOOKS:
        raise AssertionError(f"Unrecognized environment hooks: {overrides}")
    return dict(mdp={k: getattr(meter, k) for k in fields},
                env={k: v for k, v in sorted(vars(meter.env).items()) if k not in HOOKS})


class MissionSnapshot:
    """Pickle all physical state at once to preserve parent/chunk aliases."""

    def __init__(self, meter, fields):
        self.fields = fields
        state = dict(physical=physical_state(meter, fields), rng=capture_rng())
        self.blob = pickle.dumps(state, protocol=5)
        self.initial_hash = semantic_hash(state)
        self.digests = {k: getattr(meter, k).copy() for k in DIGESTS}
        self.schedule_hash = meter.schedule_hash

    def restore(self):
        state = pickle.loads(self.blob)
        physical = state["physical"]
        # New audit callbacks must capture the new environment, not the original replay.
        meter = MissionPilotMeter(physical["mdp"]["config"])
        meter.env.__dict__.update(physical["env"])
        meter.__dict__.update(physical["mdp"])
        meter.schedule_hash = self.schedule_hash
        for key, digest in self.digests.items():
            setattr(meter, key, digest.copy())
        restore_rng(state["rng"])
        if semantic_hash(dict(physical=physical_state(meter, self.fields), rng=capture_rng())) != self.initial_hash:
            raise AssertionError("Full physical state/RNG restore failed")
        return meter

    def save(self, path):
        Path(path).write_bytes(gzip.compress(self.blob, compresslevel=3))
        if semantic_hash(pickle.loads(gzip.decompress(Path(path).read_bytes()))) != self.initial_hash:
            raise AssertionError("Snapshot disk roundtrip failed")


def macro_step(meter, sac, delta):
    """The mission rollout service sequence, without learning, tail or resets."""
    env, cfg = meter.env, meter.config
    if env.deployment_resource_frames != 0 or env.service_resource_frames != 20:
        raise AssertionError("This diagnostic requires the frozen continuous mission timing")
    gamma = cfg["training"]["gamma_per_resource_frame"]
    before, start, q0 = meter.cost_sum, env._current_time(), env.uav_coords.copy()
    policy = ObservedPolicy(sac, "predictive_centroid")
    policy.meter = meter
    pred, _ = policy.macro_action(meter.macro_observation(), deterministic=True)
    pred_request = meter.requested_displacement.copy()
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    raw = pred + delta
    clipped = np.clip(raw.reshape(3, 2), -1, 1)
    projected = clipped / np.maximum(np.linalg.norm(clipped, axis=1, keepdims=True), 1)
    # Explicitly distinguish the uncapped Predictive request from the intervened raw request.
    meter.requested_displacement = 20 * raw.reshape(3, 2)
    info = meter.start_macro(raw)
    commands = meter.commands[-3:]
    control = dict(a_pred=pred.copy(), delta=delta.copy(), a_raw=raw.copy(),
                   a_projected=projected.ravel(), predictive_uncapped_request_m=pred_request,
                   requested_displacement_m=20*raw.reshape(3, 2),
                   executed_displacement_m=info["actual_displacement_m"].copy(),
                   start_position_m=q0, target_position_m=info["projected_uav_coords"].copy(),
                   component_clip=np.any(raw.reshape(3, 2) != clipped, axis=1),
                   unit_disk_projection=np.linalg.norm(clipped, axis=1)>1,
                   saturation=np.linalg.norm(projected, axis=1)>=1-1e-9,
                   cap_request=np.linalg.norm(raw.reshape(3, 2), axis=1)>1+1e-9,
                   map_projection=np.array([r["map_clipped"] for r in commands], dtype=bool),
                   safety_projection=np.array([r["safety_changed"] for r in commands], dtype=bool))
    state = meter.prepare_service()
    frames = []
    for i in range(env.service_resource_frames):
        action = sac.sac_actor.choose_action(state, deterministic=True)
        frames.append(meter.step_frame(action, gamma))
        if i < env.service_resource_frames - 1:
            state = meter.prepare_service()
    rewards = [frame["reward"] for frame in frames]
    macro_reward = sum(gamma ** i * r for i, r in enumerate(rewards)) / len(rewards)
    physical_rows = []
    rc = cfg["reward"]
    for frame in frames:
        costs = []
        for entry in frame["infos"]:
            h = float(entry["task_holding_cost_weighted"])
            e = float(entry["total_energy"]) + meter.propulsion_delta_share
            cost = holding_energy_cost(h, e, rc)
            costs.append(cost)
            physical_rows.append(dict(time_s=start+(len(physical_rows)+1)*env.micro_dt,
                cost=cost, holding=rc["delay_weight"]*h/rc["holding_norm_task_s"],
                energy=rc["energy_weight"]*e/rc["energy_norm_j"],
                full_energy_j=float(entry["total_energy"])+meter.propulsion_share,
                energy_compute_j=float(entry["energy_comp"]), energy_upload_j=float(entry["energy_uplink"]),
                energy_u2u_j=float(entry["energy_trans"]), energy_propulsion_j=meter.propulsion_share,
                propulsion_increment_j=meter.propulsion_delta_share))
        np.testing.assert_allclose(sum(costs), frame["cost"], rtol=1e-12, atol=1e-10)
    if len(physical_rows) != 100:
        raise AssertionError("Mission macro must contain100 physical steps")
    meter.advance_macro_clock()
    np.testing.assert_allclose(sum(r["cost"] for r in physical_rows), meter.cost_sum-before, rtol=1e-12, atol=1e-9)
    return dict(start_s=start, end_s=env._current_time(), macro_reward=macro_reward,
                cost=sum(r["cost"] for r in physical_rows), physical_rows=physical_rows,
                control=control, metrics=meter.metrics())


def state_features(meter):
    state = meter.macro_observation().copy()
    env = meter.env
    counts, oldest, upload, _ = meter.task_aggregates()
    loads = env.zone_expected_offered_load_mbps
    roles = np.empty(3, dtype=object)
    roles[np.argsort(-loads, kind="stable")] = ["heavy", "medium", "light"]
    parts = meter._observation_parts()
    return dict(state=state.tolist(), feature_slices=copy.deepcopy(meter.macro_feature_slices),
                uav=env.uav_coords.tolist(), centroid=env.weighted_centroids.tolist(),
                centroid_offset=(env.weighted_centroids-env.uav_coords).tolist(),
                centroid_velocity=meter.centroid_velocity.tolist(),
                compute_queue=(env.Q_loc_high+env.Q_loc_low).tolist(),
                transfer_queue=(env.Q_tx_high+env.Q_tx_low).tolist(),
                upload_queue=upload.sum(axis=0).tolist(), unfinished=counts.tolist(), oldest_age=oldest.tolist(),
                current_load=loads.tolist(), roles=roles.tolist(), phase_age=meter.phase_age,
                u2g_mean_rate_mbps=np.expm1(parts["u2g_mean_rate"]).tolist(),
                u2u_reference_quality=np.expm1(parts["u2u_spectral_efficiency"]).tolist())


def branch_metrics(meter, macros, initial):
    if len(macros) != HORIZON or len(meter.readings) != 120 or len(meter.micro_states) != 120:
        raise AssertionError("Branch service/decision count is not6x20")
    values = objective_values(macros, meter.config["training"]["gamma_per_resource_frame"])
    values.update(summarize_meter(meter.readings, 60))
    end = meter.metrics()
    for key in ("queue_total_mbit", "queue_upload_mbit", "queue_compute_mbit", "queue_transfer_mbit", "unfinished"):
        values[key] = end[key]
    delay_sums, completed = {}, {}
    for p in ("high", "low"):
        arrived = meter.env.arrived_priority_count[p]-initial["arrived"][p]
        completed[p] = meter.env.completed_priority_count[p]-initial["completed"][p]
        delay_sums[p] = meter.env.completed_priority_delay_sum[p]-initial["delays"][p]
        available = initial["unfinished"][p]+arrived
        alive = meter.readings[-1][f"task_unfinished_{p}_count"]
        if available-completed[p] != alive:
            raise AssertionError("Branch parent count conservation failed")
        values.update({f"initial_unfinished_{p}": initial["unfinished"][p],
                       f"arrived_{p}": arrived, f"completed_{p}": completed[p], f"unfinished_{p}": alive,
                       f"completion_available_{p}": completed[p]/max(available, 1),
                       f"task_delay_{p}_ms": 1000*delay_sums[p]/max(completed[p], 1)})
    values["task_delay_weighted_ms"] = 1000*(10*delay_sums["high"]+delay_sums["low"])/max(10*completed["high"]+completed["low"], 1)
    values.update(J=values["C60"]/600, J_holding=values["C60_holding"]/600, J_energy=values["C60_energy"]/600,
                  power_w=values["full_energy_j"]/60,
                  propulsion_increment_j=sum(r["propulsion_increment_j"] for r in meter.readings),
                  u2g_served_mbit=sum(r["served_mbit"] for r in meter.uplink_readings),
                  u2g_tx_time_s=sum(r["tx_time_s"] for r in meter.uplink_readings),
                  u2u_tx_time_s=sum(r["tx_time_s"] for r in meter.link_readings),
                  sac_decisions=120, macro_decisions=6, sac_update_increment=0, ppo_updates=0,
                  replay_writes=0, replay_samples=0)
    for link in ("u2g", "u2u"):
        work = values["u2g_served_mbit"] if link == "u2g" else values["tx_served_mbit"]
        values[f"{link}_rate_time_weighted_mbps"] = work/max(values[f"{link}_tx_time_s"], 1e-12)
    np.testing.assert_allclose(sum(values[k] for k in ENERGY), values["full_energy_j"], rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(values["C60_holding"], sum(r["holding_cost"] for r in meter.readings), atol=1e-9)
    np.testing.assert_allclose(values["C60_energy"], sum(r["energy_cost"] for r in meter.readings), atol=1e-9)
    return values


def initial_counts(meter):
    counts = meter.task_aggregates()[0]
    return dict(arrived=dict(meter.env.arrived_priority_count), completed=dict(meter.env.completed_priority_count),
                delays=dict(meter.env.completed_priority_delay_sum),
                unfinished={p: int(counts[i].sum()) for i, p in enumerate(("high", "low"))})


def save_branch(folder, meter, macros, result, difference):
    folder.mkdir()
    write_json(folder/"summary.json", result)
    append_csv(folder/"physical_steps.csv", [r for m in macros for r in m["physical_rows"]])
    append_csv(folder/"macros.csv", [{k: m[k] for k in ("start_s", "end_s", "cost", "macro_reward")} for m in macros])
    for name, rows in (("queues", meter.readings), ("links", meter.link_readings), ("uplink", meter.uplink_readings),
                       ("routes", meter.route_readings), ("uavs", meter.uav_readings), ("commands", meter.commands),
                       ("crn", crn_records(meter))):
        append_csv(folder/f"{name}.csv", rows)
    controls = {k: np.asarray([m["control"][k] for m in macros]) for k in macros[0]["control"]}
    controls["first_executed_difference_from_zero_m"] = difference
    np.savez_compressed(folder/"controls.npz", **controls)
    np.savez_compressed(folder/"trajectory.npz", **{name: np.asarray([r[i] for r in meter.positions])
                        for i, name in enumerate(("time_s", "uav", "users", "centroid", "groups"))},
                        micro_state=np.asarray(meter.micro_states))
    append_csv(folder/"role_summary.csv", role_summary(meter.uav_readings, .5, result["action"], result["seed"]))
    append_csv(folder/"route_summary.csv", route_statistics(meter.route_readings, result["action"], result["seed"], 2000, "branch"))


def source_inventory():
    files = tuple(dict.fromkeys((*RUN_FILES, *DEPENDENCIES, *NEW_FILES,
             "docs/mission_micro_pilot.md", "scripts/run_mission_micro_pilot.py", "tests/test_mission_micro_pilot.py")))
    return {name: file_hash(ROOT/name) for name in files}


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite or silently repeat an existing Stage3A run")
    spec = load_spec()
    saved, _ = load_source(spec)
    cfg = mission_config(saved["config"], spec)
    physics = physics_provenance()
    assert_committed_sources(RUN_FILES)
    _, _, references, prior_hashes = verify_previous(cfg, spec)
    hashes = source_inventory()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)
    output.mkdir(parents=True)
    manifest = dict(status="PREDECLARED", actual_head=head, worktree_before=status,
                    physics_base_commit=PHYSICS_BASE_COMMIT, physics_provenance=physics,
                    config=cfg, protocol=protocol(), training=False, deterministic=True, zero_shot=True,
                    source_checkpoint=spec["source_checkpoint"], source_checkpoint_sha256=spec["checkpoint_sha256"],
                    source_training_seed=101, source_window=2000,
                    source_manifest_sha256=file_hash((ROOT/spec["source_checkpoint"]).parent.parent/"manifest.json"),
                    baseline_sources=prior_hashes, source_hashes_before=hashes,
                    completed_states=0, completed_branches=0, reserved_scenes_executed=[],
                    ppo_td3_instantiated=False, sac_updates=0, replay_writes=0, replay_samples=0)
    write_json(output/"manifest.json", manifest)
    write_json(output/"protocol.json", protocol())
    for name in hashes:
        target = output/"source_evidence"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    fields = physical_fields(cfg)
    states, rows, checks = [], [], []
    started = time.perf_counter()
    with isolated_rng():
        sac = ReadOnlySAC(saved)
    del saved
    with isolated_rng(), no_learning(sac), torch.no_grad():
        for seed in SCENES:
            replay = MissionPilotMeter(cfg)
            replay.reset(seed, 90)
            snapshots, features, replay_macros, end_hashes = {}, {}, [], {}
            for i in range(86):
                t = i*10
                if t in TIMES:
                    # Features are read before the snapshot so read-only caches are identical on restore.
                    features[t] = state_features(replay)
                    snapshots[t] = MissionSnapshot(replay, fields)
                replay_macros.append(macro_step(replay, sac, np.zeros(6)))
                if (i+1)*10 in [t+60 for t in TIMES]:
                    end_hashes[(i+1)*10] = semantic_hash(physical_state(replay, fields))
            assert_crn_equal(references[seed]["crn"][:1720], crn_records(replay))
            with np.load(references[seed]["directory"]/"trajectory.npz") as trace:
                for column, name in enumerate(("time_s", "uav", "users", "centroid", "groups")):
                    np.testing.assert_array_equal(np.asarray([r[column] for r in replay.positions]), trace[name][:8601])
            old_frames = read_csv(references[seed]["directory"]/"queues.csv")[:1720]
            for old, new in zip(old_frames, replay.readings):
                for key in new:
                    np.testing.assert_allclose(new[key], float(old[key]), rtol=1e-12, atol=1e-8, err_msg=key)
            prefix = output/"predictive_replay"/str(seed)
            prefix.mkdir(parents=True)
            append_csv(prefix/"crn.csv", crn_records(replay))
            append_csv(prefix/"macros.csv", [{k: m[k] for k in ("start_s", "end_s", "cost", "macro_reward")} for m in replay_macros])
            replay_hash = semantic_hash(physical_state(replay, fields))
            print(f"scene={seed}: Predictive replay0..860s matches archived trajectory/CRN/ledgers", flush=True)
            for t in TIMES:
                root = output/"states"/f"{seed}_t{t}"
                root.mkdir(parents=True)
                snap = snapshots[t]
                snap.save(root/"physical_snapshot.pkl.gz")
                feature = dict(seed=seed, start_s=t, initial_state_sha256=snap.initial_hash, **features[t])
                write_json(root/"state.json", feature)
                states.append(feature)
                zero_values = zero_crn = zero_controls = zero_positions = zero_rng = None
                for action in actions():
                    validate_branch(seed, t, action)
                    meter = snap.restore()
                    initial = initial_counts(meter)
                    macros = [macro_step(meter, sac, action["delta"] if k == 0 else np.zeros(6)) for k in range(HORIZON)]
                    if (meter.env._current_time() != t+60 or meter.integration_steps != (t+60)*10
                            or meter.reset_count != 0 or meter.mobility_states != 120 or len(meter.positions) != 600):
                        raise AssertionError("Clone clock/physical-step/mobility/position boundary mismatch")
                    values = branch_metrics(meter, macros, initial)
                    crn = crn_records(meter)
                    positions = np.asarray([r[1] for r in meter.positions])
                    control = macros[0]["control"]
                    if action["name"] == "zero":
                        expected = objective_values(replay_macros[t//10:t//10+6], cfg["training"]["gamma_per_resource_frame"])
                        for key in expected:
                            np.testing.assert_allclose(values[key], expected[key], rtol=1e-12, atol=1e-8, err_msg=key)
                        assert_crn_equal(crn_records(replay)[t*2:(t+60)*2], crn)
                        if semantic_hash(physical_state(meter, fields)) != end_hashes[t+60]:
                            raise AssertionError("Zero branch does not reproduce full uninterrupted Predictive terminal state")
                        zero_values, zero_crn = values.copy(), crn
                        zero_controls, zero_positions = copy.deepcopy(control), positions.copy()
                        zero_rng = copy.deepcopy(meter.env.np_random.get_state())
                    else:
                        assert_crn_equal(zero_crn, crn)
                    if not state_equal(meter.env.np_random.get_state(), zero_rng):
                        raise AssertionError("Branch environment RNG mismatch")
                    if not state_equal(capture_rng(), pickle.loads(snap.blob)["rng"]):
                        raise AssertionError("Deterministic branch consumed global RNG")
                    if semantic_hash(physical_state(replay, fields)) != replay_hash:
                        raise AssertionError("Branch mutated the original Predictive replay")
                    difference = control["executed_displacement_m"]-zero_controls["executed_displacement_m"]
                    first_error = float(np.max(np.linalg.norm(positions[:100]-zero_positions[:100], axis=2)))
                    all_error = float(np.max(np.linalg.norm(positions-zero_positions, axis=2)))
                    ineffective = action["name"] != "zero" and all_error <= POSITION_ATOL
                    result = dict(seed=seed, start_s=t, action=action["name"], dimension=action["dimension"], sign=action["sign"],
                                  initial_state_sha256=snap.initial_hash, **values,
                                  executed_difference_m=float(np.linalg.norm(difference)),
                                  first_macro_trajectory_difference_m=first_error, full_trajectory_difference_m=all_error,
                                  ineffective=ineffective)
                    for flag in ("component_clip", "unit_disk_projection", "saturation", "cap_request", "map_projection", "safety_projection"):
                        result[f"{flag}_uavs"] = int(control[flag].sum())
                    for key in values:
                        if isinstance(values[key], (int, float, np.number)):
                            result[f"delta_{key}"] = values[key]-zero_values[key]
                    result["relative_C60_percent"] = 100*result["delta_C60"]/zero_values["C60"]
                    if not all(np.isfinite(v) for v in result.values() if isinstance(v, (float, int, np.number))):
                        raise FloatingPointError("Nonfinite branch metric")
                    if ineffective:
                        np.testing.assert_allclose(values["C60"], zero_values["C60"], atol=C_ATOL, rtol=0)
                    save_branch(root/action["name"], meter, macros, result, difference)
                    append_csv(output/"per_branch.csv", [result])
                    rows.append(result)
                    checks.append(dict(seed=seed, start_s=t, action=action["name"], initial_state_equal=True,
                                       crn_equal=True, environment_rng_equal=True, global_rng_unchanged=True,
                                       original_unchanged=True, zero_terminal_equal=action["name"] == "zero"))
                sac.assert_unchanged()
                manifest.update(status="RUNNING", completed_states=len(states), completed_branches=len(rows),
                                elapsed_seconds=time.perf_counter()-started)
                write_json(output/"manifest.json", manifest)
                print(f"Completed {len(states)}/16 states, {len(rows)}/208 branches: scene={seed} t={t}", flush=True)
            del snapshots, replay
    if len(rows) != 208 or len(states) != 16:
        raise AssertionError("Stage3A fixed matrix incomplete")
    if source_inventory() != hashes or any(file_hash(Path(p)) != h for p, h in prior_hashes.items()):
        raise AssertionError("Source or historical baseline changed")
    if file_hash(ROOT/spec["source_checkpoint"]) != spec["checkpoint_sha256"]:
        raise AssertionError("SAC source changed")
    write_json(output/"states.json", states)
    write_json(output/"checks.json", checks)
    manifest.update(status="COMPLETE_208_BRANCHES_STOPPED", elapsed_seconds=time.perf_counter()-started,
                    completed_states=16, completed_branches=208, branch_macros=1248, branch_physical_steps=124800,
                    prefix_replay_macros=172, sac_frozen=sac.assert_unchanged(), crn_all_equal=True,
                    baseline_reproduced=True, source_hashes_after=source_inventory(), source_unchanged=True,
                    worktree_after=subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True))
    write_json(output/"manifest.json", manifest)
    from .mission_opportunity_analysis import analyze
    analyze(output)
    return str(output)
