"""Fixed B-centered local interventions; no learning, new physics or policy search."""

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .bandwidth import summarize_meter
from .experiment import append_csv, isolated_rng, write_json
from .mdp import QueueMDP, holding_energy_cost
from .ppo_residual_b import ResidualMeter, baseline_b, project_action, prepare_residual_source
from .ppo_frozen import DEFAULT_SOURCE, SOURCE_DIGESTS
from .resumption import capture_rng, restore_rng, state_equal
from .scene_prescreen import ROOT, assert_crn_equal, source_hashes
from .trajectory_check import ReadOnlySAC, no_learning, crn_records
from .priority_grid import file_hash
from .wireless_check import ENERGY


RUN_HEAD = "d7686b0c9fd980bf76e04b87385ee1d19b19dba5"
PILOT = ROOT / "results/ppo_residual_b_pilot120_seed101"
SCENES = (33600, 33601, 33610, 33611, 33612)
TIMES = (100, 200, 300, 400, 500, 600, 700, 800)
EPSILON, HORIZON = .05, 6
C_ATOL, R_ATOL = 1e-8, 1e-10
HOOKS = {"_enqueue_compute_chunk", "_register_arrived_parent_task", "_integrate_user_positions"}
DIGESTS = ("arrivals_digest", "motion_digest", "users_digest", "centroid_digest")


def actions():
    result = [dict(name="zero", dimension=0, sign=0, delta=[0.] * 6)]
    for j in range(6):
        for sign, label in ((1, "plus"), (-1, "minus")):
            delta = np.zeros(6)
            delta[j] = sign * EPSILON
            result.append(dict(name=f"e{j+1}_{label}", dimension=j+1, sign=sign, delta=delta.tolist()))
    return result


def validate_branch(seed, start, action, horizon=HORIZON):
    if type(seed) is not int or seed not in SCENES or start not in TIMES or horizon != HORIZON:
        raise ValueError("Only the fixed forty development states and six-macro horizon are allowed")
    if action not in actions():
        raise ValueError("Action is not one of the thirteen predeclared residuals")


def physical_fields(config):
    # Native QueueMDP fields, including the two created by reset; no reset or step.
    with isolated_rng():
        return tuple(sorted(set(vars(QueueMDP(config))) - {"env"} | {"schedule", "previous_centroid"}))


def physical_state(meter, fields):
    overrides = {k for k, v in vars(meter.env).items() if callable(v)}
    if overrides != HOOKS:
        raise AssertionError(f"Unexpected instance hooks; cannot safely snapshot: {overrides}")
    return dict(mdp={k: getattr(meter, k) for k in fields},
                env={k: v for k, v in sorted(vars(meter.env).items()) if k not in HOOKS})


def semantic_hash(value):
    """Value fingerprint independent of object IDs; pickle separately preserves aliases."""
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            visit(x.detach().cpu().numpy())
        elif isinstance(x, np.ndarray):
            h.update(b"array" + x.dtype.str.encode() + str(x.shape).encode() + x.tobytes())
        elif isinstance(x, np.generic):
            visit(x.item())
        elif isinstance(x, np.random.RandomState):
            visit(x.get_state())
        elif isinstance(x, np.random.Generator):
            visit(x.bit_generator.state)
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=repr):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)) or type(x).__name__ == "deque":
            h.update(type(x).__name__.encode() + str(len(x)).encode())
            for item in x:
                visit(item)
        elif isinstance(x, (set, frozenset)):
            visit(sorted(x, key=repr))
        elif x is None or isinstance(x, (str, bytes, bool, int, float)):
            h.update(type(x).__name__.encode() + repr(x).encode())
        elif hasattr(x, "__dict__"):
            h.update((type(x).__module__ + "." + type(x).__qualname__).encode())
            visit(vars(x))
        else:
            raise TypeError(f"Unrecognized snapshot value: {type(x)}")
    visit(value)
    return h.hexdigest()


class FullSnapshot:
    def __init__(self, meter, fields):
        self.fields = fields
        state = dict(physical=physical_state(meter, fields), rng=capture_rng())
        self.blob = pickle.dumps(state, protocol=5)
        self.initial_hash = semantic_hash(state)
        self.digests = {k: getattr(meter, k).copy() for k in DIGESTS}
        self.metadata = {k: copy.deepcopy(getattr(meter, k)) for k in (
            "scene_seed", "requested_macros", "schedule_hash", "initial_positions")}

    def restore(self):
        state = pickle.loads(self.blob)
        physical = state["physical"]
        # Fresh callbacks capture the NEW meter and NEW environment, never the source.
        meter = ResidualMeter(physical["mdp"]["config"])
        meter.env.__dict__.update(physical["env"])
        meter.__dict__.update(physical["mdp"])
        meter.__dict__.update(copy.deepcopy(self.metadata))
        for k, digest in self.digests.items():
            setattr(meter, k, digest.copy())
        restore_rng(state["rng"])
        restored = dict(physical=physical_state(meter, self.fields), rng=capture_rng())
        if semantic_hash(restored) != self.initial_hash:
            raise AssertionError("Full physical/RNG state did not restore exactly")
        return meter

    def save(self, path):
        # The full physical state/RNG is portable; cumulative audit hashes are not physics.
        Path(path).write_bytes(gzip.compress(self.blob, compresslevel=3))
        if semantic_hash(pickle.loads(gzip.decompress(Path(path).read_bytes()))) != self.initial_hash:
            raise AssertionError("Snapshot disk roundtrip failed")


def macro_step(meter, sac, delta):
    """Exact nonlearning service sequence from experiment.rollout, no terminal tail."""
    cfg, env = meter.config, meter.env
    gamma = cfg["training"]["gamma_per_resource_frame"]
    before = meter.cost_sum
    start = env._current_time()
    meter.start_macro(delta)
    deployment = [meter.step_frame(np.zeros(meter.micro_action_dim), gamma)
                  for _ in range(env.deployment_resource_frames)]
    frames = list(deployment)
    state = meter.prepare_service()
    for i in range(env.service_resource_frames):
        raw = sac.sac_actor.choose_action(state, deterministic=True)
        frames.append(meter.step_frame(raw, gamma))
        if i < env.service_resource_frames - 1:
            state = meter.prepare_service()
    if len(frames) != 20 or sum(len(f["infos"]) for f in frames) != 100:
        raise AssertionError("The frozen protocol requires 20 frames and 100 steps per macro")
    rewards = [f["reward"] for f in frames]
    reward = sum(gamma ** i * r for i, r in enumerate(rewards)) / len(rewards)
    physical_rows = []
    rc = cfg["reward"]
    for frame in frames:
        reconstructed = []
        for info in frame["infos"]:
            holding = rc["delay_weight"] * float(info["task_holding_cost_weighted"]) / rc["holding_norm_task_s"]
            objective_energy = float(info["total_energy"]) + meter.propulsion_delta_share
            energy = rc["energy_weight"] * objective_energy / rc["energy_norm_j"]
            cost = holding_energy_cost(info["task_holding_cost_weighted"], objective_energy, rc)
            reconstructed.append(cost)
            physical_rows.append(dict(time_s=start+(len(physical_rows)+1)*env.micro_dt,
                cost=cost, holding=holding, energy=energy,
                full_energy_j=float(info["total_energy"])+meter.propulsion_share,
                energy_compute_j=float(info["energy_comp"]), energy_upload_j=float(info["energy_uplink"]),
                energy_u2u_j=float(info["energy_trans"]), energy_propulsion_j=meter.propulsion_share))
        np.testing.assert_allclose(sum(reconstructed), frame["cost"], rtol=1e-12, atol=1e-10)
    meter.advance_macro_clock()
    metrics = meter.metrics()
    np.testing.assert_allclose(sum(x["cost"] for x in physical_rows), meter.cost_sum-before, atol=1e-9, rtol=1e-12)
    return dict(start_s=start, end_s=env._current_time(), macro_reward=reward,
                cost=sum(x["cost"] for x in physical_rows), physical_rows=physical_rows,
                trajectory=dict(uav=env.uav_coords.copy(), users=env.user_coords.copy(),
                                centroid=env.weighted_centroids.copy()), metrics=metrics)


def objective_values(macros, gamma):
    steps = [p for m in macros for p in m["physical_rows"]]
    c = sum(p["cost"] for p in steps)
    holding, energy = (sum(p[k] for p in steps) for k in ("holding", "energy"))
    macro_gamma = gamma ** 20
    training_return = sum(macro_gamma ** i * m["macro_reward"] for i, m in enumerate(macros))
    direct = -sum((gamma ** .2) ** i * p["cost"] for i, p in enumerate(steps)) / 20
    np.testing.assert_allclose(c, holding+energy, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(training_return, direct, rtol=1e-12, atol=1e-10)
    return dict(C60=c, C60_holding=holding, C60_energy=energy,
                ppo_return60=training_return, macro_gamma=macro_gamma,
                full_energy_j=sum(p["full_energy_j"] for p in steps),
                **{k: sum(p[k] for p in steps) for k in ENERGY})


def state_features(meter):
    state = meter.macro_observation().copy()
    counts, oldest, upload, _ = meter.task_aggregates()
    env = meter.env
    return dict(state=state.tolist(), uav=env.uav_coords.tolist(), centroid=env.weighted_centroids.tolist(),
        centroid_offset=(env.weighted_centroids-env.uav_coords).tolist(),
        compute_queue=(env.Q_loc_high+env.Q_loc_low).tolist(),
        transfer_queue=(env.Q_tx_high+env.Q_tx_low).tolist(), upload_queue=upload.sum(axis=0).tolist(),
        current_load=env.zone_expected_offered_load_mbps.tolist(), phase_age=meter.phase_age,
        unfinished=counts.tolist(), oldest_age=oldest.tolist())


def branch_metrics(meter, macros, initial_completed, initial_delays):
    result = objective_values(macros, meter.config["training"]["gamma_per_resource_frame"])
    result.update(summarize_meter(meter.readings, 60))
    terminal = meter.metrics()
    for key in ("queue_total_mbit", "queue_upload_mbit", "queue_compute_mbit", "queue_transfer_mbit", "unfinished"):
        result[key] = terminal[key]
    counts, delays = {}, {}
    for priority in ("high", "low"):
        counts[priority] = meter.env.completed_priority_count[priority]-initial_completed[priority]
        delays[priority] = meter.env.completed_priority_delay_sum[priority]-initial_delays[priority]
        result[f"completed_{priority}_count"] = counts[priority]
        result[f"completed_{priority}_delay_ms"] = 1000*delays[priority]/max(counts[priority], 1)
        result[f"unfinished_{priority}_count"] = int(terminal[f"task_alive_{priority}_count"])
    result["completed_weighted_delay_ms"] = 1000*(10*delays["high"]+delays["low"])/max(10*counts["high"]+counts["low"],1)
    result.update(power_w=result["full_energy_j"]/60, start_s=macros[0]["start_s"], end_s=macros[-1]["end_s"],
        projection_macro_count=sum(bool(c["projected"].any()) for c in meter.controls),
        projection_uav_count=sum(int(c["projected"].sum()) for c in meter.controls),
        B_projection_count=sum(int(c["B_projection"].sum()) for c in meter.controls),
        physical_steps=len(macros)*100, micro_routing_decisions=len(meter.route_readings)//6)
    if result["physical_steps"] != 600 or result["micro_routing_decisions"] != 108:
        raise AssertionError("Branch horizon or SAC routing count changed")
    return result


def save_branch(directory, meter, macros, result):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"metrics.json", result)
    append_csv(directory/"physical_steps.csv", [p for m in macros for p in m["physical_rows"]])
    append_csv(directory/"macros.csv", [{k:m[k] for k in ("start_s","end_s","macro_reward","cost")} for m in macros])
    for name, rows in (("frames",meter.readings),("routes",meter.route_readings),("links",meter.link_readings),
                       ("uavs",meter.uav_readings),("deployments",meter.deployments),("crn",crn_records(meter))):
        append_csv(directory/f"{name}.csv", rows)
    fields = ("state", "current_uav", "current_centroid", "current_load", "a_B", "sampled_delta",
              "pre_projection", "a_exec", "projected", "actual_displacement")
    np.savez_compressed(directory/"controls.npz", **{k:np.asarray([x[k] for x in meter.controls]) for k in fields})
    np.savez_compressed(directory/"trajectory.npz", **{k:np.asarray([m["trajectory"][k] for m in macros])
                                                       for k in ("uav","users","centroid")})


def branch_record(seed, action, initial_hash, values, zero_values):
    return dict(seed=seed,action=action["name"],dimension=action["dimension"],sign=action["sign"],
        initial_state_sha256=initial_hash,**values,
        **{"delta_"+k:values[k]-zero_values[k] for k in ("C60","C60_holding","C60_energy","ppo_return60")})


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Will not overwrite or silently repeat an existing diagnosis")
    previous = json.loads((PILOT/"manifest.json").read_text(encoding="utf-8"))
    if previous["actual_head"] != RUN_HEAD or previous["status"] != "RESIDUAL_PPO120_DEVELOPMENT_COMPLETE":
        raise ValueError("Wrong residual pilot source")
    for path, digest in previous["source_hashes"].items():
        if file_hash(ROOT/path) != digest:
            raise ValueError(f"Source differs from the approved pilot: {path}")
    cfg, saved, provenance = prepare_residual_source()
    if cfg != previous["config"]:
        raise ValueError("Effective configuration differs from the residual pilot")
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    diff = subprocess.check_output(["git","diff",RUN_HEAD,"--","hrl_uav","configs","scripts","tests"],cwd=ROOT,text=True)
    if diff:
        raise ValueError("Tracked implementation differs; report before mixing versions")
    output.mkdir(parents=True)
    hashes = source_hashes()
    protocol = dict(branch_times_s=list(TIMES), scenes=list(SCENES), actions=actions(), epsilon=EPSILON,
        branch_macros=6, branch_count=520, branch_state_count=40, control="B=0.5; residual only in first macro",
        state_timing="after previous macro clock advance, before this macro deployment",
        replay="one B prefix/continuation per scene to 860s, reset horizon90 matching historical B",
        C60="sum of 600 original physical-step costs, holding plus original incremental energy objective",
        ppo_return60="sum_{k=0..5}(gamma^20)^k * [sum_{f=0..19}gamma^f*r_frame/20]",
        frame_reward="native step_frame: negative physical costs discounted with gamma_step=gamma^(1/5)",
        tail="finite 60-second discounted reward prefix only, no V bootstrap and no extra deployment",
        numerical_tolerances=dict(C60_absolute=C_ATOL,return_absolute=R_ATOL),
        selection="all 13 retained; report ties, not global optimality; no extra states/epsilon/horizon")
    write_json(output/"protocol.json",protocol)
    manifest = dict(status="PREFLIGHT", actual_head=head,
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        worktree=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        base_head=RUN_HEAD, config=cfg, source_hashes=hashes, provenance=provenance,
        source_checkpoint=str(DEFAULT_SOURCE), source_checkpoint_sha256=SOURCE_DIGESTS["checkpoint"],
        prior_manifest_sha256=file_hash(PILOT/"manifest.json"), protocol_sha256=file_hash(output/"protocol.json"),
        training=False, ppo_network_instantiated=False, replay_instantiated=False,
        learning_updates=0, reserved_scenes_executed=[], completed_states=0)
    write_json(output/"manifest.json",manifest)
    torch.set_num_threads(1)
    sac = ReadOnlySAC(saved)
    del saved
    fields = physical_fields(cfg)
    all_rows, states, checks = [], [], []
    started = time.perf_counter()
    with isolated_rng(), no_learning(sac):
        for seed in SCENES:
            replay = ResidualMeter(cfg)
            replay.reset(seed,90)
            snapshots, features, replay_macros, end_hashes = {}, {}, [], {}
            for i in range(86):
                t = i*10
                if t in TIMES:
                    features[t] = state_features(replay)
                    snapshots[t] = FullSnapshot(replay,fields)
                replay_macros.append(macro_step(replay,sac,np.zeros(6)))
                if (i+1)*10 in [t+60 for t in TIMES]:
                    end_hashes[(i+1)*10] = semantic_hash(physical_state(replay,fields))
            replay_digest = semantic_hash(physical_state(replay,fields))
            prefix_dir = output/"B_replay"/str(seed)
            prefix_dir.mkdir(parents=True)
            append_csv(prefix_dir/"crn.csv",crn_records(replay))
            append_csv(prefix_dir/"macros.csv",[{k:m[k] for k in ("start_s","end_s","macro_reward","cost")} for m in replay_macros])
            np.savez_compressed(prefix_dir/"trajectory.npz",**{k:np.asarray([m["trajectory"][k] for m in replay_macros])
                                                               for k in ("uav","users","centroid")})
            for t in TIMES:
                root = output/"states"/f"{seed}_t{t}"
                root.mkdir(parents=True)
                snap = snapshots[t]
                snap.save(root/"physical_snapshot.pkl.gz")
                feature = dict(seed=seed,start_s=t,initial_state_sha256=snap.initial_hash,**features[t])
                write_json(root/"state.json",feature)
                states.append(feature)
                zero_crn, zero_values, zero_env_rng = None, None, None
                for action in actions():
                    validate_branch(seed,t,action)
                    meter = snap.restore()
                    initial_completed = dict(meter.env.completed_priority_count)
                    initial_delays = dict(meter.env.completed_priority_delay_sum)
                    macros = [macro_step(meter,sac,np.asarray(action["delta"]) if k==0 else np.zeros(6))
                              for k in range(HORIZON)]
                    if meter.env._current_time()!=t+60 or meter.integration_steps!=(t+60)*10 or meter.reset_calls!=0:
                        raise AssertionError("Clone reset/time/horizon boundary mismatch")
                    values = branch_metrics(meter,macros,initial_completed,initial_delays)
                    crn = crn_records(meter)
                    if action["name"]=="zero":
                        baseline_macros = replay_macros[t//10:t//10+6]
                        expected = objective_values(baseline_macros,cfg["training"]["gamma_per_resource_frame"])
                        for key in expected:
                            np.testing.assert_allclose(values[key],expected[key],rtol=1e-12,atol=1e-8,err_msg=key)
                        assert_crn_equal(crn_records(replay)[t*2:(t+60)*2],crn)
                        if semantic_hash(physical_state(meter,fields))!=end_hashes[t+60]:
                            raise AssertionError("Zero branch full terminal physical state differs from uninterrupted B")
                        zero_crn, zero_values = crn,values.copy()
                        zero_env_rng = copy.deepcopy(meter.env.np_random.get_state())
                    else:
                        assert_crn_equal(zero_crn,crn)
                    if not state_equal(meter.env.np_random.get_state(), zero_env_rng):
                        raise AssertionError("Branch environment RNG differs from zero continuation")
                    if semantic_hash(physical_state(replay,fields))!=replay_digest:
                        raise AssertionError("A cloned branch mutated the original B replay")
                    # Complete RNG includes Python/NumPy/Torch; environment-owned RNG is in physical state.
                    if not state_equal(capture_rng(),pickle.loads(snap.blob)["rng"]):
                        raise AssertionError("Deterministic branch changed global RNG")
                    result = branch_record(seed,action,snap.initial_hash,values,zero_values)
                    if not all(np.isfinite(v) for v in result.values() if isinstance(v,(int,float,np.number))):
                        raise FloatingPointError("Nonfinite branch result")
                    save_branch(root/action["name"],meter,macros,result)
                    all_rows.append(result)
                    append_csv(output/"per_branch.csv",[result])
                    checks.append(dict(seed=seed,start_s=t,action=action["name"],initial_state_equal=True,
                        CRN_equal=True,full_zero_terminal_equal=True if action["name"]=="zero" else None,
                        original_unchanged=True,environment_rng_equal=True,
                        global_rng_unchanged=True,initial_state_sha256=snap.initial_hash))
                sac.assert_unchanged()
                manifest.update(status="RUNNING",completed_states=len(states),completed_branches=len(all_rows),
                                elapsed_seconds=time.perf_counter()-started)
                write_json(output/"manifest.json",manifest)
                print(f"Completed {len(states)}/40 states, {len(all_rows)}/520 branches: scene={seed} t={t}",flush=True)
            del snapshots,replay
    if len(all_rows)!=520 or len(states)!=40:
        raise AssertionError("Fixed experiment matrix was not completed")
    frozen = sac.assert_unchanged()
    if any(file_hash(ROOT/p)!=v for p,v in hashes.items()) or file_hash(DEFAULT_SOURCE)!=SOURCE_DIGESTS["checkpoint"]:
        raise AssertionError("Source changed during experiment")
    write_json(output/"states.json",states)
    write_json(output/"checks.json",checks)
    manifest.update(status="COMPLETE",elapsed_seconds=time.perf_counter()-started,source_unchanged=True,
                    sac_frozen_checks=frozen,executed_branch_macros=3120,executed_branch_physical_steps=312000,
                    B_replay_macros=430,completed_states=40,completed_branches=520)
    write_json(output/"manifest.json",manifest)
    return output
