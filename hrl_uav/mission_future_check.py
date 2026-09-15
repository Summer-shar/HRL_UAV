"""Stage3B: validate already selected directions on five fresh conditional futures."""

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .experiment import append_csv, isolated_rng, write_json
from .mission_opportunity import (
    ROOT, SCENES, TIMES, HORIZON, C_ATOL, R_ATOL, POSITION_ATOL,
    MissionPilotMeter, actions, physical_fields, physical_state, semantic_hash,
    macro_step, branch_metrics, initial_counts, save_branch, source_inventory as stage3a_sources,
    DIGESTS, crn_records,
)
from .mission_provenance import physics_provenance, assert_committed_sources
from .mission_smoke import load_spec, load_source, mission_config
from .priority_grid import file_hash
from .resumption import capture_rng, restore_rng, state_equal
from .scene_prescreen import assert_crn_equal
from .trajectory_check import ReadOnlySAC, no_learning, forbidden
from .wireless_check import read_csv

STAGE3A_COMMIT = "47115d7c2f5cda972e52a514b70c2c7ca784a5ad"
SOURCE = ROOT / "results/mission_predictive_stage3a"
OUTPUT = ROOT / "results/mission_predictive_stage3b"
LABELS = ("zero", "stage3a_best", "fixed_best", "opposite")
FUTURES = (1, 2, 3, 4, 5)
NEW_FILES = ("hrl_uav/mission_future_check.py", "hrl_uav/mission_future_analysis.py",
             "scripts/run_mission_future_check.py", "tests/test_mission_future_check.py",
             "docs/mission_predictive_stage3b.md")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def future_seed(scene, start, future):
    if scene not in SCENES or start not in TIMES or future not in FUTURES:
        raise ValueError("Only16 fixed states x5 futures are allowed")
    return 610000000 + (scene-33600)*10000 + start*10 + future


def frozen_actions(selected):
    by_name = {a["name"]: a for a in actions()}
    if selected not in by_name or selected == "zero":
        raise ValueError("Stage3A direction must be one of the12 nonzero directions")
    best = by_name[selected]
    opposite = next(a for a in actions() if a["dimension"] == best["dimension"] and a["sign"] == -best["sign"])
    return [dict(label=label, **copy.deepcopy(action)) for label, action in zip(LABELS,
            (by_name["zero"], best, by_name["e5_plus"], opposite))]


def protocol(plan):
    return dict(stage="3B frozen-candidate conditional future validation", training=False,
                states=16, futures_per_state=5, branches=320, macros_per_branch=6,
                physical_steps_per_branch=600, total_branch_macros=1920,
                epsilon=.05, actions=list(LABELS), fixed_best="e5_plus",
                source=str(SOURCE), stage3a_code_commit=STAGE3A_COMMIT, plan=plan,
                future_seed_formula="610000000+(scene-33600)*10000+branch_time_s*10+future_index(1..5)",
                randomized="only future env.np_random innovations: task Poisson arrivals and user-local motion",
                conditioned="full saved physical state INCLUDING pre-generated load schedule and fixed mission routes",
                initial_rng_exception="restore full Stage3A state first, then reseed the existing env RandomState; no reset/step",
                other_rng="Python/global NumPy/Torch states restored unchanged; inference deterministic",
                audit_hashes="new hashes start empty at branch; compare suffix records, not Stage3A cumulative prefixes",
                no_selection="Stage3A-selected action frozen per state; no new search or validation-based action replacement",
                duplicates="four labels each run, even when actual selected/fixed/opposite vectors coincide",
                scores="native C60=sum600 costs; holding/energy decomposition; same native6-macro truncated PPO return",
                validation_data="5 new futures only; original Stage3A future retained solely as selection-data comparison",
                hierarchy="first mean5 futures within each state, then mean16 states; 80 pairs are not80 training runs",
                c_atol=C_ATOL, return_atol=R_ATOL, trajectory_atol_m=POSITION_ATOL,
                train=False, complete_tail=False, replay_writes=0, replay_samples=0,
                excluded_scenes=[33610,33611,33612,*range(33700,33710)],
                stop="320 branches only; no new actions/futures/scenes; no training, model edits or post-run push")


def verify_selection(cfg, source=SOURCE):
    source = Path(source)
    manifest = read_json(source/"manifest.json")
    if (manifest["status"] != "COMPLETE_208_BRANCHES_STOPPED" or manifest["config"] != cfg
            or manifest["completed_states"] != 16 or manifest["completed_branches"] != 208
            or not manifest["source_unchanged"] or not manifest["crn_all_equal"]
            or not all(manifest["sac_frozen"].values()) or manifest["reserved_scenes_executed"]):
        raise ValueError("Stage3A source is incomplete or incompatible")
    if manifest["source_hashes_before"] != manifest["source_hashes_after"]:
        raise ValueError("Stage3A source changed while running")
    if stage3a_sources() != manifest["source_hashes_after"]:
        raise ValueError("Related code changed since Stage3A; do not mix models")
    analysis = read_json(source/"analysis.json")
    if analysis["best_fixed_direction"]["action"] != "e5_plus":
        raise ValueError("Fixed direction source mismatch")
    selected = read_csv(source/"per_state.csv")
    if len(selected) != 16 or {(int(r["seed"]), int(r["start_s"])) for r in selected} != {(s,t) for s in SCENES for t in TIMES}:
        raise ValueError("Original16-state selection table mismatch")
    hashes = {str(source/n): file_hash(source/n) for n in
              ("manifest.json", "protocol.json", "per_state.csv", "analysis.json", "states.json", "fixed_directions.csv")}
    plan = []
    for row in selected:
        scene, start = int(row["seed"]), int(row["start_s"])
        folder = source/"states"/f"{scene}_t{start}"
        state = read_json(folder/"state.json")
        chosen = read_json(folder/row["best_action"]/"summary.json")
        if (state["seed"] != scene or state["start_s"] != start or chosen["ineffective"]
                or chosen["initial_state_sha256"] != state["initial_state_sha256"]
                or chosen["action"] != row["best_action"]
                or chosen["delta_C60"] != float(row["best_delta_C60"])):
            raise ValueError("Saved selection and exact branch evidence disagree")
        candidates = frozen_actions(row["best_action"])
        for path in (folder/"state.json", folder/"physical_snapshot.pkl.gz",
                     folder/row["best_action"]/"summary.json", folder/"zero"/"trajectory.npz"):
            hashes[str(path)] = file_hash(path)
        plan.append(dict(scene=scene, start_s=start, selected_direction=row["best_action"],
                         stage3a_delta_C60=chosen["delta_C60"], stage3a_delta_holding=chosen["delta_C60_holding"],
                         stage3a_delta_energy=chosen["delta_C60_energy"],
                         stage3a_relative_percent=chosen["relative_C60_percent"],
                         snapshot=str(folder/"physical_snapshot.pkl.gz"),
                         snapshot_sha256=file_hash(folder/"physical_snapshot.pkl.gz"),
                         initial_state_sha256=state["initial_state_sha256"], candidates=candidates,
                         futures=[dict(future=k, rng_seed=future_seed(scene,start,k)) for k in FUTURES]))
    seeds = [f["rng_seed"] for p in plan for f in p["futures"]]
    if len(set(seeds)) != 80:
        raise AssertionError("Future seeds are not unique")
    return manifest, plan, hashes


def without_rng(physical):
    # Shallow view only for hashing; the pickle graph itself retains all aliases.
    return dict(mdp=physical["mdp"], env={k:v for k,v in physical["env"].items() if k != "np_random"})


class SavedBranch:
    """Restore the original complete disk snapshot; vary future innovations only."""

    def __init__(self, path, expected_hash, fields):
        self.blob = gzip.decompress(Path(path).read_bytes())
        self.fields = fields
        state = pickle.loads(self.blob)
        if semantic_hash(state) != expected_hash:
            raise AssertionError("Saved complete snapshot hash mismatch")
        self.original_hash = expected_hash
        self.physical_hash_without_rng = semantic_hash(without_rng(state["physical"]))
        self.original_environment_rng = state["physical"]["env"]["np_random"].get_state()

    def restore(self, rng_seed):
        state = pickle.loads(self.blob)
        physical = state["physical"]
        meter = MissionPilotMeter(physical["mdp"]["config"])
        meter.env.__dict__.update(physical["env"])
        meter.__dict__.update(physical["mdp"])
        meter.schedule_hash = hashlib.sha256(np.asarray(meter.schedule.phase_index_by_macro, dtype="<i8").tobytes()).hexdigest()
        # Independent suffix hashes are intentional; no prefix replay is needed.
        for key in DIGESTS:
            setattr(meter, key, hashlib.sha256())
        restore_rng(state["rng"])
        if semantic_hash(dict(physical=physical_state(meter,self.fields), rng=capture_rng())) != self.original_hash:
            raise AssertionError("Original Stage3A state did not restore exactly")
        meter.env.np_random.seed(rng_seed)
        if state_equal(meter.env.np_random.get_state(), self.original_environment_rng):
            raise AssertionError("Future reused the original environment RNG stream")
        if semantic_hash(without_rng(physical_state(meter,self.fields))) != self.physical_hash_without_rng:
            raise AssertionError("Changing the innovation seed altered physical/history/load state")
        if not state_equal(capture_rng(), state["rng"]):
            raise AssertionError("Future seeding changed global RNG")
        return meter


def source_inventory():
    return dict(stage3a_sources(), **{p:file_hash(ROOT/p) for p in NEW_FILES})


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Existing Stage3B output; refuse overwrite or implicit rerun")
    spec = load_spec()
    saved, _ = load_source(spec)
    cfg = mission_config(saved["config"])
    physics = physics_provenance()
    assert_committed_sources(tuple(stage3a_sources()))
    selection_manifest, plan, original_hashes = verify_selection(cfg)
    hashes = source_inventory()
    head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git","merge-base","--is-ancestor",STAGE3A_COMMIT,head],cwd=ROOT,check=True)
    output.mkdir(parents=True)
    manifest = dict(status="PREDECLARED", actual_head=head,
                    worktree_before=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True),
                    stage3a_archive_commit=STAGE3A_COMMIT, selection_run_head=selection_manifest["actual_head"],
                    physics_base_commit=selection_manifest["physics_base_commit"], physics_provenance=physics,
                    source_checkpoint=spec["source_checkpoint"], source_checkpoint_sha256=spec["checkpoint_sha256"],
                    config=cfg, protocol=protocol(plan), training=False, training_seed_count=0,
                    deterministic_frozen_sac=True, sac_updates=0, ppo_updates=0, td3_updates=0,
                    replay_writes=0, replay_samples=0, reserved_scenes_executed=[],
                    source_hashes_before=hashes, stage3a_files_before=original_hashes,
                    completed_branches=0, completed_state_futures=0)
    write_json(output/"manifest.json", manifest)
    write_json(output/"protocol.json", protocol(plan))
    write_json(output/"frozen_candidates.json", plan)
    for name in hashes:
        target = output/"source_evidence"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    fields = physical_fields(cfg)
    rows, checks = [], []
    started = time.perf_counter()
    with isolated_rng():
        sac = ReadOnlySAC(saved)
    del saved
    with isolated_rng(), no_learning(sac), torch.no_grad(), \
         patch("hrl_uav.learning.DirectHRL.__init__", forbidden), \
         patch("hrl_uav.agents.ppo_agent.ActorNet.__init__", forbidden), \
         patch("hrl_uav.agents.ppo_agent.CriticNet.__init__", forbidden):
        for item in plan:
            scene, start = item["scene"], item["start_s"]
            snapshot = SavedBranch(item["snapshot"],item["initial_state_sha256"],fields)
            future_digests = []
            with np.load(SOURCE/"states"/f"{scene}_t{start}"/"zero"/"trajectory.npz") as old:
                old_user_positions = old["users"].copy()
            for future in item["futures"]:
                k, rng_seed = future["future"], future["rng_seed"]
                folder = output/"states"/f"{scene}_t{start}"/f"future{k}"
                folder.mkdir(parents=True)
                zero_values = zero_crn = zero_control = zero_positions = zero_rng = None
                fresh_hash = None
                repeated = {}
                for candidate in item["candidates"]:
                    meter = snapshot.restore(rng_seed)
                    current_hash = semantic_hash(dict(physical=physical_state(meter,fields),rng=capture_rng()))
                    if fresh_hash is None:
                        fresh_hash = current_hash
                    elif fresh_hash != current_hash:
                        raise AssertionError("Four candidate initial states differ")
                    initial = initial_counts(meter)
                    macros = [macro_step(meter,sac,candidate["delta"] if i == 0 else np.zeros(6)) for i in range(HORIZON)]
                    if (meter.env._current_time() != start+60 or meter.integration_steps != (start+60)*10
                            or meter.reset_count != 0 or len(meter.positions) != 600 or meter.mobility_states != 120):
                        raise AssertionError("Branch timing/reset/mobility conservation failed")
                    values = branch_metrics(meter,macros,initial)
                    crn = crn_records(meter)
                    control = macros[0]["control"]
                    positions = np.asarray([p[1] for p in meter.positions])
                    if candidate["label"] == "zero":
                        zero_values, zero_crn, zero_control, zero_positions = values.copy(),crn,copy.deepcopy(control),positions.copy()
                        zero_rng = copy.deepcopy(meter.env.np_random.get_state())
                        users = np.asarray([p[2] for p in meter.positions])
                        if np.array_equal(users,old_user_positions):
                            raise AssertionError("Fresh future reproduced the original entire user trajectory")
                        signature = tuple(crn[-1][key] for key in
                            ("arrivals_sha256","users_microsteps_sha256","mobility_velocity_states_sha256"))
                        if any(any(a == b for a,b in zip(signature,old)) for old in future_digests):
                            raise AssertionError("Different futures share a complete arrival/mobility stream")
                        future_digests.append(signature)
                    else:
                        assert_crn_equal(zero_crn,crn)
                    if not state_equal(meter.env.np_random.get_state(),zero_rng):
                        raise AssertionError("Within-future RNG mismatch")
                    if not state_equal(capture_rng(),pickle.loads(snapshot.blob)["rng"]):
                        raise AssertionError("Global RNG consumed during frozen inference")
                    if semantic_hash(pickle.loads(snapshot.blob)) != snapshot.original_hash:
                        raise AssertionError("Original disk snapshot mutated")
                    difference = control["executed_displacement_m"]-zero_control["executed_displacement_m"]
                    trajectory_difference = float(np.linalg.norm(positions-zero_positions,axis=2).max())
                    result = dict(seed=scene,start_s=start,future=k,future_rng_seed=rng_seed,
                                  action=candidate["label"],direction=candidate["name"],dimension=candidate["dimension"],
                                  sign=candidate["sign"],original_state_hash=snapshot.original_hash,
                                  initial_state_hash=fresh_hash,**values,
                                  executed_difference_m=float(np.linalg.norm(difference)),
                                  full_trajectory_difference_m=trajectory_difference,
                                  ineffective=candidate["label"] != "zero" and trajectory_difference<=POSITION_ATOL)
                    for flag in ("component_clip","unit_disk_projection","saturation","cap_request","map_projection","safety_projection"):
                        result[f"{flag}_uavs"] = int(control[flag].sum())
                    for key in values:
                        if isinstance(values[key],(int,float,np.number)):
                            result[f"delta_{key}"] = values[key]-zero_values[key]
                    result["relative_C60_percent"] = 100*result["delta_C60"]/zero_values["C60"]
                    if not all(np.isfinite(v) for v in result.values() if isinstance(v,(int,float,np.number))):
                        raise FloatingPointError("Nonfinite result")
                    terminal_hash = semantic_hash(physical_state(meter,fields))
                    key = tuple(candidate["delta"])
                    duplicate = key in repeated
                    if duplicate:
                        previous_values, previous_hash = repeated[key]
                        if previous_values != values or previous_hash != terminal_hash:
                            raise AssertionError("Identical frozen directions produced different results")
                    else:
                        repeated[key] = (values.copy(), terminal_hash)
                    save_branch(folder/candidate["label"],meter,macros,result,difference)
                    append_csv(output/"per_branch.csv",[result])
                    rows.append(result)
                    checks.append(dict(seed=scene,start_s=start,future=k,action=candidate["label"],
                        initial_state_equal=True,physical_history_unchanged=True,crn_equal=True,
                        environment_rng_equal=True,global_rng_unchanged=True,
                        duplicate_direction=duplicate,duplicate_check_passed=duplicate,
                        terminal_state_sha256=terminal_hash))
                sac.assert_unchanged()
                manifest.update(status="RUNNING",completed_branches=len(rows),completed_state_futures=len(rows)//4,
                                elapsed_seconds=time.perf_counter()-started)
                write_json(output/"manifest.json",manifest)
            print(f"scene={scene} t={start}: five fresh futures complete, {len(rows)}/320 branches",flush=True)
    if len(rows) != 320:
        raise AssertionError("Fixed320-branch design incomplete")
    if source_inventory() != hashes or any(file_hash(Path(p)) != h for p,h in original_hashes.items()):
        raise AssertionError("Code or original Stage3A evidence changed")
    if file_hash(ROOT/spec["source_checkpoint"]) != spec["checkpoint_sha256"]:
        raise AssertionError("SAC checkpoint changed")
    write_json(output/"checks.json",checks)
    manifest.update(status="COMPLETE_320_BRANCHES_STOPPED",completed_branches=320,completed_state_futures=80,
                    elapsed_seconds=time.perf_counter()-started,branch_macros=1920,branch_physical_steps=192000,
                    source_hashes_after=source_inventory(),source_unchanged=True,stage3a_unchanged=True,
                    sac_frozen=sac.assert_unchanged(),crn_all_equal=True,fresh_future_streams_distinct=True,
                    worktree_after=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True))
    write_json(output/"manifest.json",manifest)
    from .mission_future_analysis import analyze
    analyze(output)
    return str(output)
