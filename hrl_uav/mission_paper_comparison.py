"""Bounded, no-learning routing ablations and frozen-policy sensitivity checks."""

from contextlib import ExitStack
import copy
import gc
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .agents.sac_agent import Memory
from .bandwidth import UniformRouting, summarize_links
from .experiment import append_csv, isolated_rng, write_json
from .mission_pilot import ObservedPolicy, crn_records
from .mission_ppo_qualification import (
    METRICS, read_json, restore_boundary, transaction, verify_transaction,
)
from .mission_ppo_stage4a import (
    MissionMacroPPO, Stage4Meter, collect_stream, stage_config, summarize,
)
from .mission_provenance import physics_provenance
from .mission_smoke import ROOT, load_source, load_spec
from .priority_grid import file_hash
from .resumption import capture_rng, state_equal
from .routing import project_rows_to_simplex
from .scene_prescreen import LocalRouting, assert_crn_equal, role_summary
from .trajectory_check import ReadOnlySAC, forbidden, no_learning
from .wireless_check import read_csv
from .wireless_pilot import route_statistics


HISTORY = ROOT / "results/mission_ppo_stage4a_qualification"
FIXED_HISTORY = ROOT / "results/mission_rpgm_v1_pilot"
PPO_SEEDS = (201, 202, 203)
SCENES = (33600, 33601)
MICROS = ("SAC", "Local", "Uniform", "Random")
CONDITIONS = (("base", 1., 6.), ("load080", .8, 6.), ("load110", 1.1, 6.),
              ("bandwidth3", 1., 3.), ("bandwidth45", 1., 4.5))
NEW_FILES = ("hrl_uav/mission_paper_comparison.py", "scripts/run_mission_paper_comparison.py",
             "tests/test_mission_paper_comparison.py", "docs/mission_paper_comparison.md")


def protocol():
    return dict(
        purpose="shared frozen PPO policy micro ablation; frozen-policy operating sensitivity",
        conditions=[dict(name=n, load_multiplier=l, bandwidth_mhz=b) for n, l, b in CONDITIONS],
        ppo_seeds=list(PPO_SEEDS), ppo_endpoint_window=120, sac_seed=101, sac_window=2000,
        scenes=list(SCENES), scene_status="reused development scenes", seconds=900,
        micro_rules=list(MICROS), predictive_reference="SAC only, causal lead10s",
        cases=130, reused_cases=8, new_evaluations=122,
        random_rule="independent Dirichlet(1,1,1) per source/priority per0.5s decision",
        random_rng="private PCG64 SeedSequence([86701, scene]); shared across conditions/PPO seeds",
        training=False, replay_writes=0, replay_samples=0, reserved_scenes_executed=[],
        no_retraining_per_parameter=True, no_best_selection=True,
        scope="Not independently retrained complete methods; not joint HRL; no SCA or TD3 result",
        continued600="report all three recorded endpoints separately; do not select a winning seed",
    )


def condition_config(cfg, name):
    matches = [(l, b) for n, l, b in CONDITIONS if n == name]
    if len(matches) != 1:
        raise ValueError("Undeclared parameter condition")
    load, bandwidth = matches[0]
    result = copy.deepcopy(cfg)
    result["phase_loads"] = (np.asarray(cfg["phase_loads"]) * load).tolist()
    result["environment"]["u2u_bandwidth_mhz"] = bandwidth
    return result


class RandomRouting:
    """Uniform distribution over each feasible route simplex, not uniform routing."""

    def __init__(self, scene):
        if scene not in SCENES:
            raise ValueError("Only the two declared development scenes are allowed")
        self.rng = np.random.default_rng(np.random.SeedSequence([86701, scene]))
        self.calls = 0

    def choose_action(self, state, deterministic=False):
        if not deterministic:
            raise AssertionError("Evaluation interface flag must remain deterministic")
        self.calls += 1
        return self.rng.dirichlet(np.ones(3), size=(2, 3)).ravel()


def micro_handle(name, scene, sac):
    if name not in MICROS:
        raise ValueError("Unknown micro rule")
    if name == "SAC":
        return sac
    rule = {"Local": LocalRouting, "Uniform": UniformRouting}.get(name)
    return SimpleNamespace(sac_actor=rule() if rule else RandomRouting(scene))


def evaluate_case(policy, sac, condition, micro, scene, *, predictive=False, macros=90):
    if scene not in SCENES or micro not in MICROS or macros not in (2, 90):
        raise ValueError("Out-of-protocol case")
    if predictive and micro != "SAC":
        raise ValueError("Predictive reference is fixed to SAC")
    cfg = condition_config(policy.config, condition)
    meter = Stage4Meter(cfg)
    handle = micro_handle(micro, scene, sac)
    controller = ObservedPolicy(sac, "predictive_centroid") if predictive else policy
    if predictive:
        controller.meter = meter
    view = SimpleNamespace(sac=handle, macro_action=controller.macro_action)
    before, rng_before = policy.ppo_state(), capture_rng()
    with ExitStack() as stack:
        stack.enter_context(isolated_rng())
        stack.enter_context(no_learning(sac))
        stack.enter_context(torch.no_grad())
        stack.enter_context(patch.object(Memory, "__init__", forbidden))
        stack.enter_context(patch.object(policy, "update_ppo", forbidden))
        for optimizer in (policy.ppo_actor_optimizer, policy.ppo_critic_optimizer):
            stack.enter_context(patch.object(optimizer, "step", forbidden))
        result = collect_stream(cfg, view, scene, train=False, macros=macros,
                                meter_factory=lambda _: meter)
    if not state_equal(before, policy.ppo_state()) or not state_equal(rng_before, capture_rng()):
        raise AssertionError("Evaluation changed PPO/RNG/optimizer/rollout state")
    summary, _, macro_rows, observations, windows, actions = result
    row, tracking = summarize(summary, meter, policy, scene)
    row.update({k: v for k, v in tracking[0].items() if k not in ("policy", "seed", "uav")})
    row["u2u_tx_time_s"] = sum(r["tx_time_s"] for r in meter.link_readings)
    row["u2u_rate_time_weighted_mbps"] = row["tx_served_mbit"] / max(row["u2u_tx_time_s"], 1e-12)
    for key in METRICS:
        if not np.isfinite(row[key]):
            raise AssertionError(f"Missing/nonfinite metric: {key}")
    for priority in ("high", "low"):
        if row[f"task_arrived_{priority}_count"] != (row[f"task_completed_{priority}_count"]
                                                     + row[f"task_unfinished_{priority}_count"]):
            raise AssertionError("Parent task conservation failure")
    if micro in ("Local", "Uniform"):
        raw = (LocalRouting() if micro == "Local" else UniformRouting()).choose_action(None)
        expected = project_rows_to_simplex(raw.reshape(2, 3, 3))
        np.testing.assert_allclose([a[1] for a in actions], np.tile(expected, (macros*20, 1, 1, 1)),
                                   atol=1e-7, rtol=0)
    if micro == "Local" and (row["tx_served_mbit"] != 0 or row["energy_u2u_j"] != 0):
        raise AssertionError("Local transmitted U2U data")
    if micro == "Random":
        if handle.sac_actor.calls != macros * 20:
            raise AssertionError("Random decision count mismatch")
        np.testing.assert_allclose([a[0].reshape(2, 3, 3) for a in actions], [a[1] for a in actions],
                                   atol=1e-7, rtol=0)
    row.update(condition=condition, micro=micro, macro="Predictive" if predictive else "PPO120",
               ppo_seed=0 if predictive else policy.seed, ppo_updates_during_evaluation=0,
               sac_updates_during_evaluation=0, training=False, reused=False,
               load_multiplier=next(l for n, l, _ in CONDITIONS if n == condition),
               bandwidth_mhz=cfg["environment"]["u2u_bandwidth_mhz"])
    checks = dict(**sac.assert_unchanged(), ppo_state_unchanged=True, global_rng_unchanged=True,
                  replay_writes=0, replay_samples=0, ppo_update_increment=0, sac_update_increment=0,
                  route_checked=True, parent_conservation=True)
    return row, meter, tracking, macro_rows, windows, observations, actions, checks


def save_result(directory, result):
    row, meter, tracking, macros, windows, observations, actions, checks = result
    name = f"{row['macro']}+{row['micro']}"
    for filename, records in (
        ("queues", meter.readings), ("commands", meter.commands), ("uplink", meter.uplink_readings),
        ("links", meter.link_readings), ("uavs", meter.uav_readings), ("routes", meter.route_readings),
        ("macro_metrics", macros), ("windows", windows), ("trajectory_summary", tracking),
        ("role_summary", role_summary(meter.uav_readings, .5, name, row["seed"])),
        ("route_summary", route_statistics(meter.route_readings, name, row["seed"], None, "frozen_micro_ablation")),
        ("link_summary", summarize_links(meter.link_readings, row["duration_s"], row["seed"], row["bandwidth_mhz"])),
    ):
        append_csv(directory / f"{filename}.csv", records)
    write_json(directory / "summary.json", row)
    write_json(directory / "crn.json", crn_records(meter))
    write_json(directory / "freeze_check.json", checks)
    np.savez_compressed(directory / "trajectory.npz", time_s=[r[0] for r in meter.positions],
                        uav=[r[1] for r in meter.positions], users=[r[2] for r in meter.positions],
                        centroid=[r[3] for r in meter.positions], groups=[r[4] for r in meter.positions],
                        micro_state=meter.micro_states)
    np.savez_compressed(directory / "macro_samples.npz", **dict(zip(
        ("state", "raw_action", "old_logprob", "reward", "next_state", "continuation"),
        (np.asarray(x) for x in zip(*observations)))))
    np.savez_compressed(directory / "micro_actions.npz", raw=[a[0] for a in actions], executed=[a[1] for a in actions])


def assert_saved_prefix(result, directory):
    meter = result[1]
    with np.load(directory / "trajectory.npz") as saved:
        for index, field in enumerate(("time_s", "uav", "users", "centroid", "groups")):
            np.testing.assert_array_equal([r[index] for r in meter.positions], saved[field][:len(meter.positions)])
        np.testing.assert_array_equal(meter.micro_states, saved["micro_state"][:len(meter.micro_states)])
    assert_crn_equal(read_json(directory / "crn.json")[:len(meter.readings)], crn_records(meter))
    previous = read_csv(directory / "queues.csv")[:len(meter.readings)]
    for actual, old in zip(meter.readings, previous):
        for key in ("time_s", "queue_total_mbit", "holding_cost", "energy_cost"):
            np.testing.assert_allclose(actual[key], float(old[key]), rtol=1e-12, atol=1e-8)


def table_row(row, directory):
    fields = ("condition", "macro", "micro", "ppo_seed", "seed", "load_multiplier", "bandwidth_mhz", "reused")
    return dict(**{k: row[k] for k in fields}, **{k: row[k] for k in METRICS}, directory=str(directory))


def aggregate(rows, output):
    per_init, summary, paired = [], [], []
    for condition, _, _ in CONDITIONS:
        for macro, micro in [("PPO120", m) for m in MICROS] + [("Predictive", "SAC")]:
            group = [r for r in rows if (r["condition"], r["macro"], r["micro"]) == (condition, macro, micro)]
            seeds = PPO_SEEDS if macro == "PPO120" else (0,)
            means = []
            for ppo_seed in seeds:
                selected = [r for r in group if r["ppo_seed"] == ppo_seed]
                if {r["seed"] for r in selected} != set(SCENES) or len(selected) != 2:
                    raise AssertionError("Missing/duplicate development case")
                record = dict(condition=condition, macro=macro, micro=micro, ppo_seed=ppo_seed,
                              **{k: float(np.mean([r[k] for r in selected])) for k in METRICS})
                per_init.append(record)
                means.append(record)
            summary.append(dict(condition=condition, macro=macro, micro=micro, scene_count=2,
                                macro_initializations=3 if macro == "PPO120" else 0,
                                **{k: float(np.mean([r[k] for r in means])) for k in METRICS},
                                **{f"std_{k}": float(np.std([r[k] for r in means], ddof=1)) if len(means)>1 else None
                                   for k in METRICS}))
        for ppo_seed in PPO_SEEDS:
            for scene in SCENES:
                target = next(r for r in rows if (r["condition"], r["macro"], r["micro"], r["ppo_seed"], r["seed"])
                              == (condition, "PPO120", "SAC", ppo_seed, scene))
                for micro in MICROS[1:]:
                    reference = next(r for r in rows if (r["condition"], r["macro"], r["micro"], r["ppo_seed"], r["seed"])
                                     == (condition, "PPO120", micro, ppo_seed, scene))
                    paired.append(dict(condition=condition, ppo_seed=ppo_seed, seed=scene, reference=micro,
                                       relative_J_percent=100*(target["J"]-reference["J"])/reference["J"],
                                       **{f"delta_{k}": target[k]-reference[k] for k in METRICS}))
    append_csv(output / "per_seed.csv", rows)
    append_csv(output / "per_training_seed.csv", per_init)
    append_csv(output / "summary.csv", summary)
    append_csv(output / "paired_deltas.csv", paired)
    return summary


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new independent result directory; no overwrite")
    physics = physics_provenance()
    prior = read_json(HISTORY / "manifest.json")
    if prior["status"] != "COMPLETE_3X120_STOPPED" or not prior["source_unchanged"]:
        raise ValueError("Incomplete/unlocked PPO source")
    lock = prior["lock"]
    for name, digest in lock["sources"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Source differs from qualification: {name}")
    protected = {n: file_hash(ROOT/n) for n in dict.fromkeys((*lock["sources"], *NEW_FILES))}
    spec = load_spec()
    saved, _ = load_source(spec)
    with isolated_rng():
        sac = ReadOnlySAC(saved)
        policies = {}
        evidence = {str(HISTORY / "manifest.json"): file_hash(HISTORY / "manifest.json")}
        for seed in PPO_SEEDS:
            cfg = stage_config(saved["config"], seed)
            if cfg != lock["configs"][str(seed)]:
                raise ValueError("Source PPO configuration mismatch")
            folder = HISTORY / f"seed{seed}/boundary040"
            verify_transaction(folder)
            payload = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
            ppo = MissionMacroPPO(cfg, seed, sac)
            if restore_boundary(ppo, payload, lock) != 40:
                raise ValueError("Not final window120")
            policies[seed] = ppo
            evidence[str(folder / "checkpoint.pt")] = file_hash(folder / "checkpoint.pt")
            verify_transaction(HISTORY / f"seed{seed}/monitor120")
    del saved, payload
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="SMOKE_PENDING", protocol=protocol(), actual_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        worktree=subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
        physics=physics, source_sha256=spec["checkpoint_sha256"], evidence=evidence,
        sources_before=protected, training=False, checkpoint_selection="all three original window120 endpoints",
        configs={n: {str(s): condition_config(policies[s].config, n) for s in PPO_SEEDS} for n, _, _ in CONDITIONS},
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, threads=torch.get_num_threads()),
        results_completed=0, source_unchanged=None, reserved_scenes_executed=[])
    write_json(output / "manifest.json", manifest)
    start = time.perf_counter()
    rows, refs = [], {}
    try:
        smoke_refs = {}
        for condition, _, _ in CONDITIONS:
            for micro in MICROS:
                result = evaluate_case(policies[201], sac, condition, micro, 33600, macros=2)
                records = crn_records(result[1])
                if condition in smoke_refs:
                    assert_crn_equal(smoke_refs[condition], records)
                else:
                    smoke_refs[condition] = records
                if condition == "base" and micro == "SAC":
                    assert_saved_prefix(result, HISTORY / "seed201/monitor120/33600")
                with transaction(output / "smoke" / f"{condition}_{micro}") as directory:
                    save_result(directory, result)
                del result
        result = evaluate_case(policies[201], sac, "base", "SAC", 33600, predictive=True, macros=2)
        assert_saved_prefix(result, FIXED_HISTORY / "predictive_centroid_33600")
        del result
        manifest.update(status="RUNNING", smoke_passed=True)
        write_json(output / "manifest.json", manifest)
        for condition, load, bandwidth in CONDITIONS:
            cases = [(s, m, False) for s in PPO_SEEDS for m in MICROS] + [(201, "SAC", True)]
            for ppo_seed, micro, predictive in cases:
                for scene in SCENES:
                    macro = "Predictive" if predictive else "PPO120"
                    identity = 0 if predictive else ppo_seed
                    destination = output / "cases" / f"{condition}_{macro}{identity}_{micro}_{scene}"
                    if condition == "base" and micro == "SAC":
                        source = (FIXED_HISTORY / f"predictive_centroid_{scene}" if predictive else
                                  HISTORY / f"seed{ppo_seed}/monitor120/{scene}")
                        row = read_json(source / "summary.json")
                        if row["duration_s"] != 900 or row["replay_writes"] or row["sac_update_increment"]:
                            raise ValueError("Incompatible baseline duration/learning semantics")
                        if predictive:
                            old_manifest = read_json(FIXED_HISTORY / "manifest.json")
                            old_cfg = copy.deepcopy(old_manifest["config"])
                            new_cfg = copy.deepcopy(policies[ppo_seed].config)
                            old_cfg.pop("training")
                            new_cfg.pop("training")
                            if old_cfg != new_cfg or old_manifest["source_checkpoint_sha256"] != spec["checkpoint_sha256"]:
                                raise ValueError("Predictive source config mismatch")
                        records = read_json(source / "crn.json")
                        row.update(condition=condition, macro=macro, micro=micro, ppo_seed=identity,
                                   load_multiplier=load, bandwidth_mhz=bandwidth, reused=True)
                        evidence[str(source / "summary.json")] = file_hash(source / "summary.json")
                        evidence[str(source / "crn.json")] = file_hash(source / "crn.json")
                        with transaction(destination) as directory:
                            write_json(directory / "summary.json", row)
                            write_json(directory / "reused.json", dict(source=str(source),
                                files={str(p): file_hash(p) for p in source.iterdir() if p.is_file()}))
                            write_json(directory / "crn.json", records)
                    else:
                        result = evaluate_case(policies[ppo_seed], sac, condition, micro, scene, predictive=predictive)
                        row, meter = result[:2]
                        records = crn_records(meter)
                        with transaction(destination) as directory:
                            save_result(directory, result)
                        del result, meter
                    if len(records) != 1800:
                        raise AssertionError("Incomplete900s CRN")
                    key = (condition, scene)
                    if key in refs:
                        assert_crn_equal(refs[key], records)
                    else:
                        refs[key] = records
                    # Same load under another bandwidth must preserve the original exogenous stream.
                    if load == 1. and condition != "base":
                        assert_crn_equal(refs[("base", scene)], records)
                    rows.append(table_row(row, destination))
                    manifest.update(results_completed=len(rows), elapsed_seconds=time.perf_counter()-start)
                    write_json(output / "manifest.json", manifest)
                    if len(rows) % 10 == 0:
                        print(f"Completed {len(rows)}/130 cases; no training", flush=True)
                    gc.collect()
        if len(rows) != 130 or sum(r["reused"] for r in rows) != 8:
            raise AssertionError("Evaluation budget mismatch")
        aggregate(rows, output)
        for path, digest in evidence.items():
            if file_hash(path) != digest:
                raise AssertionError(f"Source evidence changed: {path}")
        after = {n: file_hash(ROOT/n) for n in protected}
        if after != protected or physics_provenance() != physics:
            raise AssertionError("Source/physics changed during run")
        manifest.update(status="COMPLETE_STOPPED", sources_after=after, source_unchanged=True,
                        evidence=evidence, crn_all_equal=True, frozen_sac=sac.assert_unchanged(),
                        new_evaluations=122, reused_evaluations=8, elapsed_seconds=time.perf_counter()-start)
        write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="FAILED_STOPPED", error=repr(exc), elapsed_seconds=time.perf_counter()-start)
        write_json(output / "manifest.json", manifest)
        raise
    return output
