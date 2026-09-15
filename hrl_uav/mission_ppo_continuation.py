"""Fixed window120->600 continuation; original learning and physics stay frozen."""

from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
import torch

from . import mission_ppo_qualification as q
from .experiment import append_csv, isolated_rng, write_json
from .mission_ppo_stage4a import MissionMacroPPO, Stage4Meter, frozen_sac_guard
from .resumption import state_equal
from .trajectory_check import ReadOnlySAC


SOURCE_COMMIT = "9f98e5142b563e134a2f90bd2b6f555fd7ce4bf1"
SOURCE = q.ROOT / "results/mission_ppo_stage4a_qualification"
STAGE = "mission_direct_ppo_continuation120to600_v1"
START_INDEX, END_INDEX = 40, 200
TRAIN_SCENES = tuple(range(72000, 72200))
MONITORS = (240, 360, 480, 600)
FILES = ("hrl_uav/mission_ppo_continuation.py", "scripts/run_mission_ppo_continuation.py",
         "tests/test_mission_ppo_continuation.py", "docs/mission_ppo_continuation.md")


def protocol():
    return dict(stage=STAGE, source_commit=SOURCE_COMMIT, source_window=120, final_window=600,
                added_windows=480, ppo_seeds=list(q.PPO_SEEDS), sac_seed=101, sac_window=2000,
                training_scenes=list(TRAIN_SCENES[START_INDEX:]), independent_of_ppo_seed=True,
                stream_seconds=900, window_seconds=300, reset="only between900s streams",
                macro_samples_total=18000, updates_total=60, added_updates=48, batch_size=300,
                force=False, monitors=list(MONITORS), evaluation_scenes=list(q.VALIDATION_SCENES),
                forbidden_scenes=[33610, 33611, 33612, *range(33700, 33710)],
                final_selection="window600 only", sac="deterministic frozen, no replay",
                config="identical to source; cumulative budget is external runner metadata",
                progress="cumulative macros/3600 with unchanged joint_windows=120; std reaches1/30 at window200",
                training_storage="lossless stream.zip with per-member SHA256; checkpoints at every900s boundary",
                evaluation_storage="original case format", resume="same directory, complete900s boundaries only")


def collect_stream(cfg, policy, scene, *, train, macros=90, macro_offset=0,
                   meter_factory=Stage4Meter, record=True):
    """The original collector body, with only the explicit scene/budget guard extended.

    A structural regression test compares everything from meter construction onward
    against the frozen Stage4A implementation. Progress deliberately remains /3600.
    """
    allowed = TRAIN_SCENES if train else q.VALIDATION_SCENES
    if (type(scene) is not int or scene not in allowed or type(macros) is not int
            or macros not in range(1, 91) or type(macro_offset) is not int
            or macro_offset < 0 or macro_offset + macros > 18000):
        raise ValueError("Out-of-protocol scene, horizon or macro budget")
    if train and (macro_offset % 90 or scene != TRAIN_SCENES[macro_offset // 90]):
        raise ValueError("Training scene order must follow the fixed900s schedule")
    meter = meter_factory(cfg)
    meter.reset(scene, 90)
    if (meter.env.deployment_resource_frames != 0 or meter.env.service_resource_frames != 20
            or not meter.env.continuous_mission):
        raise ValueError("Stage4A requires the unchanged20-service-frame mission protocol")
    gamma = cfg["training"]["gamma_per_resource_frame"]
    rows, observations, window_rows, routes = [], [], [], []
    initial_collected = policy.collected_macros if train else 0
    cost_before = energy_before = 0.
    for macro in range(macros):
        state = meter.macro_observation()
        raw, logprob = policy.macro_action(state, enabled=True, deterministic=not train)
        meter.start_macro(raw)
        rewards = []
        micro_state = meter.prepare_service()
        for frame in range(20):
            with torch.no_grad():
                micro = policy.sac.sac_actor.choose_action(micro_state, deterministic=True)
            reading = meter.step_frame(micro, gamma)
            rewards.append(reading["reward"])
            if record:
                routes.append((micro.copy(), reading["route"].copy()))
            if frame != 19:
                micro_state = meter.prepare_service()
        reward = sum(gamma**i * r for i, r in enumerate(rewards)) / len(rewards)
        meter.advance_macro_clock()
        next_state = meter.macro_observation()
        transition = (state.copy(), raw.copy(), logprob, reward, next_state.copy(), float(macro < macros-1))
        if train:
            policy.macro_rollout.append(transition)
            policy.collected_macros += 1
            policy.update_ppo((macro_offset + macro + 1) / 3600.)
        if record:
            observations.append(transition)
            rows.append(dict(macro=macro+1, **meter.metrics()))
        if (macro+1) % 30 == 0 or macro == macros-1:
            metrics = meter.metrics()
            duration = (macro % 30 + 1)*10
            window_rows.append(dict(scene=scene, window_end_macro=macro+1, time_s=metrics["time_s"],
                                    window_J=(meter.cost_sum-cost_before)/(duration*10),
                                    window_power_w=(meter.total_power_energy-energy_before)/duration,
                                    queue_total_mbit=metrics["queue_total_mbit"], reset_count=meter.reset_count))
            cost_before, energy_before = meter.cost_sum, meter.total_power_energy
    if (meter.reset_count != 1 or meter.integration_steps != macros*100
            or meter.env.resource_frame_prepare_count != macros*20
            or meter.env.mobility_update_count != macros*20):
        raise AssertionError("Mission decision/time/reset conservation failed")
    if train and (policy.collected_macros-initial_collected != macros
                  or policy.collected_macros != policy.consumed_macros+len(policy.macro_rollout)):
        raise AssertionError("Macro replay/batch conservation failed")
    summary = dict(meter.metrics(), boundary_transitions=0, final_pending_dropped=0,
                   macro_decisions=macros, sac_decisions=20*macros,
                   replay_writes=0, replay_samples=0, sac_update_increment=0)
    return summary, meter, rows, observations, window_rows, routes


def validate_boundary(payload, policy, lock):
    index = payload.get("next_scene_index")
    if (payload.get("stage") != STAGE or payload.get("boundary") is not True
            or type(index) is not int or index not in range(START_INDEX, END_INDEX+1)
            or payload.get("current_window") != index*3 or payload.get("replay", "missing") is not None
            or payload.get("ppo_seed") != policy.seed or payload.get("config") != policy.config
            or payload.get("lock") != lock):
        raise ValueError("Incompatible continuation boundary/config/source")
    state = payload["ppo"]
    if any(len(state[k]) != 2 for k in ("optimizers", "gradients", "training_modes")):
        raise ValueError("Missing PPO network/optimizer state")
    updates, gradients, collected, consumed = state["counters"]
    if (collected != index*90 or consumed != updates*300 or updates != collected//300
            or len(state["rollout"]) != collected-consumed
            or not updates <= gradients <= 10*updates):
        raise ValueError("PPO stream/batch/gradient conservation failure")
    for k, transition in enumerate(state["rollout"], start=consumed):
        s, a, lp, reward, ns, continuation = transition
        if (np.shape(s) != (67,) or np.shape(ns) != (67,) or np.shape(a) != (6,)
                or continuation != float(k % 90 != 89)
                or not all(np.all(np.isfinite(x)) for x in (s, a, lp, reward, ns))):
            raise ValueError("Corrupt pending macro rollout")
    for net in (state["actor"], state["critic"]):
        if not all(torch.isfinite(t).all() for t in net.values()):
            raise ValueError("Nonfinite PPO checkpoint")
    cfg = policy.cfg
    fraction = min(1., (consumed/30)/cfg["ppo_std_decay_windows"])
    log_std = (1-fraction)*np.log(cfg["ppo_initial_std"]) + fraction*np.log(cfg["ppo_final_std"])
    np.testing.assert_allclose(state["actor"]["log_std"].cpu().numpy(), log_std, rtol=0, atol=2e-7)


def boundary_payload(policy, index, lock):
    payload = dict(stage=STAGE, boundary=True, next_scene_index=index,
                   current_window=index*3, ppo_seed=policy.seed, config=policy.config,
                   lock=copy.deepcopy(lock), ppo=policy.ppo_state(), replay=None)
    validate_boundary(payload, policy, lock)
    return payload


def restore_boundary(policy, payload, lock):
    validate_boundary(payload, policy, lock)
    state = payload["ppo"]
    for net, key in ((policy.macro_actor, "actor"), (policy.macro_critic, "critic")):
        net.load_state_dict(state[key], strict=True)
    for optimizer, saved in zip((policy.ppo_actor_optimizer, policy.ppo_critic_optimizer), state["optimizers"]):
        optimizer.load_state_dict(saved)
    for net, grads, modes in zip((policy.macro_actor, policy.macro_critic),
                                state["gradients"], state["training_modes"]):
        if len(list(net.parameters())) != len(grads) or len(list(net.modules())) != len(modes):
            raise ValueError("Missing gradient/module-mode state")
        for p, grad in zip(net.parameters(), grads):
            p.grad = None if grad is None else grad.to(p.device).clone()
        for module, mode in zip(net.modules(), modes):
            module.training = mode
    policy.macro_rollout = copy.deepcopy(state["rollout"])
    policy.rng = copy.deepcopy(state["rng"])
    (policy.ppo_updates, policy.ppo_gradient_steps,
     policy.collected_macros, policy.consumed_macros) = state["counters"]
    policy.last_ppo_stats = copy.deepcopy(state["last_stats"])
    policy.batch_log.clear()
    policy.update_log.clear()
    if not state_equal(state, policy.ppo_state()):
        raise AssertionError("Boundary restore is not bitwise equivalent")
    policy.sac.assert_unchanged()
    return payload["next_scene_index"]


def archive_case(attempt, policy, scene, window, result):
    """Persist the original statistics, then losslessly archive generated temp files."""
    with tempfile.TemporaryDirectory(prefix="case-", dir=attempt) as temp:
        folder = Path(temp) / "stream"
        row = q.persist_case(folder, policy, scene, window, result)
        members = {p.name: q.file_hash(p) for p in folder.iterdir() if p.is_file()}
        archive = attempt / "stream.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name in sorted(members):
                z.write(folder / name, name)
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != set(members) or z.testzip() is not None:
                raise AssertionError("Lossless stream archive verification failed")
            for name, digest in members.items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise AssertionError(f"Archived member changed: {name}")
        write_json(attempt / "stream_archive.json", dict(files=members, lossless=True))
    return row


def archive_json(boundary, name):
    with zipfile.ZipFile(Path(boundary) / "stream.zip") as z:
        return json.loads(z.read(name))


def aggregate(rows, baselines):
    if (len(rows) != 6 or {(r["ppo_seed"], r["seed"]) for r in rows}
            != {(s, e) for s in q.PPO_SEEDS for e in q.VALIDATION_SCENES}
            or any(r["window"] != 600 for r in rows)):
        raise ValueError("Only three final window600 policies x two scenes may be aggregated")
    if len(baselines) != 8 or {(r["policy"], r["seed"]) for r in baselines} != {
            (n, s) for n in q.BASELINES for s in q.VALIDATION_SCENES}:
        raise ValueError("Missing/duplicate frozen comparators")
    per_seed, overall, pairs = [], [], []
    for seed in q.PPO_SEEDS:
        subset = [r for r in rows if r["ppo_seed"] == seed]
        per_seed.append(dict(policy="direct_ppo", ppo_seed=seed, scene_count=2, window=600,
                             **{k: float(np.mean([r[k] for r in subset])) for k in q.METRICS}))
        for row in subset:
            for old in (b for b in baselines if b["seed"] == row["seed"]):
                pairs.append(dict(ppo_seed=seed, seed=row["seed"], comparator=old["policy"], window=600,
                                  **{k: row[k]-old[k] for k in q.METRICS}))
    for key in q.METRICS:
        values = [r[key] for r in per_seed]
        overall.append(dict(policy="direct_ppo", metric=key, training_initializations=3, scene_count=2,
                            window=600, mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                            min=min(values), max=max(values)))
        for name in q.BASELINES:
            values = [b[key] for b in baselines if b["policy"] == name]
            overall.append(dict(policy=name, metric=key, training_initializations=0, scene_count=2,
                                window=600, mean=float(np.mean(values)), std=None, min=None, max=None))
    return per_seed, overall, pairs


class ContinuationRunner(q.QualificationRunner):
    def __init__(self, output, lock, baselines, references, *, collector=collect_stream,
                 source=SOURCE, **kwargs):
        super().__init__(output, lock, baselines, references, collector=collector, **kwargs)
        self.source = Path(source)

    def run_seed(self, policy):
        root = self.output / f"seed{policy.seed}"
        root.mkdir(exist_ok=True)
        initial = root / "boundary040"
        if not initial.exists():
            if any(root.glob("boundary[0-9][0-9][0-9]")):
                raise ValueError("Missing continuation start boundary")
            source = self.source / f"seed{policy.seed}/boundary040"
            q.verify_transaction(source)
            payload = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
            if q.restore_boundary(policy, payload, self.lock["source_lock"]) != 40:
                raise ValueError("Continuation must start at window120")
            q.verify_transaction(self.source / f"seed{policy.seed}/monitor120")
            endpoint = torch.load(self.source / f"seed{policy.seed}/monitor120/ppo.pt", map_location="cpu", weights_only=False)
            if not state_equal(endpoint, policy.ppo_state()):
                raise AssertionError("Restored source does not match its evaluated window120")
            with q.transaction(initial) as attempt:
                torch.save(boundary_payload(policy, 40, self.lock), attempt / "checkpoint.pt")
                write_json(attempt / "origin.json", dict(source=str(source / "checkpoint.pt"),
                    sha256=q.file_hash(source / "checkpoint.pt"), all_ppo_state_bitwise=True,
                    inherited_updates=12, inherited_macros=3600, replay=None))
        indices = sorted(int(p.name[-3:]) for p in root.glob("boundary[0-9][0-9][0-9]"))
        if indices != list(range(40, max(indices)+1)) or max(indices) > END_INDEX:
            raise ValueError("Noncontiguous continuation boundaries")
        for index in indices:
            q.verify_transaction(root / f"boundary{index:03d}")
        index = restore_boundary(policy, torch.load(root / f"boundary{indices[-1]:03d}/checkpoint.pt",
                                                   map_location="cpu", weights_only=False), self.lock)
        if index != indices[-1]:
            raise ValueError("Boundary directory/counter mismatch")
        for window in MONITORS:
            if window < index*3:
                q.verify_transaction(root / f"monitor{window:03d}")
        while True:
            if index*3 in MONITORS:
                self.monitor(policy, index*3, root)
                rows = [q.read_json(root / f"monitor{index*3:03d}/{s}/summary.json") for s in q.VALIDATION_SCENES]
                print(f"MONITOR seed={policy.seed} window={index*3} J={np.mean([r['J'] for r in rows]):.9f} updates={policy.ppo_updates}", flush=True)
            if index == END_INDEX:
                break
            if shutil.disk_usage(self.output).free < 512*1024**2:
                raise OSError("Less than512MiB free; resume from last committed stream after restoring disk headroom")
            destination = root / f"boundary{index+1:03d}"
            with q.transaction(destination) as attempt:
                policy.batch_log.clear()
                policy.update_log.clear()
                with frozen_sac_guard(policy.sac), policy.exploration_rng():
                    result = self.collector(policy.config, policy, TRAIN_SCENES[index],
                                            train=True, macros=90, macro_offset=index*90)
                if policy.seed != q.PPO_SEEDS[0]:
                    reference = self.output / f"seed{q.PPO_SEEDS[0]}/boundary{index+1:03d}"
                    q.assert_crn_equal(archive_json(reference, "crn.json"), q.crn_records(result[1]))
                row = archive_case(attempt, policy, TRAIN_SCENES[index], (index+1)*3, result)
                if (row["duration_s"] != 900 or any(r["forced"] or r["transitions"] != 300
                                                   for r in policy.update_log)):
                    raise AssertionError("Continuation requires900s streams and full unforced300-sample batches")
                append_csv(attempt / "ppo_updates.csv", policy.update_log)
                if len(policy.update_log) != len(policy.batch_log):
                    raise AssertionError("Incomplete update journal")
                for update, batch in zip(policy.update_log, policy.batch_log):
                    np.savez_compressed(attempt / f"batch{update['update']:02d}.npz", **batch)
                torch.save(boundary_payload(policy, index+1, self.lock), attempt / "checkpoint.pt")
                del result
            index += 1
            q.atomic_json(self.output / "progress.json", dict(ppo_seed=policy.seed, window=index*3,
                          ppo_updates=policy.ppo_updates, collected_macros=policy.collected_macros,
                          pending_macros=len(policy.macro_rollout), boundary=str(destination)))
        if (policy.ppo_updates != 60 or policy.collected_macros != 18000
                or policy.consumed_macros != 18000 or policy.macro_rollout):
            raise AssertionError("Final continuation budget mismatch")

    def finalize(self):
        final = self.output / "final"
        if final.exists():
            q.verify_transaction(final)
            return
        rows, history, updates, origin_pairs = [], [], [], []
        for seed in q.PPO_SEEDS:
            for window in (120, *MONITORS):
                base = self.source if window == 120 else self.output
                folder = base / f"seed{seed}/monitor{window:03d}"
                q.verify_transaction(folder)
                history.extend(q.read_json(folder / str(scene) / "summary.json") for scene in q.VALIDATION_SCENES)
            rows.extend(r for r in history if r["ppo_seed"] == seed and r["window"] == 600)
            for index in range(1, END_INDEX+1):
                base = self.source if index <= 40 else self.output
                path = base / f"seed{seed}/boundary{index:03d}/ppo_updates.csv"
                if path.exists():
                    updates.extend(dict(r, ppo_seed=seed, window=int(r["update"])*10) for r in q.read_csv(path))
        for row in rows:
            old = next(r for r in history if r["ppo_seed"] == row["ppo_seed"] and r["seed"] == row["seed"] and r["window"] == 120)
            origin_pairs.append(dict(ppo_seed=row["ppo_seed"], seed=row["seed"], comparator="same_PPO_window120",
                                     **{k: row[k]-old[k] for k in q.METRICS}))
        per_seed, overall, pairs = aggregate(rows, self.baselines)
        if len(updates) != 180:
            raise AssertionError("Expected60 cumulative PPO updates per initialization")
        with q.transaction(final) as attempt:
            for name, records in (("per_scene", rows), ("per_training_seed", per_seed), ("summary", overall),
                                  ("paired_deltas", pairs), ("paired_vs120", origin_pairs),
                                  ("learning_curve", history), ("ppo_updates", updates)):
                append_csv(attempt / f"{name}.csv", records)
            lines = ["# PPO-only continuation to window600", "", "Final endpoint600, not best checkpoint.",
                     "Three paired continuations from their window120 endpoints; same frozen SAC101.",
                     "Only33600/33601 development evaluation. No joint learning or reserved evaluation.", "",
                     "| PPO seed | J | Holding | Energy | Weighted ms | High ms | Low ms | Power W | Tracking m |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in per_seed:
                lines.append(f"| {row['ppo_seed']} | " + " | ".join(f"{row[k]:.6f}" for k in
                    ("J", "J_holding", "J_energy", "task_delay_weighted_ms", "task_delay_high_ms",
                     "task_delay_low_ms", "power_w", "business_error_m_mean")) + " |")
            lines.extend(("", "All completion counts, queues, energy and comparator deltas remain in the CSVs.",
                          "Mean/std use three training-initialization means, not six independent training runs."))
            (attempt / "result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def preflight():
    saved, source_lock, baselines, references, head = q.preflight()
    manifest = q.read_json(SOURCE / "manifest.json")
    if (manifest.get("status") != "COMPLETE_3X120_STOPPED" or manifest.get("actual_head") != SOURCE_COMMIT
            or head != SOURCE_COMMIT or manifest["lock"] != source_lock):
        raise ValueError("Original completed qualification, configuration or runtime does not match")
    evidence = {str(SOURCE / "manifest.json"): q.file_hash(SOURCE / "manifest.json")}
    folders = [SOURCE / "final"]
    for seed in q.PPO_SEEDS:
        folders.extend(SOURCE / f"seed{seed}/{name}" for name in ("boundary000", "boundary040", "monitor000", "monitor120"))
        gate = q.read_json(SOURCE / f"seed{seed}/monitor000/gate.json")
        if set(gate) != {str(s) for s in q.VALIDATION_SCENES} or not all(g["full900s"] for g in gate.values()):
            raise ValueError("Historical window0 gate missing")
    for folder in folders:
        q.verify_transaction(folder)
        for path in folder.rglob("*"):
            if path.is_file():
                evidence[str(path)] = q.file_hash(path)
    lock = dict(protocol=protocol(), source_lock=source_lock, source_evidence=evidence,
                new_sources={name: q.file_hash(q.ROOT / name) for name in FILES},
                configs=source_lock["configs"], actual_head=head)
    return saved, lock, baselines, references


def verify_sources(lock):
    q.verify_sources(lock["source_lock"])
    for path, digest in {**lock["source_evidence"], **{str(q.ROOT / p): h for p, h in lock["new_sources"].items()}}.items():
        if q.file_hash(Path(path)) != digest:
            raise ValueError(f"Continuation source/evidence changed: {path}")


@contextmanager
def output_lease(output):
    # OS lock is released even on process failure; the marker itself is retained.
    with (Path(output) / ".runner.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(output, *, resume=False):
    output = Path(output).resolve()
    if output.exists() != resume:
        raise ValueError("Use a new output directory, or --resume on the same existing continuation")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with output_lease(output):
        saved, lock, baselines, references = preflight()
        if resume:
            manifest = q.read_json(output / "manifest.json")
            if manifest["lock"] != lock:
                raise ValueError("Continuation source/config/runtime/protocol changed")
            manifest["resume_count"] += 1
        else:
            manifest = dict(status="RUNNING", lock=lock, protocol=protocol(), actual_head=lock["actual_head"],
                branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=q.ROOT, text=True).strip(),
                training=True, continuation=True, resume_count=0, fresh_initializations=0,
                continued_ppo_seeds=list(q.PPO_SEEDS), source_window=120, final_window=600,
                forbidden_scenes_executed=[], sac_learning=False, source_unchanged=None)
        manifest["status"] = "RUNNING"
        q.atomic_json(output / "manifest.json", manifest)
        try:
            with isolated_rng():
                sac = ReadOnlySAC(saved)
            del saved
            gc.collect()
            runner = ContinuationRunner(output, lock, baselines, references)
            for seed in q.PPO_SEEDS:
                verify_sources(lock)
                policy = MissionMacroPPO(lock["configs"][str(seed)], seed, sac)
                runner.run_seed(policy)
                del policy
            verify_sources(lock)
            runner.finalize()
            manifest.update(status="COMPLETE_3X600_STOPPED", source_unchanged=True, frozen_sac=sac.assert_unchanged())
        except BaseException as exc:
            manifest.update(status="INTERRUPTED_OR_FAILED", error=f"{type(exc).__name__}: {exc}")
            q.atomic_json(output / "manifest.json", manifest)
            raise
        q.atomic_json(output / "manifest.json", manifest)
    return output
