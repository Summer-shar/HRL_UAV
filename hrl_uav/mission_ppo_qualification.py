"""Fixed Stage4A scheduler and stream-boundary recovery, not new learning math."""

from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import platform
import subprocess
import uuid

import numpy as np
import torch

from .experiment import append_csv, isolated_rng, write_json
from .mission_pilot import crn_records, save_case
from .mission_ppo_stage4a import (
    BASELINES, MONITORS, NEW_FILES, PPO_SEEDS, TRAIN_SCENES, VALIDATION_SCENES,
    MissionMacroPPO, collect_stream, frozen_sac_guard, protocol as smoke_protocol,
    stage_config, summarize, verify_baselines,
)
from .mission_provenance import PHYSICS_FILES, assert_committed_sources, physics_provenance
from .mission_smoke import ROOT, load_source, load_spec
from .priority_grid import file_hash
from .resumption import capture_rng, state_equal
from .scene_prescreen import assert_crn_equal
from .trajectory_check import ReadOnlySAC
from .wireless_check import ENERGY, PRIORITY_MEASURES, read_csv


ARCHIVED_STAGE4A = "cfffe7dfca8cfe81e527dfd1636fd95bfcafc8f2"
STAGE = "mission_direct_ppo_qualification_stream_boundary_v1"
FILES = (*NEW_FILES, "hrl_uav/mission_ppo_qualification.py",
         "scripts/run_mission_ppo_qualification.py", "tests/test_mission_ppo_qualification.py",
         "docs/mission_ppo_qualification.md")
DEPENDENCIES = tuple(dict.fromkeys((*PHYSICS_FILES, *FILES,
    "hrl_uav/mission_smoke.py", "hrl_uav/mission_pilot.py", "hrl_uav/mission_pilot_analysis.py",
    "hrl_uav/mission_micro_pilot.py", "hrl_uav/mission_provenance.py", "hrl_uav/resumption.py",
    "hrl_uav/qualification.py", "hrl_uav/priority_grid.py")))
HISTORY = ROOT / "results/mission_rpgm_v1_pilot"
METRICS = (
    "J", "J_holding", "J_energy", "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
    "task_completion_rate", *PRIORITY_MEASURES, "unfinished", "queue_total_mbit",
    "queue_upload_mbit", "queue_compute_mbit", "queue_transfer_mbit",
    "mean_queue_total_mbit", "mean_queue_upload_mbit", "mean_queue_compute_mbit", "mean_queue_transfer_mbit",
    "late_queue_slope_mbit_s", "power_w", *ENERGY, "business_error_m_mean", "group_error_m_mean",
    "distance_m", "speed_mps_mean", "speed_mps_p90", "actual_at_20m_row_fraction",
    "u2g_rate_time_weighted_mbps", "u2u_rate_time_weighted_mbps", "u2u_tx_time_s",
    "tx_throughput_mbps_wall", "compute_throughput_mbps_wall", "cpu_utilization_service",
)


def protocol():
    p = smoke_protocol()
    p.pop("smoke")
    p.pop("formal_training_authorized")
    p.update(stage=STAGE, implementation_only_until_separately_authorized=True,
             continuation="complete900s stream boundaries only; retry interrupted stream from last boundary",
             window0_gate="two full900s deterministic evaluations against archived Hover, before learning",
             force_updates=False, archived_stage4a=ARCHIVED_STAGE4A,
             aggregation="mean of two scenes per PPO initialization, then sample std over three initializations")
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, data)
    temporary.replace(path)


@contextmanager
def transaction(destination):
    """Commit a complete artifact directory; retain interrupted attempts for inspection."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = destination.with_name(".attempt-" + destination.name + "-" + uuid.uuid4().hex)
    attempt.mkdir()
    yield attempt
    hashes = {p.relative_to(attempt).as_posix(): file_hash(p)
              for p in attempt.rglob("*") if p.is_file()}
    write_json(attempt / "complete.json", dict(files=hashes))
    # Same-parent rename only; no pre-existing output is replaced.
    attempt.rename(destination)


def verify_transaction(directory):
    directory = Path(directory)
    hashes = read_json(directory / "complete.json")["files"]
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p != directory / "complete.json"}
    if actual != set(hashes):
        raise ValueError(f"Incomplete or unexpected artifact files: {directory}")
    for name, digest in hashes.items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {directory / name}")


def boundary_payload(policy, next_index, lock):
    if type(next_index) is not int or next_index not in range(41):
        raise ValueError("Not a complete900s boundary")
    payload = dict(stage=STAGE, boundary=True, next_scene_index=next_index,
                   current_window=3*next_index, ppo_seed=policy.seed, config=policy.config,
                   lock=copy.deepcopy(lock), ppo=policy.ppo_state(), replay=None)
    validate_boundary(payload, policy, lock)
    return payload


def validate_boundary(payload, policy, lock):
    index = payload.get("next_scene_index")
    if (payload.get("stage") != STAGE or payload.get("boundary") is not True
            or type(index) is not int or index not in range(41)
            or payload.get("current_window") != index*3 or payload.get("replay", "missing") is not None
            or payload.get("ppo_seed") != policy.seed or payload.get("config") != policy.config
            or payload.get("lock") != lock):
        raise ValueError("Incompatible Stage4A boundary/config/source; smoke checkpoints cannot resume")
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
    if updates == 0 and any(o["state"] for o in state["optimizers"]):
        raise ValueError("Initial PPO optimizers must be fresh")
    cfg = policy.cfg
    fraction = min(1., (consumed/30)/cfg["ppo_std_decay_windows"])
    log_std = (1-fraction)*np.log(cfg["ppo_initial_std"]) + fraction*np.log(cfg["ppo_final_std"])
    np.testing.assert_allclose(state["actor"]["log_std"].cpu().numpy(), log_std, rtol=0, atol=2e-7)


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


def evaluate_full(policy, scene):
    before, rng = policy.ppo_state(), capture_rng()
    with isolated_rng(), frozen_sac_guard(policy.sac), torch.no_grad():
        result = collect_stream(policy.config, policy, scene, train=False, macros=90)
    if not state_equal(before, policy.ppo_state()) or not state_equal(rng, capture_rng()):
        raise AssertionError("Monitor changed PPO, pending rollout, gradients, optimizer or RNG")
    return result


def compare_records(expected, actual, label):
    if len(expected) != len(actual) or not expected:
        raise AssertionError(f"{label}: missing/different record count")
    for left, right in zip(expected, actual):
        for key, value in left.items():
            if key not in right:
                raise AssertionError(f"{label}: missing column {key}")
            try:
                numeric = float(value)
            except (ValueError, TypeError):
                if str(value) != str(right[key]):
                    raise AssertionError(f"{label}: {key} mismatch")
            else:
                if not np.isfinite(numeric) or not np.isfinite(float(right[key])):
                    raise AssertionError(f"{label}: nonfinite {key}")
                np.testing.assert_allclose(float(right[key]), numeric, rtol=1e-12, atol=1e-8,
                                           err_msg=f"{label}: {key}")


def window0_gate(result, row, reference):
    """Fail closed on full trajectory/state/route/accounting differences, not just mean J."""
    summary, meter, _, observations, _, _ = result
    if summary["time_s"] != 900 or len(observations) != 90:
        raise AssertionError("Window0 gate requires the full900s")
    if (len(meter.positions) != 9001 or len(meter.micro_states) != 1800
            or len(meter.readings) != 1800 or len(meter.commands) != 270
            or len(meter.route_readings) != 10800):
        raise AssertionError("Incomplete window0 trajectory/state/queue/route records")
    np.testing.assert_array_equal([o[1] for o in observations], np.zeros((90, 6)))
    reference = Path(reference)
    with np.load(reference / "trajectory.npz", allow_pickle=False) as trace:
        for index, key in enumerate(("time_s", "uav", "users", "centroid", "groups")):
            np.testing.assert_array_equal([p[index] for p in meter.positions], trace[key])
        np.testing.assert_array_equal(meter.micro_states, trace["micro_state"])
    for name, actual in (("commands", meter.commands), ("queues", meter.readings),
                         ("routes", meter.route_readings)):
        compare_records(read_csv(reference / f"{name}.csv"), actual, name)
    old = read_json(reference / "summary.json")
    compare_records([{k: old[k] for k in METRICS}], [row], "summary")
    assert_crn_equal(read_json(reference / "crn.json"), crn_records(meter))
    return dict(full900s=True, zero_actions=True, trajectory_and_micro_state_bitwise=True,
                routes_queues_completion_energy_match=True, rtol=1e-12, atol=1e-8)


def persist_case(directory, policy, scene, window, result):
    summary, meter, macros, observations, windows, routes = result
    row, tracking = summarize(summary, meter, policy, scene)
    row.update({k: v for k, v in tracking[0].items() if k not in ("policy", "seed", "uav")})
    row.update(window=window, u2u_tx_time_s=sum(r["tx_time_s"] for r in meter.link_readings))
    row["u2u_rate_time_weighted_mbps"] = row["tx_served_mbit"]/max(row["u2u_tx_time_s"], 1e-12)
    for key in METRICS:
        if not np.isfinite(row[key]):
            raise AssertionError(f"Missing/nonfinite reporting metric: {key}")
    save_case(directory, "direct_ppo", scene, row, meter, tracking, macros, policy.sac)
    append_csv(directory / "windows.csv", windows)
    fields = zip(*observations)
    np.savez_compressed(directory / "macro_samples.npz", **dict(zip(
        ("state", "raw_action", "old_logprob", "reward", "next_state", "continuation"),
        (np.asarray(x) for x in fields))))
    np.savez_compressed(directory / "sac_actions.npz", raw=[x[0] for x in routes], executed=[x[1] for x in routes])
    return row


def aggregate(final_rows, baselines):
    expected = {(s, e) for s in PPO_SEEDS for e in VALIDATION_SCENES}
    if (len(final_rows) != 6 or {(r["ppo_seed"], r["seed"]) for r in final_rows} != expected
            or any(r["window"] != 120 for r in final_rows)):
        raise ValueError("Final aggregation requires exactly three window120 policies x two scenes")
    if len(baselines) != 8 or {(r["policy"], r["seed"]) for r in baselines} != {
            (n, s) for n in BASELINES for s in VALIDATION_SCENES}:
        raise ValueError("Missing/duplicate frozen comparators")
    per_seed, overall, pairs = [], [], []
    for seed in PPO_SEEDS:
        rows = [r for r in final_rows if r["ppo_seed"] == seed]
        per_seed.append(dict(policy="direct_ppo", ppo_seed=seed, scene_count=2,
                             **{k: float(np.mean([r[k] for r in rows])) for k in METRICS}))
        for row in rows:
            for baseline in (b for b in baselines if b["seed"] == row["seed"]):
                pairs.append(dict(ppo_seed=seed, seed=row["seed"], comparator=baseline["policy"],
                                  **{k: row[k]-baseline[k] for k in METRICS}))
    for key in METRICS:
        values = [r[key] for r in per_seed]
        overall.append(dict(policy="direct_ppo", metric=key, training_initializations=3, scene_count=2,
                            mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                            min=min(values), max=max(values)))
        for name in BASELINES:
            values = [b[key] for b in baselines if b["policy"] == name]
            overall.append(dict(policy=name, metric=key, training_initializations=0, scene_count=2,
                                mean=float(np.mean(values)), std=None, min=None, max=None))
    return per_seed, overall, pairs


class QualificationRunner:
    """A committed directory is the only recovery boundary; partial attempts never count."""

    def __init__(self, output, lock, baselines, references, *, collector=collect_stream,
                 evaluator=evaluate_full, persister=persist_case, gate=window0_gate):
        self.output, self.lock = Path(output), lock
        self.baselines, self.references = baselines, references
        self.collector, self.evaluator, self.persister, self.gate = collector, evaluator, persister, gate

    def monitor(self, policy, window, root):
        destination = root / f"monitor{window:03d}"
        if destination.exists():
            verify_transaction(destination)
            state = torch.load(destination / "ppo.pt", map_location="cpu", weights_only=False)
            if not state_equal(state, policy.ppo_state()):
                raise ValueError("Monitor belongs to a different PPO boundary state")
            return
        with transaction(destination) as attempt:
            gates, pairs = {}, []
            for scene in VALIDATION_SCENES:
                result = self.evaluator(policy, scene)
                assert_crn_equal(self.references[("hover", scene)], crn_records(result[1]))
                row = self.persister(attempt / str(scene), policy, scene, window, result)
                if window == 0:
                    gates[str(scene)] = self.gate(result, row, HISTORY / f"hover_{scene}")
                for old in (b for b in self.baselines if b["seed"] == scene):
                    pairs.append(dict(window=window, ppo_seed=policy.seed, seed=scene, comparator=old["policy"],
                                      **{k: row[k]-old[k] for k in METRICS}))
                del result
            append_csv(attempt / "paired_deltas.csv", pairs)
            write_json(attempt / "gate.json", gates)
            torch.save(policy.ppo_state(), attempt / "ppo.pt")

    def run_seed(self, policy):
        root = self.output / f"seed{policy.seed}"
        root.mkdir(exist_ok=True)
        initial = root / "boundary000"
        if not initial.exists():
            if any(root.glob("boundary[0-9][0-9][0-9]")):
                raise ValueError("Missing initial boundary")
            with transaction(initial) as attempt:
                torch.save(boundary_payload(policy, 0, self.lock), attempt / "checkpoint.pt")
        indices = sorted(int(p.name[-3:]) for p in root.glob("boundary[0-9][0-9][0-9]"))
        if indices != list(range(max(indices)+1)) or max(indices) > 40:
            raise ValueError("Noncontiguous committed stream boundaries")
        for index in indices:
            verify_transaction(root / f"boundary{index:03d}")
        path = root / f"boundary{indices[-1]:03d}" / "checkpoint.pt"
        index = restore_boundary(policy, torch.load(path, map_location="cpu", weights_only=False), self.lock)
        if index != indices[-1]:
            raise ValueError("Boundary directory/counter mismatch")
        # Every previous monitor must be complete before a later stream can exist.
        for window in MONITORS:
            if window < index*3:
                verify_transaction(root / f"monitor{window:03d}")
        while True:
            if index*3 in MONITORS:
                self.monitor(policy, index*3, root)
            if index == 40:
                break
            destination = root / f"boundary{index+1:03d}"
            with transaction(destination) as attempt:
                policy.batch_log.clear()
                policy.update_log.clear()
                with frozen_sac_guard(policy.sac), policy.exploration_rng():
                    result = self.collector(policy.config, policy, TRAIN_SCENES[index],
                                            train=True, macros=90, macro_offset=index*90)
                if policy.seed != PPO_SEEDS[0]:
                    reference = self.output / f"seed{PPO_SEEDS[0]}/boundary{index+1:03d}/stream/crn.json"
                    assert_crn_equal(read_json(reference), crn_records(result[1]))
                row = self.persister(attempt / "stream", policy, TRAIN_SCENES[index], (index+1)*3, result)
                if (row["duration_s"] != 900 or any(r["forced"] or r["transitions"] != 300
                                                   for r in policy.update_log)):
                    raise AssertionError("Formal runner must use900s streams and full unforced300-sample updates")
                append_csv(attempt / "ppo_updates.csv", policy.update_log)
                if len(policy.update_log) != len(policy.batch_log):
                    raise AssertionError("PPO update journal is incomplete")
                for update, batch in zip(policy.update_log, policy.batch_log):
                    np.savez_compressed(attempt / f"batch{update['update']:02d}.npz", **batch)
                torch.save(boundary_payload(policy, index+1, self.lock), attempt / "checkpoint.pt")
                del result
            index += 1
        if policy.ppo_updates != 12 or policy.collected_macros != 3600 or policy.macro_rollout:
            raise AssertionError("Final qualification budget mismatch")

    def finalize(self):
        final = self.output / "final"
        if final.exists():
            verify_transaction(final)
            return
        rows = []
        for seed in PPO_SEEDS:
            folder = self.output / f"seed{seed}/monitor120"
            verify_transaction(folder)
            rows.extend(read_json(folder / str(scene) / "summary.json") for scene in VALIDATION_SCENES)
        per_seed, overall, pairs = aggregate(rows, self.baselines)
        with transaction(final) as attempt:
            for name, values in (("per_scene", rows), ("per_training_seed", per_seed),
                                 ("summary", overall), ("paired_deltas", pairs)):
                append_csv(attempt / f"{name}.csv", values)
            lines = ["# Stage4A development qualification", "", "Final endpoint: window120 only.",
                     "Each PPO seed first averages33600/33601; std is sample std across201/202/203.",
                     "Fixed comparators have no training-initialization std. These are reused development scenes.", "",
                     "| Controller | PPO seed | J | Holding | Energy | Weighted ms | High ms | Low ms | Power W | Tracking m |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            display = per_seed + [dict(policy=n, ppo_seed="fixed", **{
                k: float(np.mean([b[k] for b in self.baselines if b["policy"] == n])) for k in METRICS}) for n in BASELINES]
            for row in display:
                keys = ("J", "J_holding", "J_energy", "task_delay_weighted_ms", "task_delay_high_ms",
                        "task_delay_low_ms", "power_w", "business_error_m_mean")
                lines.append(f"| {row['policy']} | {row['ppo_seed']} | " + " | ".join(f"{row[k]:.6f}" for k in keys) + " |")
            lines += ["", "See per_scene and paired_deltas for completion counts/rates, queues and energy tradeoffs.",
                      "No automatic extension, joint learning, best-checkpoint selection or reserved evaluation."]
            (attempt / "result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def preflight(*, require_committed=True):
    physics = physics_provenance()
    head = (assert_committed_sources(DEPENDENCIES) if require_committed else
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    spec = load_spec()
    saved, source_manifest = load_source(spec)
    prior, references, hashes = verify_baselines(saved["config"], spec)
    baselines = []
    previous_csv = read_csv(HISTORY / "per_seed.csv")
    for name in BASELINES:
        for scene in VALIDATION_SCENES:
            folder = HISTORY / f"{name}_{scene}"
            if not all(read_json(folder / "freeze_check.json").values()):
                raise ValueError("Historical frozen-SAC check failed")
            row = read_json(folder / "summary.json")
            for key in METRICS:
                if not np.isfinite(row[key]):
                    raise ValueError(f"Missing baseline metric: {name}/{scene}/{key}")
            matches = [r for r in previous_csv if r["policy"] == name and int(r["seed"]) == scene]
            compare_records([{k: row[k] for k in METRICS}], matches, "historical summary/CSV")
            assert_crn_equal(references[("hover", scene)], references[(name, scene)])
            baselines.append(row)
            for path in folder.iterdir():
                if path.is_file():
                    hashes[str(path)] = file_hash(path)
    checkpoint = ROOT / spec["source_checkpoint"]
    hashes[str(checkpoint)] = file_hash(checkpoint)
    hashes[str(checkpoint.parent.parent / "manifest.json")] = file_hash(checkpoint.parent.parent / "manifest.json")
    lock = dict(protocol=protocol(), physics=physics,
                sources={p: file_hash(ROOT / p) for p in DEPENDENCIES}, evidence=hashes,
                configs={str(s): stage_config(saved["config"], s) for s in PPO_SEEDS},
                source_checkpoint=spec["source_checkpoint"], source_sha256=spec["checkpoint_sha256"],
                source_config=source_manifest["config"], baseline_head=prior["actual_head"],
                runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                             cuda=torch.version.cuda, device_count=torch.cuda.device_count(),
                             devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                             deterministic=torch.are_deterministic_algorithms_enabled(),
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             cudnn_benchmark=torch.backends.cudnn.benchmark))
    return saved, lock, baselines, references, head


def verify_sources(lock):
    if physics_provenance() != lock["physics"]:
        raise ValueError("Physics changed during qualification")
    for path, digest in {**{str(ROOT / p): h for p, h in lock["sources"].items()}, **lock["evidence"]}.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f"Source or reused result changed: {path}")


def run(output, *, resume=False, require_committed=True):
    """Calling this function starts the full preregistered budget; not used in acceptance."""
    output = Path(output).resolve()
    if output.exists() != resume:
        raise ValueError("Use a new output directory, or explicit --resume on its existing run")
    saved, lock, baselines, references, head = preflight(require_committed=require_committed)
    if resume:
        manifest = read_json(output / "manifest.json")
        if manifest.get("lock") != lock:
            raise ValueError("Resume source/config/protocol/runtime lock differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(status="RUNNING", actual_head=head, lock=lock, protocol=protocol(),
                        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                        training=True, planned_training_initializations=list(PPO_SEEDS),
                        frozen_sac_seed=101, reserved_scenes_executed=[], source_unchanged=None)
        atomic_json(output / "manifest.json", manifest)
    try:
        with isolated_rng():
            sac = ReadOnlySAC(saved)
        del saved
        gc.collect()
        runner = QualificationRunner(output, lock, baselines, references)
        for seed in PPO_SEEDS:
            verify_sources(lock)
            policy = MissionMacroPPO(lock["configs"][str(seed)], seed, sac)
            runner.run_seed(policy)
            del policy
        verify_sources(lock)
        runner.finalize()
        manifest.update(status="COMPLETE_3X120_STOPPED", source_unchanged=True, frozen_sac=sac.assert_unchanged(),
                        completed_training_initializations=list(PPO_SEEDS), final_window=120)
    except BaseException as exc:
        manifest.update(status="INTERRUPTED_OR_FAILED", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    return output
