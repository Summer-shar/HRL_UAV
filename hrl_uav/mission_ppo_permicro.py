"""Per-micro-controller PPO qualification: PPO trained under CVX/Random/Local feedback.

Same fixed protocol as the archived SAC qualification (3 fresh PPO seeds x 40
900 s streams, monitor windows 0/30/60/90/120, window120 endpoint only, no best
selection), except the micro routing controller is injected per stream via
collect_stream(..., micro_controller=...). SAC101 is loaded frozen and never
acts; its freeze is asserted at every update and stream boundary. Pairing uses
a hover reference run with the SAME micro controller, not the archived
SAC-micro baselines: only same-micro comparisons isolate the PPO contribution.
"""

import copy
import gc
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from torch import nn

from .agents.ppo_agent import ActorNet, CriticNet
from .agents.sac_agent import device
from .experiment import append_csv, isolated_rng, snapshot, write_json
from .micro_controllers import FixedRuleMicro, RandomMicro, cvx_micro
from .mission_pilot import crn_records, save_case
from .mission_ppo_qualification import (
    DEPENDENCIES as QUAL_DEP, HISTORY, METRICS, QualificationRunner,
    atomic_json, compare_records, read_json, transaction, verify_transaction,
)
from .mission_ppo_qualification import protocol as qual_protocol
from .mission_ppo_stage4a import (
    MONITORS, PPO_SEEDS, TRAIN_SCENES, VALIDATION_SCENES, MissionMacroPPO,
    Stage4Meter, collect_stream, frozen_sac_guard, stage_config, summarize,
    verify_baselines,
)
from .mission_provenance import assert_committed_sources, physics_provenance
from .mission_smoke import ROOT, load_source, load_spec
from .priority_grid import file_hash
from .resumption import capture_rng, state_equal
from .scene_prescreen import assert_crn_equal
from .trajectory_check import ReadOnlySAC
from .wireless_check import read_csv


MICROS = ("cvx", "random", "local")
MICRO_LABELS = {"cvx": "Xu-CVX-R", "random": "Random", "local": "Local"}
EXTRA_SOURCES = ("hrl_uav/micro_controllers.py", "hrl_uav/xu_cvx_r.py",
                 "hrl_uav/mission_xu_cvx_r.py", "hrl_uav/mission_ppo_permicro.py",
                 "scripts/run_mission_ppo_permicro.py", "tests/test_mission_ppo_permicro.py")
DEPENDENCIES = tuple(dict.fromkeys((*QUAL_DEP, *EXTRA_SOURCES)))


def stage_name(micro):
    if micro not in MICROS:
        raise ValueError(f"Unknown micro controller: {micro}")
    return f"mission_direct_ppo_micro_{micro}_qualification_stream_boundary_v1"


def protocol(micro, cfg):
    p = qual_protocol()
    training = cfg["training"]
    p.update(
        stage=stage_name(micro), micro_controller=MICRO_LABELS[micro],
        micro_learning="none; fixed rule or convex solver controller, no micro training",
        sac_role="SAC101 loaded frozen but never queried; freeze asserted per update/stream",
        baselines=["hover with the same injected micro controller"],
        window0_gate=("zero-action structural gate: bitwise UAV/user/centroid trajectory and "
                      "CRN vs archived hover, plus metric equality vs the same-micro hover reference"),
        ppo_settings={k: training[k] for k in (
            "ppo_initial_std", "ppo_final_std", "ppo_std_decay_windows", "ppo_update_macros",
            "ppo_epochs", "ppo_actor_lr", "ppo_critic_lr", "ppo_clip", "ppo_target_kl")},
        continuation="complete900s stream boundaries only; retry interrupted stream from last boundary",
        force_updates=False,
        aggregation="mean of two scenes per PPO initialization, then sample std over three initializations",
        implementation_only_until_separately_authorized=True)
    return p


def make_controller(micro, cfg, scene):
    """One fresh controller per stream; Random is seeded per scene for CRN pairing."""
    if micro == "random":
        return RandomMicro(scene)
    if micro == "local":
        return FixedRuleMicro("Local")
    if micro == "cvx":
        return cvx_micro(Stage4Meter(cfg))  # Probe meter supplies static constants only.
    raise ValueError(f"Unknown micro controller: {micro}")


def boundary_payload(policy, next_index, lock, stage):
    if type(next_index) is not int or next_index not in range(41):
        raise ValueError("Not a complete900s boundary")
    payload = dict(stage=stage, boundary=True, next_scene_index=next_index,
                   current_window=3*next_index, ppo_seed=policy.seed, config=policy.config,
                   lock=copy.deepcopy(lock), ppo=policy.ppo_state(), replay=None)
    validate_boundary(payload, policy, lock, stage)
    return payload


def validate_boundary(payload, policy, lock, stage):
    index = payload.get("next_scene_index")
    if (payload.get("stage") != stage or payload.get("boundary") is not True
            or type(index) is not int or index not in range(41)
            or payload.get("current_window") != index*3 or payload.get("replay", "missing") is not None
            or payload.get("ppo_seed") != policy.seed or payload.get("config") != policy.config
            or payload.get("lock") != lock):
        raise ValueError("Incompatible per-micro boundary/config/source")
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


def restore_boundary(policy, payload, lock, stage):
    validate_boundary(payload, policy, lock, stage)
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


def evaluate_full(policy, scene, micro):
    controller = make_controller(micro, policy.config, scene)
    before, rng = policy.ppo_state(), capture_rng()
    with isolated_rng(), frozen_sac_guard(policy.sac), torch.no_grad():
        result = collect_stream(policy.config, policy, scene, train=False, macros=90,
                                micro_controller=controller)
    if not state_equal(before, policy.ppo_state()) or not state_equal(rng, capture_rng()):
        raise AssertionError("Monitor changed PPO, pending rollout, gradients, optimizer or RNG")
    return result, controller


def window0_gate(result, row, reference, baseline_row=None):
    """Structural gate: exogenous trajectory/CRN must match archived hover bitwise.

    Queues, routes and metrics differ from archived hover by construction (the
    micro controller differs); when baseline_row is given, metrics must instead
    equal the same-micro hover reference at full precision.
    """
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
    compare_records(read_csv(reference / "commands.csv"), meter.commands, "commands")
    assert_crn_equal(read_json(reference / "crn.json"), crn_records(meter))
    gate = dict(full900s=True, zero_actions=True, exogenous_trajectory_bitwise=True,
                crn_equal_archived_hover=True, rtol=1e-12, atol=1e-8)
    if baseline_row is not None:
        compare_records([{k: baseline_row[k] for k in METRICS}], [row], "hover_same_micro")
        gate["metrics_equal_same_micro_hover"] = True
    return gate


def persist_case(directory, policy, scene, window, result, micro):
    summary, meter, macros, observations, windows, routes = result
    row, tracking = summarize(summary, meter, policy, scene)
    row.update({k: v for k, v in tracking[0].items() if k not in ("policy", "seed", "uav")})
    row.update(window=window, micro_controller=MICRO_LABELS[micro],
               u2u_tx_time_s=sum(r["tx_time_s"] for r in meter.link_readings))
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
    np.savez_compressed(directory / "micro_actions.npz",
                        raw=[x[0] for x in routes], executed=[x[1] for x in routes])
    return row


def aggregate(final_rows, baselines):
    expected = {(s, e) for s in PPO_SEEDS for e in VALIDATION_SCENES}
    if (len(final_rows) != 6 or {(r["ppo_seed"], r["seed"]) for r in final_rows} != expected
            or any(r["window"] != 120 for r in final_rows)):
        raise ValueError("Final aggregation requires exactly three window120 policies x two scenes")
    if (len(baselines) != 2 or {(r["policy"], r["seed"]) for r in baselines}
            != {("hover", s) for s in VALIDATION_SCENES}):
        raise ValueError("Missing/duplicate same-micro hover references")
    per_seed, overall, pairs = [], [], []
    for seed in PPO_SEEDS:
        rows = [r for r in final_rows if r["ppo_seed"] == seed]
        per_seed.append(dict(policy="direct_ppo", ppo_seed=seed, scene_count=2,
                             **{k: float(np.mean([r[k] for r in rows])) for k in METRICS}))
        for row in rows:
            baseline = next(b for b in baselines if b["seed"] == row["seed"])
            pairs.append(dict(ppo_seed=seed, seed=row["seed"], comparator="hover",
                              **{k: row[k]-baseline[k] for k in METRICS}))
    for key in METRICS:
        values = [r[key] for r in per_seed]
        overall.append(dict(policy="direct_ppo", metric=key, training_initializations=3, scene_count=2,
                            mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                            min=min(values), max=max(values)))
        values = [b[key] for b in baselines]
        overall.append(dict(policy="hover", metric=key, training_initializations=0, scene_count=2,
                            mean=float(np.mean(values)), std=None, min=None, max=None))
    return per_seed, overall, pairs


class HoverMacro:
    """Zero-displacement macro control for the same-micro hover reference."""

    def __init__(self, config, sac):
        self.config, self.cfg = config, config["training"]
        self.seed = 0
        self.sac = sac
        self.ppo_updates = 0
        with isolated_rng():
            self.macro_actor = ActorNet(67, 6).to(device)
            self.macro_critic = CriticNet(67).to(device)
            nn.init.zeros_(self.macro_actor.out.weight)
            nn.init.zeros_(self.macro_actor.out.bias)
            self.macro_actor.log_std.requires_grad_(False)
            self.macro_actor.log_std.data.fill_(np.log(self.cfg["ppo_initial_std"]))
        self.initial_actor = snapshot(self.macro_actor)

    def macro_action(self, state, enabled=True, deterministic=False):
        if not deterministic or not enabled:
            raise ValueError("The hover reference is deterministic only")
        if np.asarray(state).shape != (67,):
            raise ValueError("Hover reference expects the 67D macro observation")
        return np.zeros(6, dtype=np.float32), 0.0


class PerMicroRunner(QualificationRunner):
    """QualificationRunner with per-stream micro injection and parameterized boundaries."""

    def __init__(self, output, lock, baselines, references, *, micro):
        self.micro, self.stage = micro, stage_name(micro)
        super().__init__(output, lock, baselines, references,
                         evaluator=lambda policy, scene: evaluate_full(policy, scene, micro)[0],
                         gate=self._gate)

    def _gate(self, result, row, reference):
        baseline = next(b for b in self.baselines if b["seed"] == row["seed"])
        return window0_gate(result, row, reference, baseline)

    def run_seed(self, policy):
        root = self.output / f"seed{policy.seed}"
        root.mkdir(exist_ok=True)
        initial = root / "boundary000"
        if not initial.exists():
            if any(root.glob("boundary[0-9][0-9][0-9]")):
                raise ValueError("Missing initial boundary")
            with transaction(initial) as attempt:
                torch.save(boundary_payload(policy, 0, self.lock, self.stage),
                           attempt / "checkpoint.pt")
        indices = sorted(int(p.name[-3:]) for p in root.glob("boundary[0-9][0-9][0-9]"))
        if indices != list(range(max(indices)+1)) or max(indices) > 40:
            raise ValueError("Noncontiguous committed stream boundaries")
        for index in indices:
            verify_transaction(root / f"boundary{index:03d}")
        path = root / f"boundary{indices[-1]:03d}" / "checkpoint.pt"
        index = restore_boundary(policy, torch.load(path, map_location="cpu", weights_only=False),
                                 self.lock, self.stage)
        if index != indices[-1]:
            raise ValueError("Boundary directory/counter mismatch")
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
                scene = TRAIN_SCENES[index]
                controller = make_controller(self.micro, policy.config, scene)
                with frozen_sac_guard(policy.sac), policy.exploration_rng():
                    result = collect_stream(policy.config, policy, scene, train=True, macros=90,
                                            macro_offset=index*90, micro_controller=controller)
                if policy.seed != PPO_SEEDS[0]:
                    reference = self.output / f"seed{PPO_SEEDS[0]}/boundary{index+1:03d}/stream/crn.json"
                    assert_crn_equal(read_json(reference), crn_records(result[1]))
                row = persist_case(attempt / "stream", policy, scene, (index+1)*3, result, self.micro)
                if (row["duration_s"] != 900 or any(r["forced"] or r["transitions"] != 300
                                                   for r in policy.update_log)):
                    raise AssertionError("Formal runner must use900s streams and full unforced300-sample updates")
                append_csv(attempt / "ppo_updates.csv", policy.update_log)
                if len(policy.update_log) != len(policy.batch_log):
                    raise AssertionError("PPO update journal is incomplete")
                for update, batch in zip(policy.update_log, policy.batch_log):
                    np.savez_compressed(attempt / f"batch{update['update']:02d}.npz", **batch)
                write_json(attempt / "micro_audit.json", controller.assert_unchanged())
                torch.save(boundary_payload(policy, index+1, self.lock, self.stage),
                           attempt / "checkpoint.pt")
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
            label = MICRO_LABELS[self.micro]
            lines = [f"# Per-micro PPO qualification: PPO+{label}", "",
                     "Final endpoint: window120 only. Each PPO seed first averages33600/33601; "
                     "std is sample std across201/202/203.",
                     f"The micro controller is {label} during BOTH training and evaluation; "
                     "SAC101 stays frozen and unused.",
                     "The hover comparator runs the same micro controller. "
                     "These are reused development scenes.", "",
                     "| Controller | PPO seed | J | Holding | Energy | Weighted ms | High ms | Low ms | Power W | Tracking m |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            display = per_seed + [dict(policy="hover", ppo_seed="fixed", **{
                k: float(np.mean([b[k] for b in self.baselines])) for k in METRICS})]
            for row in display:
                keys = ("J", "J_holding", "J_energy", "task_delay_weighted_ms", "task_delay_high_ms",
                        "task_delay_low_ms", "power_w", "business_error_m_mean")
                lines.append(f"| {row['policy']} | {row['ppo_seed']} | " + " | ".join(f"{row[k]:.6f}" for k in keys) + " |")
            lines += ["", "See per_scene and paired_deltas for completion counts/rates, queues and energy tradeoffs.",
                      "No automatic extension, joint learning, best-checkpoint selection or reserved evaluation."]
            (attempt / "result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_baselines(output, lock, micro, sac, references):
    """Same-micro hover references on both validation scenes; transaction-committed."""
    rows = []
    for scene in VALIDATION_SCENES:
        destination = output / "baselines" / f"hover_{scene}"
        if destination.exists():
            verify_transaction(destination)
            rows.append(read_json(destination / "summary.json"))
            continue
        cfg = lock["configs"][str(PPO_SEEDS[0])]
        with transaction(destination) as attempt:
            policy = HoverMacro(cfg, sac)
            controller = make_controller(micro, cfg, scene)
            rng = capture_rng()
            with isolated_rng(), frozen_sac_guard(sac), torch.no_grad():
                result = collect_stream(cfg, policy, scene, train=False, macros=90,
                                        micro_controller=controller)
            if not state_equal(rng, capture_rng()):
                raise AssertionError("Hover reference changed RNG")
            row = persist_case(attempt, policy, scene, 0, result, micro)
            row.update(policy="hover", ppo_seed="none")
            write_json(attempt / "summary.json", row)
            gate = window0_gate(result, row, HISTORY / f"hover_{scene}")
            write_json(attempt / "gate.json", gate)
            write_json(attempt / "micro_audit.json", controller.assert_unchanged())
            assert_crn_equal(references[("hover", scene)], crn_records(result[1]))
        rows.append(row)
    return rows


def preflight(micro, *, require_committed=True):
    physics = physics_provenance()
    head = (assert_committed_sources(DEPENDENCIES) if require_committed else
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    spec = load_spec()
    saved, source_manifest = load_source(spec)
    prior, references, hashes = verify_baselines(saved["config"], spec)
    for name in ("hover",):
        for scene in VALIDATION_SCENES:
            folder = HISTORY / f"{name}_{scene}"
            if not all(read_json(folder / "freeze_check.json").values()):
                raise ValueError("Historical frozen-SAC check failed")
            row = read_json(folder / "summary.json")
            for key in METRICS:
                if not np.isfinite(row[key]):
                    raise ValueError(f"Missing baseline metric: {name}/{scene}/{key}")
            for path in folder.iterdir():
                if path.is_file():
                    hashes[str(path)] = file_hash(path)
    checkpoint = ROOT / spec["source_checkpoint"]
    hashes[str(checkpoint)] = file_hash(checkpoint)
    hashes[str(checkpoint.parent.parent / "manifest.json")] = file_hash(checkpoint.parent.parent / "manifest.json")
    configs = {str(s): stage_config(saved["config"], s) for s in PPO_SEEDS}
    lock = dict(protocol=protocol(micro, configs[str(PPO_SEEDS[0])]), physics=physics,
                sources={p: file_hash(ROOT / p) for p in DEPENDENCIES}, evidence=hashes,
                configs=configs,
                source_checkpoint=spec["source_checkpoint"], source_sha256=spec["checkpoint_sha256"],
                source_config=source_manifest["config"], baseline_head=prior["actual_head"],
                runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                             cuda=torch.version.cuda, device_count=torch.cuda.device_count(),
                             devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                             deterministic=torch.are_deterministic_algorithms_enabled(),
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             cudnn_benchmark=torch.backends.cudnn.benchmark))
    return saved, lock, references, head


def verify_sources(lock):
    if physics_provenance() != lock["physics"]:
        raise ValueError("Physics changed during qualification")
    for path, digest in {**{str(ROOT / p): h for p, h in lock["sources"].items()}, **lock["evidence"]}.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f"Source or reused result changed: {path}")


def run(output, micro, *, resume=False, require_committed=True):
    """Calling this function starts the full preregistered budget; not used in acceptance."""
    if micro not in MICROS:
        raise ValueError(f"micro must be one of {MICROS}")
    output = Path(output).resolve()
    if output.exists() != resume:
        raise ValueError("Use a new output directory, or explicit --resume on its existing run")
    saved, lock, references, head = preflight(micro, require_committed=require_committed)
    if resume:
        manifest = read_json(output / "manifest.json")
        if manifest.get("lock") != lock:
            raise ValueError("Resume source/config/protocol/runtime lock differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(status="RUNNING", actual_head=head, lock=lock,
                        micro_controller=MICRO_LABELS[micro],
                        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                        training=True, planned_training_initializations=list(PPO_SEEDS),
                        frozen_sac_seed=101, reserved_scenes_executed=[], source_unchanged=None)
        atomic_json(output / "manifest.json", manifest)
    try:
        with isolated_rng():
            sac = ReadOnlySAC(saved)
        del saved
        gc.collect()
        baselines = run_baselines(output, lock, micro, sac, references)
        manifest["micro_baselines"] = {
            str(scene): file_hash(output / "baselines" / f"hover_{scene}" / "summary.json")
            for scene in VALIDATION_SCENES}
        atomic_json(output / "manifest.json", manifest)
        runner = PerMicroRunner(output, lock, baselines, references, micro=micro)
        for seed in PPO_SEEDS:
            verify_sources(lock)
            policy = MissionMacroPPO(lock["configs"][str(seed)], seed, sac)
            runner.run_seed(policy)
            del policy
        verify_sources(lock)
        runner.finalize()
        manifest.update(status="COMPLETE_3X120_STOPPED", source_unchanged=True,
                        frozen_sac=sac.assert_unchanged(),
                        completed_training_initializations=list(PPO_SEEDS), final_window=120)
    except BaseException as exc:
        manifest.update(status="INTERRUPTED_OR_FAILED", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    return output
