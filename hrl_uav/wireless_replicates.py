"""Fresh independent SAC initializations on the frozen seed101 scenario sequence."""

from contextlib import redirect_stdout
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .experiment import append_csv, parameter_delta, rollout, write_json
from .mdp import QueueMDP
from .priority_grid import file_hash
from .resumption import capture_rng, state_equal
from .scene_prescreen import ROOT, assert_crn_equal, source_hashes
from .wireless_long import PILOT, cached_results, rows, stream_plan, window_optimization_summary
from .wireless_pilot import (ENERGY, MEASURES, OPT_KEYS, PilotMeter, checked_agent,
                             evaluate_one, grouped_means, route_statistics)


REFERENCE = ROOT / "results/sac_only_wireless6_long_seed101"
OBSERVE = (120, 200, 400, 800, 1200, 1600, 2000)
INITIALIZATIONS = (102, 103)
SCENES = (33600, 33601, 33610, 33611, 33612)
SUMMARY = ROOT / "results/sac_only_wireless6_multiseed_summary"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fresh_config(reference, seed):
    if seed not in INITIALIZATIONS:
        raise ValueError("Only the two authorized fresh initializations are supported")
    cfg = copy.deepcopy(reference)
    cfg["training"]["seed"] = seed
    if (cfg["training"]["pretrain_windows"] != 2000 or cfg["training"]["joint_windows"] != 0
            or cfg["pilot"]["training_scene_seed_base"] != 71000):
        raise ValueError("Frozen budget or environment seed schedule differs")
    return cfg


def fresh_agent(cfg, output):
    seed = cfg["training"]["seed"]
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = QueueMDP(cfg)
    model.reset(71000, 90)
    agent, macro, actor, counters = checked_agent(model, cfg, output)
    score = agent.sac_actor.choose_action(model.micro_observation(), deterministic=True)
    saved = agent.checkpoint(cfg)
    error = float(np.max(np.abs(model.route(score) - 1 / 3)))
    initial = dict(checkpoint=None, parameter_source="fresh native initialization", teacher=False,
        training_seed=seed, first_training_scene=71000, uniform_max_abs_error=error,
        replay_count=agent.replay.memory_counter, replay_transition_count=len(saved["replay"]["transitions"]),
        optimizer_state_sizes=[len(s["state"]) for s in saved["optimizers"]],
        updates=saved["updates"], alpha=agent.entropy.alpha.detach().item(),
        log_alpha=agent.entropy.log_alpha.detach().item(), target_entropy=agent.entropy.target_entropy,
        learning_rates=[s["param_groups"][0]["lr"] for s in saved["optimizers"][:3]],
        stage=agent.stage, stage_initial_updates=agent.stage_initial_updates,
        actor_sha256=hashlib.sha256(b"".join(v.numpy().tobytes() for v in actor.values())).hexdigest(),
        critic_sha256=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
            for v in saved["sac_critic"].values())).hexdigest(),
        native_zero_output_heads=True, observations=agent.observation_spec,
        replay_rng_seed=seed + 731, device=str(next(agent.macro_actor.parameters()).device))
    checks = dict(empty_replay=initial["replay_count"] == initial["replay_transition_count"] == 0,
        empty_optimizers=not any(initial["optimizer_state_sizes"]),
        zero_updates=not any(saved["updates"].values()), initial_uniform=error < 1e-7,
        alpha_from_config=abs(initial["alpha"] - cfg["training"]["initial_alpha"]) < 1e-7,
        initial_stage=agent.stage == "pretrain" and agent.stage_initial_updates == 0,
        replay_rng=state_equal(saved["replay"]["rng_state"], np.random.default_rng(seed + 731).bit_generator.state),
        rates=initial["learning_rates"] == cfg["training"]["pretrain_lr"])
    if not all(checks.values()):
        raise AssertionError(f"Fresh initialization failed: {checks}")
    initial.update(status="FRESH_PASS", checks=checks)
    return agent, macro, actor, counters, initial


def record_window_optimization(output, window, cursor):
    if (output / "optimization_stats.csv").exists():
        window_optimization_summary(output, window, cursor)
    else:
        append_csv(output / "optimization_windows.csv", [dict(window=window, events=0,
            **{f"{key}_{stat}": "" for key in OPT_KEYS for stat in ("mean", "max", "p95")})])


def save_fresh_snapshot(agent, cfg, path, window, scene, at_end, meter, pending):
    data = agent.checkpoint(cfg)
    data["initialization"] = dict(seed=cfg["training"]["seed"], source="fresh", parent_checkpoint=None)
    data["sampling"] = dict(window=window, stream_seed=scene, at_stream_end=at_end,
        stream_time_s=meter.env._current_time(), pending=pending, rng=capture_rng(),
        full_environment_saved=False)
    if at_end:
        data["run_state"] = dict(completed_windows=window, at_rollout_boundary=True,
                                 next_training_scene_seed=scene + 1, rng=capture_rng())
    temp = path.with_suffix(".pt.tmp")
    torch.save(data, temp)
    temp.replace(path)
    return dict(window=window, path=str(path), sha256=file_hash(path), at_stream_end=at_end,
                seamless_resume=at_end, replay_count=agent.replay.memory_counter)


def training_reference(scene):
    folder = PILOT if scene < 71040 else REFERENCE
    return folder / "training_streams" / f"{scene}_crn_with_tail.csv"


def optimization_periods(records):
    result = []
    for low, high in zip((0, *OBSERVE[:-1]), OBSERVE):
        group = [r for r in records if low < r["window"] <= high]
        item = dict(start_exclusive=low, end_inclusive=high, events=len(group))
        for key in OPT_KEYS:
            values = np.array([r[key] for r in group if r[key] != ""], dtype=float)
            item[key] = (dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)),
                              max=float(values.max()), last=float(values[-1])) if len(values) else None)
        result.append(item)
    return result


def finish_seed(output, manifest):
    validation = rows(output / "validation.csv")
    endpoint = rows(output / "endpoint.csv")
    history = [dict(window=w, mean=grouped_means([r for r in validation if r["checkpoint_window"] == w])["SAC"])
               for w in OBSERVE]
    report = dict(status="FRESH_COMPLETE", training_seed_count=1,
        initialization_seed=manifest["training_initialization_seed"], checkpoint_source="fresh",
        final_window=2000, validation_history=history,
        endpoint_mean=grouped_means(endpoint)["SAC"], endpoint_rows=endpoint,
        optimization_periods=optimization_periods(rows(output / "optimization_stats.csv")),
        learning_audit=manifest["learning_audit"], source_unchanged=True,
        reference_results_unchanged=True, reserved_scenes_executed=[], elapsed_seconds=manifest["elapsed_seconds"])
    write_json(output / "report.json", report)
    lines = [f"# SAC-only seed={report['initialization_seed']} 从零训练", "",
        "固定累计2000窗口；没有加载seed101权重，没有教师或PPO，只有training.seed不同。", "",
        "| 窗口 | 验证J | 加权完成时延/ms | 低优完成时延/ms | 期末积压/Mbit |",
        "|---|---:|---:|---:|---:|"]
    for item in history:
        m = item["mean"]
        lines.append(f"|{item['window']}|{m['J']:.6f}|{m['task_delay_weighted_ms']:.3f}|"
                     f"{m['task_delay_low_ms']:.3f}|{m['queue_total_mbit']:.3f}|")
    lines += ["", "验证为33600/33601各900秒；终点另补33610–33612，五场景均值如下：", "",
              "```json", json.dumps(report["endpoint_mean"], ensure_ascii=False, indent=2), "```", "",
              "300/600秒只记日志不重置，流尾真实部署补全经验。观察点不用于挑选终点或调参。",
              "完整路由分组见route_stats.csv；训练场景逐帧与seed101进行CRN核对，见training_pairing.csv。",
              "本报告为重复使用的开发场景结果，不是未见测试集；不单独宣称收敛或普遍优势。", ""]
    (output / "result.md").write_text("\n".join(lines), encoding="utf-8")


def run_fresh(seed):
    prior = load_json(REFERENCE / "manifest.json")
    if prior["status"] != "LONG_COMPLETE":
        raise ValueError("seed101 is not complete")
    changed = [p for p, digest in prior["source_hashes"].items() if file_hash(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Relevant source changed since seed101: {changed}")
    cfg = fresh_config(prior["config"], seed)
    fixed, dev_fixed, references, protected = cached_results(load_json(PILOT / "manifest.json"))
    protected.update({str(p): file_hash(p) for p in REFERENCE.rglob("*") if p.is_file()})
    plan = list(stream_plan(0, 71000))
    for _, _, scene in plan:
        if not training_reference(scene).is_file():
            raise FileNotFoundError(f"Missing seed101 training CRN: {training_reference(scene)}")
    output = ROOT / f"results/sac_only_wireless6_long_seed{seed}"
    output.mkdir(exist_ok=False)
    for directory in ("checkpoints", "resume_anchors", "training_streams"):
        (output / directory).mkdir()
    sources = source_hashes()
    manifest = dict(status="INITIALIZING", actual_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        worktree_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        source_hashes=sources, config=cfg, reference_manifest=str(REFERENCE / "manifest.json"),
        reference_manifest_sha256=file_hash(REFERENCE / "manifest.json"),
        protected_files=protected, training_initialization_seed=seed, training_seed_count=1,
        training=True, checkpoint_source="fresh", checkpoint=None, teacher=False, macro_enabled=False,
        completed_windows=0, target_windows=2000, final_checkpoint_window=2000,
        training_scene_seeds=[s for _, _, s in plan], observation_windows=list(OBSERVE),
        validation_seeds=[33600, 33601], final_development_seeds=list(SCENES),
        reserved_scenes_executed=[], evaluation_train=False, evaluation_complete_tail=False,
        training_complete_tail=True, baseline_reused=True, checkpoints=[], latest_resume_anchor=None,
        runtime_wallclock_limit_seconds=None,
        rng_sources=dict(algorithm_seed=seed, replay_seed=seed + 731,
            environment="QueueMDP.reset(scene_seed), local RandomState; starts at71000 for every initialization",
            schedule="build_load_switch_schedule(scene_seed); independent of training.seed"),
        comparison_scope="17 evaluations: two scenes at seven checkpoints plus three more at final2000",
        gradient_norm_semantics="critic_grad/actor_grad are pre-clip norms; native grad_clip=1")
    if manifest["branch"] != "codex/u2u-model":
        raise ValueError("Unexpected branch")
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        agent, initial_macro, initial_actor, counters, fresh = fresh_agent(cfg, output)
        original_fresh = load_json(PILOT / "manifest.json")["fresh"]
        if fresh["actor_sha256"] == original_fresh["actor_sha256"]:
            raise AssertionError("Fresh actor unexpectedly matches initialization101")
        manifest["fresh"] = fresh
        write_json(output / "fresh_audit.json", fresh)
        append_csv(output / "fixed_baselines.csv", [dict(r, reused=True,
            source_path=str(PILOT / ("validation_fixed_baselines.csv" if r["seed"] < 33610 else "pilot_endpoint.csv")),
            source_sha256=protected[str(PILOT / ("validation_fixed_baselines.csv" if r["seed"] < 33610 else "pilot_endpoint.csv"))])
            for r in fixed + dev_fixed])
        cursor, last_validation = {}, None

        def progress(status="TRAINING"):
            manifest.update(status=status, elapsed_seconds=time.perf_counter() - started)
            write_json(output / "manifest.json", manifest)
            write_json(output / "progress.json", dict(status=status, training_seed=seed,
                completed_windows=manifest["completed_windows"], target_windows=2000,
                elapsed_seconds=manifest["elapsed_seconds"], replay_count=agent.replay.memory_counter,
                critic_updates=agent.sac_updates, critic_gradient_steps=agent.sac_critic_gradient_steps,
                target_updates=agent.sac_target_updates, actor_updates=agent.sac_actor_updates,
                ppo_updates=agent.ppo_updates, alpha=agent.entropy.alpha.detach().item(),
                last_validation=last_validation, latest_resume_anchor=manifest["latest_resume_anchor"]))

        def save_validate(window, meter, pending, at_end):
            nonlocal last_validation
            manifest["checkpoints"].append(save_fresh_snapshot(agent, cfg,
                output / "checkpoints" / f"window{window}.pt", window, agent.pilot_stream_seed, at_end, meter, pending))
            progress("DEVELOPMENT_EVALUATION")
            before_rng, before = capture_rng(), copy.deepcopy(agent.checkpoint(cfg))
            selected = []
            for scene in (33600, 33601):
                r, _ = evaluate_one(cfg, agent, scene, output, "SAC", window, "validation", references)
                append_csv(output / "validation.csv", [r])
                selected.append(r)
                if window == 2000:
                    append_csv(output / "endpoint.csv", [r])
            last_validation = dict(window=window, mean=grouped_means(selected)["SAC"])
            if window == 2000:
                for scene in (33610, 33611, 33612):
                    r, _ = evaluate_one(cfg, agent, scene, output, "SAC", window, "development", references)
                    append_csv(output / "endpoint.csv", [r])
            if not state_equal(before, agent.checkpoint(cfg)) or not state_equal(before_rng, capture_rng()):
                raise AssertionError("Evaluation changed learning state or RNG")
            progress()

        def on_window(row, meter, pending, last):
            window = row["window"]
            row.update(replay_count=agent.replay.memory_counter,
                critic_grad=agent.last_sac_stats.get("critic_grad", ""),
                actor_loss=agent.last_sac_stats.get("actor_loss", ""),
                actor_grad=agent.last_sac_stats.get("actor_grad", ""))
            append_csv(output / "training.csv", [row])
            record_window_optimization(output, window, cursor)
            manifest["completed_windows"] = window
            if window in OBSERVE and not last:
                save_validate(window, meter, pending, False)
            agent.pilot_window = window + 1
            progress()

        progress()
        for offset, nwindows, scene in plan:
            agent.pilot_stream_seed, agent.pilot_window = scene, offset + 1
            count_before = agent.replay.memory_counter
            duration = 300 * nwindows
            with (output / "training_stdout.txt").open("a", encoding="utf-8", buffering=1) as stream_log:
                with redirect_stdout(stream_log), patch("hrl_uav.experiment.QueueMDP",
                    lambda settings: PilotMeter(settings, record_routes=False)):
                    summary, _, _, _, meter = rollout(cfg, agent, scene, 30 * nwindows,
                        macro_enabled=False, train=True, window_offset=offset,
                        complete_tail=True, on_window=on_window)
            if (summary["boundary_transitions"] != 30 * nwindows or summary["final_pending_dropped"] != 0
                or agent.replay.memory_counter - count_before != 540 * nwindows or meter.reset_count != 1
                or summary["time_s"] != duration or meter.env._current_time() != duration + 1):
                raise AssertionError("Stream boundary/replay conservation failed")
            reference = training_reference(scene)
            assert_crn_equal(rows(reference), meter.crn_frames)
            append_csv(output / "training_pairing.csv", [dict(training_seed=seed, scene=scene,
                reference_path=str(reference), reference_sha256=file_hash(reference),
                frames=len(meter.crn_frames), arrivals_equal=True, motion_equal=True,
                schedule_equal=True, centroid_equal=True)])
            np.testing.assert_allclose(summary["boundary_discount_min"], cfg["training"]["gamma_per_resource_frame"] ** 3,
                                       rtol=0, atol=1e-12)
            prefix = [r for r in meter.readings if r["time_s"] <= duration]
            np.testing.assert_allclose(sum(r[k] for r in prefix for k in ENERGY),
                                       summary["power_w"] * duration, rtol=1e-12, atol=1e-6)
            end = offset + nwindows
            append_csv(output / "boundary_audit.csv", [dict(stream_seed=scene, end_window=end,
                summary_time_s=duration, environment_time_s=meter.env._current_time(),
                boundary_transitions=summary["boundary_transitions"], discount=summary["boundary_discount_min"],
                pending_dropped=0, replay_added=agent.replay.memory_counter - count_before,
                reset_count=meter.reset_count, summary_excludes_tail=True, deployment_radio_zero=True)])
            append_csv(output / "training_streams" / f"{scene}_service_with_tail.csv", meter.readings)
            append_csv(output / "training_streams" / f"{scene}_crn_with_tail.csv", meter.crn_frames)
            if agent.ppo_updates or agent.macro_rollout or parameter_delta(initial_macro, agent.macro_actor) != 0:
                raise AssertionError("PPO changed")
            manifest["latest_resume_anchor"] = save_fresh_snapshot(agent, cfg,
                output / "resume_anchors" / f"latest_slot{(scene - 71000) % 2}.pt",
                end, scene, True, meter, None)
            if end in OBSERVE:
                save_validate(end, meter, None, True)
            progress()
            if any(offset < w <= end for w in range(100, 2001, 100)):
                print(f"[seed{seed} window{end}] elapsed={manifest['elapsed_seconds']:.1f}s "
                      f"updates={agent.sac_updates} alpha={agent.entropy.alpha.detach().item():.6f} "
                      f"last_validation={last_validation}", flush=True)
        if source_hashes() != sources or any(file_hash(p) != digest for p, digest in protected.items()):
            raise AssertionError("Source or seed101 artifacts changed during training")
        if agent.replay.memory_counter != 1080000 or agent.sac_updates != 53951 or agent.sac_actor_updates != 26925:
            raise AssertionError("Final native update budget mismatch")
        manifest.update(source_hashes_after=source_hashes(), source_unchanged=True,
            reference_results_unchanged=True, learning_audit=dict(**counters,
                critic_updates=agent.sac_updates, critic_gradient_steps=agent.sac_critic_gradient_steps,
                target_updates=agent.sac_target_updates, actor_updates=agent.sac_actor_updates,
                ppo_updates=agent.ppo_updates, replay_count=agent.replay.memory_counter,
                macro_delta=parameter_delta(initial_macro, agent.macro_actor),
                sac_actor_delta=parameter_delta(initial_actor, agent.sac_actor.action_net)))
        progress("WRITING_REPORT")
        finish_seed(output, manifest)
        progress("FRESH_COMPLETE")
        print(f"[FRESH_COMPLETE seed{seed}] elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    except BaseException as error:
        manifest.update(status="STOPPED_ERROR", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        write_json(output / "progress.json", dict(status="STOPPED_ERROR", error=repr(error),
            completed_windows=manifest["completed_windows"], latest_resume_anchor=manifest["latest_resume_anchor"]))
        raise


def nested_stats(per_seed):
    if len({r["training_seed"] for r in per_seed}) != len(per_seed):
        raise ValueError("Expected one already scene-averaged row per training initialization")
    return [dict(metric=key, mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                 min=float(np.min(values)), max=float(np.max(values)), training_seed_count=len(values), ddof=1)
            for key in MEASURES for values in [[r[key] for r in per_seed]]]


def aggregate():
    output = SUMMARY
    output.mkdir(exist_ok=False)
    origin = load_json(REFERENCE / "report.json")
    manifests, reports, endpoint_rows, source_paths = {}, {}, {}, []
    for seed in (101, 102, 103):
        folder = ROOT / f"results/sac_only_wireless6_long_seed{seed}"
        manifests[seed], reports[seed] = load_json(folder / "manifest.json"), load_json(folder / "report.json")
        if reports[seed]["status"] not in ("LONG_COMPLETE", "FRESH_COMPLETE"):
            raise ValueError(f"Incomplete training initialization {seed}")
        cfg = copy.deepcopy(manifests[seed]["config"])
        cfg["training"]["seed"] = 101
        if cfg != manifests[101]["config"]:
            raise ValueError("Effective configurations differ beyond initialization seed")
        endpoint_rows[seed] = ([r for r in rows(folder / "validation.csv") if r["checkpoint_window"] == 2000]
            + [r for r in rows(folder / "development_1600_2000.csv") if r["checkpoint_window"] == 2000]
            if seed == 101 else rows(folder / "endpoint.csv"))
        if len(endpoint_rows[seed]) != 5 or {int(r["seed"]) for r in endpoint_rows[seed]} != set(SCENES):
            raise ValueError("Endpoint matrix incomplete")
        source_paths += [folder / "manifest.json", folder / "report.json", folder / "route_stats.csv"]
    baseline = rows(PILOT / "validation_fixed_baselines.csv") + [r for r in rows(PILOT / "pilot_endpoint.csv") if r["policy"] != "SAC"]
    fixed_means = grouped_means(baseline)
    per_seed = []
    for seed in (101, 102, 103):
        means = grouped_means(endpoint_rows[seed])["SAC"]
        per_seed.append(dict(training_seed=seed, scene_count=5, **means,
            **{f"minus_{policy}_{key}": means[key] - fixed_means[policy][key]
               for policy in ("Uniform", "Fixed-A", "Fixed-B") for key in MEASURES}))
    append_csv(output / "per_training_seed.csv", per_seed)
    append_csv(output / "summary.csv", nested_stats(per_seed))
    append_csv(output / "baseline_scene_means.csv", [dict(policy=p, **m) for p, m in fixed_means.items()])
    append_csv(output / "per_scene.csv", [dict(training_seed=s, **r) for s, rs in endpoint_rows.items() for r in rs])
    route_summary, learning = [], []
    for seed in (101, 102, 103):
        folder = ROOT / f"results/sac_only_wireless6_long_seed{seed}"
        raw = []
        keys = ["priority", "source", "role", "window_300s", "local_ratio", "raw_mean", "raw_std", "raw_saturation",
                *[f"destination_{j}" for j in range(3)], *[f"raw_{j}" for j in range(3)]]
        for scene in SCENES:
            split = "validation" if scene < 33610 else "development"
            path = folder / "evaluations" / f"{split}_SAC_2000_{scene}_routes.csv"
            source_paths.append(path)
            raw += [{key: r[key] for key in keys} for r in rows(path)]
        for r in route_statistics(raw, "SAC", seed, 2000, "five_development_scenes_pooled"):
            r["training_seed"] = r.pop("seed")
            route_summary.append(r)
        history = reports[seed]["validation_history"]
        values = [r["mean"]["J"] for r in history]
        at1600 = next(r["mean"] for r in history if r["window"] == 1600)
        at2000 = next(r["mean"] for r in history if r["window"] == 2000)
        item = dict(training_seed=seed, status=reports[seed]["status"],
            first_observed_validation_J_below_1=next((r["window"] for r in history if r["mean"]["J"] < 1), "none"),
            observation_J_increases=sum(b > a for a, b in zip(values, values[1:])),
            max_adjacent_observation_J_increase=max((b-a for a, b in zip(values, values[1:])), default=0),
            validation_J1600=at1600["J"], validation_J2000=at2000["J"],
            validation_delta_J2000_minus1600=at2000["J"]-at1600["J"],
            validation_delay1600_ms=at1600["task_delay_weighted_ms"],
            validation_delay2000_ms=at2000["task_delay_weighted_ms"],
            validation_delta_delay2000_minus1600_ms=at2000["task_delay_weighted_ms"]-at1600["task_delay_weighted_ms"])
        early, late = reports[seed]["optimization_periods"][0], reports[seed]["optimization_periods"][-1]
        for key in OPT_KEYS:
            for period, label in ((early, "first120"), (late, "last400")):
                for stat in ("mean", "p95", "max", "last"):
                    item[f"{label}_{key}_{stat}"] = period[key][stat] if period[key] else ""
        learning.append(item)
    append_csv(output / "route_summary.csv", route_summary)
    append_csv(output / "learning_summary.csv", learning)
    manifest = dict(status="MULTISEED_COMPLETE", training_initializations=[101, 102, 103],
        training_seed_count=3, scenes=list(SCENES), evaluation_duration_s=900,
        aggregation="Arithmetic mean of 5 scenes within each training seed; sample std across 3 seed means, ddof=1",
        route_aggregation="Pool deterministic decision/source samples across 5 scenes separately within each training seed",
        initialization_scope="algorithm RNG: network initialization, exploration and replay sampling; same exogenous training sequence",
        source_hashes=source_hashes(), input_hashes={str(p): file_hash(p) for p in source_paths},
        final_checkpoint_window=2000, reserved_scenes_executed=[], ppo_updates=0,
        same_training_environment_sequence=True, observation_schedule=list(OBSERVE),
        selection="Fixed final2000 for each initialization; no best selection",
        observation_comparison="Two common validation scenes at all observation checkpoints; no extra1600 three-scene run",
        fixed_baselines_reused=True, figures="Descriptive, no confidence interval/significance claim")
    write_json(output / "manifest.json", manifest)
    lines = ["# 候选无线模型6MHz：SAC-only三个训练初始化", "",
        "seed102和103从零初始化，未加载seed101参数、优化器或经验；三个seed的物理与训练配置仅training.seed不同。",
        "每seed固定取2000窗口，先对33600/33601/33610/33611/33612五个开发场景平均，再按三个训练初始化汇总。标准差用ddof=1，不把15条场景记录当成15次训练。", "",
        "## 最终结果", "",
        "| 训练seed | J | 加权ms | 高优ms | 低优ms | 高/低完成率 | 期末Mbit | 功率W |",
        "|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in per_seed:
        lines.append(f"|{r['training_seed']}|{r['J']:.6f}|{r['task_delay_weighted_ms']:.3f}|{r['task_delay_high_ms']:.3f}|"
                     f"{r['task_delay_low_ms']:.3f}|{r['task_completion_high_rate']:.6f}/{r['task_completion_low_rate']:.6f}|"
                     f"{r['queue_total_mbit']:.3f}|{r['power_w']:.3f}|")
    lines += ["", "| 指标 | 三seed均值 | 标准差 | 最小 | 最大 |", "|---|---:|---:|---:|---:|"]
    for r in nested_stats(per_seed):
        if r["metric"] in ("J", "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
                           "task_completion_high_rate", "task_completion_low_rate", "queue_total_mbit", "power_w"):
            lines.append(f"|{r['metric']}|{r['mean']:.6f}|{r['std']:.6f}|{r['min']:.6f}|{r['max']:.6f}|")
    lines += ["", "## 固定规则对照", "",
        "下列差值先按同场景配对再对五场景平均；负J/负时延代表本开发集物理代价/完成均值较低。", "",
        "| 训练seed | 对照 | ΔJ | Δ加权ms | Δ期末Mbit | Δ功率W |", "|---|---|---:|---:|---:|---:|"]
    for r in per_seed:
        for policy in ("Uniform", "Fixed-A", "Fixed-B"):
            values = [r[f"minus_{policy}_{key}"] for key in ("J", "task_delay_weighted_ms", "queue_total_mbit", "power_w")]
            lines.append(f"|{r['training_seed']}|{policy}|" + "|".join(f"{v:+.6f}" for v in values) + "|")
    lines += ["", "## 路由重复性", "",
        "各行在同一训练seed内合并五场景的执行路由样本；不是按任务数据量加权的吞吐比例。角色随负载阶段变化。", "",
        "| seed | 优先级 | 来源/角色 | 本地均值 | 标准差 | P10/P50/P90 |", "|---|---|---|---:|---:|---|"]
    for r in route_summary:
        if r["window_300s"] == 0:
            lines.append(f"|{r['training_seed']}|{r['priority']}|{r['source']}/{r['role']}|{r['local_mean']:.4f}|{r['local_std']:.4f}|"
                         f"{r['local_q10']:.4f}/{r['local_q50']:.4f}/{r['local_q90']:.4f}|")
    lines += ["", "完整300秒分组在route_summary.csv；不同编号或角色差异不是机制因果证明。", "",
        "## 训练稳定性", "", "| seed | 首次观察到验证J<1 | 验证J上升次数 | J1600→2000 | 时延1600→2000/ms |",
        "|---|---|---:|---|---|"]
    for r in learning:
        lines.append(f"|{r['training_seed']}|{r['first_observed_validation_J_below_1']}|{r['observation_J_increases']}|"
            f"{r['validation_J1600']:.6f}→{r['validation_J2000']:.6f}|"
            f"{r['validation_delay1600_ms']:.3f}→{r['validation_delay2000_ms']:.3f}|")
    lines += ["", "只比较指定观察点，首次J<1不是精确越界时间；训练原始曲线另存。梯度范数是裁剪前统计，grad_clip=1保持不变。",
        "各优化指标前120窗口与后400窗口的均值/P95/最大/末值均见learning_summary.csv，不只报告全程最大值。",
        "评价时延仅统计完成任务；完成率、未完成数、期末积压及后段队列斜率保留在per_scene.csv和per_training_seed.csv。",
        "所有新增训练流的外生任务、运动、调度和质心指纹逐帧对齐seed101；不要求队列/实际服务轨迹相同。",
        "仅验证当前固定6MHz模型和已使用开发场景的训练重复性，不宣称未见场景泛化、全局最优或已收敛。没有运行保留集、3MHz或PPO，也没有修改论文、提交或推送。", ""]
    (output / "result.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[MULTISEED_COMPLETE] {output}", flush=True)

