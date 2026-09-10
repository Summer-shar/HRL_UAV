"""Fixed-budget continuation of the completed candidate-wireless SAC pilot."""

import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .experiment import append_csv, parameter_delta, rollout, snapshot, write_json
from .mdp import QueueMDP
from .priority_grid import file_hash
from .resumption import capture_rng, restore_agent, state_equal
from .scene_prescreen import ROOT, source_hashes
from .wireless_pilot import (ENERGY, MEASURES, OPT_KEYS, PilotMeter, checked_agent,
                             evaluate_one, finite, grouped_means)


PILOT = ROOT / "results/sac_only_wireless6_pilot_seed101"
ORIGIN = PILOT / "checkpoints/window120.pt"
ORIGIN_SHA = "80b7ae77f0b698248acc978ec3881828375db3ee2218174bba55d9c3518a347b"
OBSERVE = (200, 400, 800, 1200, 1600, 2000)
TOTAL = 2000


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        result = list(csv.DictReader(handle))
    for row in result:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return result


def stream_plan(start=120, next_seed=71040):
    for offset in range(start, TOTAL, 3):
        yield offset, min(3, TOTAL-offset), next_seed + (offset-start)//3


def normalize_checkpoint(saved, origin=True):
    """Translate saved pilot RNG metadata, without inventing any missing state."""
    required = ("sac_actor", "sac_critic", "sac_target", "log_alpha", "macro_actor",
                "macro_critic", "optimizers", "updates", "training_state", "replay",
                "observations", "micro_state_dim", "macro_state_dim", "stage", "config")
    missing = [name for name in required if name not in saved]
    if missing:
        raise ValueError(f"Exact restore missing fields: {missing}")
    for name in ("stage_initial_updates", "last_sac_stats", "last_ppo_stats", "macro_rollout"):
        if name not in saved["training_state"]:
            raise ValueError(f"Exact restore missing training_state.{name}")
    for name in ("sac_critic", "sac_critic_gradients", "sac_target", "sac_actor", "ppo", "ppo_gradients"):
        if name not in saved["updates"]:
            raise ValueError(f"Exact restore missing updates.{name}")
    for name in ("transitions", "rng_state", "memory_counter", "capacity", "transition_width",
                 "recent_window", "recent_ratio"):
        if name not in saved["replay"]:
            raise ValueError(f"Exact restore missing replay.{name}")
    if len(saved["optimizers"]) != 5 or any(not s["state"] for s in saved["optimizers"][:3]):
        raise ValueError("Missing trained SAC optimizer state")
    if saved["stage"] != "pretrain" or saved["training_state"]["stage_initial_updates"] != 0:
        raise ValueError("Stage or lifetime actor warmup changed")
    if saved["updates"]["ppo"] != 0 or saved["training_state"]["macro_rollout"]:
        raise ValueError("PPO state is not empty")
    if origin:
        s = saved["pilot_sampling"]
        if (s["window"] != 120 or s["stream_seed"] != 71039 or s["stream_time_s"] != 901
                or not s["at_stream_end"] or s["pending"] is not None):
            raise ValueError("Pilot checkpoint is not the completed window120 tail boundary")
        if saved["updates"] != dict(sac_critic=3191, sac_critic_gradients=3191,
                                    sac_target=3191, sac_actor=1545, ppo=0, ppo_gradients=0):
            raise ValueError("Pilot update counters differ")
        if saved["replay"]["memory_counter"] != 64800:
            raise ValueError("Pilot replay counter differs")
        run_state = dict(completed_windows=120, at_rollout_boundary=True,
                         next_training_scene_seed=71040, rng=s["rng"])
    else:
        run_state = saved["run_state"]
        if (not run_state["at_rollout_boundary"] or saved["origin_sha256"] != ORIGIN_SHA
                or run_state["completed_windows"] % 3 != 0):
            raise ValueError("Not a usable true-stream resume anchor")
    if set(run_state["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete Python/NumPy/Torch/CUDA RNG")
    restored = dict(saved, run_state=run_state)
    return restored


def exact_restore(agent, saved, checkpoint_path, origin=True):
    data = normalize_checkpoint(saved, origin)
    state = data["run_state"]
    audit = restore_agent(agent, data, dict(
        checkpoint=str(checkpoint_path), checkpoint_sha256=file_hash(checkpoint_path),
        replay_counter=data["replay"]["memory_counter"], start_window=state["completed_windows"],
        next_training_scene_seed=state["next_training_scene_seed"],
        rng_mode="saved_pilot_sampling_rng" if origin else "saved_stream_anchor_rng"))
    checks = audit["exact_restore_checks"]
    checks.update(
        complete_replay=state_equal(agent.replay.state_dict(), data["replay"]),
        update_counters=state_equal(agent.checkpoint(agent.full_config, False)["updates"], data["updates"]),
        complete_training_state=state_equal(agent.checkpoint(agent.full_config, False)["training_state"],
                                           data["training_state"]),
        stage_pretrain=agent.stage == "pretrain",
        lifetime_actor_warmup=agent.stage_initial_updates == 0 and agent.sac_actor_warmup() == 100,
        alpha=state_equal(agent.entropy.alpha.detach(), data["log_alpha"].to(agent.entropy.log_alpha.device).exp()),
        macro_optimizer_unchanged=all(not o.state for o in (agent.ppo_actor_optimizer, agent.ppo_critic_optimizer)),
    )
    for i, optimizer in enumerate((agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer)):
        checks[f"sac_optimizer_{i}_full_state"] = state_equal(optimizer.state_dict(), data["optimizers"][i])
    for name in ("python", "numpy", "torch", "cuda"):
        checks[f"rng_{name}"] = state_equal(capture_rng()[name], state["rng"][name])
    if not all(checks.values()):
        raise AssertionError(f"Exact restore failed: {checks}")
    audit.update(status="EXACT_RESTORE_PASS", before_any_new_environment_experience=True,
                 alpha=agent.entropy.log_alpha.exp().item(), target_entropy=agent.entropy.target_entropy,
                 replay_counter=agent.replay.memory_counter, stage=agent.stage,
                 next_training_scene_seed=state["next_training_scene_seed"],
                 rng_mapping="pilot_sampling.rng -> run_state.rng; all bytes preserved")
    return audit


def cached_results(manifest):
    changed = [p for p, sha in manifest["source_hashes"].items() if file_hash(ROOT/p) != sha]
    if changed:
        raise ValueError(f"Pilot source mismatch; reuse/continuation stopped: {changed}")
    fixed = rows(PILOT / "validation_fixed_baselines.csv")
    endpoint = [r for r in rows(PILOT / "pilot_endpoint.csv") if r["policy"] != "SAC"]
    expected = set(manifest["config"]["pilot"]["fixed_rules"])
    for group, seeds in ((fixed, [33600, 33601]), (endpoint, [33610, 33611, 33612])):
        if {(r["policy"], int(r["seed"])) for r in group} != {(p, s) for p in expected for s in seeds}:
            raise ValueError("Incomplete cached fixed baseline matrix")
        for r in group:
            if r["time_s"] != 900 or r["final_pending_dropped"] != 0:
                raise ValueError("Cached evaluation scope differs")
    references, hashes = {}, {}
    for seed in (33600, 33601, 33610, 33611, 33612):
        split = "validation" if seed < 33610 else "endpoint"
        base = PILOT / "evaluations" / f"{split}_Uniform_0_{seed}"
        crn, trajectory = Path(str(base)+"_crn.csv"), Path(str(base)+"_trajectory.npz")
        with np.load(trajectory) as data:
            geometry = {k: data[k].copy() for k in ("uav", "users", "centroid", "macro_action")}
        references[seed] = (rows(crn), geometry)
        for p in (crn, trajectory):
            hashes[str(p)] = file_hash(p)
    for p in (PILOT/"manifest.json", PILOT/"report.json", PILOT/"validation_fixed_baselines.csv",
              PILOT/"pilot_endpoint.csv", PILOT/"training.csv", PILOT/"validation.csv",
              PILOT/"route_stats.csv", PILOT/"optimization_stats.csv", ORIGIN):
        hashes[str(p)] = file_hash(p)
    return fixed, endpoint, references, hashes


def save_snapshot(agent, cfg, path, window, scene, at_end, meter, pending):
    data = agent.checkpoint(cfg)
    data["origin_sha256"] = ORIGIN_SHA
    data["long_sampling"] = dict(window=window, stream_seed=scene, at_stream_end=at_end,
        stream_time_s=meter.env._current_time(), pending=pending,
        rng=capture_rng(), full_environment_saved=False)
    if at_end:
        data["run_state"] = dict(completed_windows=window, at_rollout_boundary=True,
                                 next_training_scene_seed=scene+1, rng=capture_rng())
    temporary = path.with_suffix(".pt.tmp")
    torch.save(data, temporary)
    temporary.replace(path)
    return dict(window=window, path=str(path), sha256=file_hash(path), at_stream_end=at_end,
                seamless_resume=at_end, replay_count=agent.replay.memory_counter)


def window_optimization_summary(output, window, cursor):
    # Read each new event once; long training must not repeatedly parse its full history.
    with (output/"optimization_stats.csv").open(encoding="utf-8-sig", newline="") as handle:
        if "offset" in cursor:
            handle.seek(cursor["offset"])
        else:
            cursor["fields"] = next(csv.reader([handle.readline()]))
        content = handle.read()
        cursor["offset"] = handle.tell()
    selected = list(csv.DictReader(io.StringIO(content), fieldnames=cursor["fields"]))
    selected = [r for r in selected if int(float(r["window"])) == window]
    result = dict(window=window, events=len(selected))
    for key in OPT_KEYS:
        v = [float(r[key]) for r in selected if r[key] != ""]
        for label, value in (("mean", np.mean(v) if v else None), ("max", max(v) if v else None),
                             ("p95", np.percentile(v, 95) if v else None)):
            result[f"{key}_{label}"] = "" if value is None else float(value)
    append_csv(output/"optimization_windows.csv", [result])


def analyze(output, manifest, fixed, development_fixed, origin_report):
    validation = rows(output/"validation.csv")
    development = rows(output/"development_1600_2000.csv")
    history = [origin_report["validation_history"][-1]] + [dict(window=w,
        mean=grouped_means([r for r in validation if r["checkpoint_window"] == w])["SAC"])
        for w in OBSERVE]
    means = {str(w): grouped_means([r for r in development if r["checkpoint_window"] == w]
                                  + development_fixed) for w in (1600, 2000)}
    deltas = []
    for r in development:
        for b in development_fixed:
            if b["seed"] == r["seed"]:
                deltas.append(dict(window=r["checkpoint_window"], seed=r["seed"], baseline=b["policy"],
                                   **{k: r[k]-b[k] for k in MEASURES}))
    append_csv(output/"development_paired_deltas.csv", deltas)
    routes = rows(output/"route_stats.csv")
    route_history = []
    for w in (120, *OBSERVE):
        source = rows(PILOT/"route_stats.csv") if w == 120 else routes
        for priority in ("high", "low"):
            r = [x for x in source if x["checkpoint_window"] == w and x["split"] == "validation"
                 and x["source"] == "all" and x["role"] == "all"
                 and x["window_300s"] == 0 and x["priority"] == priority]
            route_history.append(dict(window=w, priority=priority,
                **{k: float(np.mean([x[k] for x in r])) for k in
                   ("local_mean", "local_std", "local_q10", "local_q50", "local_q90",
                    "destination_0_mean", "destination_1_mean", "destination_2_mean", "raw_saturation")}))
    optimization = rows(output/"optimization_stats.csv")
    optimization_before = rows(PILOT/"optimization_stats.csv")
    opt_periods = []
    for lo, hi in ((0, 120), (120, 200), (200, 400), (400, 800), (800, 1200), (1200, 1600), (1600, 2000)):
        group = [r for r in (optimization_before if hi == 120 else optimization) if lo < r["window"] <= hi]
        period = dict(start_exclusive=lo, end_inclusive=hi, events=len(group))
        for k in OPT_KEYS:
            v = np.array([r[k] for r in group if r[k] != ""], dtype=float)
            period[k] = dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)), max=float(v.max()),
                             last=float(v[-1])) if len(v) else None
        opt_periods.append(period)
    report = dict(status="LONG_COMPLETE", training_seed_count=1, initialization_seed=101,
        origin_checkpoint=str(ORIGIN), origin_sha256=ORIGIN_SHA, final_window=2000,
        validation_history=history, development_means=means, development_rows=development,
        development_deltas=deltas, fixed_validation_means=grouped_means(fixed),
        route_history=route_history, optimization_periods=opt_periods,
        learning_audit=manifest["learning_audit"], source_unchanged=True, pilot_results_unchanged=True,
        reserved_scenes_executed=[], elapsed_seconds=manifest["elapsed_seconds"],
        gradient_norm_semantics="actor_grad and critic_grad are pre-clip norms returned by clip_grad_norm_; limit=1",
        scope="One continued initialization; reused development scenes; final fixed at 2000; descriptive only")
    write_json(output/"report.json", report)
    lines = ["# SAC-only 累计2000窗口续训结果", "",
        "从唯一window120检查点保留网络、优化器、replay、更新计数及随机状态续训。"
        "一个初始化seed=101；最终策略固定window2000，中间成绩没有用于选点或调参。", "",
        "## 长期开发验证", "",
        "33600/33601各900s的算术平均；J为平均物理代价。完成时延仅针对已完成任务。", "",
        "| 窗口 | J | 加权ms | 高优ms | 低优ms | 高/低完成率 | 高/低未完成数 | 期末Mbit | 后300s队列斜率Mbit/s |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for x in history:
        r=x["mean"]
        lines.append(f"|{x['window']}|{r['J']:.6f}|{r['task_delay_weighted_ms']:.3f}|{r['task_delay_high_ms']:.3f}|"
                     f"{r['task_delay_low_ms']:.3f}|{r['task_completion_high_rate']:.6f}/{r['task_completion_low_rate']:.6f}|"
                     f"{r['task_unfinished_high_count']:.1f}/{r['task_unfinished_low_count']:.1f}|"
                     f"{r['queue_total_mbit']:.3f}|{r['late_queue_slope_mbit_s']:.6f}|")
    values = [r["mean"]["J"] for r in history]
    lines += ["", f"120到2000验证J变化：{values[-1]-values[0]:+.6f}；"
              f"相邻观察点中J上升{sum(b>a for a,b in zip(values,values[1:]))}次。"
              "training.csv逐窗口曲线保留，不用单个终点代替学习轨迹。", "",
              "## 1600和2000开发诊断", "",
              "33610–33612已经在120窗口使用，此处均为开发场景，未运行33700–33709。", "",
              "| 窗口 | 方法 | J | 加权ms | 高优ms | 低优ms | 低优完成率 | 低优未完成 | 期末Mbit | 后300s斜率 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w, methods in means.items():
        for label,r in methods.items():
            lines.append(f"|{w}|{label}|{r['J']:.6f}|{r['task_delay_weighted_ms']:.3f}|{r['task_delay_high_ms']:.3f}|"
                         f"{r['task_delay_low_ms']:.3f}|{r['task_completion_low_rate']:.6f}|{r['task_unfinished_low_count']:.1f}|"
                         f"{r['queue_total_mbit']:.3f}|{r['late_queue_slope_mbit_s']:.6f}|")
    lines += ["", "Uniform/Fixed-A/Fixed-B仅为所测固定规则。逐场景完成率、队列和能量见CSV；"
              "未完成不一律解释为超时。来源匹配后复用固定规则，未新增其评估。", "",
              "## 路由变化", "",
              "下表先在各场景汇总，再对两个验证场景取平均；分位数不是池化后的分位数。", "",
              "| 窗口 | 优先级 | 本地均值 | 场景内标准差均值 | P10/P50/P90均值 | 目的0/1/2均值 | 原分数饱和比例 |",
              "|---|---|---:|---:|---|---|---:|"]
    for r in route_history:
        lines.append(f"|{r['window']}|{r['priority']}|{r['local_mean']:.5f}|{r['local_std']:.5f}|"
                     + "/".join(f"{r[f'local_q{q}']:.5f}" for q in (10,50,90)) + "|"
                     + "/".join(f"{r[f'destination_{j}_mean']:.5f}" for j in range(3))
                     + f"|{r['raw_saturation']:.5f}|")
    lines += ["", "标准差、角色/源UAV/300s分组描述路由变化，并不单独证明对某个状态变量的因果依赖。"
              "不能因平均比例接近固定规则便断言策略等价；也不把任何预定本地比例当作学习目标。", "",
              "## 优化统计", "",
              "critic_grad和actor_grad是clip_grad_norm_返回的裁剪前范数，裁剪上限始终为1。"
              "temperature_grad记录温度参数梯度范数；温度优化保持原实现。", "",
              "| 窗口区间 | 评价损失均值/P95/最大 | 目标std均值/P95 | 评价梯度P95/最大 | 动作梯度P95/最大 | alpha末值 |",
              "|---|---|---|---|---|---:|"]
    for p in opt_periods:
        def fmt(k, fields):
            return "/".join(f"{p[k][f]:.4g}" for f in fields) if p[k] else "缺失"
        lines.append(f"|({p['start_exclusive']},{p['end_inclusive']}]|"+fmt("critic_loss",("mean","p95","max"))+"|"
                     +fmt("target_std",("mean","p95"))+"|"+fmt("critic_grad",("p95","max"))+"|"
                     +fmt("actor_grad",("p95","max"))+"|"+fmt("alpha",("last",))+"|")
    lines += ["", "这些数值反映拟合过程，不能单独判定评价网络准确。没有运行额外失败机制审计。", "",
              "## 运行和证据范围", "",
              f"本轮新增1880窗口，训练场景71040–71666，最后一条流600s；耗时{manifest['elapsed_seconds']/3600:.3f}小时。",
              "各完整流300/600s不重置；尾部1s部署只补全经验，不计入结束窗口指标。"
              "中途观察点保留pending快照，不能无缝恢复；真正流末另存恢复锚点。",
              "restore_audit.json逐值记录恢复结果；boundary_audit.csv记录所有流边界。"
              "manifest记录物理和配置指纹、固定结果复用来源、检查点血缘。",
              "本轮没有新的独立训练重复、未见测试集验证或显著性检验，不能写已收敛、全局最优或泛化良好。"
              "仅完成规定训练与开发评价；未启用PPO、改变参数或继续到2000以后。", ""]
    (output/"result.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_long(output, audit_only=False, resume_anchor=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "resume_anchors", "training_streams"):
        (output/folder).mkdir()
    prior = json.loads((PILOT/"manifest.json").read_text(encoding="utf-8"))
    origin_report = json.loads((PILOT/"report.json").read_text(encoding="utf-8"))
    if (prior["status"] != "PILOT_COMPLETE" or file_hash(ORIGIN) != ORIGIN_SHA
            or prior["checkpoints"][-1]["sha256"] != ORIGIN_SHA):
        raise ValueError("Unique pilot origin hash/status mismatch")
    fixed, development_fixed, references, protected = cached_results(prior)
    cfg = copy.deepcopy(prior["config"])
    cfg["training"]["pretrain_windows"] = TOTAL
    path = Path(resume_anchor).resolve() if resume_anchor else ORIGIN
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected_cfg = copy.deepcopy(cfg)
    expected_cfg["training"]["pretrain_windows"] = saved["config"]["training"]["pretrain_windows"]
    if saved["config"] != expected_cfg:
        raise ValueError("Restoration would change a frozen configuration field")
    normalized = normalize_checkpoint(saved, resume_anchor is None)
    start = normalized["run_state"]["completed_windows"]
    next_seed = normalized["run_state"]["next_training_scene_seed"]
    if not (120 <= start < TOTAL and next_seed == 71000+start//3):
        raise ValueError("Continuation scene/window mismatch")
    sources = source_hashes()
    manifest = dict(status="RESTORING", actual_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        worktree_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        source_hashes=sources, pilot_source_hashes=prior["source_hashes"], config=cfg,
        origin_checkpoint=str(ORIGIN), origin_sha256=ORIGIN_SHA, restore_from=str(path),
        restore_sha256=file_hash(path), completed_windows=start, target_windows=TOTAL,
        training_seed_count=1, training_initialization_seed=101, training=True, macro_enabled=False,
        training_scene_seeds=[s for _,_,s in stream_plan(start,next_seed)],
        observation_windows=list(OBSERVE), validation_seeds=[33600,33601],
        development_seeds=[33610,33611,33612], development_windows=[1600,2000],
        reserved_scenes_executed=[], final_checkpoint_window=TOTAL,
        evaluation_train=False, evaluation_complete_tail=False, training_complete_tail=True,
        runtime_wallclock_limit_seconds=None,
        runtime_policy="No inherited max_stage_seconds; atomic alternating stream anchors support external restart",
        resume_anchor_policy="Two rolling stream anchors; retain named anchors at first boundary after observations",
        reused_baseline_files=protected, baseline_reused=True,
        gradient_norm_semantics="critic_grad/actor_grad are pre-clip; native clip_grad_norm_ limit stays 1",
        source_scope="New continuation runner/tests only; every pilot source fingerprint must match",
        checkpoints=[], retained_resume_anchors=[], latest_resume_anchor=None)
    if manifest["branch"] != "codex/u2u-model":
        raise ValueError("Unexpected branch")
    write_json(output/"manifest.json", manifest)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        # Constructor supplies shapes only; reset/arrivals occur after the exact audit.
        mdp = QueueMDP(cfg)
        agent, _, _, counters = checked_agent(mdp, cfg, output)
        agent.full_config = cfg
        audit = exact_restore(agent, saved, path, resume_anchor is None)
        if str(next(agent.macro_actor.parameters()).device) != prior["device"]:
            raise ValueError("Device differs from pilot")
        write_json(output/"restore_audit.json", audit)
        initial_macro = snapshot(agent.macro_actor)
        initial_actor = snapshot(agent.sac_actor.action_net)
        del saved, normalized
        if audit_only:
            manifest["status"] = "EXACT_RESTORE_PASS"
            write_json(output/"manifest.json", manifest)
            print("[EXACT_RESTORE_PASS] No new experience generated", flush=True)
            return manifest
        # A recovery uses a new directory and copies only completed, audited prefix rows.
        if resume_anchor:
            parent = path.parent.parent
            previous = json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
            if previous["source_hashes"] != sources:
                raise ValueError("Continuation runner/source changed since resume anchor")
            manifest["parent_run"] = str(parent)
            for name,key in (("training.csv","window"),("optimization_stats.csv","window"),
                             ("optimization_windows.csv","window"),("validation.csv","checkpoint_window"),
                             ("route_stats.csv","checkpoint_window"),("boundary_audit.csv","end_window"),
                             ("development_1600_2000.csv","checkpoint_window")):
                if (parent/name).exists():
                    append_csv(output/name, [r for r in rows(parent/name) if r[key] <= start])
            manifest["checkpoints"] = [r for r in previous["checkpoints"] if r["window"] <= start]
        append_csv(output/"validation_fixed_baselines.csv", [dict(r,reused=True,
            source_path=str(PILOT/"validation_fixed_baselines.csv"),source_sha256=protected[str(PILOT/"validation_fixed_baselines.csv")]) for r in fixed])
        append_csv(output/"development_fixed_baselines.csv", [dict(r,reused=True,
            source_path=str(PILOT/"pilot_endpoint.csv"),source_sha256=protected[str(PILOT/"pilot_endpoint.csv")]) for r in development_fixed])
        manifest["status"] = "TRAINING"
        last_validation = None
        optimization_cursor = {}
        if resume_anchor and (output/"optimization_stats.csv").exists():
            with (output/"optimization_stats.csv").open(encoding="utf-8-sig", newline="") as handle:
                optimization_cursor["fields"] = next(csv.reader([handle.readline()]))
                handle.read()
                optimization_cursor["offset"] = handle.tell()

        def progress(status="TRAINING"):
            manifest.update(status=status, elapsed_seconds=time.perf_counter()-started)
            write_json(output/"manifest.json", manifest)
            write_json(output/"progress.json", dict(status=status, completed_windows=manifest["completed_windows"],
                target_windows=TOTAL, current_stream_seed=getattr(agent,"pilot_stream_seed",next_seed),
                elapsed_seconds=manifest["elapsed_seconds"], replay_count=agent.replay.memory_counter,
                critic_updates=agent.sac_updates, critic_gradient_steps=agent.sac_critic_gradient_steps,
                target_updates=agent.sac_target_updates, actor_updates=agent.sac_actor_updates,
                ppo_updates=agent.ppo_updates, alpha=agent.entropy.log_alpha.exp().item(),
                last_validation=last_validation, latest_resume_anchor=manifest["latest_resume_anchor"]))

        def save_validate(window, meter, pending, at_end):
            nonlocal last_validation
            item = save_snapshot(agent,cfg,output/"checkpoints"/f"window{window}.pt",
                                 window,agent.pilot_stream_seed,at_end,meter,pending)
            manifest["checkpoints"].append(item)
            progress("DEVELOPMENT_EVALUATION")
            before_rng = capture_rng()
            before = copy.deepcopy(agent.checkpoint(cfg))
            validation_rows=[]
            for scene in cfg["validation_seeds"]:
                r,_=evaluate_one(cfg,agent,scene,output,"SAC",window,"validation",references)
                validation_rows.append(r)
                append_csv(output/"validation.csv",[r])
            last_validation = dict(window=window,mean=grouped_means(validation_rows)["SAC"])
            if window in (1600,2000):
                for scene in cfg["test_seeds"]:
                    r,_=evaluate_one(cfg,agent,scene,output,"SAC",window,"development",references)
                    append_csv(output/"development_1600_2000.csv",[r])
            if not state_equal(before,agent.checkpoint(cfg)) or not state_equal(before_rng,capture_rng()):
                raise AssertionError("Evaluation changed learning state or RNG")
            progress()

        def on_window(row,meter,pending,last):
            window=row["window"]
            row.update(replay_count=agent.replay.memory_counter,
                       critic_grad=agent.last_sac_stats.get("critic_grad",""),
                       actor_loss=agent.last_sac_stats.get("actor_loss",""),
                       actor_grad=agent.last_sac_stats.get("actor_grad",""))
            append_csv(output/"training.csv",[row])
            window_optimization_summary(output,window,optimization_cursor)
            manifest["completed_windows"]=window
            if window in OBSERVE and not last:
                save_validate(window,meter,pending,False)
            agent.pilot_window=window+1
            progress()

        progress()
        for offset,nwindows,scene in stream_plan(start,next_seed):
            agent.pilot_stream_seed,agent.pilot_window=scene,offset+1
            count_before=agent.replay.memory_counter
            duration=300*nwindows
            with patch("hrl_uav.experiment.QueueMDP",lambda settings:PilotMeter(settings,record_routes=False)):
                summary,_,_,_,meter=rollout(cfg,agent,scene,30*nwindows,macro_enabled=False,
                    train=True,window_offset=offset,complete_tail=True,on_window=on_window)
            if (summary["boundary_transitions"]!=30*nwindows or summary["final_pending_dropped"]!=0
                    or agent.replay.memory_counter-count_before!=540*nwindows or meter.reset_count!=1
                    or summary["time_s"]!=duration or meter.env._current_time()!=duration+1):
                raise AssertionError("Training stream boundary or replay conservation failed")
            np.testing.assert_allclose(summary["boundary_discount_min"],cfg["training"]["gamma_per_resource_frame"]**3,rtol=0,atol=1e-12)
            prefix=[r for r in meter.readings if r["time_s"]<=duration]
            np.testing.assert_allclose(sum(r[k] for r in prefix for k in ENERGY),summary["power_w"]*duration,rtol=1e-12,atol=1e-6)
            end=offset+nwindows
            append_csv(output/"boundary_audit.csv",[dict(stream_seed=scene,end_window=end,
                summary_time_s=summary["time_s"],environment_time_s=meter.env._current_time(),
                boundary_transitions=summary["boundary_transitions"],discount=summary["boundary_discount_min"],
                pending_dropped=0,replay_added=agent.replay.memory_counter-count_before,reset_count=meter.reset_count,
                summary_excludes_tail=True,deployment_radio_zero=True)])
            append_csv(output/"training_streams"/f"{scene}_service_with_tail.csv",meter.readings)
            append_csv(output/"training_streams"/f"{scene}_crn_with_tail.csv",meter.crn_frames)
            if agent.ppo_updates or agent.macro_rollout or parameter_delta(initial_macro,agent.macro_actor)!=0:
                raise AssertionError("PPO changed")
            slot=(scene-71040)%2
            anchor=save_snapshot(agent,cfg,output/"resume_anchors"/f"latest_slot{slot}.pt",end,scene,True,meter,None)
            manifest["latest_resume_anchor"]=anchor
            if any(offset < w <= end for w in OBSERVE):
                anchor=save_snapshot(agent,cfg,output/"resume_anchors"/f"resume_after_{end}.pt",end,scene,True,meter,None)
                manifest["retained_resume_anchors"].append(anchor)
            if end in OBSERVE:
                save_validate(end,meter,None,True)
            progress()
        after=source_hashes()
        if after!=sources or any(file_hash(Path(p))!=sha for p,sha in protected.items()):
            raise AssertionError("Source or protected pilot artifacts changed during continuation")
        manifest.update(source_hashes_after=after,source_unchanged=True,pilot_results_unchanged=True,
            learning_audit=dict(**counters,critic_updates=agent.sac_updates,
                critic_gradient_steps=agent.sac_critic_gradient_steps,target_updates=agent.sac_target_updates,
                actor_updates=agent.sac_actor_updates,ppo_updates=agent.ppo_updates,
                replay_count=agent.replay.memory_counter,macro_delta=parameter_delta(initial_macro,agent.macro_actor),
                sac_actor_delta_from120=parameter_delta(initial_actor,agent.sac_actor.action_net)))
        progress("WRITING_REPORT")
        analyze(output,manifest,fixed,development_fixed,origin_report)
        progress("LONG_COMPLETE")
        print(f"[LONG_COMPLETE] {output} elapsed={manifest['elapsed_seconds']:.1f}s",flush=True)
        return manifest
    except BaseException as error:
        manifest.update(status="STOPPED_ERROR",error=repr(error),elapsed_seconds=time.perf_counter()-started)
        write_json(output/"manifest.json",manifest)
        write_json(output/"progress.json",dict(status="STOPPED_ERROR",error=repr(error),
            completed_windows=manifest["completed_windows"],latest_resume_anchor=manifest["latest_resume_anchor"]))
        raise
