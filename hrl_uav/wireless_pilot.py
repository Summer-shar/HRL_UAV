"""Bounded fresh SAC pilot. Native updates and physical statistics are reused."""

import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from .bandwidth import summarize_links, summarize_meter
from .experiment import (append_csv, evaluate, isolated_rng, parameter_delta,
                         rollout, snapshot, write_json)
from .learning import DirectHRL
from .mdp import QueueMDP
from .priority_grid import PriorityRouting, file_hash
from .resumption import capture_rng
from .scene_prescreen import ROOT, assert_crn_equal, role_summary, source_hashes
from .wireless_check import AuditMeter, ENERGY, PRIORITY_MEASURES


CONFIG = ROOT / "configs/sac_only_wireless6_pilot.json"
SOURCE = ROOT / "results/priority_fixed_routing_grid/manifest.json"
CHECKPOINTS = [20, 40, 60, 80, 100, 120]
MEASURES = (
    "J", "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
    "task_completion_rate", *PRIORITY_MEASURES, "mean_queue_upload_mbit",
    "mean_queue_compute_mbit", "mean_queue_transfer_mbit", "queue_total_mbit",
    "oldest_age_s", "power_w", *ENERGY, "tx_throughput_mbps_wall",
    "compute_throughput_mbps_wall", "cpu_utilization_service", "late_queue_slope_mbit_s",
)
OPT_KEYS = ("critic_loss", "critic_loss_last_fit", "critic_grad", "target_std",
            "actor_loss", "actor_grad", "alpha", "temperature_grad")


def finite(value, name):
    if isinstance(value, torch.Tensor):
        ok = bool(torch.isfinite(value).all())
    else:
        ok = bool(np.isfinite(np.asarray(value)).all())
    if not ok:
        raise FloatingPointError(f"Non-finite {name}; stopping without parameter changes")


def verify_source_config(cfg):
    prior = json.loads(SOURCE.read_text(encoding="utf-8"))
    before = prior["source_hashes"]
    # The only pre-existing file changed for this runner is the observation-only hook.
    mismatches = [name for name, digest in before.items()
                  if file_hash(ROOT/name) != digest and name != "hrl_uav/experiment.py"]
    if mismatches:
        raise ValueError(f"Source changed since the fixed-routing grid: {mismatches}")
    base = next(c["config"] for c in prior["conditions"] if c["policy"] == "h1of3_l1of3_b6")
    for key in ("schema", "environment", "observation", "cpu_ghz", "phase_loads",
                "phase_dwell", "waypoint_radius_m", "max_displacement_m", "reward"):
        if cfg[key] != base[key]:
            raise ValueError(f"Frozen physical/observation configuration differs: {key}")
    expected_training = dict(base["training"], seed=101, pretrain_windows=120, joint_windows=0,
                             micro_algorithm="sac", sac_critic_steps_per_update=1)
    if cfg["training"] != expected_training:
        raise ValueError("Native training settings differ from the authorized configuration")
    if (cfg["validation_seeds"] != [33600, 33601]
            or cfg["test_seeds"] != [33610, 33611, 33612]
            or cfg["pilot"]["checkpoint_windows"] != CHECKPOINTS
            or cfg["pilot"]["reserved_final_seeds"] != list(range(33700, 33710))):
        raise ValueError("Unregistered scene or checkpoint schedule")
    return {"grid_manifest": str(SOURCE), "grid_manifest_sha256": file_hash(SOURCE),
            "grid_base_head": prior["base_head"], "all_prior_sources_except_hook_match": True,
            "source_hashes_before": before,
            "instrumentation_change": "experiment.rollout: optional on_window callback only"}


class PilotMeter(AuditMeter):
    def __init__(self, config, record_routes=True):
        super().__init__(config)
        self.check_local = False
        self.record_routes = record_routes
        self.route_readings = []

    def step_frame(self, action, gamma_frame=.99):
        frame = super().step_frame(action, gamma_frame)
        finite(frame["reward"], "frame reward")
        if self.record_routes and frame["phase"] == "service":
            for p, priority in enumerate(("high", "low")):
                for source in range(self.N):
                    route = frame["route"][p, source]
                    raw = np.asarray(action).reshape(2, self.N, self.N)[p, source]
                    role = self.uav_readings[-self.N+source]["role"]
                    self.route_readings.append({
                        "time_s": self.env._current_time(), "priority": priority,
                        "source": source, "role": role,
                        "window_300s": (self.env.global_micro_step-1)//3000 + 1,
                        "phase_index": int(self.schedule.phase_index_by_macro[
                            min(self.macro_index, len(self.schedule.phase_index_by_macro)-1)]),
                        "phase_age": self.phase_age, "local_ratio": float(route[source]),
                        **{f"destination_{j}": float(route[j]) for j in range(self.N)},
                        **{f"raw_{j}": float(raw[j]) for j in range(self.N)},
                        "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
                        "raw_saturation": float((np.abs(raw) > .95).mean()),
                    })
        return frame


def route_statistics(rows, label, seed, window, split):
    groups = {}
    for r in rows:
        for period in (0, int(r["window_300s"])):
            for source, role in (("all", "all"), (str(r["source"]), "all"), ("all", r["role"])):
                groups.setdefault((r["priority"], source, role, period), []).append(r)
    result = []
    for (priority, source, role, period), selected in sorted(groups.items()):
        local = np.array([r["local_ratio"] for r in selected])
        raw = np.array([[r[f"raw_{j}"] for j in range(3)] for r in selected])
        result.append({
            "split": split, "policy": label, "checkpoint_window": window, "seed": seed,
            "priority": priority, "source": source, "role": role, "window_300s": period,
            "samples": len(local), "local_mean": float(local.mean()), "local_std": float(local.std()),
            **{f"local_q{q}": float(np.percentile(local, q)) for q in (10, 50, 90)},
            **{f"destination_{j}_mean": float(np.mean([r[f"destination_{j}"] for r in selected]))
               for j in range(3)},
            "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
            "raw_saturation": float((np.abs(raw) > .95).mean()),
        })
    return result


def checked_agent(mdp, cfg, output):
    """Wrap native calls with read-only assertions, not a second learning algorithm."""
    agent = DirectHRL.from_mdp(mdp, cfg)
    initial_macro = snapshot(agent.macro_actor)
    initial_sac = snapshot(agent.sac_actor.action_net)
    audit = {"gradient_checks": 0, "critic_projected_input_checks": 0,
             "replay_route_checks": 0, "finite_update_checks": 0}
    networks = (agent.sac_actor.action_net, agent.sac_critic.critic_v,
                agent.sac_critic.target_critic_v, agent.macro_actor, agent.macro_critic)

    def gradient_check(grad):
        finite(grad, "gradient")
        audit["gradient_checks"] += 1
        return grad

    for net in networks[:2]:
        for parameter in net.parameters():
            parameter.register_hook(gradient_check)
    agent.entropy.log_alpha.register_hook(gradient_check)

    # Check the actual tensor entering both first layers, after native projection.
    def first_layer_input(module, inputs):
        route = inputs[0][:, -agent.action_dim:].reshape(-1, 3)
        finite(route, "executed-route critic input")
        if not bool(((route >= -1e-6) & (route <= 1+1e-6)).all()):
            raise AssertionError("Critic did not receive execution simplex coordinates")
        torch.testing.assert_close(route.sum(-1), torch.ones_like(route[:, 0]), rtol=0, atol=2e-6)
        audit["critic_projected_input_checks"] += 1

    for net in (agent.sac_critic.critic_v, agent.sac_critic.target_critic_v):
        if net.route_width != 3:
            raise AssertionError("Native executed-route critic is disabled")
        net.in_to_y1.register_forward_pre_hook(first_layer_input)
        net.q2_in_to_y1.register_forward_pre_hook(first_layer_input)

    for obj in (agent.sac_actor, agent.entropy):
        original = obj.learn

        def checked_learn(loss, *args, _original=original, **kwargs):
            finite(loss, "actor/temperature loss")
            return _original(loss, *args, **kwargs)

        obj.learn = checked_learn

    native_store = agent.store

    def store(state, action, route, reward, next_state, discount):
        for name, value in (("state", state), ("action", action), ("route", route),
                            ("reward", reward), ("next_state", next_state), ("discount", discount)):
            finite(value, name)
        if not (0 < discount <= 1):
            raise AssertionError("Invalid replay discount")
        np.testing.assert_allclose(mdp.route(action), route, rtol=0, atol=2e-7)
        audit["replay_route_checks"] += 1
        return native_store(state, action, route, reward, next_state, discount)

    agent.store = store
    native_update = agent.update_sac

    def update():
        before = agent.sac_updates
        native_update()
        if agent.sac_updates == before:
            return
        for net in networks:
            for parameter in net.parameters():
                finite(parameter, "network parameter")
        finite(agent.entropy.log_alpha, "log_alpha")
        finite(agent.entropy.log_alpha.exp(), "alpha")
        for name, value in agent.last_sac_stats.items():
            finite(value, name)
        if agent.macro_rollout or agent.ppo_updates != 0:
            raise AssertionError("PPO training was enabled")
        audit["finite_update_checks"] += 1
        stats = {key: agent.last_sac_stats.get(key, "") for key in OPT_KEYS}
        stats["temperature_grad"] = (float(agent.entropy.log_alpha.grad.detach().norm())
                                     if "actor_loss" in agent.last_sac_stats else "")
        append_csv(output / "optimization_stats.csv", [{
            "stream_seed": agent.pilot_stream_seed, "window": agent.pilot_window,
            "replay_count": agent.replay.memory_counter,
            "sac_critic_updates": agent.sac_updates,
            "sac_critic_gradient_steps": agent.sac_critic_gradient_steps,
            "sac_target_updates": agent.sac_target_updates,
            "sac_actor_updates": agent.sac_actor_updates, "ppo_updates": agent.ppo_updates,
            **stats,
        }])

    agent.update_sac = update
    agent.update_ppo = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("PPO forbidden"))
    return agent, initial_macro, initial_sac, audit


def evaluate_one(cfg, agent, seed, output, label, window, split, references):
    label_on_disk = f"{split}_{label}_{window}"
    folder = output / "evaluations"
    folder.mkdir(exist_ok=True)
    meters = []

    def factory(config):
        meter = PilotMeter(config)
        meters.append(meter)
        return meter

    with patch("hrl_uav.experiment.QueueMDP", factory):
        row = evaluate(cfg, agent, [seed], 90, folder, label_on_disk, macro_enabled=False)[0]
    meter = meters[0]
    if meter.reset_count != 1 or row["time_s"] != 900 or meter.env._current_time() != 900:
        raise AssertionError("Evaluation reset/tail/window mismatch")
    row.update(policy=label, checkpoint_window=window, split=split,
               **summarize_meter(meter.readings, 900))
    row.update({key: meter.readings[-1][key] for key in PRIORITY_MEASURES})
    row.update({key: sum(r[key] for r in meter.readings) for key in ENERGY})
    tail = [r for r in meter.readings if r["time_s"] > 600]
    row["late_queue_slope_mbit_s"] = float(np.polyfit(
        [r["time_s"] for r in tail],
        [r["queue_compute_mbit"]+r["queue_transfer_mbit"]+r["queue_upload_mbit"] for r in tail], 1)[0])
    np.testing.assert_allclose(sum(row[k] for k in ENERGY), row["power_w"]*900, rtol=1e-12, atol=1e-6)
    for key in MEASURES:
        finite(row[key], key)
    for suffix, records in (("service", meter.readings), ("uavs", meter.uav_readings),
                            ("links", meter.link_readings), ("crn", meter.crn_frames),
                            ("routes", meter.route_readings)):
        append_csv(folder / f"{label_on_disk}_{seed}_{suffix}.csv", records)
    with np.load(folder / f"{label_on_disk}_{seed}_trajectory.npz") as trajectory:
        geometry = {key: trajectory[key].copy() for key in ("uav", "users", "centroid", "macro_action")}
    if seed in references:
        frames, reference_geometry = references[seed]
        assert_crn_equal(frames, meter.crn_frames)
        for key in geometry:
            np.testing.assert_array_equal(geometry[key], reference_geometry[key])
    else:
        references[seed] = (copy.deepcopy(meter.crn_frames), geometry)
    append_csv(output / "pairing_audit.csv", [{"split": split, "policy": label,
        "window": window, "seed": seed, "frames": len(meter.crn_frames), "crn_equal": True,
        "centroid_equal": True, "reset_count": meter.reset_count, "complete_tail": False}])
    route_stats = route_statistics(meter.route_readings, label, seed, window, split)
    append_csv(output / "route_stats.csv", route_stats)
    append_csv(output / "role_summary.csv", [dict(split=split, checkpoint_window=window, **r)
        for r in role_summary(meter.uav_readings, .5, label, seed)])
    append_csv(output / "link_summary.csv", [dict(split=split, policy=label, checkpoint_window=window, **r)
        for r in summarize_links(meter.link_readings, 900, seed, 6)])
    print(f"[{split} {label} w{window} scene {seed}] J={row['J']:.6f} "
          f"delay={row['task_delay_weighted_ms']:.2f}ms Qend={row['queue_total_mbit']:.2f}", flush=True)
    return row, route_stats


def grouped_means(rows):
    return {label: {key: float(np.mean([r[key] for r in rows if r["policy"] == label]))
                    for key in MEASURES} for label in sorted({r["policy"] for r in rows})}


def report_text(report):
    lines = ["# 候选无线模型 SAC-only 120窗口资格训练", "",
             "本轮只有一个从零训练初始化(seed=101)，全部采用6MHz候选无线模型。"
             "没有PPO训练、教师、旧权重或中途调参。固定取window120，不选择中间成绩。", "",
             "## 1. 初始化与数据隔离", "",
             f"Fresh路由相对Uniform最大绝对差：{report['fresh']['uniform_max_abs_error']:.12g}；"
             f"PPO参数最大变化：{report['macro_delta']:.12g}；PPO更新：0。",
             "训练71000–71039共40条连续900s流，300/600s不清队列。"
             "窗口内模拟36000s，另有40次1s部署仅用于补全尾经验，不进入窗口成绩。",
             "开发验证33600/33601；资格终点33610/33611/33612只在120窗口后首次运行。"
             "33700–33709仅保留，未运行。Fixed-A/B来自旧开发网格，不是固定最优。", "",
             "## 2. 周期开发验证", "",
             "| 窗口 | J | 加权ms | 高优ms | 低优ms | 高优完成率 | 低优完成率 | 期末Mbit |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["validation_history"]:
        m = item["mean"]
        keys = ("J", "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
                "task_completion_high_rate", "task_completion_low_rate", "queue_total_mbit")
        lines.append(f"| {item['window']} | " + " | ".join(f"{m[k]:.6f}" for k in keys) + " |")
    history = report["validation_history"]
    delta = history[-1]["mean"]["J"] - history[0]["mean"]["J"]
    lines += ["", f"window120减window20的验证J：{delta:+.6f}。完整六个读数全部保留；"
              "它们只描述本次初始化的学习过程，不作超出预算的外推。", "",
              "## 3–5. 固定基线与资格终点", ""]
    for title, means in (("开发验证：SAC120与缓存固定规则", report["validation_final_means"]),
                         ("资格终点：三个场景均值", report["endpoint_means"])):
        lines += [f"### {title}", "",
                  "| 方法 | J | 加权ms | 高优ms | 低优ms | 高优完成率 | 低优完成率 | 高/低优未完成 | 期末Mbit | 功率W |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for label, r in means.items():
            lines.append(f"| {label} | {r['J']:.6f} | {r['task_delay_weighted_ms']:.3f} | "
                         f"{r['task_delay_high_ms']:.3f} | {r['task_delay_low_ms']:.3f} | "
                         f"{r['task_completion_high_rate']:.6f} | {r['task_completion_low_rate']:.6f} | "
                         f"{r['task_unfinished_high_count']:.2f}/{r['task_unfinished_low_count']:.2f} | "
                         f"{r['queue_total_mbit']:.3f} | {r['power_w']:.3f} |")
    lines += ["", "终点逐场景配对差(均为SAC120减固定规则；负J表示本次物理代价更低)：", "",
              "| 场景 | 对照 | ΔJ | Δ加权ms | Δ低优完成率 | Δ期末Mbit |",
              "|---|---|---:|---:|---:|---:|"]
    for r in report["endpoint_deltas"]:
        lines.append(f"| {r['seed']} | {r['baseline']} | {r['J']:+.6f} | "
                     f"{r['task_delay_weighted_ms']:+.3f} | {r['task_completion_low_rate']:+.6f} | "
                     f"{r['queue_total_mbit']:+.3f} |")
    lines += ["", "## 6–7. 实际执行路由", "",
              "| 场景 | 优先级 | 角色 | 本地均值 | 标准差 | P10/P50/P90 | 目的0/1/2均值 |",
              "|---|---|---|---:|---:|---|---|"]
    for r in report["endpoint_route_summary"]:
        lines.append(f"| {r['seed']} | {r['priority']} | {r['role']} | {r['local_mean']:.4f} | "
                     f"{r['local_std']:.4f} | {r['local_q10']:.4f}/{r['local_q50']:.4f}/{r['local_q90']:.4f} | "
                     + "/".join(f"{r[f'destination_{j}_mean']:.4f}" for j in range(3)) + " |")
    lines += ["", "上述角色每帧按当前负载排序。标准差包括该角色在不同时间和UAV上的差异，"
              "不等同于单个状态变量的因果效应。route_stats.csv另有逐源、逐300s窗口和全场景统计；"
              "evaluations/*_routes.csv与trajectory.npz保留逐服务帧原分数及实际路由。", "",
              "## 8. 学习与积压风险", "",
              f"更新与有限性检查：`{json.dumps(report['learning_audit'], ensure_ascii=False)}`。",
              f"优化统计范围：`{json.dumps(report['optimization_ranges'], ensure_ascii=False)}`。",
              "未触发非有限数值停止条件不等于评价网络估值准确；当前未做Q排序因果审计。"
              "alpha、梯度、损失全部在optimization_stats.csv。饱和比例定义为原分数abs(score)>0.95。",
              "终点完成率/未完成量并列报告；平均积压与后300s斜率见pilot_endpoint.csv。"
              "未完成任务不是一律超时，completed-only时延不能代替任务完成和积压检查。", "",
              "## 统计与复现边界", "",
              "J为原物理步代价平均，不等同于训练折扣soft return或completed-only时延。"
              "表格先按场景求值再算术平均，未池化任务，也未计算显著性。",
              "平均队列取含部署的0.5s帧末样本；期末队列为900s。"
              "功率包含CPU、上传、U2U、推进完整能量并除900s；CPU利用率含可计算的部署物理步。"
              "链路带宽/功率/噪声/速率的条件均值按活跃物理步；吞吐除墙钟时间。",
              "20/40/80/100窗口checkpoint位于连续流内部，仅为策略/学习状态快照；"
              "保存pending与采样位置，但不冒充包含完整环境的无缝恢复点。"
              "60/120窗口在真实尾部补全后保存。验证隔离并恢复训练RNG，不修改训练环境。",
              "固定规则不需要网络；模型配置和原结果保持不变。"
              "本轮完成后已停止，未运行保留集、3MHz、PPO或额外训练。", ""]
    return "\n".join(lines)


def run_pilot(output, smoke=False, smoke_report=None):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    provenance = verify_source_config(cfg)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    sources = source_hashes()
    protected = {str(p.relative_to(ROOT)): file_hash(p)
                 for folder in (SOURCE.parent, ROOT / "results/wireless_model_fixed_rule_check")
                 for p in folder.rglob("*") if p.is_file()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "codex/u2u-model":
        raise ValueError("Unexpected branch")
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    diff = subprocess.check_output(["git", "diff", "--", "hrl_uav", "tests", "configs", "scripts"], cwd=ROOT, text=True)
    if smoke:
        # Only the three authorized smoke budget fields differ, plus independent RNG seeds.
        cfg["training"].update(pretrain_windows=6, batch_size=16, replay_warmup=16, seed=9101)
        scene_base = 81000
    else:
        if smoke_report is None:
            raise ValueError("Completed isolated smoke is required")
        passed_smoke = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
        if passed_smoke["status"] != "SMOKE_COMPLETE" or passed_smoke["source_hashes_after"] != sources:
            raise ValueError("Smoke incomplete or source/config changed since smoke")
        scene_base = cfg["pilot"]["training_scene_seed_base"]
    windows = cfg["training"]["pretrain_windows"]
    train_seeds = list(range(scene_base, scene_base+windows//3))
    manifest = dict(provenance, actual_head=head, branch=branch, worktree_status=status,
        tracked_diff=diff, source_hashes=sources, config=cfg, checkpoint_source="fresh",
        checkpoint=None, teacher=False, training_seed_count=1, training=True, smoke=smoke,
        training_initialization_seed=cfg["training"]["seed"], training_scene_seeds=train_seeds,
        validation_seeds=[] if smoke else cfg["validation_seeds"],
        pilot_endpoint_seeds=[] if smoke else cfg["test_seeds"],
        reserved_final_seeds=cfg["pilot"]["reserved_final_seeds"], reserved_scenes_executed=[],
        evaluation_train=False, evaluation_complete_tail=False, training_complete_tail=True,
        macro_enabled=False, status="INITIALIZING", protected_result_hashes=protected,
        smoke_report=None if smoke else str(smoke_report),
        rng_sources={"python_random": cfg["training"]["seed"], "numpy_global": cfg["training"]["seed"],
            "torch_cpu_and_cuda": cfg["training"]["seed"],
            "replay_numpy_default_rng": cfg["training"]["seed"]+731,
            "environment_rng": "QueueMDP.reset(scene_seed); environment-local RandomState",
            "load_schedule_rng": "existing build_load_switch_schedule seed from QueueMDP.reset",
            "validation_rng_isolation": "existing isolated_rng restores Python/NumPy/Torch/CUDA"})
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        if smoke:
            test = subprocess.run([sys.executable, "-m", "unittest", "tests.test_wireless_pilot", "-v"],
                                  cwd=ROOT, text=True, capture_output=True)
            (output / "integration_test.txt").write_text(test.stdout+test.stderr, encoding="utf-8")
            if test.returncode != 0:
                raise AssertionError("Combined integration test failed; training not started")
        torch.set_num_threads(1)
        seed = cfg["training"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model_env = QueueMDP(cfg)
        model_env.reset(train_seeds[0], 90)
        agent, initial_macro, initial_sac, audit = checked_agent(model_env, cfg, output)
        raw = agent.sac_actor.choose_action(model_env.micro_observation(), deterministic=True)
        error = float(np.max(np.abs(model_env.route(raw)-1/3)))
        optimizers = (agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer,
                      agent.ppo_actor_optimizer, agent.ppo_critic_optimizer)
        fresh = {"uniform_max_abs_error": error, "raw_action": raw.tolist(),
                 "replay_count": agent.replay.memory_counter,
                 "optimizer_state_sizes": [len(o.state) for o in optimizers],
                 "sac_learning_rates": [o.param_groups[0]["lr"] for o in optimizers[:3]],
                 "alpha": agent.entropy.log_alpha.exp().item(), "target_entropy": agent.entropy.target_entropy,
                 "critic_steps_per_update": agent.critic_steps_per_update,
                 "observations": agent.observation_spec,
                 "actor_sha256": hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial_sac.values())).hexdigest()}
        manifest.update(fresh=fresh, device=str(next(agent.macro_actor.parameters()).device))
        write_json(output / "manifest.json", manifest)
        if error > 1e-7 or agent.replay.memory_counter or any(fresh["optimizer_state_sizes"]):
            raise AssertionError("Fresh initialization is not empty-state Uniform; stopping")
        references, fixed, validation, endpoint, all_routes, checkpoints = {}, [], [], [], [], []
        if not smoke:
            for label, alphas in cfg["pilot"]["fixed_rules"].items():
                for scene in cfg["validation_seeds"]:
                    row, routes = evaluate_one(cfg, PriorityRouting(*alphas), scene, output,
                                               label, 0, "validation", references)
                    fixed.append(row)
                    append_csv(output / "validation_fixed_baselines.csv", [row])
        manifest["status"] = "TRAINING"
        write_json(output / "manifest.json", manifest)

        def save_validate(window, meter, pending, stream_end):
            point = output / "checkpoints" / f"window{window:03d}.pt"
            data = agent.checkpoint(cfg)
            data["pilot_sampling"] = {"window": window, "stream_seed": agent.pilot_stream_seed,
                "stream_time_s": meter.env._current_time(), "at_stream_end": stream_end,
                "pending": pending, "rng": capture_rng(),
                "full_environment_saved": False,
                "scope": "Policy/learning snapshot; midstream points are not seamless-resume files"}
            torch.save(data, point)
            checkpoints.append({"window": window, "path": str(point), "sha256": file_hash(point),
                                "at_stream_end": stream_end, "replay_count": agent.replay.memory_counter})
            if not smoke:
                before_actor, before_alpha = snapshot(agent.sac_actor.action_net), agent.entropy.log_alpha.detach().clone()
                replay_count = agent.replay.memory_counter
                for scene in cfg["validation_seeds"]:
                    row, routes = evaluate_one(cfg, agent, scene, output, "SAC", window, "validation", references)
                    validation.append(row)
                    all_routes.extend(routes)
                    append_csv(output / "validation.csv", [row])
                assert parameter_delta(before_actor, agent.sac_actor.action_net) == 0
                assert torch.equal(before_alpha, agent.entropy.log_alpha)
                assert replay_count == agent.replay.memory_counter

        def on_window(row, meter, pending, last):
            row = dict(row, replay_count=agent.replay.memory_counter,
                       critic_grad=agent.last_sac_stats.get("critic_grad", ""),
                       actor_loss=agent.last_sac_stats.get("actor_loss", ""),
                       actor_grad=agent.last_sac_stats.get("actor_grad", ""))
            append_csv(output / "training.csv", [row])
            window = row["window"]
            if not smoke and window in CHECKPOINTS and not last:
                save_validate(window, meter, pending, False)
            agent.pilot_window = window+1
            manifest.update(completed_windows=window, checkpoints=checkpoints)
            write_json(output / "manifest.json", manifest)

        for offset, scene in zip(range(0, windows, 3), train_seeds):
            agent.pilot_stream_seed, agent.pilot_window = scene, offset+1
            count_before = agent.replay.memory_counter
            with patch("hrl_uav.experiment.QueueMDP", lambda settings: PilotMeter(settings, record_routes=False)):
                summary, _, _, _, meter = rollout(cfg, agent, scene, 90, macro_enabled=False,
                    train=True, window_offset=offset, complete_tail=True, on_window=on_window)
            if (summary["boundary_transitions"] != 90 or summary["final_pending_dropped"] != 0
                    or agent.replay.memory_counter-count_before != 1620 or meter.reset_count != 1
                    or summary["time_s"] != 900 or meter.env._current_time() != 901):
                raise AssertionError("Training continuation/transition count failed")
            gamma = cfg["training"]["gamma_per_resource_frame"]
            np.testing.assert_allclose(summary["boundary_discount_min"], gamma**3, rtol=0, atol=1e-12)
            prefix = [r for r in meter.readings if r["time_s"] <= 900]
            np.testing.assert_allclose(sum(r[k] for r in prefix for k in ENERGY), summary["power_w"]*900,
                                       rtol=1e-12, atol=1e-6)
            append_csv(output / "boundary_audit.csv", [{"stream_seed": scene, "end_window": offset+3,
                "summary_time_s": summary["time_s"], "environment_time_s": meter.env._current_time(),
                "boundary_transitions": summary["boundary_transitions"],
                "discount": summary["boundary_discount_min"], "pending_dropped": 0,
                "replay_added": agent.replay.memory_counter-count_before, "reset_count": meter.reset_count,
                "summary_excludes_tail": True, "deployment_radio_zero": True}])
            folder = output / "training_streams"
            folder.mkdir(exist_ok=True)
            append_csv(folder / f"{scene}_service_with_tail.csv", meter.readings)
            append_csv(folder / f"{scene}_crn_with_tail.csv", meter.crn_frames)
            if (not smoke and offset+3 in CHECKPOINTS) or (smoke and offset+3 == windows):
                save_validate(offset+3, meter, None, True)
        if not smoke:
            manifest["status"] = "ENDPOINT_EVALUATION"
            write_json(output / "manifest.json", manifest)
            for label, policy in [("SAC", agent)] + [(k, PriorityRouting(*a)) for k, a in cfg["pilot"]["fixed_rules"].items()]:
                for scene in cfg["test_seeds"]:
                    row, routes = evaluate_one(cfg, policy, scene, output, label, 120 if label == "SAC" else 0,
                                               "endpoint", references)
                    endpoint.append(row)
                    all_routes.extend(routes)
                    append_csv(output / "pilot_endpoint.csv", [row])
        macro_delta = parameter_delta(initial_macro, agent.macro_actor)
        if macro_delta != 0 or agent.ppo_updates or agent.macro_rollout or any(o.state for o in optimizers[3:]):
            raise AssertionError("PPO state changed in micro-only pilot")
        if agent.sac_actor_updates == 0 or audit["critic_projected_input_checks"] == 0:
            raise AssertionError("No native actor/critic update occurred")
        after = source_hashes()
        if after != sources or any(file_hash(ROOT/p) != sha for p, sha in protected.items()):
            raise AssertionError("Source or protected historical result changed during run")
        audit.update(sac_critic_updates=agent.sac_updates, sac_actor_updates=agent.sac_actor_updates,
                     sac_critic_gradient_steps=agent.sac_critic_gradient_steps,
                     sac_target_updates=agent.sac_target_updates, ppo_updates=0,
                     replay_count=agent.replay.memory_counter, macro_delta=macro_delta,
                     sac_actor_delta=parameter_delta(initial_sac, agent.sac_actor.action_net))
        manifest.update(status="SMOKE_COMPLETE" if smoke else "PILOT_COMPLETE", checkpoints=checkpoints,
                        source_hashes_after=after, source_unchanged=True, protected_results_unchanged=True,
                        elapsed_seconds=time.perf_counter()-started, learning_audit=audit)
        write_json(output / "manifest.json", manifest)
        if smoke:
            write_json(output / "report.json", dict(status="SMOKE_COMPLETE", fresh=fresh, learning_audit=audit,
                        scope="Functional smoke only; not an algorithm performance result"))
        else:
            import csv
            with (output / "optimization_stats.csv").open(encoding="utf-8-sig", newline="") as f:
                optimization = list(csv.DictReader(f))
            ranges = {k: {"min": min(float(r[k]) for r in optimization if r[k] != ""),
                          "max": max(float(r[k]) for r in optimization if r[k] != ""),
                          "last": next(float(r[k]) for r in reversed(optimization) if r[k] != "")}
                      for k in OPT_KEYS}
            history = [{"window": window, "mean": grouped_means([r for r in validation if r["checkpoint_window"] == window])["SAC"]}
                       for window in CHECKPOINTS]
            deltas = []
            for scene in cfg["test_seeds"]:
                sac = next(r for r in endpoint if r["policy"] == "SAC" and r["seed"] == scene)
                for base in (r for r in endpoint if r["policy"] != "SAC" and r["seed"] == scene):
                    deltas.append({"seed": scene, "baseline": base["policy"],
                                   **{key: sac[key]-base[key] for key in MEASURES}})
            append_csv(output / "endpoint_paired_deltas.csv", deltas)
            report = dict(status="PILOT_COMPLETE", fresh=fresh, macro_delta=macro_delta,
                training_seed_count=1, learning_audit=audit, optimization_ranges=ranges,
                validation_history=history, validation_rows=validation, fixed_validation_rows=fixed,
                validation_final_means=grouped_means(fixed+[r for r in validation if r["checkpoint_window"] == 120]),
                endpoint_means=grouped_means(endpoint), endpoint_rows=endpoint, endpoint_deltas=deltas,
                endpoint_route_summary=[r for r in all_routes if r["split"] == "endpoint" and r["policy"] == "SAC"
                    and r["window_300s"] == 0 and r["source"] == "all"],
                crn_paired=True, reserved_seeds_executed=[], source_unchanged=True,
                elapsed_seconds=manifest["elapsed_seconds"])
            write_json(output / "report.json", report)
            (output / "result.md").write_text(report_text(report), encoding="utf-8")
        print(f"[{manifest['status']}] {output} ({manifest['elapsed_seconds']:.1f}s)", flush=True)
        return manifest
    except Exception as error:
        manifest.update(status="STOPPED_ERROR", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        write_json(output / "manifest.json", manifest)
        raise
