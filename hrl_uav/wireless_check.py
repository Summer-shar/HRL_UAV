"""Bounded fixed-rule wireless comparison; reuse rollout, CRN and resource meters."""

import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from .bandwidth import UniformRouting, summarize_links, summarize_meter
from .env import UAV_MEC_Env
from .experiment import append_csv, evaluate, isolated_rng, rollout, write_json
from .scene_prescreen import (
    ROOT, SEEDS, LocalRouting, PairedMeter, assert_crn_equal, role_summary, source_hashes,
)


PSD = dict(noise_psd_w_per_hz=1.95e-19, u2u_tx_budget_w=1.0,
           u2u_reference_bandwidth_hz=512820.5128205128, u2u_reference_power_w=1.0)
ENERGY = ("energy_compute_j", "energy_upload_j", "energy_u2u_j", "energy_propulsion_j")
BASE_MEASURES = (
    "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "task_completion_rate",
    "mean_queue_upload_mbit", "mean_queue_compute_mbit", "mean_queue_transfer_mbit",
    "queue_total_mbit", "queue_compute_mbit", "queue_transfer_mbit", "J", "power_w",
    "tx_throughput_mbps_wall", "compute_throughput_mbps_wall", "mean_active_links_service",
    "mean_rate_per_active_link_mbps", "cpu_utilization_service", "task_arrived_count",
    "task_completed_count", "unfinished", "time_s",
)
PRIORITY_MEASURES = tuple(f"task_{kind}_{priority}_{unit}" for priority in ("high", "low")
                          for kind, unit in (("arrived", "count"), ("completed", "count"),
                                             ("unfinished", "count"), ("completion", "rate")))
MEASURES = BASE_MEASURES + PRIORITY_MEASURES + ENERGY
LINK_EXTRA = ("power_w_when_active", "noise_w_when_active", "tx_time_s", "tx_energy_j")
TOLERANCE = {"rtol": 1e-12, "atol": 1e-8}


def arm_configs(config):
    arms = {}
    for rule, mhz in (("local", 6), ("uniform", 3), ("uniform", 6)):
        for mode in (0, 1):
            cfg = copy.deepcopy(config)
            cfg["environment"].update(
                compute_during_deployment=True, pause_service_during_flight=True,
                u2u_bandwidth_mhz=mhz,
                wireless_model="psd_per_uav_budget" if mode else "legacy_fixed_noise")
            if mode:
                cfg["environment"].update(PSD)
            arms[f"w{mode}_{rule}_b{mhz}"] = cfg
    return arms


def fixed_rollout(*args, **kwargs):
    if kwargs.get("train", False) or kwargs.get("complete_tail", False) or kwargs.get("macro_enabled", True):
        raise ValueError("This check forbids learning, tail extension and PPO deployment")
    return rollout(*args, **dict(kwargs, train=False, complete_tail=False))


class AuditMeter(PairedMeter):
    """Read-only additions for energy components and parent-count conservation."""

    def __init__(self, config):
        super().__init__(config)
        self.reset_count = 0
        self.local_reference_rates = []
        self.local_state_samples = []

    def reset(self, seed, macro_count):
        self.reset_count += 1
        return super().reset(seed, macro_count)

    def step_frame(self, action, gamma_frame=0.99):
        frame = super().step_frame(action, gamma_frame)
        record = self.readings[-1]
        for name, info_key in zip(ENERGY[:3], ("energy_comp", "energy_uplink", "energy_trans")):
            record[name] = sum(float(info[info_key]) for info in frame["infos"])
        record["energy_propulsion_j"] = self.propulsion_share * len(frame["infos"])
        # CPU and radio flags are independent; radio must always pause at deployment.
        if frame["phase"] == "deployment":
            if record["energy_u2u_j"] != 0 or record["energy_upload_j"] != 0:
                raise AssertionError("Deployment radio energy is nonzero")
            if any(float(np.sum(info.get("tx_service_matrix_mbits", 0))) != 0 for info in frame["infos"]):
                raise AssertionError("Deployment U2U service is nonzero")
        alive = {priority: 0 for priority in ("high", "low")}
        for parent in self.env.parent_tasks.values():
            if parent.finish_time is None:
                alive[parent.priority] += 1
        for priority in alive:
            arrived = int(self.env.arrived_priority_count[priority])
            completed = int(self.env.completed_priority_count[priority])
            if arrived != completed + alive[priority]:
                raise AssertionError(f"Parent conservation failed for {priority}")
            record[f"task_arrived_{priority}_count"] = arrived
            record[f"task_completed_{priority}_count"] = completed
            record[f"task_unfinished_{priority}_count"] = alive[priority]
            record[f"task_completion_{priority}_rate"] = completed / max(arrived, 1)
        if self.check_local:
            self.local_reference_rates.append(self.env.compute_u2g_rate_bps())
            self.local_state_samples.append(np.r_[self.env.Q_loc_high, self.env.Q_loc_low,
                                                   self.env.Q_tx_high.ravel(), self.env.Q_tx_low.ravel(),
                                                   record["queue_upload_mbit"]])
        return frame


def assert_local_equal(old_row, new_row, old_rates, new_rates, old_states, new_states):
    np.testing.assert_allclose(new_rates, old_rates, **TOLERANCE)
    np.testing.assert_allclose(new_states, old_states, **TOLERANCE)
    for key in MEASURES:
        np.testing.assert_allclose(new_row[key], old_row[key], **TOLERANCE, err_msg=key)
    for row in (old_row, new_row):
        for key in ("energy_u2u_j", "tx_served_mbit", "queue_transfer_mbit"):
            if row[key] != 0:
                raise AssertionError(f"Local routing has nonzero {key}")
    return {"passed": True, "tolerance": TOLERANCE,
            "rate_max_abs_error_bps": float(np.max(np.abs(np.asarray(new_rates)-old_rates))),
            "queue_max_abs_error_mbit": float(np.max(np.abs(np.asarray(new_states)-old_states))),
            "metric_max_abs_errors": {key: abs(new_row[key]-old_row[key]) for key in MEASURES}}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def historical_context(arms, source):
    manifest_path = source / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prior["status"] != "PRESCREEN_COMPLETE" or prior["seeds"] != SEEDS or prior["duration_s"] != 900:
        raise ValueError("Historical P1 reference is incomplete or has different scenes/duration")
    mappings = {"w0_local_b6": "p1_local", "w0_uniform_b3": "p1_uniform_b3", "w0_uniform_b6": "p1_uniform_b6"}
    for current, historical in mappings.items():
        cfg = copy.deepcopy(arms[current])
        cfg["environment"].pop("wireless_model")
        if cfg != prior["arm_configs"][historical]:
            raise ValueError(f"Effective configuration differs from historical {historical}")
    files = (manifest_path, source / "per_seed.csv", source / "report.json")
    return {"path": str(source), "base_head": prior["base_head"], "mappings": mappings,
            "configs_equal_except_explicit_legacy_mode": True,
            "source_hashes": prior["source_hashes"],
            "file_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def write_result(output, means, pairs, local_audits, historical, smoke):
    m = {row["policy"]: row for row in means}
    lines = ["# 无线模型固定规则对照", "",
             f"范围：{'六臂各20秒 smoke' if smoke else '六臂 × 3场景 × 连续900秒'}；"
             "部署期CPU开启，通信暂停；宏观动态质心，无学习。33500–33502为复用开发场景，不是全新最终测试集。",
             "各表为逐场景指标的算术平均，完成时延只统计已完成父任务。", "",
             "## 1. 全本地参考校准", "",
             f"逐帧上行速率最大绝对差 {max(a['rate_max_abs_error_bps'] for a in local_audits):.12g} bit/s；"
             f"队列最大绝对差 {max(a['queue_max_abs_error_mbit'] for a in local_audits):.12g} Mbit。"
             "检查容差 rtol=1e-12、atol=1e-8（按对应单位）。两臂U2U服务和发送能量严格为0。",
             "全本地新旧模型的任务计数、时延、J和功率均通过同容差检查。", "",
             "## 2. 与已有P1预筛选的对应", "",
             f"历史基准 {historical['base_head']}；有效配置仅新增显式旧无线模式名称。"
             "历史源码指纹与当前指纹分别记录，不冒充同一源码。"]
    if smoke:
        lines.append("本次只有20秒，尚未与900秒历史指标作数值对照。")
    else:
        errors = historical["metric_max_abs_errors"]
        lines.append(f"对照全部共同数值列最大绝对差 {max(errors.values(), default=0):.12g}；"
                     f"容差检查：{historical['all_within_tolerance']}。逐场景差见 historical_p1_deltas.csv，"
                     "外生过程与历史CRN记录逐帧核对通过。")
    lines += ["", "## 3. 候选模式的3/6 MHz均分", "",
              "| 方法 | 加权ms | 高优ms | 低优ms | 高优完成率 | 低优完成率 | 平均上传Mbit | 平均计算Mbit | 平均转发Mbit | 期末总Mbit | J | 功率W |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in means:
        keys = ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
                "task_completion_high_rate", "task_completion_low_rate", "mean_queue_upload_mbit",
                "mean_queue_compute_mbit", "mean_queue_transfer_mbit", "queue_total_mbit", "J", "power_w")
        lines.append("| " + row["policy"] + " | " + " | ".join(f"{row[k]:.6f}" for k in keys) + " |")
    lines += ["", "| 候选均分 | 高/低优未完成数 | 计算能量J | 上传能量J | U2U能量J | 推进能量J | 实际U2U吞吐Mbps | 平均活跃链路数 | 活跃链路平均速率Mbps |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("w1_uniform_b3", "w1_uniform_b6"):
        r = m[label]
        lines.append(f"| {label} | {r['task_unfinished_high_count']:.3f}/{r['task_unfinished_low_count']:.3f} | "
                     + " | ".join(f"{r[k]:.6f}" for k in ENERGY + ("tx_throughput_mbps_wall", "mean_active_links_service", "mean_rate_per_active_link_mbps")) + " |")
    a, b = m["w1_uniform_b6"], m["w1_uniform_b3"]
    lines += ["", f"候选6MHz减3MHz：平均转发积压 {a['mean_queue_transfer_mbit']-b['mean_queue_transfer_mbit']:+.6f} Mbit，"
              f"平均计算积压 {a['mean_queue_compute_mbit']-b['mean_queue_compute_mbit']:+.6f} Mbit，"
              f"低优完成率 {(a['task_completion_low_rate']-b['task_completion_low_rate'])*100:+.6f} 个百分点。",
              "", "## 4. 新无线模型相对旧模型", ""]
    for rule in ("local_b6", "uniform_b3", "uniform_b6"):
        a, b = m[f"w1_{rule}"], m[f"w0_{rule}"]
        lines.append(f"- {rule}：加权完成时延 {a['task_delay_weighted_ms']-b['task_delay_weighted_ms']:+.6f} ms；"
                     f"低优完成率 {(a['task_completion_low_rate']-b['task_completion_low_rate'])*100:+.6f} 个百分点；"
                     f"平均转发队列 {a['mean_queue_transfer_mbit']-b['mean_queue_transfer_mbit']:+.6f} Mbit；"
                     f"期末总积压 {a['queue_total_mbit']-b['queue_total_mbit']:+.6f} Mbit；"
                     f"U2U能量 {a['energy_u2u_j']-b['energy_u2u_j']:+.6f} J。")
    lines += ["", "## 5. 证据边界与统计分母", "",
              "这是整套无线定义改变的影响，不是噪声或功率单因素消融，也不是算法提升。"
              "实际吞吐是给定规则和到达过程下实际传出的数据量，不能当作网络固定容量或推导通用卸载比例上限。",
              "不能因completed-only均值较低而忽略未完成任务；本报告并列计数、完成率和积压。"
              "只有两个固定路由规则与三个复用开发场景，不能称Uniform最优，不评价未运行的PPO/SAC，"
              "不作无限时域稳定、硬件有效或最终泛化主张。",
              "- per_seed.csv：任务按高/低优出生、完成和未完成计数；均值完成率为各场景比率的算术平均。",
              "- 平均队列：全部0.5s帧末等间隔采样，含部署；期末队列为900s（smoke为20s）时刻。",
              "- J沿用原物理步代价平均；功率与各分项能量包含完整推进账，能量和除完整墙钟时长等于功率。",
              "- uav_summary.csv：逐UAV计算吞吐除完整墙钟时间，CPU利用率分母含部署期CPU可服务步。",
              "- role_summary.csv：按每帧负载排序归属heavy/medium/light，角色不是固定UAV编号。",
              "- link_summary.csv：带宽/功率/噪声/速率为该有向链路活跃物理步条件均值；发送时间和能量为窗口累计。"
              "实际发送吞吐除完整墙钟时间。资源帧记录中的 *_sum 是物理步样本求和，不是时间平均。",
              "- 旧模式未提供分配功率/噪声的矩阵记录，其新增候选链路列留空；旧模式系统U2U能量直接读取原info，不反推。",
              "- 每场景仅reset一次，300/600秒处不清队列；train=false、complete_tail=false，无流末额外部署。",
              "- 共同随机数检查使用任务出生明细、物理步运动累计、负载调度和几何指纹，另核对轨迹NPZ；跨规则不要求路由相同。",
              "", "运行结束后停止；未修改模型、带宽公式、奖励、论文或历史成绩。", ""]
    (output / "result.md").write_text("\n".join(lines), encoding="utf-8")


def run_check(config, output, smoke=False, smoke_report=None):
    output = Path(output).resolve()
    arms, sources = arm_configs(config), source_hashes()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "codex/u2u-model":
        raise ValueError("Unexpected branch")
    if not smoke:
        proof = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
        if (proof["status"] != "SMOKE_COMPLETE" or proof["arm_configs"] != arms
                or proof["source_hashes"] != sources or proof["base_head"] != head
                or not proof["crn_equal"] or not all(a["passed"] for a in proof["local_calibration"])):
            raise ValueError("A matching six-arm successful smoke is required")
    history = historical_context(arms, ROOT / "results/scene_prescreen_deployment_compute")
    history["source_files_changed_since_p1"] = [p for p, h in history["source_hashes"].items() if sources.get(p) != h]
    seeds, macros = ([33500], 2) if smoke else (SEEDS, 90)
    output.mkdir(parents=True, exist_ok=False)
    defaults = {name: p.default for name, p in inspect.signature(UAV_MEC_Env).parameters.items()}
    manifest = dict(
        status="RUNNING", base_head=head, branch=branch, arm_configs=arms,
        effective_environments={name: {**defaults, **cfg["environment"], "effective_cpu_ghz": cfg["cpu_ghz"]}
                                for name, cfg in arms.items()}, source_hashes=sources,
        worktree_status=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True),
        tracked_diff=subprocess.check_output(["git", "diff", "HEAD", "--", "hrl_uav", "scripts", "tests", "configs"], cwd=ROOT, text=True),
        seeds=seeds, scene_scope="Reused development scenes, not unseen final tests", duration_s=macros*10,
        macro_count=macros, training=False, training_seed_count=0, checkpoint=None,
        macro_enabled=False, train=False, complete_tail=False, historical_reference=history,
        smoke_report=str(smoke_report) if smoke_report else None, local_tolerance=TOLERANCE,
    )
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    rows, roles, links, uavs, calibrations, historic_deltas = [], [], [], [], [], []
    references, local_references = {}, {}
    balance_error = 0.0
    historical_lookup = {(r["policy"], int(r["seed"])): r for r in read_csv(Path(history["path"]) / "per_seed.csv")}
    try:
        for label, cfg in arms.items():
            local = "_local_" in label
            for seed in seeds:
                print(f"[start {label} seed={seed} duration={macros*10}s]", flush=True)
                meters = []

                def factory(settings):
                    meter = AuditMeter(settings)
                    meter.check_local = local
                    meters.append(meter)
                    return meter

                with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", factory), \
                     patch("hrl_uav.experiment.rollout", fixed_rollout), \
                     patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("No learning networks")):
                    row = evaluate(cfg, LocalRouting() if local else UniformRouting(), [seed], macros,
                                   output, label, macro_enabled=False)[0]
                meter, env = meters[0], meters[0].env
                if env.wireless_model != cfg["environment"]["wireless_model"] or not env.compute_during_deployment or not env.pause_service_during_flight:
                    raise AssertionError("Effective mode mismatch")
                if meter.reset_count != 1 or row["time_s"] != macros*10 or len(meter.readings) != macros*20:
                    raise AssertionError("Continuous-stream time/reset mismatch")
                if seed in references:
                    assert_crn_equal(references[seed], meter.crn_frames)
                    with np.load(output/f"w0_local_b6_{seed}_trajectory.npz") as original, np.load(output/f"{label}_{seed}_trajectory.npz") as current:
                        for key in ("uav", "users", "centroid", "macro_action"):
                            np.testing.assert_array_equal(original[key], current[key])
                else:
                    references[seed] = meter.crn_frames
                with np.load(output/f"{label}_{seed}_trajectory.npz") as trajectory:
                    expected = np.tile(np.eye(3), (2, 1, 1)) if local else np.full((2, 3, 3), 1/3)
                    np.testing.assert_allclose(trajectory["executed_route"], np.broadcast_to(expected, trajectory["executed_route"].shape))
                row.update(wireless_model=env.wireless_model, compute_during_deployment=True,
                           bandwidth_mhz=env.B_u2u_total/1e6, routing="local" if local else "uniform",
                           **summarize_meter(meter.readings, row["time_s"]))
                # Keep the same CSV columns for both modes; read actual legacy energy too.
                row["tx_energy_j"] = sum(r["energy_u2u_j"] for r in meter.readings)
                row["tx_power_w_wall"] = row["tx_energy_j"]/row["time_s"]
                row.update({key: sum(r[key] for r in meter.readings) for key in ENERGY})
                row.update({key: meter.readings[-1][key] for key in PRIORITY_MEASURES})
                np.testing.assert_allclose(sum(row[k] for k in ENERGY), row["power_w"]*row["time_s"], rtol=1e-12, atol=1e-6)
                if local:
                    rates, states = np.asarray(meter.local_reference_rates), np.asarray(meter.local_state_samples)
                    np.savez_compressed(output/f"{label}_{seed}_calibration.npz", rates_bps=rates, queues_mbit=states)
                    if label.startswith("w0"):
                        local_references[seed] = (row, rates, states)
                    else:
                        old_row, old_rates, old_states = local_references[seed]
                        calibrations.append(dict(seed=seed, **assert_local_equal(old_row, row, old_rates, rates, old_states, states)))
                if not smoke and label in history["mappings"]:
                    old_label = history["mappings"][label]
                    old_row = historical_lookup[old_label, seed]
                    delta = {"policy": label, "historical_policy": old_label, "seed": seed}
                    for key in old_row.keys() & row.keys():
                        if key in ("seed", "policy", "routing", "compute_during_deployment"):
                            continue
                        try:
                            delta[key] = float(row[key]) - float(old_row[key])
                        except (ValueError, TypeError):
                            pass
                    historic_deltas.append(delta)
                    prior_crn = read_csv(Path(history["path"]) / f"{old_label}_{seed}_crn.csv")
                    for old, current in zip(prior_crn, meter.crn_frames):
                        for key in current:
                            if key.endswith("sha256") and old[key] != current[key]:
                                raise AssertionError(f"Historical CRN mismatch: {key}")
                    if len(prior_crn) != len(meter.crn_frames):
                        raise AssertionError("Historical CRN length mismatch")
                for suffix, data in (("service", meter.readings), ("uavs", meter.uav_readings),
                                     ("links", meter.link_readings), ("crn", meter.crn_frames)):
                    append_csv(output/f"{label}_{seed}_{suffix}.csv", data)
                role_rows = role_summary(meter.uav_readings, env.resource_frame_dt, label, seed)
                roles.extend(role_rows)
                for r in summarize_links(meter.link_readings, row["time_s"], seed, row["bandwidth_mhz"]):
                    links.append({"policy": label, **r, **{k: r.get(k) for k in LINK_EXTRA}})
                for uav in range(env.N):
                    detail = [r for r in meter.uav_readings if r["uav"] == uav]
                    uavs.append(dict(policy=label, seed=seed, uav=uav,
                                     computed_mbit=sum(r["computed_mbit"] for r in detail),
                                     compute_throughput_mbps_wall=sum(r["computed_mbit"] for r in detail)/row["time_s"],
                                     cpu_utilization_compute_available=sum(r["cpu_utilization_sum"] for r in detail)/sum(r["compute_steps"] for r in detail)))
                balance_error = max(balance_error, max(abs(r["balance_error"]) for r in meter.link_readings),
                                    max(abs(r["compute_balance_error"]) for r in meter.uav_readings))
                rows.append(row)
                append_csv(output/"per_seed.csv", [row])
                print(f"[done {label} seed={seed}] delay={row['task_delay_weighted_ms']:.3f}ms "
                      f"completion={row['task_completion_rate']:.6f} terminalQ={row['queue_total_mbit']:.3f}Mbit "
                      f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if sources != source_hashes():
            raise AssertionError("Source/config changed during run")
        for name, digest in history["file_hashes"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise AssertionError("Historical result changed")
        means = [{"policy": label, "scene_count": len(seeds), **{key: float(np.mean([r[key] for r in rows if r["policy"] == label])) for key in MEASURES}} for label in arms]
        lookup = {(r["policy"], r["seed"]): r for r in rows}
        comparisons = [(f"w1_{s}", f"w0_{s}") for s in ("local_b6", "uniform_b3", "uniform_b6")]
        comparisons += [(f"w{w}_uniform_b6", f"w{w}_uniform_b3") for w in (0, 1)]
        comparisons += [(f"w1_uniform_b{b}", "w1_local_b6") for b in (3, 6)]
        pairs = [{"treatment": a, "control": b, "seed": seed, **{key: lookup[a, seed][key]-lookup[b, seed][key] for key in MEASURES}} for a, b in comparisons for seed in seeds]
        if historic_deltas:
            keys = sorted(set(historic_deltas[0]) - {"policy", "historical_policy", "seed"})
            history["metric_max_abs_errors"] = {key: max(abs(r[key]) for r in historic_deltas) for key in keys}
            history["all_within_tolerance"] = all(
                np.isclose(lookup[r["policy"], r["seed"]][k], float(historical_lookup[r["historical_policy"], r["seed"]][k]), **TOLERANCE)
                for r in historic_deltas for k in keys)
            # Stable explicit schema, independent of hash randomization/set order.
            historic_deltas = [{k: r[k] for k in ("policy", "historical_policy", "seed", *keys)} for r in historic_deltas]
            append_csv(output/"historical_p1_deltas.csv", historic_deltas)
        for name, data in (("summary", means), ("paired_deltas", pairs), ("role_summary", roles),
                           ("link_summary", links), ("uav_summary", uavs)):
            append_csv(output/f"{name}.csv", data)
        report = dict(status="SMOKE_COMPLETE" if smoke else "FIXED_RULE_CHECK_COMPLETE", base_head=head,
                      arm_configs=arms, source_hashes=sources, crn_equal=True, local_calibration=calibrations,
                      queue_balance_max_error=balance_error, parent_conservation=True,
                      means=means, rows=rows, paired_deltas=pairs, historical_reference=history,
                      elapsed_seconds=time.perf_counter()-started)
        write_result(output, means, pairs, calibrations, history, smoke)
        write_json(output/"report.json", report)
        manifest.update(status=report["status"], source_unchanged=True, source_hashes_after=source_hashes(),
                        crn_equal=True, parent_conservation=True, queue_balance_max_error=balance_error,
                        local_calibration=calibrations, elapsed_seconds=report["elapsed_seconds"])
        write_json(output/"manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", source_unchanged=sources == source_hashes())
        write_json(output/"manifest.json", manifest)
        raise
