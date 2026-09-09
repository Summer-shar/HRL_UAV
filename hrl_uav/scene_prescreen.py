"""Six fixed-rule arms using existing resource meters and evaluation paths."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from .bandwidth import MeteredMDP, UniformRouting, bandwidth_config, summarize_links, summarize_meter
from .experiment import append_csv, evaluate, isolated_rng, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE_HEAD = "07a727d8189be4f73c0218fca47603d8cbf6956b"
SEEDS = [33500, 33501, 33502]
MEASURES = (
    "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
    "task_completion_rate", "mean_queue_upload_mbit", "mean_queue_compute_mbit",
    "mean_queue_transfer_mbit", "queue_total_mbit", "queue_compute_mbit",
    "queue_transfer_mbit", "J", "power_w", "tx_throughput_mbps_wall",
    "compute_throughput_mbps_wall", "mean_active_links_service",
    "mean_rate_per_active_link_mbps", "cpu_utilization_service",
    "task_arrived_count", "task_completed_count", "unfinished", "time_s",
)
ROLE_FLOWS = (
    "assigned_source_mbit", "assigned_destination_mbit", "assigned_local_mbit",
    "actual_tx_in_mbit", "actual_tx_out_mbit", "actual_compute_input_mbit", "computed_mbit",
)


class LocalRouting(UniformRouting):
    """All six priority/source rows select their own UAV, without an env branch."""

    def choose_action(self, state, deterministic=False):
        return np.tile(2 * np.eye(3) - 1, (2, 1, 1)).ravel().astype(np.float32)


def arm_configs(config):
    arms = {}
    for enabled in (False, True):
        for rule, mhz in (("local", config["environment"]["u2u_bandwidth_mhz"]),
                          ("uniform_b3", 3), ("uniform_b6", 6)):
            cfg = bandwidth_config(config, mhz)
            cfg["environment"]["compute_during_deployment"] = enabled
            arms[f"p{int(enabled)}_{rule}"] = cfg
    reference = copy.deepcopy(config)
    reference["environment"].pop("compute_during_deployment", None)
    reference["environment"].pop("u2u_bandwidth_mhz")
    for cfg in arms.values():
        other = copy.deepcopy(cfg)
        other["environment"].pop("compute_during_deployment")
        other["environment"].pop("u2u_bandwidth_mhz")
        if other != reference:
            raise AssertionError("Unregistered arm configuration change")
    return arms


def digest_update(digest, *values):
    for value in values:
        array = np.asarray(value, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())


class PairedMeter(MeteredMDP):
    """Read-only CRN fingerprints in addition to the existing physical counters."""

    def __init__(self, config):
        super().__init__(config, record_links=True)
        self.arrivals_digest = hashlib.sha256()
        self.motion_digest = hashlib.sha256()
        self.crn_frames = []
        register = self.env._register_arrived_parent_task
        integrate = self.env._integrate_user_positions

        def record_arrival(parent):
            fields = (parent.task_id, parent.user_id, parent.source_uav, parent.task_type,
                      parent.priority, parent.birth_time, parent.size_mbits, parent.cpu_cycles)
            self.arrivals_digest.update((json.dumps(fields, separators=(",", ":")) + "\n").encode())
            return register(parent)

        def record_motion():
            result = integrate()
            digest_update(self.motion_digest, self.env.global_micro_step,
                          self.env.user_coords, self.env.uav_coords)
            return result

        self.env._register_arrived_parent_task = record_arrival
        self.env._integrate_user_positions = record_motion

    def reset(self, seed, macro_count):
        state = super().reset(seed, macro_count)
        digest_update(self.motion_digest, self.env.global_micro_step,
                      self.env.user_coords, self.env.uav_coords)
        self.schedule_hash = hashlib.sha256(
            np.asarray(self.schedule.phase_index_by_macro, dtype="<i8").tobytes()).hexdigest()
        return state

    def step_frame(self, action, gamma_frame=0.99):
        frame = super().step_frame(action, gamma_frame)
        geometry, load = hashlib.sha256(), hashlib.sha256()
        digest_update(geometry, self.env.user_coords, self.env.uav_coords, self.env.weighted_centroids)
        digest_update(load, self.macro_index, self.phase_age,
                      self.env.zone_expected_offered_load_mbps)
        self.crn_frames.append({
            "time_s": self.env._current_time(), "task_arrived_count": self.env.arrived_task_count,
            "arrivals_sha256": self.arrivals_digest.hexdigest(),
            "motion_sha256": self.motion_digest.hexdigest(),
            "geometry_sha256": geometry.hexdigest(), "load_sha256": load.hexdigest(),
            "schedule_sha256": self.schedule_hash,
        })
        return frame


def assert_crn_equal(reference, other):
    if len(reference) != len(other):
        raise AssertionError("CRN frame count mismatch")
    for index, (left, right) in enumerate(zip(reference, other)):
        for key in left:
            if left[key] != right[key]:
                raise AssertionError(f"CRN mismatch at frame {index}: {key}")


def role_summary(rows, frame_dt, label, seed):
    # Same role-weighted sums/denominators as run_bandwidth; no new queue metrics.
    summaries = []
    for role in ("heavy", "medium", "light"):
        selected = [r for r in rows if r["role"] == role]
        duration = len(selected) * frame_dt
        compute_steps = max(sum(r.get("compute_steps", r["service_steps"]) for r in selected), 1)
        result = {
            "policy": label, "seed": seed, "role": role, "duration_s": duration,
            "uav_ids": ",".join(str(i) for i in sorted({r["uav"] for r in selected})),
            "expected_source_load_mbps": float(np.mean([r["expected_source_load_mbps"] for r in selected])),
        }
        for key in ROLE_FLOWS:
            result[key] = sum(r[key] for r in selected)
            result[key.replace("_mbit", "_mbps_wall")] = result[key] / duration
        for key in ("queue_compute_mbit", "queue_compute_low_mbit", "queue_transfer_in_mbit"):
            result[f"mean_{key}"] = float(np.mean([r[key] for r in selected]))
        result["cpu_utilization_service"] = sum(r["cpu_utilization_sum"] for r in selected) / compute_steps
        result["cpu_low_utilization_service"] = sum(r["cpu_low_utilization_sum"] for r in selected) / compute_steps
        summaries.append(result)
    return summaries


def source_hashes():
    files = [p for directory in ("hrl_uav", "scripts", "configs", "tests")
             for p in (ROOT / directory).rglob("*") if p.suffix in (".py", ".json") and p.is_file()]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def resource_budget(arms, seed):
    budgets = {}
    with isolated_rng():
        for label in ("p0_local", "p1_local"):
            meter = MeteredMDP(arms[label])
            meter.reset(seed, 2)
            env = meter.env
            duty = 1.0 if env.compute_during_deployment else 1 - env.deployment_dt / env.macro_dt
            capacity = env.F_max_array / (env.C * 1e6) * duty
            loads = env.zone_expected_offered_load_mbps.copy()
            budgets[label[:2]] = {
                "phase_zero_source_load_mbps": loads.tolist(), "total_load_mbps": float(loads.sum()),
                "cpu_nominal_mbps_wall_per_uav": capacity.tolist(),
                "cpu_nominal_total_mbps_wall": float(capacity.sum()),
                "uniform_expected_assigned_mbps_per_uav": float(loads.sum() / env.N),
                "uniform_expected_transfer_mbps_wall": float(loads.sum() * (1 - 1 / env.N)),
                "cpu_duty": duty,
            }
    return budgets


def write_result(output, means, pairs, budgets, smoke):
    by_name = {row["policy"]: row for row in means}
    lines = ["# 部署期计算场景预筛选", "",
             "短 smoke，仅核对流程。" if smoke else "固定规则评估：3 个配对场景种子，每场景连续 900 s。未训练或加载任何网络。",
             "", "各列为场景均值。加权时延仅统计已完成父任务，必须与完成率和期末积压同时读取。",
             "", "|臂|加权/ms|高优/ms|低优/ms|完成率|平均上传/Mbit|平均计算/Mbit|平均转发/Mbit|期末总队列/Mbit|J|功率/W|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in means:
        fields = [row[k] for k in MEASURES[:10] if k not in ("queue_compute_mbit", "queue_transfer_mbit")]
        lines.append(f"|{row['policy']}|" + "|".join(f"{v:.6f}" for v in fields)
                     + f"|{row['J']:.6f}|{row['power_w']:.3f}|")
    lines += ["", "## A. 开启部署期 CPU 的配对变化", "", "以下均为 P1 减 P0；负时延差表示缩短。"]
    for suffix in ("local", "uniform_b3", "uniform_b6"):
        p0, p1 = by_name[f"p0_{suffix}"], by_name[f"p1_{suffix}"]
        lines.append(f"- {suffix}：加权时延 {p1['task_delay_weighted_ms']-p0['task_delay_weighted_ms']:+.3f} ms；"
                     f"完成率 {(p1['task_completion_rate']-p0['task_completion_rate'])*100:+.4f} 个百分点；"
                     f"期末队列 {p1['queue_total_mbit']-p0['queue_total_mbit']:+.3f} Mbit；"
                     f"功率 {p1['power_w']-p0['power_w']:+.3f} W。")
    lines += ["", "## B. P1 下 Local 与 Uniform", ""]
    for name in ("p1_uniform_b3", "p1_uniform_b6"):
        a, b = by_name[name], by_name["p1_local"]
        lines.append(f"- {name} 减 p1_local：加权时延 {a['task_delay_weighted_ms']-b['task_delay_weighted_ms']:+.3f} ms；"
                     f"完成率 {(a['task_completion_rate']-b['task_completion_rate'])*100:+.4f} 个百分点；"
                     f"期末队列 {a['queue_total_mbit']-b['queue_total_mbit']:+.3f} Mbit。")
    lines += ["", "## C. 3 到 6 MHz 的积压变化", ""]
    for mode in ("p0", "p1"):
        a, b = by_name[f"{mode}_uniform_b6"], by_name[f"{mode}_uniform_b3"]
        lines.append(f"- {mode}：平均转发队列 {a['mean_queue_transfer_mbit']-b['mean_queue_transfer_mbit']:+.3f} Mbit；"
                     f"平均计算队列 {a['mean_queue_compute_mbit']-b['mean_queue_compute_mbit']:+.3f} Mbit；"
                     f"U2U 吞吐 {a['tx_throughput_mbps_wall']-b['tx_throughput_mbps_wall']:+.3f} Mbps；"
                     f"计算吞吐 {a['compute_throughput_mbps_wall']-b['compute_throughput_mbps_wall']:+.3f} Mbps。")
    lines += ["", "## D. 条件依赖与解释边界", "",
              "这里只比较全本地与固定均分两条规则。若带宽变化改变两者的时延、完成率或积压差距，"
              "可以描述规则表现对带宽条件的依赖；不能据此给出最优路由或学习算法优劣结论。",
              "固定均分是否缓解任务分配不均衡，要结合 role_summary.csv 中分配到达、实际计算输入、"
              "计算服务和角色队列读取，不能把均分的分配量等同于即时到达目的 CPU 的工作量。",
              "", "## 资源账与配对", ""]
    for mode, b in budgets.items():
        lines.append(f"- {mode}：每架名义周期平均 CPU 能力 {b['cpu_nominal_mbps_wall_per_uav']} Mbps；"
                     f"总到达 {b['total_load_mbps']:.6f} Mbps；均分期望跨 UAV 转发 {b['uniform_expected_transfer_mbps_wall']:.6f} Mbps。")
    lines += ["- 上述是名义能力账，不是队列稳定性证明。",
              "- 六臂按同一 seed 逐帧核对任务出生明细哈希、逐物理步运动累计哈希、几何和负载；"
              "另核对现有评估轨迹 NPZ。Local 与 Uniform 的路由不同是实验变量，不要求跨规则路由相同。",
              "- 平均队列使用现有 0.5 s 帧末采样；queue_total_mbit 是 900 s 期末值。角色随当前阶段负载排序，非固定 UAV 编号。",
              "- CPU 利用率分母为 CPU 可服务物理步：P0 排除部署，P1 包含部署；不是同一可用时间分母。",
              "- 完整逐场景数据见 per_seed.csv，逐场景配对差见 paired_deltas.csv，角色与链路分别见 role_summary.csv、link_summary.csv。",
              "- 只有 3 个有限场景，不作显著性、无限时域稳定性或 Uniform 最优的断言。不自动修改任何参数。", ""]
    (output / "result.md").write_text("\n".join(lines), encoding="utf-8")


def run_prescreen(config, output, smoke=False, smoke_report=None):
    output = Path(output).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != BASE_HEAD or branch != "codex/u2u-model":
        raise ValueError("Fixed git baseline does not match")
    arms, sources = arm_configs(config), source_hashes()
    if not smoke:
        prior = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
        if (prior["status"] != "SMOKE_COMPLETE" or prior["arm_configs"] != arms
                or prior["source_hashes"] != sources or not prior["crn_equal"]):
            raise ValueError("Matching successful smoke is required before formal evaluation")
    seeds, macros = ([33500], 2) if smoke else (SEEDS, 90)
    output.mkdir(parents=True, exist_ok=False)
    budget = resource_budget(arms, seeds[0])
    manifest = {
        "status": "RUNNING", "base_head": head, "branch": branch, "arm_configs": arms,
        "source_hashes": sources, "seeds": seeds, "macro_count": macros, "duration_s": macros * 10,
        "mode": "smoke" if smoke else "prescreen", "training": False, "training_seed_count": 0,
        "checkpoint": None, "macro_enabled": False, "budget": budget,
        "smoke_report": str(smoke_report) if smoke_report else None,
        "queue_sampling": "Mean: existing 0.5 s post-frame samples; queue_total_mbit: terminal.",
        "only_environment_changes": ["compute_during_deployment", "u2u_bandwidth_mhz (uniform arms only)"],
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    rows, roles, links, audits = [], [], [], []
    references = {}
    max_error = 0.0
    try:
        for label, cfg in arms.items():
            local = label.endswith("local")
            for seed in seeds:
                print(f"[start {label} seed={seed} seconds={macros * 10}]", flush=True)
                instances = []

                def factory(settings):
                    meter = PairedMeter(settings)
                    instances.append(meter)
                    return meter

                with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", factory):
                    row = evaluate(cfg, LocalRouting() if local else UniformRouting(),
                                   [seed], macros, output, label, macro_enabled=False)[0]
                meter = instances[0]
                if row["time_s"] != macros * meter.env.macro_dt:
                    raise AssertionError("Evaluation duration mismatch")
                if seed in references:
                    assert_crn_equal(references[seed], meter.crn_frames)
                    with np.load(output / f"p0_local_{seed}_trajectory.npz") as base, \
                         np.load(output / f"{label}_{seed}_trajectory.npz") as candidate:
                        for key in ("uav", "users", "centroid", "macro_action"):
                            np.testing.assert_array_equal(base[key], candidate[key])
                else:
                    references[seed] = meter.crn_frames
                with np.load(output / f"{label}_{seed}_trajectory.npz") as trajectory:
                    expected = np.tile(np.eye(3), (2, 1, 1)) if local else np.full((2, 3, 3), 1 / 3)
                    np.testing.assert_allclose(trajectory["executed_route"],
                                               np.broadcast_to(expected, trajectory["executed_route"].shape))
                for suffix, data in (("service", meter.readings), ("uavs", meter.uav_readings),
                                     ("links", meter.link_readings), ("crn", meter.crn_frames)):
                    append_csv(output / f"{label}_{seed}_{suffix}.csv", data)
                mhz = cfg["environment"]["u2u_bandwidth_mhz"]
                row.update(compute_during_deployment=cfg["environment"]["compute_during_deployment"],
                           bandwidth_mhz=mhz, routing="local" if local else "uniform",
                           **summarize_meter(meter.readings, row["time_s"]))
                if local and (row["tx_served_mbit"] != 0 or row["queue_transfer_mbit"] != 0):
                    raise AssertionError("LocalRouting used U2U")
                role_rows = role_summary(meter.uav_readings, meter.env.resource_frame_dt, label, seed)
                if not smoke and any(r["uav_ids"] != "0,1,2" for r in role_rows):
                    raise AssertionError("Load-switch roles did not rotate across all UAVs")
                roles.extend(role_rows)
                links.extend({"policy": label, **r} for r in summarize_links(meter.link_readings, row["time_s"], seed, mhz))
                max_error = max(max_error, max(abs(r["balance_error"]) for r in meter.link_readings),
                                max(abs(r["compute_balance_error"]) for r in meter.uav_readings))
                rows.append(row)
                audits.append({"policy": label, "seed": seed, **meter.crn_frames[-1]})
                append_csv(output / "per_seed.csv", [row])
                append_csv(output / "role_summary.csv", role_rows)
                print(f"[done {label} seed={seed}] delay={row['task_delay_weighted_ms']:.3f} ms "
                      f"complete={row['task_completion_rate']:.6f} "
                      f"Qcpu={row['mean_queue_compute_mbit']:.3f} Qtx={row['mean_queue_transfer_mbit']:.3f} "
                      f"elapsed={(time.perf_counter()-started)/60:.2f} min", flush=True)
        if source_hashes() != sources:
            raise AssertionError("Sources or configuration changed during evaluation")
        means = [{"policy": label, "scene_count": len(seeds),
                  **{key: float(np.mean([r[key] for r in rows if r["policy"] == label])) for key in MEASURES}}
                 for label in arms]
        comparisons = [(f"p1_{s}", f"p0_{s}") for s in ("local", "uniform_b3", "uniform_b6")]
        comparisons += [(f"p1_uniform_b{b}", "p1_local") for b in (3, 6)]
        comparisons += [(f"p{p}_uniform_b6", f"p{p}_uniform_b3") for p in (0, 1)]
        lookup = {(r["policy"], r["seed"]): r for r in rows}
        pairs = [{"treatment": a, "control": b, "seed": seed,
                  **{key: lookup[a, seed][key] - lookup[b, seed][key] for key in MEASURES}}
                 for a, b in comparisons for seed in seeds]
        append_csv(output / "summary.csv", means)
        append_csv(output / "paired_deltas.csv", pairs)
        append_csv(output / "link_summary.csv", links)
        append_csv(output / "crn_summary.csv", audits)
        report = {
            "status": "SMOKE_COMPLETE" if smoke else "PRESCREEN_COMPLETE",
            "arm_configs": arms, "source_hashes": sources, "crn_equal": True,
            "queue_balance_max_error": max_error, "means": means, "rows": rows,
            "budget": budget, "paired_deltas": pairs, "role_summaries": roles,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_result(output, means, pairs, budget, smoke)
        write_json(output / "report.json", report)
        manifest.update(status=report["status"], crn_equal=True, source_unchanged=True,
                        queue_balance_max_error=max_error, elapsed_seconds=report["elapsed_seconds"])
        write_json(output / "manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
