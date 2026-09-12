"""Bounded continuous-mission functional smoke, never a training entry point."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from .bandwidth import summarize_links, summarize_meter
from .experiment import append_csv, isolated_rng, rollout, write_json
from .mdp import QueueMDP
from .mission_provenance import PHYSICS_BASE_COMMIT, physics_provenance
from .ppo_frozen import MacroPairMeter, SOURCE_DIGESTS, exogenous_records
from .priority_grid import file_hash
from .scene_prescreen import ROOT, assert_crn_equal, digest_update, role_summary
from .trajectory_check import ReadOnlySAC, forbidden, no_learning
from .wireless_check import ENERGY, PRIORITY_MEASURES
from .wireless_pilot import PilotMeter, route_statistics


BASE = "92d3214788ee94de82ae3c0186fdd3ac1566e765"
SPEC_PATH = ROOT / "configs/mission_rpgm_v1_smoke.json"
CONTROLLERS = ("hover", "current_centroid", "progressive_centroid", "predictive_centroid", "blind_patrol")
CORE_FILES = ("hrl_uav/env.py", "hrl_uav/mdp.py", "hrl_uav/experiment.py", "hrl_uav/learning.py",
              "hrl_uav/agents/sac_agent.py", "hrl_uav/routing.py", "hrl_uav/load_switching.py",
              "hrl_uav/bandwidth.py", "hrl_uav/ppo_frozen.py", "hrl_uav/trajectory_check.py",
              "hrl_uav/scene_prescreen.py", "hrl_uav/wireless_check.py", "hrl_uav/wireless_pilot.py")


def load_spec():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if (spec["base_commit"] != BASE or spec["scene_seed"] != 33600 or spec["duration_s"] != 60
            or spec["controllers"] != list(CONTROLLERS) or spec["progressive_coefficient"] != .5
            or spec["predictive_lead_s"] != 10 or spec["blind_patrol"]["speed_mps"] != 2
            or spec["blind_patrol"]["leg_duration_s"] != 30 or spec["blind_patrol"]["world_axis"] != "x"):
        raise ValueError("Only the fixed one-scene, five-controller, 60 s smoke is authorized")
    return spec


def mission_config(source_config, spec=None):
    spec = load_spec() if spec is None else spec
    cfg = copy.deepcopy(source_config)
    cfg["environment"].update(copy.deepcopy(spec["environment_overrides"]))
    env = cfg["environment"]
    if (cfg["observation"] != {"layout": "split_v1", "routing_ready": False}
            or env["wireless_model"] != "psd_per_uav_budget" or env["u2u_bandwidth_mhz"] != 6
            or cfg["cpu_ghz"] != 10 or cfg["max_displacement_m"] != 20):
        raise ValueError("Unexpected SAC101 physical or observation configuration")
    return cfg


def load_source(spec):
    source = ROOT / spec["source_checkpoint"]
    manifest_path = source.parent.parent / "manifest.json"
    if file_hash(source) != SOURCE_DIGESTS["checkpoint"] or file_hash(source) != spec["checkpoint_sha256"]:
        raise ValueError("SAC101 checkpoint digest mismatch")
    if file_hash(manifest_path) != SOURCE_DIGESTS["manifest.json"]:
        raise ValueError("SAC101 source manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if (saved["config"] != manifest["config"] or manifest["status"] != "LONG_COMPLETE"
            or manifest["training_initialization_seed"] != 101
            or saved["run_state"]["completed_windows"] != 2000
            or not saved["run_state"]["at_rollout_boundary"]
            or saved["updates"]["ppo"] != 0 or saved["stage"] != "pretrain"):
        raise ValueError("Not the declared SAC-only window2000 source")
    # This is a new environment, not a falsified same-stage continuation.
    for name in CORE_FILES:
        if name in manifest["source_hashes"] and name not in ("hrl_uav/env.py", "hrl_uav/mdp.py"):
            if file_hash(ROOT / name) != manifest["source_hashes"][name]:
                raise ValueError(f"Unregistered SAC/source change: {name}")
    return saved, manifest


class MissionPolicy:
    def __init__(self, sac, name):
        if name not in CONTROLLERS:
            raise ValueError("Unknown smoke controller")
        self.sac_actor = sac.sac_actor
        self.name, self.meter, self.macro_calls = name, None, 0

    def macro_action(self, state, enabled=True, deterministic=False):
        if not deterministic:
            raise AssertionError("Mission smoke only permits deterministic actions")
        index = self.macro_calls
        self.macro_calls += 1
        if self.name == "hover":
            delta = np.zeros((3, 2))
        elif self.name == "blind_patrol":
            # No env/state/geometry access. All sources execute the same path.
            sign = 1.0 if (index // 3) % 2 == 0 else -1.0
            delta = np.tile([20.0 * sign, 0.0], (3, 1))
        else:
            env = self.meter.env
            delta = env.weighted_centroids - env.uav_coords
            if self.name == "progressive_centroid":
                delta = .5 * delta
            elif self.name == "predictive_centroid":
                delta = delta + 10.0 * self.meter.centroid_velocity
        # Preserve direction when the requested target exceeds the 20 m ball.
        delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True) / 20.0, 1.0)
        return (delta / 20.0).ravel(), 0.0

    store = update_sac = update_ppo = forbidden


class MissionMeter(MacroPairMeter, PilotMeter):
    """Use existing queue, route, energy and CRN meters; add motion records only."""

    def __init__(self, config):
        super().__init__(config)
        self.positions, self.commands, self.micro_states = [], [], []
        self.centroid_digest = hashlib.sha256()
        integrate, snapshot = self.env._integrate_user_positions, self.env.record_trajectory_snapshot

        def record_centroids():
            result = integrate()
            digest_update(self.centroid_digest, self.env.global_micro_step,
                          self.env.weighted_centroids, self.env.mobility_group_centers)
            return result

        def record_positions():
            snapshot()
            self.positions.append((self.env._current_time(), self.env.uav_coords.copy(),
                                   self.env.user_coords.copy(), self.env.weighted_centroids.copy(),
                                   self.env.mobility_group_centers.copy()))

        self.env._integrate_user_positions = record_centroids
        self.env.record_trajectory_snapshot = record_positions

    def reset(self, seed, macro_count):
        result = super().reset(seed, macro_count)
        if not self.positions:
            self.env.record_trajectory_snapshot()
        return result

    def start_macro(self, action):
        before = self.env.uav_coords.copy()
        info = super().start_macro(action)
        for i in range(self.N):
            row = dict(macro=self.macro_index, time_s=self.env._current_time(), uav=i,
                       distance_m=float(np.linalg.norm(info["actual_displacement_m"][i])),
                       speed_mps=float(self.env.current_macro_v[i]),
                       energy_propulsion_j=float(info["E_macro_per_uav"][i]),
                       propulsion_increment_j=float(info["E_propulsion_increment_per_uav"][i]),
                       transient_energy_j=0.0, separation_scale=info["separation_projection_scale"])
            for key, value in (("start", before[i]), ("target", info["projected_uav_coords"][i]),
                               ("raw", np.asarray(action).reshape(3, 2)[i]),
                               ("centroid", self.env.weighted_centroids[i])):
                row.update({f"{key}_{axis}": float(value[k]) for k, axis in enumerate(("x", "y"))})
            self.commands.append(row)
        return info

    def step_frame(self, action, gamma_frame=.99):
        self.micro_states.append(self.micro_observation().copy())
        frame = super().step_frame(action, gamma_frame)
        rw = self.reward_config
        row = self.readings[-1]
        row["holding_cost"] = sum(float(i["task_holding_cost_weighted"]) for i in frame["infos"]) * rw["delay_weight"] / rw["holding_norm_task_s"]
        row["propulsion_increment_j"] = self.propulsion_delta_share * len(frame["infos"])
        row["energy_cost"] = (sum(row[k] for k in ENERGY[:3]) + row["propulsion_increment_j"]) * rw["energy_weight"] / rw["energy_norm_j"]
        if not np.isclose(row["holding_cost"] + row["energy_cost"], frame["cost"], atol=1e-10, rtol=1e-12):
            raise AssertionError("Mission objective components do not close")
        self.crn_frames[-1]["centroid_group_microsteps_sha256"] = self.centroid_digest.hexdigest()
        if frame["phase"] != "service" or row["service_steps"] != 5 or row["compute_steps"] != 5:
            raise AssertionError("Continuous mission has an unexpected service gap")
        return frame


def run_controller(cfg, sac, name, seed=33600, seconds=60):
    if seed != 33600 or seconds not in (20, 60):
        raise ValueError("Only scene33600 at 20/60 s is permitted, never reserved scenes or 900 s")
    meter = MissionMeter(cfg)
    policy = MissionPolicy(sac, name)
    policy.meter = meter
    with isolated_rng(), no_learning(sac), torch.no_grad(), patch("hrl_uav.experiment.QueueMDP", return_value=meter):
        summary, _, _, _, _ = rollout(cfg, policy, seed, seconds // 10, macro_enabled=True,
                                      train=False, record=False, complete_tail=False)
    if (meter.integration_steps != seconds * 10 or len(meter.micro_states) != seconds * 2
            or policy.macro_calls != seconds // 10 or meter.reset_count != 1
            or meter.env.resource_frame_prepare_count != seconds * 2
            or meter.env.mobility_update_count != seconds * 2):
        raise AssertionError("Mission smoke clock/decision count mismatch")
    summary.update(policy=name, seed=seed, duration_s=seconds, **summarize_meter(meter.readings, seconds))
    summary.update({key: meter.readings[-1][key] for key in PRIORITY_MEASURES})
    summary.update({key: sum(r[key] for r in meter.readings) for key in ENERGY})
    summary.update(J_holding=sum(r["holding_cost"] for r in meter.readings) / meter.integration_steps,
                   J_energy=sum(r["energy_cost"] for r in meter.readings) / meter.integration_steps,
                   propulsion_increment_j=sum(r["propulsion_increment_j"] for r in meter.readings),
                   fleet_distance_m=sum(r["distance_m"] for r in meter.commands),
                   boundary_user_hits=meter.env.mission_user_boundary_hits,
                   sac_decisions=len(meter.micro_states), macro_decisions=policy.macro_calls,
                   sac_update_increment=0, ppo_updates=0, replay_writes=0, replay_samples=0)
    positions = np.asarray([p[1] for p in meter.positions])
    centroids = np.asarray([p[3] for p in meter.positions])
    summary["mean_uav_centroid_distance_m"] = float(np.linalg.norm(positions - centroids, axis=2).mean())
    np.testing.assert_allclose(summary["J"], summary["J_holding"] + summary["J_energy"], atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(summary["power_w"], sum(summary[k] for k in ENERGY) / seconds, atol=1e-8)
    return summary, meter


def paired_records(meter):
    records = exogenous_records(meter)
    for row, original in zip(records, meter.crn_frames):
        row["centroid_group_microsteps_sha256"] = original["centroid_group_microsteps_sha256"]
    return records


def plot_smoke(output, meters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if f in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.sans-serif": [chosen], "axes.unicode_minus": False})
    titles = {"hover": "固定悬停", "current_centroid": "当前质心追踪", "progressive_centroid": "渐进质心 0.5",
              "predictive_centroid": "历史速度预测质心", "blind_patrol": "无业务巡航（仅诊断）"}
    colors = ("#157f72", "#b23b4a", "#4356ad")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, name in zip(axes.ravel(), CONTROLLERS):
        trace = meters[name].positions
        uav, users, centroid, groups = [np.asarray([p[k] for p in trace]) for k in (1, 2, 3, 4)]
        allocations = meters[name].env.user_allocations
        for i, color in enumerate(colors):
            ax.scatter(users[0, allocations == i, 0], users[0, allocations == i, 1], s=9, color=color, alpha=.2)
            for user in np.where(allocations == i)[0]:
                ax.plot(users[:, user, 0], users[:, user, 1], color=color, alpha=.12, lw=.5)
            ax.plot(uav[:, i, 0], uav[:, i, 1], color=color, lw=2, label=f"UAV {i+1}")
            ax.plot(centroid[:, i, 0], centroid[:, i, 1], color=color, ls=":", lw=.9)
            ax.plot(groups[:, i, 0], groups[:, i, 1], color=color, ls="--", lw=1.3)
            ax.scatter(*uav[0, i], color=color, marker="s", s=28)
            ax.scatter(*uav[-1, i], color=color, marker="x", s=40)
        ax.set(title=titles[name], xlabel="x / m", ylabel="y / m", xlim=(280, 740), ylim=(280, 720))
        ax.set_aspect("equal")
        ax.grid(alpha=.18)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(.03, .9, "场景33600，真实0–60秒轨迹\n\n实线：UAV；方形：起点；叉号：终点\n虚线：任务群中心；点线：业务质心\n浅色细线：地面用户\n\n每条mission路线仍按900秒定义。\n此图只画实际运行的前60秒，未加速路线。\n不是收敛、长期性能或泛化证据。", va="top", fontsize=12, linespacing=1.8)
    fig.suptitle("连续运动 mission V1：冻结 SAC 的短程功能检查", fontsize=16)
    fig.savefig(output / "mission_smoke_trajectory.png", dpi=170)
    fig.savefig(output / "mission_smoke_trajectory.pdf")
    plt.close(fig)


def write_report(output, rows, pairs, checks):
    columns = ("J", "J_holding", "J_energy", "power_w", "task_delay_high_ms", "task_delay_low_ms",
               "task_completion_high_rate", "task_completion_low_rate", "queue_total_mbit")
    lines = ["# mission_rpgm_v1 短程功能检查", "", "仅场景33600，各控制器60秒；路线仍按900秒定义，未运行900秒。",
             "SAC101 window2000确定性零样本迁移；无PPO网络、无学习、无经验读写或采样。",
             "", "| 控制器 | J | 驻留项 | 能量项 | 完整功率/W | 高优/ms | 低优/ms | 高优完成率 | 低优完成率 | 期末积压/Mbit |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("|" + r["policy"] + "|" + "|".join(f"{r[k]:.6f}" for k in columns) + "|")
    lines += ["", "## 无业务巡航诊断", "", "以下差值为Blind Patrol减对应控制器，保留真实通信、路由和队列变化。"]
    for p in pairs:
        lines.append(f"- 相对{p['reference']}：ΔJ={p['delta_J']:+.8f}，驻留项={p['delta_J_holding']:+.8f}，"
                     f"能量项={p['delta_J_energy']:+.8f}，推进能量={p['delta_energy_propulsion_j']:+.3f} J。")
    lines += ["", f"短程能量驱动排序风险标记：{checks['energy_driven_wins']}。",
              "标记定义：巡航J更低、驻留项不更低且能量项更低（数值容差1e-10）；仅为诊断，不是正式门禁。",
              "没有标记也不能排除长期巡航收益；巡航输给追踪不证明所有轨迹收益都来自通信或队列。",
              "准稳态模型忽略加减速/转弯能量，相对悬停增量允许为负；无业务巡航不是算法对比基线。",
              "", "## 数据口径", "", "完成时延只统计已完成父任务；需并列完成率、未完成数及积压。60秒从空队列开始，不能证明稳定。",
              "queues.csv为0.5秒帧末积压；trajectory.npz为0.1秒实际轨迹；routes.csv为确定性执行路由。",
              "J为原物理步代价平均；完整功率含全部推进能量。链路指标复用原统计器，速率按活跃步，吞吐按完整60秒。",
              "crn.json记录逐任务出生、逐物理步用户运动/群中心/业务质心及负载配对。",
              "", "## 停止", "", "全部学习状态核验不变；仅执行指定短检查。未运行保留场景、未训练、未自动追加实验。"]
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output):
    spec = load_spec()
    if Path(output).exists():
        raise FileExistsError("Use a new smoke output directory; historical output is never overwritten")
    physics = physics_provenance()
    saved, source_manifest = load_source(spec)
    cfg = mission_config(saved["config"], spec)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = (*CORE_FILES, "hrl_uav/mission_smoke.py", "configs/mission_rpgm_v1_smoke.json",
               "scripts/run_mission_rpgm_v1_smoke.py", "tests/test_mission_rpgm_v1.py")
    hashes = {name: file_hash(ROOT / name) for name in tracked}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", actual_head=head, base_commit=BASE,
                    physics_base_commit=PHYSICS_BASE_COMMIT, physics_hashes=physics,
                    branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                    worktree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                    protocol=spec, config=cfg, source_checkpoint_sha256=spec["checkpoint_sha256"],
                    source_manifest_sha256=SOURCE_DIGESTS["manifest.json"], source_config=source_manifest["config"],
                    original_source_hashes=source_manifest["source_hashes"], source_hashes_before=hashes,
                    training=False, training_seed_count=0, transfer="deterministic zero-shot, not continuation",
                    evaluation_train=False, complete_tail=False, reserved_scenes_executed=[],
                    statistical_scope="one reused development scene, 60 s empty-queue smoke only")
    # Preregister every controller before any environment is evaluated.
    write_json(output / "manifest.json", manifest)
    with isolated_rng():
        sac = ReadOnlySAC(saved)
    rows, meters, checks, reference = [], {}, {}, None
    try:
        for name in CONTROLLERS:
            row, meter = run_controller(cfg, sac, name)
            records = paired_records(meter)
            if reference is None:
                reference = records
            assert_crn_equal(reference, records)
            directory = output / name
            directory.mkdir()
            append_csv(directory / "queues.csv", meter.readings)
            append_csv(directory / "commands.csv", meter.commands)
            append_csv(directory / "routes.csv", meter.route_readings)
            append_csv(directory / "role_summary.csv", role_summary(meter.uav_readings, .5, name, 33600))
            append_csv(directory / "route_summary.csv", route_statistics(meter.route_readings, name, 33600, 2000, "mission_smoke"))
            append_csv(directory / "link_summary.csv", summarize_links(meter.link_readings, 60, 33600, 6))
            write_json(directory / "crn.json", records)
            trace = meter.positions
            np.savez_compressed(directory / "trajectory.npz", time_s=np.asarray([p[0] for p in trace]),
                                uav=np.asarray([p[1] for p in trace]), users=np.asarray([p[2] for p in trace]),
                                centroid=np.asarray([p[3] for p in trace]), groups=np.asarray([p[4] for p in trace]),
                                micro_state=np.asarray(meter.micro_states))
            write_json(directory / "freeze_check.json", sac.assert_unchanged())
            rows.append(row)
            meters[name] = meter
            append_csv(output / "per_policy.csv", [row])
        blind = rows[-1]
        keys = ("J", "J_holding", "J_energy", "power_w", *ENERGY, "task_delay_high_ms", "task_delay_low_ms",
                "task_completion_high_rate", "task_completion_low_rate", "queue_total_mbit")
        pairs = [{"policy": "blind_patrol", "reference": r["policy"],
                  **{f"delta_{k}": blind[k] - r[k] for k in keys}} for r in rows[:-1]]
        checks["energy_driven_wins"] = [p["reference"] for p in pairs if p["delta_J"] < -1e-10
                                        and p["delta_J_holding"] >= -1e-10 and p["delta_J_energy"] < -1e-10]
        checks.update(crn_paired=True, frozen_sac=sac.assert_unchanged(), longer_horizon_verified=False)
        append_csv(output / "paired_deltas.csv", pairs)
        write_json(output / "checks.json", checks)
        plot_smoke(output, meters)
        write_report(output, rows, pairs, checks)
        manifest["source_hashes_after"] = {name: file_hash(ROOT / name) for name in tracked}
        manifest["source_unchanged_during_smoke"] = hashes == manifest["source_hashes_after"]
        if not manifest["source_unchanged_during_smoke"]:
            raise AssertionError("Source changed during smoke")
        manifest["status"] = "SMOKE_COMPLETE_NOT_PERFORMANCE_VALIDATION"
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="FAILED_STOPPED", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    return {"output": str(output), "status": manifest["status"], "energy_driven_wins": checks["energy_driven_wins"]}
