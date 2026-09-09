"""Fresh micro-only comparison: the sole config difference is ready-task state."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .experiment import write_json
from .qualification import run_mainline


ARMS = ("control_104", "ready_116")
ROLES = ("heavy", "medium", "light")


def arm_configs(config):
    control, treatment = copy.deepcopy(config), copy.deepcopy(config)
    # This command retains the explicit historical 104/116 input comparison.
    for arm in (control, treatment):
        arm.setdefault("observation", {})["layout"] = "shared_v1"
    control.setdefault("observation", {})["routing_ready"] = False
    treatment.setdefault("observation", {})["routing_ready"] = True
    check = copy.deepcopy(treatment)
    check["observation"]["routing_ready"] = False
    assert check == control
    return control, treatment


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def action_summary(folder, report, frame_dt):
    matrices, role_matrices, workloads = [], [], []
    for evaluation in report["final"]:
        seed = int(evaluation["seed"])
        prefix = f"{report['final_label']}_{seed}"
        with np.load(folder/f"{prefix}_trajectory.npz") as saved:
            route = saved["executed_route"].copy()
            assert not np.any(saved["macro_action"])
        rows = read_csv(folder/f"{prefix}_uavs.csv")
        service = sorted((r for r in rows if int(r["service_steps"]) > 0),
                         key=lambda r: (float(r["time_s"]), int(r["uav"])))
        assert len(service) == 3*len(route)
        order = np.argsort(np.array([ROLES.index(r["role"]) for r in service]).reshape(-1, 3), axis=1)
        roles = np.stack([route[i][:, ids][:, :, ids] for i, ids in enumerate(order)])
        matrices.append(route.mean(0))
        role_matrices.append(roles.mean(0))
        by_role = {}
        for role in ROLES:
            selected = [r for r in rows if r["role"] == role]
            by_role[role] = sum(float(r["assigned_destination_mbit"]) for r in selected)/(len(selected)*frame_dt)
        workloads.append({"seed": seed, "destination_workload_mbps_wall": by_role})
    return {"route_mean_fixed_ids": np.mean(matrices, axis=0).tolist(),
            "route_mean_roles": np.mean(role_matrices, axis=0).tolist(),
            "role_order": list(ROLES), "per_seed_workloads": workloads,
            "scope": "Decision-mean high/low routes; source rows and destination columns. Workloads use actual assigned bits per wall second."}


def paired_metrics(control, treatment):
    result = {}
    a = {int(r["seed"]): r for r in control["final"]}
    b = {int(r["seed"]): r for r in treatment["final"]}
    assert a.keys() == b.keys()
    for key in control["mean"]:
        delta = np.array([b[s][key] - a[s][key] for s in sorted(a)])
        margin = stats.t.ppf(0.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else None
        result[key] = {"ready_minus_control": delta.tolist(), "mean": float(delta.mean()),
                       "paired_t95_ci": [float(delta.mean()-margin), float(delta.mean()+margin)] if margin is not None else None}
    return result


def plot_comparison(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm, name, color in zip(ARMS, ("原状态 104维", "待路由增强 116维"), ("#bb5937", "#28768e")):
        validation = read_csv(output/arm/"validation.csv")
        windows = sorted({int(r["window"]) for r in validation})
        values = [np.mean([float(r["task_delay_weighted_ms"])/1000 for r in validation if int(r["window"]) == w]) for w in windows]
        axes[0, 0].plot(windows, values, "o-", color=color, label=name, markersize=3)
        training = read_csv(output/arm/"training.csv")
        axes[0, 1].plot([int(r["window"]) for r in training], [float(r["window_J"]) for r in training], color=color, label=name)
        queues = [read_csv(output/arm/f"{reports[arm]['final_label']}_{int(r['seed'])}_queues.csv") for r in reports[arm]["final"]]
        times = np.array([float(r["time_s"]) for r in queues[0]])
        values = np.array([[float(r["queue_total_mbit"]) for r in q] for q in queues])
        axes[1, 0].plot(times, values.mean(0), color=color, label=name)
        axes[1, 0].fill_between(times, values.min(0), values.max(0), color=color, alpha=.1)
        flow = reports[arm]["action_summary"]["per_seed_workloads"]
        means = [np.mean([v["destination_workload_mbps_wall"][role] for v in flow]) for role in ROLES]
        x = np.arange(3) + (-.18 if arm == ARMS[0] else .18)
        bars = axes[1, 1].bar(x, means, width=.36, color=color, label=name)
        axes[1, 1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    window_seconds = float(training[0]["time_s"])/int(training[0]["window"])
    uniform = reports[ARMS[0]]["uniform_mean"]["task_delay_weighted_ms"]/1000
    axes[0, 0].axhline(uniform, linestyle=":", color="#555555", label="固定均分测试均值（非RL）")
    axes[0, 0].set(title="固定验证场景：加权完成时延", xlabel=f"微观训练窗口（每窗{window_seconds:g}秒）", ylabel="秒，越低越好", ylim=(0, None))
    axes[0, 1].set(title="训练原始目标：未完成任务持有代价＋软能耗", xlabel="微观训练窗口", ylabel="J，越低越好", ylim=(0, None))
    axes[1, 0].set(title=f"最后权重：连续{times[-1]:g}秒总积压", xlabel="仿真时间（秒）", ylabel="Mbit", ylim=(0, None))
    axes[1, 1].axhline(9, color="#a33333", linestyle="--", label="每机有效计算上限 9 Mbps")
    axes[1, 1].set(title="最后权重：分派到各角色的计算工作量", ylabel="Mbps（含部署暂停）", xticks=range(3), xticklabels=["重载区目的机", "中载区目的机", "轻载区目的机"])
    axes[1, 1].set_ylim(0, max(12, axes[1, 1].get_ylim()[1]*1.2))
    for ax in axes.flat:
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("仅改变待路由状态：两臂分别从头训练纯SAC，宏观固定质心", fontsize=14)
    n_validation = len({r["seed"] for r in validation})
    n_test = len(reports[ARMS[0]]["final"])
    fig.text(.5, .02, f"1个训练种子；{n_validation}个验证场景；{n_test}个测试场景。阴影为测试场景范围，不是训练种子置信区间。\n完成时延需与积压、完成率一起解读；不挑中间权重、不自动启动PPO。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(output/"comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


def run_observation_ablation(config, output, smoke=False):
    if config["environment"]["num_uavs"] != 3:
        raise ValueError("This scoped 104/116 comparison is defined for three UAVs")
    if config["training"].get("critic_action_representation") != "executed_route":
        raise ValueError("Keep the current executed-route Critic fixed in both arms")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configs = arm_configs(config)
    write_json(output/"protocol.json", {
        "status": "FROZEN_BEFORE_TRAINING", "smoke": smoke,
        "arms": dict(zip(ARMS, configs)),
        "only_config_difference": "observation.routing_ready",
        "new_features": {"routing_ready_work": {"dimension": 6, "scale_mbit": 10},
                         "routing_ready_count": {"dimension": 6, "scale_tasks": 10}},
        "feature_semantics": "Read completed_upload_tasks_by_uav at decision time; never consume the list or use future within-frame completions. Keep all 104 original features.",
        "training": "Fresh networks/replay/optimizers for each arm. Same seed/budget, not identical tensors across different input dimensions. No teacher or pretrained weights.",
        "stopping": "Pretrain only even if quality passes. Fixed last weights, original time cap, no budget extension. Test results never select checkpoints.",
        "statistics": "Paired scenario differences are descriptive for one training initialization. Existing evaluation seeds are reused for an exploratory comparison, not new confirmatory evidence.",
        "criteria": "Report original qualification checks unchanged, original J, completion, queue growth and delay, plus actual destination workload. Higher routing entropy alone is not success.",
    })
    reports, manifests = {}, {}
    for arm, arm_config in zip(ARMS, configs):
        write_json(output/"progress.json", {"stage": "training", "arm": arm, "completed_arms": list(reports)})
        print(f"[state comparison] starting {arm}, fresh seed={arm_config['training']['seed']}", flush=True)
        report = run_mainline(arm_config, output/arm, smoke=smoke, pretrain_only=True)
        report["action_summary"] = action_summary(output/arm, report, arm_config["environment"]["resource_frame_dt"])
        reports[arm] = report
        manifests[arm] = json.loads((output/arm/"manifest.json").read_text(encoding="utf-8"))
        write_json(output/f"{arm}_action_summary.json", report["action_summary"])
    control, treatment = (reports[arm] for arm in ARMS)
    cross_crn = {}
    for a, b in zip(control["final"], treatment["final"]):
        assert a["seed"] == b["seed"]
        arrays = [np.load(output/arm/f"{reports[arm]['final_label']}_{int(a['seed'])}_trajectory.npz") for arm in ARMS]
        cross_crn[str(int(a["seed"]))] = {
            "users": bool(np.array_equal(arrays[0]["users"], arrays[1]["users"])),
            "uav": bool(np.array_equal(arrays[0]["uav"], arrays[1]["uav"])),
            "arrivals": a["task_arrived_count"] == b["task_arrived_count"],
        }
        for array in arrays:
            array.close()
    integrity = {
        "dimensions": [manifests[a]["state_dim"] for a in ARMS] == [104, 116],
        "same_source": manifests[ARMS[0]]["source_hashes"] == manifests[ARMS[1]]["source_hashes"],
        "source_unchanged": all(r["source_hashes_unchanged"] for r in reports.values()),
        "fresh_only": all(not m["teacher"] and m["old_checkpoint"] is None for m in manifests.values()),
        "no_ppo": all(not r["joint_training_started"] and r["macro_mean_max_delta"] == 0 and all(s["ppo_updates"] == 0 for s in r["stage_audits"]) for r in reports.values()),
        "completed": all(r["status"] != "BUDGET_STOP_INCOMPLETE" for r in reports.values()),
        "paired_exogenous": all(all(v.values()) for v in cross_crn.values()),
    }
    result = {"status": "SMOKE_ONLY" if smoke else ("STATE_ABLATION_COMPLETE" if all(integrity.values()) else "INCOMPLETE_OR_INVALID"),
              "integrity": integrity, "cross_arm_crn": cross_crn,
              "arms": {arm: {"mean": r["mean"], "qualified": r["pretrain_admission"], "minutes": r["minutes"],
                              "stage_audits": r["stage_audits"], "action_summary": r["action_summary"]} for arm, r in reports.items()},
              "ready_minus_control": paired_metrics(control, treatment),
              "uniform_mean": control["uniform_mean"],
              "scope": "Exploratory single-training-seed observation intervention, not proof of global optimum or a fully observed MDP."}
    write_json(output/"report.json", result)
    plot_comparison(output, reports)
    write_json(output/"progress.json", {"stage": "complete", "status": result["status"]})
    return result
