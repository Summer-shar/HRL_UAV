"""Summarize the bounded Actor-LR ablation from actual final checkpoints and logs."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP
from scripts.audit_micro_feedback import route_probe


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    roots = [ROOT/"results"/name for name in ("split_mdp_mainline_seed1", "split_mdp_actor_lr3e5_seed1")]
    reports = [json.loads((root/"report.json").read_text()) for root in roots]
    checkpoints = [torch.load(root/"micro_pretrained.pt", map_location="cpu", weights_only=False) for root in roots]
    replay = checkpoints[0]["replay"]["transitions"]
    indices = np.linspace(0, len(replay)-1, 512, dtype=int)
    states = tensor(np.asarray(replay)[indices, :checkpoints[0]["micro_state_dim"]])
    action_stats, workload = [], []
    for checkpoint, root, report in zip(checkpoints, roots, reports):
        mdp = QueueMDP(checkpoint["config"])
        mdp.reset(94400, 90)
        agent = DirectHRL.from_mdp(mdp, checkpoint["config"])
        agent.sac_actor.action_net.load_state_dict(checkpoint["sac_actor"])
        agent.sac_critic.critic_v.load_state_dict(checkpoint["sac_critic"])
        with torch.no_grad():
            mean, _ = agent.sac_actor.action_net(states)
        action_stats.append(route_probe(agent.sac_critic.critic_v, states, mean.tanh()))
        by_seed = []
        for seed in checkpoint["config"]["test_seeds"]:
            rows = read_csv(root/f"{report['final_label']}_{seed}_uavs.csv")
            end = max(float(r["time_s"]) for r in rows)
            by_seed.append([{
                "uav": uav,
                "assigned_mbps": sum(float(r["assigned_destination_mbit"]) for r in rows if int(r["uav"]) == uav)/end,
                "computed_mbps": sum(float(r["computed_mbit"]) for r in rows if int(r["uav"]) == uav)/end,
                "terminal_compute_mbit": next(float(r["queue_compute_mbit"]) for r in reversed(rows) if int(r["uav"]) == uav),
            } for uav in range(3)])
        workload.append([{key: float(np.mean([s[uav][key] for s in by_seed]))
                          for key in ("assigned_mbps", "computed_mbps", "terminal_compute_mbit")}
                         for uav in range(3)])
    crn = []
    for seed in checkpoints[0]["config"]["test_seeds"]:
        with np.load(roots[0]/f"{reports[0]['final_label']}_{seed}_trajectory.npz") as left, np.load(roots[1]/f"{reports[1]['final_label']}_{seed}_trajectory.npz") as right:
            assert all(np.array_equal(left[key], right[key]) for key in ("users", "uav"))
        arrivals = [[float(r["task_arrived_count"]) for r in read_csv(root/f"{report['final_label']}_{seed}_queues.csv")]
                    for root, report in zip(roots, reports)]
        assert arrivals[0] == arrivals[1]
        crn.append({"seed": seed, "users_uavs_arrival_counts_equal": True})
    config_a, config_b = [json.loads(json.dumps(c["config"])) for c in checkpoints]
    config_b["training"]["pretrain_lr"][0] = config_a["training"]["pretrain_lr"][0]
    assert config_a == config_b
    assert checkpoints[0]["updates"] == checkpoints[1]["updates"]
    assert reports[1]["source_hashes_unchanged"]
    result = {
        "scope": "One-factor, one-training-seed development comparison. Reused evaluation scenes; not independent confirmatory evidence.",
        "verdict": "LOWER_LR_PARTIAL_MITIGATION_NOT_STABLE",
        "only_config_difference": "pretrain_lr[0]: 1e-4 -> 3e-5",
        "updates_equal": checkpoints[0]["updates"], "crn": crn,
        "old_mean": reports[0]["mean"], "new_mean": reports[1]["mean"],
        "old_workload_by_fixed_uav": workload[0], "new_workload_by_fixed_uav": workload[1],
        "gradient_scope": "Both final actors on the same 512 old replay states; descriptive state bank, not final deployment distribution",
        "old_action_probe": action_stats[0], "new_action_probe": action_stats[1],
        "qualification_passed": reports[1]["pretrain_admission"]["passed"],
    }
    (roots[1]/"comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(f.name == font for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for root, report, color, label in zip(roots, reports, ("#c24c3a", "#197da1"), ("原学习率 1e-4", "小学习率 3e-5")):
        training = read_csv(root/"training.csv")
        axes[0, 0].plot(np.arange(1, len(training)+1), [float(r["window_reward"]) for r in training], color=color, label=label)
        validation = read_csv(root/"validation.csv")
        windows = sorted({int(r["window"]) for r in validation})
        delay = [np.mean([float(r["task_delay_weighted_ms"])/1000 for r in validation if int(r["window"]) == window]) for window in windows]
        completion = [100*np.mean([float(r["task_completion_rate"]) for r in validation if int(r["window"]) == window]) for window in windows]
        axes[0, 1].plot(windows, delay, "o-", color=color, label=label)
        axes[1, 0].plot(windows, completion, "o-", color=color)
        curves = [read_csv(root/f"{report['final_label']}_{seed}_queues.csv") for seed in report["crn"]]
        values = np.array([[float(row["queue_total_mbit"]) for row in curve] for curve in curves])
        times = [float(row["time_s"]) for row in curves[0]]
        axes[1, 1].plot(times, values.mean(0), color=color, label=label)
        axes[1, 1].fill_between(times, values.min(0), values.max(0), color=color, alpha=.12)
    axes[0, 0].set(title="训练原始奖励（未平滑，越高越好）", xlabel="300 秒训练窗口", ylabel="窗口平均宏观奖励")
    axes[0, 0].legend(fontsize=10)
    axes[0, 1].set(title="固定验证场景：完成任务加权时延", xlabel="300 秒训练窗口", ylabel="时延 / 秒", ylim=(0, None))
    axes[1, 0].set(title="同一验证场景：未完成任务不能漏看", xlabel="300 秒训练窗口", ylabel="任务完成率 / %", ylim=(0, 102))
    axes[1, 1].set(title="末次权重连续运行：队列仍在增长", xlabel="连续运行时间 / 秒", ylabel="全部积压 / Mbit", ylim=(0, None))
    for ax in axes.flat:
        ax.grid(alpha=.16)
    fig.suptitle("纯 SAC 小学习率对照：退化减轻，但未收敛，未启动 PPO\n两臂从头训练 60 窗口；固定末次权重；右下阴影为 3 个评估场景范围", fontsize=14)
    target = roots[1]/"figures"
    target.mkdir(exist_ok=True)
    fig.savefig(target/"01_actor_lr_comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: result[key] for key in ("verdict", "updates_equal", "new_workload_by_fixed_uav")}, indent=2))
    print("Common-state low-route vertex fractions:", action_stats[0]["low_rows_vertex_fraction"], action_stats[1]["low_rows_vertex_fraction"])


if __name__ == "__main__":
    main()
