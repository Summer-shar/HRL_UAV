"""Read-only fixed-UAV audit of a completed split-state training run."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/split_mdp_mainline_seed1"
report = json.loads((OUTPUT / "report.json").read_text())
records, routing = [], []
for evaluation in report["final"]:
    seed = int(evaluation["seed"])
    prefix = f"{report['final_label']}_{seed}"
    with (OUTPUT / f"{prefix}_uavs.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    duration = float(evaluation["time_s"])
    by_uav = []
    for uav in range(3):
        values = [r for r in rows if int(r["uav"]) == uav]
        by_uav.append({
            "uav": uav,
            "assigned_mbps": sum(float(r["assigned_destination_mbit"]) for r in values) / duration,
            "computed_mbps": sum(float(r["computed_mbit"]) for r in values) / duration,
            "terminal_compute_mbit": float(values[-1]["queue_compute_mbit"]),
        })
    with np.load(OUTPUT / f"{prefix}_trajectory.npz") as trace:
        routing.append(trace["executed_route"].mean(axis=0))
    records.append({"seed": seed, "uavs": by_uav})

means = {key: np.mean([[u[key] for u in r["uavs"]] for r in records], axis=0)
         for key in ("assigned_mbps", "computed_mbps", "terminal_compute_mbit")}
routes = np.mean(routing, axis=0)
audit = {
    "scope": "Same final SAC, fixed UAV identities, all three held-out scenes; no training",
    "per_seed": records, "mean_by_fixed_uav": {k: v.tolist() for k, v in means.items()},
    "decision_mean_route_high_low_source_destination": routes.tolist(),
    "route_scope": "Decision mean, not bit-weighted flow. Roles rotate but UAV identities do not.",
}
(OUTPUT / "fixed_uav_action_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
x = np.arange(3)
for offset, key, label, color in (
    (-.18, "assigned_mbps", "被分配的工作量", "#c45338"),
    (.18, "computed_mbps", "实际计算完成量", "#1d7d9b"),
):
    bars = axes[0].bar(x + offset, means[key], width=.36, label=label, color=color)
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
axes[0].axhline(9, color="#555555", ls="--", label="单架理想平均计算上限 9 Mbps")
axes[0].set(xticks=x, xticklabels=["UAV 0", "UAV 1", "UAV 2"],
            ylabel="900 s 实际比特数 / 900 s，单位 Mbps", ylim=(0, 25),
            title="固定编号：UAV 1 过载，其他算力未充分使用")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=.15)
axes[0].set_axisbelow(True)
im = axes[1].imshow(routes[1], vmin=0, vmax=1, cmap="Blues")
for i in range(3):
    for j in range(3):
        value = routes[1, i, j]
        axes[1].text(j, i, f"{100*value:.1f}%", ha="center", va="center",
                     color="white" if value > .55 else "black")
axes[1].set(xticks=x, yticks=x, xticklabels=["UAV 0", "UAV 1", "UAV 2"],
            yticklabels=["UAV 0", "UAV 1", "UAV 2"], xlabel="目的 UAV", ylabel="来源 UAV",
            title="低优路由：各来源都集中送向编号 1")
fig.colorbar(im, ax=axes[1], shrink=.75, label="每次决策的平均路由比例")
fig.suptitle("SAC 最后权重的动作诊断：3 个测试场景均值，PPO 未训练\n左图按真实工作量计，右图按决策平均；不是按会轮换的负载角色平均", fontsize=12)
fig.savefig(OUTPUT / "figures/03_fixed_uav_actions.png", dpi=160)
plt.close(fig)
print(json.dumps(audit, indent=2))
