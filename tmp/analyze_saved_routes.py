"""Read-only action audit of saved evaluations; no training or replay changes."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hrl_uav.routing import project_rows_to_simplex_tensor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mainline_6mhz_routing_critic"
SEEDS = (33500, 33501, 33502)
ROLES = ("heavy", "medium", "light")


def main():
    records = []
    for seed in SEEDS:
        with np.load(OUTPUT / f"pretrain_final_{seed}_trajectory.npz") as saved:
            route = saved["executed_route"].copy()
            raw = saved["raw_action"].reshape(-1, 2, 3, 3).copy()
            assert np.count_nonzero(saved["macro_action"]) == 0
        meter = pd.read_csv(OUTPUT / f"pretrain_final_{seed}_uavs.csv")
        service = meter[meter.service_steps > 0].sort_values(["time_s", "uav"])
        assert len(service) == 3 * len(route)
        assert (service.groupby("time_s").size() == 3).all()
        assert np.allclose(route.sum(-1), 1, atol=1e-6)
        role_ids = np.array([ROLES.index(role) for role in service.role]).reshape(-1, 3)
        role_order = np.argsort(role_ids, axis=1)
        role_route = np.stack([route[i][:, order][:, :, order]
                               for i, order in enumerate(role_order)])
        records.append((seed, route, raw, meter, role_ids, role_route))

    routes = np.concatenate([r[1] for r in records])
    raw = np.concatenate([r[2] for r in records])
    role_routes = np.concatenate([r[5] for r in records])
    vertex = (routes > 1e-8).sum(-1) == 1
    # At a strict simplex vertex every route derivative vanishes locally.
    scores = torch.tensor(raw, requires_grad=True)
    projected = project_rows_to_simplex_tensor(scores)
    (projected * torch.arange(3, dtype=scores.dtype)).sum().backward()
    projection_error = float(np.abs(projected.detach().numpy() - routes).max())
    assert projection_error < 2e-6
    vertex_gradient = float(scores.grad[torch.tensor(vertex)].abs().max())
    assert vertex_gradient == 0

    phases = []
    for heavy_id in range(3):
        matrices, by_seed = [], []
        for seed, route, _, meter, role_ids, _ in records:
            matrices.append(route[role_ids[:, heavy_id] == 0])
            times = meter[(meter.uav == heavy_id) & (meter.role == "heavy")].time_s
            phase = meter[meter.time_s.isin(times)]
            means = {}
            for uid, rows in phase.groupby("uav"):
                duration = len(rows) * 0.5
                means[int(uid)] = {
                    "assigned_mbps_wall": float(rows.assigned_destination_mbit.sum() / duration),
                    "computed_mbps_wall": float(rows.computed_mbit.sum() / duration),
                    "compute_queue_low_mbit": float(rows.queue_compute_low_mbit.mean()),
                }
            by_seed.append({"seed": seed, "destinations": means})
        phase_mean = {
            str(uid): {key: float(np.mean([x["destinations"][uid][key] for x in by_seed]))
                       for key in by_seed[0]["destinations"][uid]}
            for uid in range(3)
        }
        phases.append({"heavy_uav": heavy_id,
                       "route_mean_per_decision": np.concatenate(matrices).mean(0).tolist(),
                       "destination_mean_across_seeds": phase_mean, "per_seed": by_seed})

    evolution = []
    validation = pd.read_csv(OUTPUT / "validation.csv")
    for window in (0, 6, 12, 18, 30, 60):
        values = []
        for seed in (33000, 33001):
            with np.load(OUTPUT / f"validation_pretrain_{window}_{seed}_trajectory.npz") as saved:
                values.append(saved["executed_route"])
        route = np.concatenate(values)
        row = validation[validation.window == window]
        evolution.append({"window": window, "route_mean_per_decision": route.mean(0).tolist(),
                          "weighted_delay_ms": float(row.task_delay_weighted_ms.mean()),
                          "terminal_queue_mbit": float(row.queue_total_mbit.mean())})

    audit = {
        "scope": "One training seed; three 900-second deterministic evaluation scenes. No new rollouts.",
        "route_mean_semantics": "Decision-weighted fractions, not bit-weighted traffic. Row=source, column=destination; high then low.",
        "rate_semantics": "Actual assigned destination Mbit / wall seconds, including deployment pauses; mean across seeds.",
        "macro_actions_all_zero": True,
        "seeds": list(SEEDS), "route_mean_fixed_ids": routes.mean(0).tolist(),
        "route_mean_load_roles": role_routes.mean(0).tolist(),
        "role_order": list(ROLES), "phases": phases,
        "fraction_single_destination_rows_high_low": vertex.mean((0, 2)).tolist(),
        "raw_fraction_abs_over_095": float(np.mean(np.abs(raw) > 0.95)),
        "projection_reconstruction_max_error": projection_error,
        "vertex_gradient_max_abs": vertex_gradient,
        "gradient_scope": "Saved deterministic actions only; not a measurement of sampled training gradients or proof of the unique failure cause.",
        "validation_evolution": evolution,
        "nominal_feasibility_illustration_not_executed": {
            "offered_mbps_heavy_medium_light": [18.975, 4.025, 1.725],
            "route_matrix": [[43 / 99, 22 / 99, 34 / 99], [0, 1, 0], [0, 0, 1]],
            "assigned_each_mbps": 24.725 / 3,
            "caveat": "Only a nominal CPU-capacity calculation; ignores queues, link service and priority timing. Not a proven feasible/optimal policy or teacher.",
        },
    }
    with (OUTPUT / "action_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for p, title in enumerate(("高优先级任务", "低优先级任务")):
        matrix = routes[:, p].mean(0)
        axes[p].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axes[p].set(xticks=range(3), yticks=range(3),
                    xticklabels=["0号", "1号", "2号"], yticklabels=["0号", "1号", "2号"],
                    xlabel="送给谁计算（目的机）", ylabel="任务来自谁（源机）", title=title)
        for i in range(3):
            for j in range(3):
                axes[p].text(j, i, f"{matrix[i,j]:.1%}", ha="center", va="center",
                             color="white" if matrix[i,j] > 0.55 else "black", fontsize=13)
    x = np.arange(3)
    for uid, color in enumerate(("#397ca8", "#cc7631", "#499873")):
        values = [p["destination_mean_across_seeds"][str(uid)]["assigned_mbps_wall"] for p in phases]
        bars = axes[2].bar(x + (uid-1)*0.24, values, width=0.24, label=f"{uid}号接到的任务", color=color)
        axes[2].bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    axes[2].axhline(9, color="#ae3333", linestyle="--", linewidth=1.5, label="每机平均计算上限 9 Mbps")
    axes[2].set(xticks=x, xticklabels=["0号重载阶段", "1号重载阶段", "2号重载阶段"],
                ylabel="分派到目的机的工作量（Mbps）", ylim=(0, 23), yticks=[0, 5, 10, 15, 20],
                title="负载切换后，拥堵被转移给了谁")
    axes[2].legend(loc="upper center", fontsize=8, frameon=False, ncol=2)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("当前纯 SAC 的实际动作：存在固定方向偏好，部分目的机被分配过量任务", fontsize=15)
    fig.text(0.5, 0.055,
             "最后权重；1个训练种子，3个900秒评估场景。矩阵为逐决策平均比例；右图来自实际任务分派账本。\n"
             "箭头方向表示不同来源任务的分派，不表示同一任务循环转发。PPO尚未启动，宏观使用质心跟踪。",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.79, bottom=0.28, wspace=0.36)
    fig.savefig(OUTPUT / "03_action_analysis.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps({"phases": [p["destination_mean_across_seeds"] for p in phases],
                      "single_destination_high_low": audit["fraction_single_destination_rows_high_low"],
                      "files": [str(OUTPUT / "action_analysis.json"), str(OUTPUT / "03_action_analysis.png")]}, indent=2))


if __name__ == "__main__":
    main()
