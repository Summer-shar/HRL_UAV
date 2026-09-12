"""Offline tables and figures from the ten predeclared mission runs only."""

import csv
from pathlib import Path

import numpy as np

from .experiment import append_csv, write_json
from .mission_smoke import CONTROLLERS

TRACKING = CONTROLLERS[1:4]
COMPARISONS = tuple((p, "hover") for p in TRACKING) + (
    ("progressive_centroid", "current_centroid"), ("predictive_centroid", "current_centroid"),
    ("predictive_centroid", "progressive_centroid"),
) + tuple(("blind_patrol", p) for p in CONTROLLERS[:-1])


def pair_rows(rows):
    result = []
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "seed"]
    for seed in (33600, 33601):
        by = {r["policy"]: r for r in rows if r["seed"] == seed}
        for policy, reference in COMPARISONS:
            a, b = by[policy], by[reference]
            row = dict(seed=seed, policy=policy, reference=reference)
            row.update({f"delta_{k}": a[k]-b[k] for k in numeric})
            for key in ("J", "J_holding", "J_energy", "power_w"):
                row[f"relative_{key}_percent"] = 100*(a[key]-b[key])/abs(b[key]) if b[key] else None
            row["energy_driven_ordering_risk"] = (policy == "blind_patrol" and row["delta_J"] < -1e-10
                                                  and row["delta_J_holding"] >= -1e-10
                                                  and row["delta_J_energy"] < -1e-10)
            result.append(row)
    return result


def table(rows, columns):
    lines = ["|" + "|".join(label for key, label in columns) + "|", "|"+"|".join("---" for _ in columns)+"|"]
    for row in rows:
        values = []
        for key, _ in columns:
            v = row[key]
            values.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        lines.append("|"+"|".join(values)+"|")
    return lines


def plot_scene(output, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if n in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.sans-serif": [chosen], "axes.unicode_minus": False, "font.size": 10})
    names = {"hover": "固定悬停", "current_centroid": "当前业务质心", "progressive_centroid": "渐进质心 0.5",
             "predictive_centroid": "历史速度预测质心", "blind_patrol": "无业务巡航（仅退化诊断）"}
    colors = ("#17665b", "#a33242", "#364e9e")
    markers = ("o", "^", "D")
    traces = {}
    for name in CONTROLLERS:
        with np.load(output / f"{name}_{seed}" / "trajectory.npz") as saved:
            traces[name] = {k: saved[k] for k in ("time_s", "uav", "users", "centroid", "groups")}
    reference = traces["hover"]
    for trace in traces.values():
        for key in ("time_s", "users", "centroid", "groups"):
            np.testing.assert_array_equal(trace[key], reference[key])

    def panel(ax, name):
        trace = traces[name]
        for j in range(39):
            ax.plot(reference["users"][:, j, 0], reference["users"][:, j, 1], color=".68", alpha=.32, lw=.45)
        for i, color in enumerate(colors):
            ax.plot(trace["groups"][:, i, 0], trace["groups"][:, i, 1], color=color, ls="--", lw=1.6)
            ax.plot(trace["centroid"][:, i, 0], trace["centroid"][:, i, 1], color=color, ls=":", lw=.8)
            q = trace["uav"][:, i]
            ax.plot(q[:, 0], q[:, 1], color=color, lw=1.8)
            ax.scatter(q[1000::1000, 0], q[1000::1000, 1], marker=markers[i], s=24,
                       facecolors="none", edgecolors=color, linewidths=.9, zorder=4)
            ax.scatter(*q[0], marker="s", s=30, color=color, zorder=5)
            ax.scatter(*q[-1], marker="x", s=45, linewidths=1.4, color=color, zorder=6)
        ax.set(title=names[name], xlim=(0, 1000), ylim=(0, 1000), xlabel="x / m", ylabel="y / m")
        ax.set_aspect("equal")
        ax.grid(alpha=.18)

    legend = [Line2D([0], [0], color=c, marker=m, label=f"UAV {i+1} / 每100 s") for i, (c,m) in enumerate(zip(colors,markers))]
    legend += [Line2D([0], [0], color=".3", ls="--", label="任务群中心"),
               Line2D([0], [0], color=".3", ls=":", label="业务质心"),
               Line2D([0], [0], color=".7", label="地面用户"),
               Line2D([0], [0], color=".3", marker="s", ls="none", label="起点"),
               Line2D([0], [0], color=".3", marker="x", ls="none", label="终点")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, name in zip(axes.ravel(), CONTROLLERS[:4]):
        panel(ax, name)
    fig.suptitle(f"连续移动任务群：真实0–900 s轨迹（开发场景{seed}）", fontsize=14)
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, .96))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"trajectory_{seed}.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    panel(ax, "blind_patrol")
    fig.suptitle(f"开发场景{seed}，900 s，非算法基线", fontsize=12)
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, .96))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"blind_patrol_{seed}.{ext}", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    line_styles = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
    for name, style in zip(CONTROLLERS, line_styles):
        trace = traces[name]
        for ax, key in zip(axes, ("centroid", "groups")):
            error = np.linalg.norm(trace["uav"]-trace[key], axis=2).mean(axis=1)
            ax.plot(trace["time_s"], error, ls=style, lw=1.1, label=names[name])
    axes[0].set(ylabel="平均业务质心距离 / m", title=f"逐物理步追踪误差（开发场景{seed}）")
    axes[1].set(ylabel="平均任务群中心距离 / m", xlabel="时间 / s", xlim=(0,900))
    axes[0].legend(ncol=2, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"tracking_error_{seed}.{ext}", dpi=300)
    plt.close(fig)


def analyze(output, rows, manifest):
    means = [{"policy": p, "scene_count": 2,
              **{k: float(np.mean([r[k] for r in rows if r["policy"] == p]))
                 for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "seed"}}
             for p in CONTROLLERS]
    pairs = pair_rows(rows)
    pair_means = []
    for p, b in COMPARISONS:
        selected = [r for r in pairs if r["policy"] == p and r["reference"] == b]
        row = dict(policy=p, reference=b)
        row.update({k: float(np.mean([r[k] for r in selected])) for k, v in selected[0].items()
                    if isinstance(v, (int, float)) and k not in ("seed", "energy_driven_ordering_risk")})
        row["energy_driven_ordering_risk"] = (p == "blind_patrol" and row["delta_J"] < -1e-10
                                               and row["delta_J_holding"] >= -1e-10 and row["delta_J_energy"] < -1e-10)
        row["both_scene_J_lower"] = all(r["delta_J"] < -1e-10 for r in selected)
        pair_means.append(row)
    for filename, data in (("summary.csv", means), ("paired_deltas.csv", pairs), ("paired_means.csv", pair_means)):
        append_csv(output / filename, data)
    risks = [r for r in pairs if r["energy_driven_ordering_risk"]]
    write_json(output / "ordering_checks.json", dict(energy_driven_risks=risks,
               scene_pair_count=2, training_count=0, policy_comparisons=pair_means))
    main_cols = [("policy","控制器"),("seed","场景"),("J","J"),("J_holding","驻留项"),("J_energy","能量项"),
                 ("task_delay_weighted_ms","加权/ms"),("task_delay_high_ms","高优/ms"),("task_delay_low_ms","低优/ms"),("power_w","完整功率/W")]
    lines = ["# Mission V1：第一阶段900秒无学习开发pilot", "", "## A. 运行版本与范围", "",
             f"运行提交：`{manifest['actual_head']}`；物理提交：`{manifest['physics_base_commit']}`。",
             f"SAC来源SHA256：`{manifest['source_checkpoint_sha256']}`。",
             "仅33600/33601，各五个固定控制器连续900秒；300/600秒未重置。两者是已使用开发场景，非保留测试集。",
             "SAC101确定性零样本迁移，未训练网络、未使用replay、未实例化PPO。所有10条均保留，未按表现提前停。",
             "物理文件指纹、完整有效配置及运行前后源码指纹见manifest；测试命令和范围见docs/mission_rpgm_v1_pilot.md。",
             "", "## B. 逐场景结果", ""] + table(rows, main_cols)
    lines += ["", "两场景算术平均（不是两次独立训练）：", ""] + table(means, [c for c in main_cols if c[0] != "seed"])
    task_cols = [("policy","控制器"),("seed","场景"),("task_arrived_high_count","高优到达"),("task_completed_high_count","高优完成"),
                 ("task_unfinished_high_count","高优未完"),("task_completion_high_rate","高优完成率"),
                 ("task_arrived_low_count","低优到达"),("task_completed_low_count","低优完成"),
                 ("task_unfinished_low_count","低优未完"),("task_completion_low_rate","低优完成率")]
    lines += ["", "## C. 完成与积压", "", "时延仅针对已完成父任务，未完成不一律解释为超时。", ""] + table(rows, task_cols)
    lines += [""] + table(rows, [("policy","控制器"),("seed","场景"),
          ("mean_queue_upload_mbit","平均上传/Mbit"),("mean_queue_compute_mbit","平均计算/Mbit"),
          ("mean_queue_transfer_mbit","平均转发/Mbit"),("queue_upload_mbit","期末上传/Mbit"),
          ("queue_compute_mbit","期末计算/Mbit"),("queue_transfer_mbit","期末转发/Mbit"),
          ("late_queue_slope_mbit_s","后300秒总队列斜率/Mbit/s")])
    lines += ["", "## D. 预设配对比较", "", "差值=左侧控制器减参考；负J表示降低。百分比以参考绝对值为分母。", ""]
    lines += table(pairs, [("policy","控制器"),("reference","参考"),("seed","场景"),("delta_J","ΔJ"),
                          ("relative_J_percent","ΔJ/%"),("delta_J_holding","Δ驻留"),("delta_J_energy","Δ能量"),
                          ("energy_driven_ordering_risk","能量驱动风险")])
    lines += ["", "两场景平均差；百分比为逐场景相对差的平均：", ""] + table(pair_means,
               [("policy","控制器"),("reference","参考"),("delta_J","平均ΔJ"),("relative_J_percent","平均ΔJ/%"),
                ("delta_J_holding","平均Δ驻留"),("delta_J_energy","平均Δ能量"),("both_scene_J_lower","两场景J均降低")])
    lines += ["", "## E. 实际通信与能量", "",
              "U2G有效速率=实际发送Mbit/累计用户发送秒；U2U活跃速率按活跃有向链路-物理步平均，均非参考链路质量。",
              "并发发送秒可超过900秒，不是墙钟停留时间；吞吐按完整900秒。方向及当前角色流见每臂links/uavs/role_summary.csv。", ""]
    lines += table(rows, [("policy","控制器"),("seed","场景"),("u2g_rate_time_weighted_mbps","U2G有效Mbps"),
                         ("u2g_served_mbit","U2G工作量/Mbit"),("u2g_tx_time_s","U2G发送秒"),
                         ("mean_rate_per_active_link_mbps","U2U活跃Mbps"),("tx_served_mbit","U2U工作量/Mbit"),
                         ("u2u_tx_time_s","U2U发送秒")])
    lines += [""] + table(rows, [("policy","控制器"),("seed","场景"),("energy_propulsion_j","完整推进/J"),
               ("propulsion_increment_j","推进增量/J"),("energy_compute_j","CPU/J"),("energy_upload_j","上行/J"),("energy_u2u_j","U2U/J")])
    lines += ["", "## F. 追踪与约束", "", "误差按0.1秒采样（含初始点），全系统三机等权。每UAV的mean/median/p90/max见trajectory_summary.csv。", ""]
    lines += table(rows, [("policy","控制器"),("seed","场景"),("business_error_m_mean","业务误差均值/m"),
               ("group_error_m_mean","群中心误差均值/m"),("fleet_distance_m","总路程/m"),("speed_mps_mean","平均速度/m/s"),
               ("cap_triggered_macros","请求超20m周期数"),("cap_triggered_rows","请求超20m行数"),
               ("unit_action_saturated_rows","单位动作饱和行数"),("safety_changed_macros","安全投影周期"),
               ("map_clipped_rows","地图裁剪行数"),("boundary_user_hits","用户边界命中"),("group_endpoint_max_error_m","群终点误差/m")])
    lines += ["", "超过20m才计cap触发；恰好20m另计单位动作饱和。分母分别90个macro/270个UAV行，比例和触发UAV数见原表。",
              "Current与Progressive可能同时饱和；不能仅从均值接近认定两者等价。commands.csv保留裁剪前请求、地图裁剪前后目标及实际目标。",
              "", "## G. 观察与解释边界", ""]
    for p in pair_means[:6]:
        lines.append(f"- {p['policy']} 相对 {p['reference']}：平均ΔJ={p['delta_J']:+.6f}，"
                     f"平均Δ驻留={p['delta_J_holding']:+.6f}；两场景J均降低={p['both_scene_J_lower']}。")
    lines += [f"- Blind Patrol能量驱动排序风险共{len(risks)}个逐场景比较；判据保留原1e-10数值容差，不作为显著性门槛。",
              "- 轨迹干预后的系统指标来自真实仿真，未用旧驻留收益替换推进能量。通信工作量、速率、路由与队列共同变化属于直接记录。",
              "- 速率改善是否是某次完成时延改善的唯一原因、哪一个父任务关键分片决定收益，尚未做因果分解；不能只凭积压大小宣布唯一瓶颈。",
              "- 追踪胜过Hover仅说明这些规则在所测场景下的移动价值；不同追踪规则的配对差才是后续轨迹决策问题的线索，不证明PPO能学会。",
              "- 即使盲巡航输给追踪，也不能排除其他无业务轨迹利用准稳态功率曲线；本轮未修改负推进增量、转弯/加减速近似或权重。",
              "- 当前SAC来自旧运动协议，68维观测未补充UAV速度；两场景结果不是迁移训练成功、泛化或队列无限时域稳定的证据。",
              "", "## H. 图与停止", "",
              "trajectory_33600/33601.png与PDF：四个业务控制器真实900秒轨迹；blind_patrol图单列诊断。",
              "tracking_error图分别展示到业务质心与任务群中心的距离；不使用未来用户轨迹生成控制动作。",
              "本轮完成10次后停止。未运行33610–33612、33700–33709或第二阶段微观规则比较，未训练任何网络。",
              "下一阶段是否获准，需要结合逐场景驻留/完成/积压和Blind风险裁决；不自动执行。", ""]
    (output / "result.md").write_text("\n".join(lines), encoding="utf-8")
    for seed in (33600, 33601):
        plot_scene(output, seed)
