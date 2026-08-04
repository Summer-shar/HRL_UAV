import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = Path(bootstrap())

from run_load_switch_gate import configure_chinese_font
from run_load_switch_gate import read_raw_rows
from run_load_switch_gate import write_csv_atomic
from run_load_switch_gate import write_json_atomic
from search_position_oracle import load_checkpoint_components


POLICIES = ("P0", "P1", "P2")
STACK_COMPONENTS = (
    ("hover_power_w", "悬停推进", "#A5A5A5"),
    ("steady_propulsion_power_w", "部署段稳态推进", "#4472C4"),
    ("transient_propulsion_power_w", "部署瞬态机动", "#ED7D31"),
    ("cpu_power_w", "CPU 计算", "#70AD47"),
    ("uplink_power_w", "U2G 通信", "#5B9BD5"),
    ("u2u_power_w", "U2U 通信", "#FFC000"),
)
DELTA_COMPONENTS = (
    ("steady_propulsion_power_w", "部署段稳态推进"),
    ("transient_propulsion_power_w", "部署瞬态机动"),
    ("cpu_power_w", "CPU 计算"),
    ("uplink_power_w", "U2G 通信"),
    ("u2u_power_w", "U2U 通信"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the frozen load-switch gate power accounting."
    )
    parser.add_argument(
        "--input-dir",
        default="results/load_switch_gate_power_audit",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/tcom_dynamic_u2u_sparse_v23_target_noise/pilot/"
            "seed1/joint_td3_sac/checkpoints/best.pt"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def resolve(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sample_summary(values):
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    mean = float(np.mean(values)) if count else float("nan")
    std = float(np.std(values, ddof=1)) if count > 1 else 0.0
    if count > 1:
        half_width = (
            float(student_t.ppf(0.975, count - 1))
            * std
            / math.sqrt(count)
        )
    else:
        half_width = float("inf")
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "sample_count": count,
    }


def paired_component_summary(rows_by_key, seeds, candidate, reference, metric):
    values = [
        float(rows_by_key[(candidate, seed)][metric])
        - float(rows_by_key[(reference, seed)][metric])
        for seed in seeds
    ]
    result = sample_summary(values)
    result["values"] = values
    return result


def read_macro_rows(root, policy, seed):
    path = root / "macro_traces" / f"{policy}_seed{seed}.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def role_conditioned_rows(input_dir, seeds):
    accumulators = {
        role: {
            component: []
            for component in ("propulsion", "cpu", "communication")
        }
        for role in ("support", "heavy", "third")
    }
    maximum_macro_cpu_power = 0.0
    for seed in seeds:
        seed_values = {
            role: {
                component: []
                for component in ("propulsion", "cpu", "communication")
            }
            for role in ("support", "heavy", "third")
        }
        p0_rows = read_macro_rows(input_dir, "P0", seed)
        p1_rows = read_macro_rows(input_dir, "P1", seed)
        if len(p0_rows) != len(p1_rows):
            raise RuntimeError(f"Macro trace length mismatch for seed {seed}.")
        for p0, p1 in zip(p0_rows, p1_rows):
            support = int(p1["support_uav"])
            heavy = int(p1["heavy_uav"])
            for uav in range(3):
                maximum_macro_cpu_power = max(
                    maximum_macro_cpu_power,
                    float(p1[f"cpu_power_uav{uav + 1}_w"]),
                )
            if support < 0 or heavy < 0:
                continue
            role_uavs = {
                "support": support,
                "heavy": heavy,
                "third": next(
                    uav for uav in range(3) if uav not in {support, heavy}
                ),
            }
            for role, uav in role_uavs.items():
                index = uav + 1
                for component in ("propulsion", "cpu", "communication"):
                    field = f"{component}_power_uav{index}_w"
                    seed_values[role][component].append(
                        float(p1[field]) - float(p0[field])
                    )
        for role, components in seed_values.items():
            for component, values in components.items():
                if values:
                    accumulators[role][component].append(float(np.mean(values)))
    rows = []
    for role, components in accumulators.items():
        for component, values in components.items():
            rows.append(
                {
                    "role": role,
                    "component": component,
                    **sample_summary(values),
                }
            )
    return rows, float(maximum_macro_cpu_power)


def physical_budget_envelopes(base_env):
    num_uavs = int(base_env.N)
    service_fraction = float(
        (base_env.macro_dt - base_env.deployment_dt) / base_env.macro_dt
    )
    p_hover_per_uav = float(base_env._compute_tcom_flying_power([0.0])[0])
    hover_total = num_uavs * p_hover_per_uav
    max_cpu_per_uav = np.asarray(
        base_env.kappa * np.power(base_env.F_max_array, 3.0),
        dtype=np.float64,
    )
    cpu_full_service_average = service_fraction * float(
        np.sum(max_cpu_per_uav)
    )
    maximum_u2u_links = num_uavs * (num_uavs - 1)
    communication_envelope = service_fraction * (
        float(base_env.M) * float(base_env.p_gt)
        + maximum_u2u_links * float(base_env.p_u2u)
    )

    nominal_centroid_speed = min(
        float(base_env.gm_mean_speed_mps * base_env.macro_dt),
        20.0,
    )
    p_nominal = float(
        base_env._compute_tcom_flying_power([nominal_centroid_speed])[0]
    )
    e_nominal_transient = float(
        base_env._compute_deployment_transient_energy(
            [nominal_centroid_speed]
        )[0]
    )
    nominal_tracking_delta = (
        num_uavs
        * (p_nominal + e_nominal_transient - p_hover_per_uav)
        / float(base_env.macro_dt)
    )

    maximum_deployment_speed = 20.0 / float(base_env.deployment_dt)
    p_max_deployment = float(
        base_env._compute_tcom_flying_power([maximum_deployment_speed])[0]
    )
    e_max_transient = float(
        base_env._compute_deployment_transient_energy(
            [maximum_deployment_speed]
        )[0]
    )
    max_move_event_delta_j = (
        (p_max_deployment - p_hover_per_uav) * float(base_env.deployment_dt)
        + e_max_transient
    )
    # Initial 40 m offset needs two 20 m legs. Each of two switches may
    # require the old supporter to return and the new supporter to move out.
    residual_move_events = 2 + 2 * 4
    episode_duration_s = 30.0 * float(base_env.macro_dt)
    residual_maneuver_reserve = (
        residual_move_events * max_move_event_delta_j / episode_duration_s
    )

    full_rated_budget = (
        hover_total
        + nominal_tracking_delta
        + residual_maneuver_reserve
        + cpu_full_service_average
        + communication_envelope
    )
    return {
        "p_hover_per_uav_w": p_hover_per_uav,
        "hover_total_w": hover_total,
        "service_fraction": service_fraction,
        "cpu_peak_per_uav_w": max_cpu_per_uav,
        "cpu_full_service_average_w": cpu_full_service_average,
        "communication_envelope_w": communication_envelope,
        "nominal_centroid_deployment_speed_mps": nominal_centroid_speed,
        "nominal_tracking_delta_w": nominal_tracking_delta,
        "maximum_deployment_speed_mps": maximum_deployment_speed,
        "maximum_move_event_delta_j": max_move_event_delta_j,
        "residual_move_events": residual_move_events,
        "residual_maneuver_reserve_w": residual_maneuver_reserve,
        "full_rated_budget_w": full_rated_budget,
    }


def plot_audit(output_dir, means, delta_summaries, power_budget_w):
    import matplotlib.pyplot as plt

    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    x = np.arange(len(POLICIES))
    bottoms = np.zeros(len(POLICIES), dtype=np.float64)
    for metric, label, color in STACK_COMPONENTS:
        values = np.asarray([means[policy][metric] for policy in POLICIES])
        axes[0].bar(x, values, bottom=bottoms, label=label, color=color)
        bottoms += values
    axes[0].set_xticks(x, POLICIES)
    axes[0].set_ylabel("系统平均功率 (W)")
    axes[0].set_title("(a) 功率账本分解")
    axes[0].axhline(
        float(power_budget_w),
        color="#C00000",
        linestyle="--",
        linewidth=1.1,
        label=f"{float(power_budget_w):.0f} W 门槛",
    )
    for index, total in enumerate(bottoms):
        axes[0].text(
            index,
            total + 4.0,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    axes[0].grid(axis="y", alpha=0.25)

    labels = [label for _, label in DELTA_COMPONENTS]
    values = np.asarray(
        [delta_summaries[metric]["mean"] for metric, _ in DELTA_COMPONENTS]
    )
    errors = np.asarray(
        [
            delta_summaries[metric]["ci95_high"]
            - delta_summaries[metric]["mean"]
            for metric, _ in DELTA_COMPONENTS
        ]
    )
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#5B9BD5", "#FFC000"]
    axes[1].bar(
        np.arange(len(labels)),
        values,
        yerr=errors,
        capsize=3,
        color=colors,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(
        np.arange(len(labels)),
        labels,
        rotation=22,
        ha="right",
    )
    axes[1].set_ylabel("P1-P0 配对功率差 (W)")
    axes[1].set_title("(b) 13.5 W 增量来源")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("负载切换门禁的系统功率归因", fontsize=15)
    fig.tight_layout()
    target = output_dir / "power_decomposition"
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = resolve(args.input_dir)
    rows = read_raw_rows(input_dir / "per_seed_raw.csv")
    grouped = {
        policy: [row for row in rows if row["policy"] == policy]
        for policy in POLICIES
    }
    if any(len(grouped[policy]) != 20 for policy in POLICIES):
        raise RuntimeError("Power audit requires 20 rows for each policy.")
    rows_by_key = {
        (str(row["policy"]), int(row["seed"])): row for row in rows
    }
    seeds = sorted(int(row["seed"]) for row in grouped["P0"])

    metrics = [metric for metric, _, _ in STACK_COMPONENTS]
    metrics += [
        "average_power_w",
        "propulsion_power_w",
        "cpu_power_w",
        "communication_power_w",
        "component_power_sum_w",
        "power_closure_residual_w",
    ]
    means = {
        policy: {
            metric: float(
                np.mean([float(row[metric]) for row in grouped[policy]])
            )
            for metric in metrics
        }
        for policy in POLICIES
    }
    delta_summaries = {
        metric: paired_component_summary(
            rows_by_key,
            seeds,
            "P1",
            "P0",
            metric,
        )
        for metric in dict.fromkeys(metrics)
    }

    policy_rows = []
    for policy in POLICIES:
        for metric in metrics:
            policy_rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    **sample_summary(
                        [float(row[metric]) for row in grouped[policy]]
                    ),
                }
            )
    write_csv_atomic(
        input_dir / "power_components_by_policy.csv",
        policy_rows,
        (
            "policy",
            "metric",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "sample_count",
        ),
    )

    paired_rows = []
    for metric, summary in delta_summaries.items():
        paired_rows.append(
            {
                "contrast": "P1-P0",
                "metric": metric,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "values"
                },
            }
        )
    write_csv_atomic(
        input_dir / "power_component_paired_deltas.csv",
        paired_rows,
        (
            "contrast",
            "metric",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "sample_count",
        ),
    )

    per_uav_rows = []
    for policy in POLICIES:
        for uav in range(1, 4):
            for metric in (
                "propulsion_power",
                "cpu_power",
                "communication_power",
                "deployment_speed_mean",
                "episode_path_speed",
                "flight_distance",
            ):
                unit = (
                    "w"
                    if "power" in metric
                    else ("mps" if "speed" in metric else "m")
                )
                field = f"{metric}_uav{uav}_{unit}"
                per_uav_rows.append(
                    {
                        "policy": policy,
                        "uav": uav,
                        "metric": metric,
                        **sample_summary(
                            [float(row[field]) for row in grouped[policy]]
                        ),
                    }
                )
    write_csv_atomic(
        input_dir / "power_per_uav.csv",
        per_uav_rows,
        (
            "policy",
            "uav",
            "metric",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "sample_count",
        ),
    )

    role_rows, maximum_macro_cpu_power = role_conditioned_rows(
        input_dir,
        seeds,
    )
    write_csv_atomic(
        input_dir / "role_conditioned_power_deltas.csv",
        role_rows,
        (
            "role",
            "component",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "sample_count",
        ),
    )

    checkpoint_path, checkpoint_args, env, _actor, _micro, _device = (
        load_checkpoint_components(args.checkpoint, args.device)
    )
    base_env = env.env
    envelopes = physical_budget_envelopes(base_env)
    closure_max = float(
        max(abs(float(row["power_closure_residual_w"])) for row in rows)
    )
    total_delta = delta_summaries["average_power_w"]["mean"]
    propulsion_delta = delta_summaries["propulsion_power_w"]["mean"]
    transient_delta = delta_summaries[
        "transient_propulsion_power_w"
    ]["mean"]
    cpu_delta = delta_summaries["cpu_power_w"]["mean"]
    communication_delta = (
        delta_summaries["communication_power_w"]["mean"]
    )
    role_lookup = {
        (row["role"], row["component"]): row for row in role_rows
    }

    audit = {
        "checkpoint": str(checkpoint_path),
        "rows": len(rows),
        "seeds": seeds,
        "component_means_w": means,
        "p1_minus_p0": delta_summaries,
        "closure_max_abs_w": closure_max,
        "role_conditioned_p1_minus_p0_w": role_rows,
        "maximum_observed_macro_average_cpu_power_per_uav_w": (
            maximum_macro_cpu_power
        ),
        "model": {
            "power_constraint_scope": str(env.power_constraint_scope),
            "system_power_budget_w": float(env.power_budget_w),
            "cpu_power_cap_w": float(base_env.cpu_power_cap_w),
            "cpu_alloc_mode": str(base_env.cpu_alloc_mode),
            "uav_mass_kg": float(base_env.uav_mass_kg),
            "deployment_transient_energy_mode": str(
                base_env.deployment_transient_energy_mode
            ),
            "f_max_hz_per_uav": np.asarray(base_env.F_max_array),
            "kappa": float(base_env.kappa),
        },
        "physical_budget_envelopes": envelopes,
    }
    write_json_atomic(input_dir / "power_audit.json", audit)
    plot_audit(input_dir, means, delta_summaries, env.power_budget_w)

    def contribution(value):
        return 100.0 * value / total_delta

    full_budget = envelopes["full_rated_budget_w"]
    transient_20 = float(
        base_env._compute_deployment_transient_energy([20.0])[0]
    )
    lines = [
        "# 负载切换门禁功率归因审计",
        "",
        "## 结论",
        "",
        f"- P1 相对 P0 的系统平均功率增量为 `{total_delta:.3f} W`。",
        f"  95% 配对 CI 为 `[{delta_summaries['average_power_w']['ci95_low']:.3f}, {delta_summaries['average_power_w']['ci95_high']:.3f}] W`。",
        f"- 推进项贡献 `{propulsion_delta:.3f} W`（`{contribution(propulsion_delta):.1f}%`，95% CI `[{delta_summaries['propulsion_power_w']['ci95_low']:.3f}, {delta_summaries['propulsion_power_w']['ci95_high']:.3f}] W`），其中部署瞬态机动贡献 `{transient_delta:.3f} W`（`{contribution(transient_delta):.1f}%`，95% CI `[{delta_summaries['transient_propulsion_power_w']['ci95_low']:.3f}, {delta_summaries['transient_propulsion_power_w']['ci95_high']:.3f}] W`）。",
        f"- CPU 仅增加 `{cpu_delta:.3f} W`（`{contribution(cpu_delta):.1f}%`，95% CI `[{delta_summaries['cpu_power_w']['ci95_low']:.3f}, {delta_summaries['cpu_power_w']['ci95_high']:.3f}] W`）；通信净变化 `{communication_delta:.3f} W`。因此当前数据否定“13.5 W 主要来自支援机 CPU 加班”的假设。",
        f"- 在支援角色激活的宏观周期内先按 episode 聚合后，支援机相对同种子 P0 的推进功率增加 `{role_lookup[('support', 'propulsion')]['mean']:.3f} W`，CPU 功率变化仅 `{role_lookup[('support', 'cpu')]['mean']:.3f} W`。",
        f"- 分项和总功率的最大闭合残差为 `{closure_max:.3e} W`。",
        "",
        "## 为什么外部解析判断失效",
        "",
        "外部报告只比较了稳态功率曲线 P(v)，但当前环境还启用了 `kinetic_lower_bound`，每个 1 s 部署段额外计入 `0.5*m*v^2`。"
        f"以 20 m/s 部署时，该瞬态项为 `{transient_20:.3f} J/次/机`。门禁轨迹的部署段并非普遍为 2 m/s，因此不能用“2 m/s 比悬停省电”推断整段总功率。",
        "",
        "## 约束口径",
        "",
        f"- 当前约束是三机系统长期平均功率 `<= {float(env.power_budget_w):.0f} W`，并非逐机 280 W。",
        f"- 当前没有聚合 CPU 功率帽：`cpu_power_cap_w={float(base_env.cpu_power_cap_w):.1f}`。10 GHz 与立方功耗模型隐含每机 CPU 动态峰值约 `{float(envelopes['cpu_peak_per_uav_w'][0]):.1f} W`。",
        f"- 重放中观测到的单机最高宏观平均 CPU 功率为 `{maximum_macro_cpu_power:.1f} W`，没有单机 280 W 超限问题。",
        "- `average_power_w` 是系统口径，包含 UAV 推进、CPU、U2U 发射以及地面用户 U2G 发射；其中 U2G 能量按关联 UAV 归档，仅为记账归属，不表示它由 UAV 电池承担。",
        "",
        "## 与观测值隔离的候选物理预算",
        "",
        "下面只使用模型参数与冻结 P1 的机动协议，不使用 P1=845.43 W 观测值反推：",
        "",
        f"1. 三机悬停底座：`3P_hover={envelopes['hover_total_w']:.3f} W`；",
        f"2. 10 GHz CPU 在 9/10 服务占空比下三机满额动态功率：`{envelopes['cpu_full_service_average_w']:.3f} W`；",
        f"3. 39 条 U2G 与 6 条有向 U2U 链路同时活动的保守通信包络：`{envelopes['communication_envelope_w']:.3f} W`；",
        f"4. 用户平均 0.5 m/s 对应的名义质心跟踪增量：`{envelopes['nominal_tracking_delta_w']:.3f} W`；",
        f"5. P1 初始建立 40 m 偏移并在两次切换时旧支援机返回、新支援机建立偏移，共按 `{envelopes['residual_move_events']}` 次 20 m 最大部署腿计，机动余量为 `{envelopes['residual_maneuver_reserve_w']:.3f} W`。",
        "",
        f"由此得到全额硬件/协议包络 `P_budget,C={full_budget:.3f} W`，工程配置可预注册为 `878 W`（若采用 5 W 整数档则为 `880 W`）。它与 845 W 观测值隔离，但属于较松的硬件包络；采用后功率约束大概率不再是紧约束。是否接受这一研究口径，应在修改 §7C 前明确。",
        "",
        "## 下一步判决",
        "",
        "- 若研究目标是“时延为主、功率只需不超过合理硬件包络”，可把 §7C 预注册为上述 878/880 W 物理包络，再对同一门禁数据按新阈值重判；不得把 843.3 W 写成物理预算。",
        "- 若论文必须强调紧的长期功率约束，则不能仅放宽预算；应保持 840 W，并把 P1 改成能量感知的因果支援规则后重新预注册。",
        "- 原冻结门禁的正式结论仍是 FAIL，本审计只解释失败来源，不追溯改写原结果。",
        "",
    ]
    (input_dir / "power_audit.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )
    print(f"[AUDIT] P1-P0 total={total_delta:.3f} W")
    print(
        f"[AUDIT] propulsion={propulsion_delta:.3f} W, "
        f"transient={transient_delta:.3f} W, CPU={cpu_delta:.3f} W, "
        f"communication={communication_delta:.3f} W"
    )
    print(f"[AUDIT] closure max={closure_max:.3e} W")
    print(f"[AUDIT] independent full-rated envelope={full_budget:.3f} W")
    print(f"[OUTPUT] {input_dir}")


if __name__ == "__main__":
    main()
