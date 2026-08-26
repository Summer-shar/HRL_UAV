"""Scan causal cooperative waypoint offsets with the frozen v51 micro policy.

For each macro period, the lightest/least-queued non-heavy UAV is the support
UAV. Positive offsets move it from its own dynamic centroid toward the current
heavy UAV; negative offsets move it away. The frozen v51 gated-SAC policy and
all physical parameters remain unchanged. The zero-offset arm is dynamic
centroid tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v45_fair_screen as v45  # noqa: E402
from audit_v51_micro_decision_frequency import make_schedule  # noqa: E402
from run_load_switch_gate import run_gate_episode  # noqa: E402
from run_v34_cooperative_mobility_gate import (  # noqa: E402
    V34MacroPolicy,
    queue_metrics,
    soft_objective_metrics,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v51_actor_autonomous_gate"
    / "formal"
    / "seed1"
    / "joint_ppo_gated_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v51_cooperative_offset_scan"
DEFAULT_OFFSETS = (-60.0, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 60.0)

PAIRED_FIELDS = (
    "task_delay_weighted_ms",
    "task_delay_high_ms",
    "task_delay_low_ms",
    "transition_window_delay_weighted_ms",
    "mean_queue_mbits",
    "terminal_queue_mbits",
    "tail_queue_slope_mbps",
    "soft_objective_j",
    "average_power_w",
    "u2g_active_time_s",
    "u2u_active_time_s",
    "coop_ratio",
    "cross_flow_mbits",
    "task_completion_rate",
    "mean_support_heavy_distance_m",
    "mean_support_heavy_spectral_efficiency",
    "mean_support_heavy_nominal_rate_mbps",
    "mean_support_centroid_residual_m",
    "mean_all_uav_centroid_residual_m",
    "heavy_to_support_flow_mbits",
)


def offset_name(offset_m: float) -> str:
    value = int(round(float(offset_m)))
    return f"offset_{value:+04d}m"


def parse_offsets(value: str) -> tuple[float, ...]:
    offsets = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not offsets or 0.0 not in offsets:
        raise ValueError("Offsets must contain a zero-offset baseline")
    if len(set(offsets)) != len(offsets):
        raise ValueError("Offsets must be unique")
    return offsets


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in {"policy", "external_trace_hash"}:
                    row[key] = value
                elif key in {
                    "seed",
                    "task_arrived_count",
                    "task_completed_count",
                    "micro_decision_count",
                    "offset_m",
                }:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def geometry_metrics(env, trace: dict) -> dict:
    distances = []
    efficiencies = []
    rates = []
    support_residuals = []
    all_residuals = []
    heavy_to_support_flow = 0.0
    for record in trace["macro_records"]:
        uavs = np.asarray(record["uav_after_m"], dtype=np.float64)
        centroids = np.asarray(record["centroids_after_m"], dtype=np.float64)
        all_residuals.extend(np.linalg.norm(uavs - centroids, axis=1).tolist())
        heavy = int(record.get("heavy_uav", -1))
        support = int(record.get("support_uav", -1))
        if heavy < 0 or support < 0:
            continue
        distance = float(np.linalg.norm(uavs[support] - uavs[heavy]))
        spectral_efficiency = float(
            env.env.compute_u2u_spectral_efficiency(uavs)[heavy, support]
        )
        nominal_rate = float(
            env.env.compute_u2u_rate_bps(uav_coords=uavs)[heavy, support] / 1e6
        )
        distances.append(distance)
        efficiencies.append(spectral_efficiency)
        rates.append(nominal_rate)
        support_residuals.append(float(np.linalg.norm(uavs[support] - centroids[support])))
        flow = np.asarray(record["tx_flow_mbits"], dtype=np.float64)
        heavy_to_support_flow += float(flow[heavy, support])

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "mean_support_heavy_distance_m": mean_or_nan(distances),
        "mean_support_heavy_spectral_efficiency": mean_or_nan(efficiencies),
        "mean_support_heavy_nominal_rate_mbps": mean_or_nan(rates),
        "mean_support_centroid_residual_m": mean_or_nan(support_residuals),
        "mean_all_uav_centroid_residual_m": mean_or_nan(all_residuals),
        "heavy_to_support_flow_mbits": float(heavy_to_support_flow),
    }


def evaluate_offset(policy: dict, offset_m: float, seed: int) -> dict:
    args = policy["args"]
    env = v45.make_eval_env(args, record_trajectory=True)
    # The zero-offset arm still runs the same causal role selector so its
    # support-heavy geometry is measured on exactly the same link definition.
    # With offset_m=0 its executed waypoint remains the dynamic centroid.
    macro = V34MacroPolicy("cooperative", offset_m, movement_trigger_m=0.0)
    result, trace = run_gate_episode(
        env,
        args,
        policy["micro_policy"],
        offset_name(offset_m),
        seed,
        macro_steps=int(args.macro_steps),
        load_switch_schedule=make_schedule(args, seed),
        macro_policy=macro,
        macro_metadata_provider=macro.metadata,
    )
    result.update(queue_metrics(trace, float(env.env.macro_dt)))
    result.update(
        soft_objective_metrics(
            trace,
            float(getattr(args, "micro_reward_scale", 1.0)),
        )
    )
    result.update(geometry_metrics(env, trace))
    arrived = max(int(result["task_arrived_count"]), 1)
    result.update(
        offset_m=int(round(offset_m)),
        task_completion_rate=float(result["task_completed_count"] / arrived),
    )
    return result


def summary_by_offset(rows: list[dict], offsets: tuple[float, ...]) -> dict:
    return {
        str(int(round(offset))): {
            field: v45.mean_ci95(
                [float(row[field]) for row in rows if float(row["offset_m"]) == offset]
            )
            for field in PAIRED_FIELDS
        }
        for offset in offsets
    }


def paired_vs_zero(rows: list[dict], offsets: tuple[float, ...], field: str) -> dict:
    zero = {
        int(row["seed"]): float(row[field])
        for row in rows
        if float(row["offset_m"]) == 0.0
    }
    output = {}
    for offset in offsets:
        if np.isclose(offset, 0.0):
            continue
        candidate = {
            int(row["seed"]): float(row[field])
            for row in rows
            if float(row["offset_m"]) == offset
        }
        seeds = sorted(set(zero) & set(candidate))
        deltas = [candidate[seed] - zero[seed] for seed in seeds]
        output[str(int(round(offset)))] = {
            **v45.mean_ci95(deltas),
            "contrast": "offset_minus_dynamic_centroid",
            "per_seed": {str(seed): value for seed, value in zip(seeds, deltas)},
        }
    return output


def crn_report(rows: list[dict], offsets: tuple[float, ...]) -> dict:
    expected = {offset_name(offset) for offset in offsets}
    mismatches = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        items = {
            str(row["policy"]): str(row["external_trace_hash"])
            for row in rows
            if int(row["seed"]) == seed
        }
        if set(items) != expected or len(set(items.values())) != 1:
            mismatches[str(seed)] = items
    return {
        "all_external_trace_hashes_match": not mismatches,
        "mismatches": mismatches,
    }


def build_report(rows, offsets, checkpoint, seeds):
    summaries = summary_by_offset(rows, offsets)
    paired = {field: paired_vs_zero(rows, offsets, field) for field in PAIRED_FIELDS}
    delay = paired["task_delay_weighted_ms"]
    practically_positive = []
    statistically_positive = []
    for key, stats in delay.items():
        gain = -float(stats["mean"])
        gain_ci_low = -float(stats["ci95_high"])
        if gain > 1.0:
            practically_positive.append(int(key))
        if gain_ci_low > 0.0:
            statistically_positive.append(int(key))
    best_key = min(
        summaries,
        key=lambda key: summaries[key]["task_delay_weighted_ms"]["mean"],
    )
    return {
        "scope": (
            "Frozen v51 gated-SAC micro checkpoint; causal dynamic-centroid or "
            "signed support-toward-heavy residual; no network training."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "evaluation_seeds": seeds,
        "offsets_m": list(offsets),
        "offset_semantics": (
            "positive=toward current heavy UAV; negative=away; zero=dynamic centroid"
        ),
        "method_summaries": summaries,
        "paired_offset_minus_centroid": paired,
        "crn": crn_report(rows, offsets),
        "best_offset_by_descriptive_mean_delay_m": int(best_key),
        "offsets_with_descriptive_gain_over_1ms": practically_positive,
        "offsets_with_delay_gain_ci_excluding_zero": statistically_positive,
        "interpretation_boundary": (
            "The scan characterizes this causal one-support-UAV rule family; it "
            "is not a global trajectory optimum and does not retrain PPO."
        ),
    }


def plot_report(report: dict, output: Path) -> None:
    v45.configure_plotting()
    offsets = np.asarray(report["offsets_m"], dtype=float)
    summaries = report["method_summaries"]
    paired = report["paired_offset_minus_centroid"]

    def means(field):
        return np.asarray([summaries[str(int(x))][field]["mean"] for x in offsets])

    delay_gain = np.zeros_like(offsets)
    delay_low = np.zeros_like(offsets)
    delay_high = np.zeros_like(offsets)
    high_gain = np.zeros_like(offsets)
    low_gain = np.zeros_like(offsets)
    for index, offset in enumerate(offsets):
        if np.isclose(offset, 0.0):
            continue
        stats = paired["task_delay_weighted_ms"][str(int(offset))]
        delay_gain[index] = -stats["mean"]
        delay_low[index] = -stats["ci95_high"]
        delay_high[index] = -stats["ci95_low"]
        high_gain[index] = -paired["task_delay_high_ms"][str(int(offset))]["mean"]
        low_gain[index] = -paired["task_delay_low_ms"][str(int(offset))]["mean"]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.0))
    axes[0, 0].errorbar(
        offsets,
        delay_gain,
        yerr=np.vstack((delay_gain - delay_low, delay_high - delay_gain)),
        marker="o",
        linewidth=2,
        capsize=4,
        color="#2878B5",
    )
    axes[0, 0].axhline(0.0, color="#B23A48", linestyle="--")
    axes[0, 0].axhline(1.0, color="#7A8B5A", linestyle=":", label="1 ms描述线")
    axes[0, 0].set_ylabel("相对质心的时延收益 / ms")
    axes[0, 0].set_title("(a) 协作偏移是否产生净时延价值")
    axes[0, 0].legend()

    distance_axis = axes[0, 1]
    distance_axis.plot(
        offsets,
        means("mean_support_heavy_distance_m"),
        "o-",
        linewidth=2,
        color="#E07B39",
        label="支援机-重载机距离",
    )
    distance_axis.set_ylabel("平均距离 / m", color="#E07B39")
    distance_axis.tick_params(axis="y", labelcolor="#E07B39")
    rate_axis = distance_axis.twinx()
    rate_axis.plot(
        offsets,
        means("mean_support_heavy_nominal_rate_mbps"),
        "s--",
        linewidth=1.8,
        color="#2E8B57",
        label="名义U2U速率",
    )
    rate_axis.set_ylabel("名义U2U速率 / Mbps", color="#2E8B57")
    rate_axis.tick_params(axis="y", labelcolor="#2E8B57")
    distance_axis.set_title("(b) 偏移引起的链路几何变化")

    axes[1, 0].plot(offsets, high_gain, "o-", linewidth=2, label="高优时延收益")
    axes[1, 0].plot(offsets, low_gain, "s-", linewidth=2, label="低优时延收益")
    axes[1, 0].axhline(0.0, color="#B23A48", linestyle="--")
    axes[1, 0].set_ylabel("相对质心的时延收益 / ms")
    axes[1, 0].set_title("(c) 高低优任务的收益来源")
    axes[1, 0].legend()

    queue_delta = np.zeros_like(offsets)
    u2u_delta = np.zeros_like(offsets)
    for index, offset in enumerate(offsets):
        if np.isclose(offset, 0.0):
            continue
        queue_delta[index] = paired["mean_queue_mbits"][str(int(offset))]["mean"]
        u2u_delta[index] = paired["u2u_active_time_s"][str(int(offset))]["mean"]
    queue_axis = axes[1, 1]
    queue_axis.plot(offsets, queue_delta, "o-", linewidth=2, color="#7A65A8")
    queue_axis.axhline(0.0, color="#B23A48", linestyle="--")
    queue_axis.set_ylabel("平均队列变化 / Mbit", color="#7A65A8")
    queue_axis.tick_params(axis="y", labelcolor="#7A65A8")
    activity_axis = queue_axis.twinx()
    activity_axis.plot(offsets, u2u_delta, "s--", linewidth=1.8, color="#4B5563")
    activity_axis.set_ylabel("U2U活动时间变化 / s", color="#4B5563")
    activity_axis.tick_params(axis="y", labelcolor="#4B5563")
    queue_axis.set_title("(d) 队列与U2U服务代价")

    for axis in axes.flat:
        axis.set_xlabel("协作偏移 / m（正值朝重载机，负值远离）")
        axis.grid(linestyle="--", alpha=0.24)
    fig.suptitle("v51 同checkpoint协作方向偏移扫描")
    fig.tight_layout()
    fig.savefig(output / "01_协作方向偏移机制扫描.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_result(report: dict, output: Path) -> None:
    lines = [
        "# v51 协作方向偏移扫描",
        "",
        "> 冻结 v51 完整方法的微观门控 SAC；所有臂不训练网络。正偏移表示支援 UAV 朝当前重载 UAV 移动，负偏移表示远离，0 m 为动态质心。",
        "",
        "| 偏移/m | 时延/ms | 相对质心收益/ms | 支援链距离/m | 名义U2U速率/Mbps | 平均队列/Mbit |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = report["method_summaries"]
    paired = report["paired_offset_minus_centroid"]["task_delay_weighted_ms"]
    for offset in report["offsets_m"]:
        key = str(int(offset))
        gain = 0.0 if offset == 0 else -paired[key]["mean"]
        lines.append(
            f"| {offset:+g} | {summaries[key]['task_delay_weighted_ms']['mean']:.3f} | "
            f"{gain:+.3f} | {summaries[key]['mean_support_heavy_distance_m']['mean']:.2f} | "
            f"{summaries[key]['mean_support_heavy_nominal_rate_mbps']['mean']:.3f} | "
            f"{summaries[key]['mean_queue_mbits']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"描述性最低时延偏移：**{report['best_offset_by_descriptive_mean_delay_m']} m**。",
            f"时延收益 CI 排除 0 的偏移：`{report['offsets_with_delay_gain_ci_excluding_zero']}`。",
            "",
            "本结果只裁决这一条因果支援规则族，不等于全局轨迹最优解。",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--offsets", default=",".join(str(value) for value in DEFAULT_OFFSETS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--cpu-ghz", type=float, default=None)
    parser.add_argument("--load-scale", type=float, default=None)
    parser.add_argument("--u2u-bandwidth-mhz", type=float, default=None)
    parser.add_argument("--load-switch-phases", default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    seeds = v45.parse_seed_spec(args.eval_seeds)
    offsets = parse_offsets(args.offsets)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "per_seed_raw.csv"
    rows = [] if args.no_resume or not raw_path.is_file() else read_csv(raw_path)
    expected_pairs = {(seed, int(round(offset))) for seed in seeds for offset in offsets}
    rows = [
        row
        for row in rows
        if (int(row["seed"]), int(row["offset_m"])) in expected_pairs
    ]
    existing = {(int(row["seed"]), int(row["offset_m"])) for row in rows}

    policy = v45.load_method_policy(
        "joint_ppo_gated_sac",
        args.checkpoint,
        args.device,
    )
    scenario_args = argparse.Namespace(**vars(policy["args"]))
    for cli_name, checkpoint_name in (
        ("cpu_ghz", "cpu_ghz"),
        ("load_scale", "load_scale"),
        ("u2u_bandwidth_mhz", "u2u_bandwidth_mhz"),
        ("load_switch_phases", "load_switch_phases"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            setattr(scenario_args, checkpoint_name, value)
    policy = dict(policy, args=scenario_args)
    for seed in seeds:
        for offset in offsets:
            key = (int(seed), int(round(offset)))
            if key in existing:
                continue
            row = evaluate_offset(policy, offset, seed)
            rows.append(row)
            rows.sort(key=lambda item: (int(item["seed"]), int(item["offset_m"])))
            write_csv(raw_path, rows)
            print(
                f"seed={seed} offset={offset:+g}m: "
                f"delay={row['task_delay_weighted_ms']:.3f}ms "
                f"distance={row['mean_support_heavy_distance_m']:.2f}m "
                f"queue={row['mean_queue_mbits']:.3f}Mbit",
                flush=True,
            )

    report = build_report(rows, offsets, args.checkpoint, seeds)
    report["scenario"] = {
        "cpu_ghz": float(getattr(scenario_args, "cpu_ghz")),
        "load_scale": float(getattr(scenario_args, "load_scale")),
        "u2u_bandwidth_mhz": float(
            getattr(scenario_args, "u2u_bandwidth_mhz")
        ),
        "load_switch_phases": str(
            getattr(scenario_args, "load_switch_phases")
        ),
    }
    if not report["crn"]["all_external_trace_hashes_match"]:
        raise RuntimeError("CRN external trace hashes differ across offsets")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_report(report, output)
    write_result(report, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "best_offset_by_descriptive_mean_delay_m": report[
                    "best_offset_by_descriptive_mean_delay_m"
                ],
                "offsets_with_delay_gain_ci_excluding_zero": report[
                    "offsets_with_delay_gain_ci_excluding_zero"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
