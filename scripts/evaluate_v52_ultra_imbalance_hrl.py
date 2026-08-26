"""Evaluate the frozen v52 seed-1 eight-method matrix with paired CRN seeds.

The evaluator uses final checkpoints only.  It also replays the complete
method with learned, dynamic-centroid, and causal 20 m support macro policies
while keeping the complete method's micro policy fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from evaluate_v51_actor_autonomous_gate import (  # noqa: E402
    forced_sca_micro_policy,
)
from audit_v51_cooperative_offset_scan import geometry_metrics  # noqa: E402
from audit_v51_micro_decision_frequency import make_schedule  # noqa: E402
from run_load_switch_gate import run_gate_episode  # noqa: E402
from run_v34_cooperative_mobility_gate import (  # noqa: E402
    V34MacroPolicy,
    queue_metrics,
    soft_objective_metrics,
)
from train_centralized_macro import (  # noqa: E402
    fixed_macro_policy,
    zero_macro_policy,
)


RESULTS_VERSION = "tcom_v52_ultra_imbalance_hrl"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v52_ultra_imbalance_seed1"
METHODS = (
    "joint_ppo_gated_sac",
    "cppo_sca",
    "joint_ppo_sac",
    "centroid_gated_sac",
    "centroid_sac",
    "fixed_sac",
    "joint_td3_sac",
    "no_coop",
)
MAIN_METHOD = "joint_ppo_gated_sac"
LABELS = {
    "joint_ppo_gated_sac": "PPO-自主门控SAC（本文）",
    "cppo_sca": "PPO-SCA",
    "joint_ppo_sac": "PPO-原始残差SAC",
    "centroid_gated_sac": "动态质心-自主门控SAC",
    "centroid_sac": "动态质心-原始SAC",
    "fixed_sac": "固定站位-SAC",
    "joint_td3_sac": "TD3-SAC",
    "no_coop": "No-Coop",
}
COLORS = {
    "joint_ppo_gated_sac": "#2878B5",
    "cppo_sca": "#7A65A8",
    "joint_ppo_sac": "#E07B39",
    "centroid_gated_sac": "#2E8B57",
    "centroid_sac": "#65A765",
    "fixed_sac": "#777777",
    "joint_td3_sac": "#C44E52",
    "no_coop": "#A6A6A6",
}
SUMMARY_FIELDS = (
    "task_delay_weighted_ms",
    "task_delay_high_ms",
    "task_delay_low_ms",
    "transition_window_delay_weighted_ms",
    "mean_queue_mbits",
    "terminal_queue_mbits",
    "tail_queue_slope_mbps",
    "peak_queue_mbits",
    "task_completion_rate",
    "soft_objective_j",
    "average_power_w",
    "coop_ratio",
    "cross_flow_mbits",
    "u2u_active_time_s",
    "cross_payload_efficiency_mbps",
    "mean_support_heavy_distance_m",
    "mean_support_heavy_nominal_rate_mbps",
    "mean_all_uav_centroid_residual_m",
    "heavy_to_support_flow_mbits",
    "support_motion_alignment_mean",
    "support_toward_fraction",
    "residual_gate_mean",
)
MECHANISM_ARMS = (
    "dynamic_centroid",
    "causal_support_20m",
    "learned_macro",
)
MECHANISM_LABELS = {
    "dynamic_centroid": "动态质心",
    "causal_support_20m": "因果支援20m",
    "learned_macro": "学习轨迹",
}
LAYER_ARMS = (
    "learned_gated",
    "centroid_gated",
    "learned_forced_sca",
    "centroid_forced_sca",
)
LAYER_LABELS = {
    "learned_gated": "学习轨迹 + 门控SAC",
    "centroid_gated": "动态质心 + 门控SAC",
    "learned_forced_sca": "学习轨迹 + 纯SCA",
    "centroid_forced_sca": "动态质心 + 纯SCA",
}


def checkpoint(stage: str, train_seed: int, method: str) -> Path:
    return (
        ROOT
        / "results"
        / RESULTS_VERSION
        / stage
        / f"seed{train_seed}"
        / method
        / "checkpoints"
        / "final.pt"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_policies(stage: str, train_seed: int, device: str) -> dict[str, dict]:
    policies = {}
    for method in METHODS:
        path = checkpoint(stage, train_seed, method)
        if not path.is_file():
            raise FileNotFoundError(path)
        policy = v45.load_method_policy(method, path, device)
        if method in {"centroid_gated_sac", "centroid_sac"}:
            policy["macro_policy"] = zero_macro_policy
        elif method == "fixed_sac":
            policy["macro_policy"] = fixed_macro_policy
        policies[method] = policy

    reference = v45.scenario_signature(policies[MAIN_METHOD]["args"])
    mismatches = {
        method: v45.scenario_signature(policy["args"])
        for method, policy in policies.items()
        if v45.scenario_signature(policy["args"]) != reference
    }
    if mismatches:
        raise RuntimeError(
            "Checkpoint scenarios differ: "
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    return policies


class RoleObservedMacroPolicy:
    """Attach the frozen causal heavy/support role definition to any policy."""

    def __init__(self, base_policy):
        self.base_policy = base_policy
        self.observer = V34MacroPolicy(
            "cooperative",
            0.0,
            movement_trigger_m=0.0,
        )

    def __call__(self, env, local_obs_batch, eval_mode):
        self.observer(env, local_obs_batch, eval_mode)
        return self.base_policy(env, local_obs_batch, eval_mode)

    def metadata(self):
        return self.observer.metadata()


def support_motion_metrics(trace: dict) -> dict:
    alignments = []
    toward = []
    movements = []
    for record in trace.get("macro_records", ()):
        heavy = int(record.get("heavy_uav", -1))
        support = int(record.get("support_uav", -1))
        before = np.asarray(record["uav_before_m"], dtype=np.float64)
        after = np.asarray(record["uav_after_m"], dtype=np.float64)
        if heavy < 0 or support < 0 or heavy == support:
            continue
        displacement = after[support] - before[support]
        direction = before[heavy] - before[support]
        move_norm = float(np.linalg.norm(displacement))
        direction_norm = float(np.linalg.norm(direction))
        movements.append(move_norm)
        if move_norm <= 1e-9 or direction_norm <= 1e-9:
            continue
        cosine = float(np.dot(displacement, direction) / (move_norm * direction_norm))
        alignments.append(float(np.clip(cosine, -1.0, 1.0)))
        toward.append(float(cosine > 0.0))
    return {
        "support_motion_alignment_mean": (
            float(np.mean(alignments)) if alignments else 0.0
        ),
        "support_toward_fraction": float(np.mean(toward)) if toward else 0.0,
        "support_movement_mean_m": float(np.mean(movements)) if movements else 0.0,
    }


def trajectory_payload(env, trace: dict) -> dict[str, np.ndarray]:
    base = env.env
    return {
        "uavs": np.asarray(base.uav_trajectory, dtype=np.float32),
        "users": np.asarray(base.users_trajectory, dtype=np.float32),
        "centroids": np.asarray(base.centroid_trajectory, dtype=np.float32),
        "user_allocations": np.asarray(base.user_allocations, dtype=np.int64),
        "macro_uavs": np.asarray(trace["uav_trajectory_m"], dtype=np.float32),
        "macro_centroids": np.asarray(
            trace["centroid_trajectory_m"], dtype=np.float32
        ),
        "macro_phases": np.asarray(trace["trajectory_phases"], dtype=np.int64),
    }


def evaluate_policy_arm(
    policy: dict,
    method: str,
    seed: int,
    macro_policy,
    metadata_provider,
    trajectory_path: Path | None = None,
) -> dict:
    args = policy["args"]
    gate_values = getattr(policy["micro_policy"], "residual_gate_values", None)
    if gate_values is not None:
        gate_values.clear()
    env = v45.make_eval_env(args, record_trajectory=trajectory_path is not None)
    if policy["micro_solver"] == "local":
        env.force_local_routing = True
    result, trace = run_gate_episode(
        env,
        args,
        policy["micro_policy"],
        method,
        seed,
        macro_steps=int(args.macro_steps),
        load_switch_schedule=make_schedule(args, seed),
        macro_policy=macro_policy,
        macro_metadata_provider=metadata_provider,
    )
    result.update(queue_metrics(trace, float(env.env.macro_dt)))
    result.update(
        soft_objective_metrics(
            trace,
            float(getattr(args, "micro_reward_scale", 1.0)),
        )
    )
    result.update(geometry_metrics(env, trace))
    result.update(support_motion_metrics(trace))
    arrived = max(int(result["task_arrived_count"]), 1)
    result["task_completion_rate"] = float(result["task_completed_count"] / arrived)
    result["cross_payload_efficiency_mbps"] = float(
        result["cross_flow_mbits"] / max(result["u2u_active_time_s"], 1e-12)
    )
    result["method"] = method
    result.pop("policy", None)
    if gate_values is not None and len(gate_values):
        gate_array = np.asarray(gate_values, dtype=np.float64)
        result["residual_gate_mean"] = float(np.mean(gate_array))
        result["residual_gate_std"] = float(np.std(gate_array))
    else:
        result["residual_gate_mean"] = 0.0 if policy["micro_solver"] == "sca" else 1.0
        result["residual_gate_std"] = 0.0
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory_path, **trajectory_payload(env, trace))
    return result


def evaluate_method(
    policy: dict,
    method: str,
    seed: int,
    trajectory_path: Path | None = None,
) -> dict:
    observed = RoleObservedMacroPolicy(policy["macro_policy"])
    return evaluate_policy_arm(
        policy,
        method,
        seed,
        observed,
        observed.metadata,
        trajectory_path,
    )


def mechanism_policy(policy: dict, arm: str):
    if arm == "learned_macro":
        observed = RoleObservedMacroPolicy(policy["macro_policy"])
        return observed, observed.metadata
    if arm == "dynamic_centroid":
        controller = V34MacroPolicy("cooperative", 0.0, movement_trigger_m=0.0)
        return controller, controller.metadata
    if arm == "causal_support_20m":
        controller = V34MacroPolicy("cooperative", 20.0, movement_trigger_m=0.0)
        return controller, controller.metadata
    raise ValueError(arm)


def normalized_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if key in {"method", "external_trace_hash"}:
                converted[key] = str(value)
            elif key in {
                "seed",
                "task_arrived_count",
                "task_completed_count",
                "micro_decision_count",
                "proxy_step_count",
            }:
                converted[key] = int(float(value))
            else:
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
        output.append(converted)
    return output


def mean_ci95(values) -> dict:
    return v45.mean_ci95([float(value) for value in values])


def method_summaries(rows: list[dict], methods: tuple[str, ...]) -> dict:
    return {
        method: {
            field: mean_ci95(
                row[field] for row in rows if row["method"] == method
            )
            for field in SUMMARY_FIELDS
        }
        for method in methods
    }


def paired_difference(rows: list[dict], left: str, right: str, field: str) -> dict:
    left_values = {
        int(row["seed"]): float(row[field])
        for row in rows
        if row["method"] == left
    }
    right_values = {
        int(row["seed"]): float(row[field])
        for row in rows
        if row["method"] == right
    }
    seeds = sorted(set(left_values) & set(right_values))
    deltas = [left_values[seed] - right_values[seed] for seed in seeds]
    return {
        **mean_ci95(deltas),
        "left": left,
        "right": right,
        "field": field,
        "per_seed": {str(seed): value for seed, value in zip(seeds, deltas)},
    }


def crn_report(rows: list[dict], methods: tuple[str, ...]) -> dict:
    mismatches = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        hashes = {
            row["method"]: row["external_trace_hash"]
            for row in rows
            if int(row["seed"]) == seed and row["method"] in methods
        }
        if len(hashes) != len(methods) or len(set(hashes.values())) != 1:
            mismatches[str(seed)] = hashes
    return {
        "all_methods_match_within_seed": not mismatches,
        "mismatches": mismatches,
    }


def build_report(rows: list[dict], seeds: list[int], policies: dict) -> dict:
    summaries = method_summaries(rows, METHODS)
    paired = {
        baseline: {
            field: paired_difference(rows, MAIN_METHOD, baseline, field)
            for field in SUMMARY_FIELDS
        }
        for baseline in METHODS
        if baseline != MAIN_METHOD
    }
    ranking = sorted(
        METHODS,
        key=lambda method: summaries[method]["task_delay_weighted_ms"]["mean"],
    )
    return {
        "scope": (
            "One training seed; final checkpoints; paired evaluation seeds. "
            "This is the frozen seed-1 screen, not a multi-training-seed claim."
        ),
        "results_version": RESULTS_VERSION,
        "training_seed": 1,
        "evaluation_seeds": seeds,
        "checkpoint_rule": "final only",
        "checkpoints": {
            method: {
                "path": str(policy["path"].resolve()),
                "sha256": sha256(policy["path"]),
            }
            for method, policy in policies.items()
        },
        "scenario_signature": v45.scenario_signature(policies[MAIN_METHOD]["args"]),
        "method_summaries": summaries,
        "complete_minus_baseline": paired,
        "delay_ranking": ranking,
        "crn": crn_report(rows, METHODS),
    }


def build_mechanism_report(rows: list[dict], seeds: list[int]) -> dict:
    summaries = method_summaries(rows, MECHANISM_ARMS)
    return {
        "scope": (
            "Read-only macro-policy replay with the final complete-method "
            "micro policy frozen; no network updates."
        ),
        "evaluation_seeds": seeds,
        "method_summaries": summaries,
        "learned_minus_centroid": {
            field: paired_difference(
                rows, "learned_macro", "dynamic_centroid", field
            )
            for field in SUMMARY_FIELDS
        },
        "support20_minus_centroid": {
            field: paired_difference(
                rows, "causal_support_20m", "dynamic_centroid", field
            )
            for field in SUMMARY_FIELDS
        },
        "crn": crn_report(rows, MECHANISM_ARMS),
    }


def layer_interaction(rows: list[dict], field: str) -> dict:
    by_arm = {
        arm: {
            int(row["seed"]): float(row[field])
            for row in rows
            if row["method"] == arm
        }
        for arm in LAYER_ARMS
    }
    seeds = sorted(set.intersection(*(set(values) for values in by_arm.values())))
    deltas = [
        (
            by_arm["learned_gated"][seed]
            - by_arm["centroid_gated"][seed]
            - by_arm["learned_forced_sca"][seed]
            + by_arm["centroid_forced_sca"][seed]
        )
        for seed in seeds
    ]
    return {
        **mean_ci95(deltas),
        "field": field,
        "per_seed": {str(seed): value for seed, value in zip(seeds, deltas)},
    }


def build_layer_report(rows: list[dict], seeds: list[int]) -> dict:
    comparisons = {
        "macro_effect_with_gated_sac": ("learned_gated", "centroid_gated"),
        "macro_effect_with_forced_sca": (
            "learned_forced_sca",
            "centroid_forced_sca",
        ),
        "micro_gate_effect_with_learned_macro": (
            "learned_gated",
            "learned_forced_sca",
        ),
        "micro_gate_effect_with_centroid": (
            "centroid_gated",
            "centroid_forced_sca",
        ),
    }
    return {
        "scope": (
            "Post-hoc read-only 2x2 attribution using the final complete-method "
            "checkpoint. It does not alter the frozen v52 verdict."
        ),
        "evaluation_seeds": seeds,
        "method_summaries": method_summaries(rows, LAYER_ARMS),
        "paired_effects": {
            name: {
                field: paired_difference(rows, left, right, field)
                for field in SUMMARY_FIELDS
            }
            for name, (left, right) in comparisons.items()
        },
        "interaction": {
            field: layer_interaction(rows, field) for field in SUMMARY_FIELDS
        },
        "crn": crn_report(rows, LAYER_ARMS),
    }


def configure_plotting() -> None:
    v45.configure_plotting()


def plot_method_summary(report: dict, output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.2))
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms"),
        ("mean_queue_mbits", "平均积压队列 / Mbit"),
        ("task_completion_rate", "任务完成率"),
        ("soft_objective_j", "软目标 J（越低越好）"),
    )
    cooperative_methods = tuple(method for method in METHODS if method != "no_coop")
    x = np.arange(len(cooperative_methods))
    labels = [LABELS[method] for method in cooperative_methods]
    for axis, (field, ylabel) in zip(axes.flat, fields):
        stats = [
            report["method_summaries"][method][field]
            for method in cooperative_methods
        ]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray(
            [item["ci95_high"] - item["mean"] for item in stats]
        )
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            color=[COLORS[method] for method in cooperative_methods],
            capsize=3,
        )
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        value_format = "%.3f" if field in {"task_completion_rate", "soft_objective_j"} else "%.1f"
        axis.bar_label(bars, fmt=value_format, padding=3, fontsize=8)
        span = max(float(np.ptp(means)), max(abs(means.mean()) * 0.01, 1e-6))
        axis.set_ylim(float(means.min() - 0.18 * span), float(means.max() + 0.38 * span))
    no_coop = report["method_summaries"]["no_coop"]
    note = (
        "No-Coop（单列，避免压扁协作方法）："
        f"时延 {no_coop['task_delay_weighted_ms']['mean']:.1f} ms；"
        f"队列 {no_coop['mean_queue_mbits']['mean']:.1f} Mbit；"
        f"完成率 {no_coop['task_completion_rate']['mean']:.3f}；"
        f"J {no_coop['soft_objective_j']['mean']:.3f}"
    )
    fig.suptitle("v52 final checkpoint：协作方法局部尺度比较", y=0.995)
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.975))
    fig.savefig(output / "01_八方法性能对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_paired_delay(report: dict, output: Path) -> None:
    configure_plotting()
    baselines = [
        method
        for method in METHODS
        if method not in {MAIN_METHOD, "no_coop"}
    ]
    stats = [
        report["complete_minus_baseline"][method]["task_delay_weighted_ms"]
        for method in baselines
    ]
    means = np.asarray([item["mean"] for item in stats])
    low = means - np.asarray([item["ci95_low"] for item in stats])
    high = np.asarray([item["ci95_high"] for item in stats]) - means
    fig, axis = plt.subplots(figsize=(10.8, 6.8))
    y = np.arange(len(baselines))
    for index, (mean, lo, hi) in enumerate(zip(means, low, high)):
        ci_low = mean - lo
        ci_high = mean + hi
        color = "#C44E52" if ci_low > 0 else "#2E8B57" if ci_high < 0 else "#777777"
        axis.errorbar(
            mean,
            index,
            xerr=np.asarray([[lo], [hi]]),
            fmt="o",
            color=color,
            capsize=4,
        )
        axis.annotate(
            f"{mean:+.2f} ms",
            (mean, index),
            xytext=(7 if mean >= 0 else -7, 0),
            textcoords="offset points",
            ha="left" if mean >= 0 else "right",
            va="center",
            fontsize=9,
        )
    axis.axvline(0.0, color="#333333", linewidth=1.2)
    axis.set_yticks(y, [LABELS[item] for item in baselines])
    axis.set_xlabel("本文方法 - 对照方法的配对时延差 / ms（负值更好）")
    axis.set_title("完整方法相对各基线的配对差及 95% CI")
    axis.grid(axis="x", linestyle="--", alpha=0.28)
    no_coop = report["complete_minus_baseline"]["no_coop"][
        "task_delay_weighted_ms"
    ]
    fig.text(
        0.5,
        0.018,
        "No-Coop 单列："
        f"{no_coop['mean']:+.1f} ms，95% CI "
        f"[{no_coop['ci95_low']:+.1f}, {no_coop['ci95_high']:+.1f}]",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    fig.savefig(output / "02_完整方法配对时延差.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectories(output: Path, seed: int) -> None:
    configure_plotting()
    methods = (
        MAIN_METHOD,
        "cppo_sca",
        "centroid_gated_sac",
        "joint_td3_sac",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.2), sharex=True, sharey=True)
    uav_colors = ("#2878B5", "#E67E22", "#2E9D55")
    for axis, method in zip(axes.flat, methods):
        with np.load(output / "trajectories" / f"{method}_seed{seed}.npz") as data:
            users = data["users"]
            uavs = data["uavs"]
            centroids = data["centroids"]
        user_stride = max(1, users.shape[0] // 300)
        for index in range(users.shape[1]):
            color = uav_colors[index % 3]
            axis.plot(
                users[::user_stride, index, 0],
                users[::user_stride, index, 1],
                color=color,
                alpha=0.10,
                linewidth=0.55,
            )
        for index, color in enumerate(uav_colors):
            axis.plot(
                centroids[:, index, 0],
                centroids[:, index, 1],
                linestyle="--",
                color=color,
                alpha=0.65,
                linewidth=1.2,
            )
            axis.plot(
                uavs[:, index, 0],
                uavs[:, index, 1],
                color=color,
                linewidth=2.3,
                label=f"UAV-{index + 1}",
            )
        axis.set_title(LABELS[method])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(linestyle="--", alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Y 坐标 / m")
    for axis in axes[-1, :]:
        axis.set_xlabel("X 坐标 / m")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"v52 代表性轨迹（共同随机种子 {seed}，不代表均值排名）")
    fig.tight_layout()
    fig.savefig(output / "03_主要方法与地面用户轨迹.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_mechanism(report: dict, output: Path) -> None:
    configure_plotting()
    fields = (
        ("task_delay_weighted_ms", "加权时延 / ms"),
        ("mean_queue_mbits", "平均队列 / Mbit"),
        ("mean_support_heavy_distance_m", "支援-重载距离 / m"),
        ("mean_support_heavy_nominal_rate_mbps", "支援链路名义速率 / Mbps"),
        ("heavy_to_support_flow_mbits", "重载→支援流量 / Mbit"),
        ("cross_payload_efficiency_mbps", "跨机载荷效率 / Mbit/(link·s)"),
    )
    x = np.arange(len(MECHANISM_ARMS))
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.7))
    for axis, (field, ylabel) in zip(axes.flat, fields):
        stats = [
            report["method_summaries"][arm][field]
            for arm in MECHANISM_ARMS
        ]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            color=("#65A765", "#E07B39", "#2878B5"),
            capsize=3,
        )
        axis.set_xticks(x, [MECHANISM_LABELS[arm] for arm in MECHANISM_ARMS], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        value_format = "%.2f" if float(np.max(np.abs(means))) < 100 else "%.1f"
        axis.bar_label(bars, fmt=value_format, padding=3, fontsize=8)
    fig.suptitle("完整方法 final 微观策略下的宏观位置机制重放")
    fig.tight_layout()
    fig.savefig(output / "04_轨迹物理机制分解.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_convergence(stage: str, train_seed: int, output: Path) -> None:
    configure_plotting()
    curve_specs = {
        MAIN_METHOD: "eval_central_macro.csv",
        "cppo_sca": "eval_central_sca.csv",
        "joint_ppo_sac": "eval_central_macro.csv",
        "centroid_gated_sac": "eval_central_macro.csv",
        "joint_td3_sac": "eval_central_td3.csv",
    }
    base = ROOT / "results" / RESULTS_VERSION / stage / f"seed{train_seed}"
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True)
    loaded = 0
    for method, filename in curve_specs.items():
        path = base / method / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        episodes = np.asarray([float(row["episode"]) for row in rows])
        delay = np.asarray([float(row["task_delay_weighted_ms"]) for row in rows])
        queue = np.asarray(
            [float(row["q_high"]) + float(row["q_low"]) for row in rows]
        )
        color = COLORS[method]
        axes[0].plot(episodes, delay, marker="o", markersize=3.2, linewidth=1.8, color=color, label=LABELS[method])
        axes[1].plot(episodes, queue, marker="o", markersize=3.2, linewidth=1.8, color=color, label=LABELS[method])
        loaded += 1
    if loaded == 0:
        plt.close(fig)
        return
    axes[0].set_ylabel("固定评估加权端到端时延 / ms")
    axes[1].set_ylabel("固定评估总队列 / Mbit")
    for axis in axes:
        axis.set_xlabel("联合训练 episode")
        axis.grid(linestyle="--", alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("v52 联合训练固定评估曲线（训练 seed=1）")
    fig.tight_layout()
    fig.savefig(output / "05_收敛与队列稳定性.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_layer_attribution(report: dict, output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms"),
        ("mean_queue_mbits", "平均积压队列 / Mbit"),
    )
    macro_labels = ("学习轨迹", "动态质心")
    gated_arms = ("learned_gated", "centroid_gated")
    sca_arms = ("learned_forced_sca", "centroid_forced_sca")
    x = np.arange(2)
    width = 0.34
    for axis, (field, ylabel) in zip(axes, fields):
        gated = [report["method_summaries"][arm][field] for arm in gated_arms]
        sca = [report["method_summaries"][arm][field] for arm in sca_arms]
        gated_means = np.asarray([item["mean"] for item in gated])
        sca_means = np.asarray([item["mean"] for item in sca])
        gated_err = np.asarray([item["ci95_high"] - item["mean"] for item in gated])
        sca_err = np.asarray([item["ci95_high"] - item["mean"] for item in sca])
        bars_gated = axis.bar(
            x - width / 2,
            gated_means,
            width,
            yerr=gated_err,
            label="自主门控SAC",
            color="#2878B5",
            capsize=3,
        )
        bars_sca = axis.bar(
            x + width / 2,
            sca_means,
            width,
            yerr=sca_err,
            label="强制纯SCA",
            color="#7A65A8",
            capsize=3,
        )
        axis.bar_label(bars_gated, fmt="%.2f", padding=3, fontsize=9)
        axis.bar_label(bars_sca, fmt="%.2f", padding=3, fontsize=9)
        combined = np.concatenate([gated_means, sca_means])
        span = max(float(np.ptp(combined)), max(abs(combined.mean()) * 0.005, 1e-6))
        axis.set_ylim(float(combined.min() - 0.2 * span), float(combined.max() + 0.45 * span))
        axis.set_xticks(x, macro_labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()
    interaction = report["interaction"]["task_delay_weighted_ms"]
    fig.suptitle(
        "v52 同一完整 checkpoint 的宏观×微观 2×2 归因\n"
        f"时延交互项 {interaction['mean']:+.2f} ms，95% CI "
        f"[{interaction['ci95_low']:+.2f}, {interaction['ci95_high']:+.2f}]"
    )
    fig.tight_layout()
    fig.savefig(output / "06_宏微观层2x2归因.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_verdict(
    report: dict,
    mechanism: dict,
    layer: dict,
    output: Path,
) -> None:
    lines = [
        "# v52 seed-1 冻结筛查结果",
        "",
        "> 单一训练种子；所有方法使用 final checkpoint；8300--8319 为配对评估种子。",
        "",
        "| 排名 | 方法 | 加权时延/ms | 95% CI | 平均队列/Mbit | 期末队列/Mbit | 完成率 | J | 功率/W |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, method in enumerate(report["delay_ranking"], start=1):
        item = report["method_summaries"][method]
        delay = item["task_delay_weighted_ms"]
        lines.append(
            f"| {rank} | {LABELS[method]} | {delay['mean']:.2f} | "
            f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
            f"{item['mean_queue_mbits']['mean']:.3f} | "
            f"{item['terminal_queue_mbits']['mean']:.3f} | "
            f"{item['task_completion_rate']['mean']:.6f} | "
            f"{item['soft_objective_j']['mean']:.6f} | "
            f"{item['average_power_w']['mean']:.2f} |"
        )
    lines.extend(["", "## 冻结主比较", ""])
    delay_pass = True
    for baseline in ("cppo_sca", "centroid_gated_sac"):
        delta = report["complete_minus_baseline"][baseline]["task_delay_weighted_ms"]
        passed = float(delta["ci95_high"]) < 0.0
        delay_pass = delay_pass and passed
        lines.append(
            f"- 本文 - {LABELS[baseline]} = {delta['mean']:+.3f} ms，"
            f"95% CI [{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}]："
            f"{'PASS' if passed else 'FAIL'}。"
        )
    final_verdict = "PASS" if delay_pass else "FAIL"
    lines.extend([
        "",
        "## 最终冻结判决",
        "",
        f"- 时延主判据：{'PASS' if delay_pass else 'FAIL'}。",
        f"- v52 seed-1 最终判决：**{final_verdict}**。",
        "- 队列与完成率稳定，但不能覆盖完整方法显著输给 PPO-SCA 的主判据。",
        "- 按冻结协议，不扩训练种子、不挑 checkpoint、不延长训练。",
    ])
    learned_delta = mechanism["learned_minus_centroid"]["task_delay_weighted_ms"]
    forced_sca_macro = layer["paired_effects"][
        "macro_effect_with_forced_sca"
    ]["task_delay_weighted_ms"]
    learned_micro = layer["paired_effects"][
        "micro_gate_effect_with_learned_macro"
    ]["task_delay_weighted_ms"]
    centroid_micro = layer["paired_effects"][
        "micro_gate_effect_with_centroid"
    ]["task_delay_weighted_ms"]
    interaction = layer["interaction"]["task_delay_weighted_ms"]
    lines.extend([
        "",
        "## 同 checkpoint 机制与层级归因",
        "",
        f"- 学习轨迹 - 动态质心 = {learned_delta['mean']:+.3f} ms，"
        f"95% CI [{learned_delta['ci95_low']:+.3f}, {learned_delta['ci95_high']:+.3f}]。",
        f"- 纯 SCA 下的学习轨迹效应 = {forced_sca_macro['mean']:+.3f} ms，"
        f"95% CI [{forced_sca_macro['ci95_low']:+.3f}, {forced_sca_macro['ci95_high']:+.3f}]。",
        f"- 学习轨迹下的自主门控效应 = {learned_micro['mean']:+.3f} ms，"
        f"95% CI [{learned_micro['ci95_low']:+.3f}, {learned_micro['ci95_high']:+.3f}]。",
        f"- 动态质心下的自主门控效应 = {centroid_micro['mean']:+.3f} ms，"
        f"95% CI [{centroid_micro['ci95_low']:+.3f}, {centroid_micro['ci95_high']:+.3f}]。",
        f"- 宏观×微观交互项 = {interaction['mean']:+.3f} ms，"
        f"95% CI [{interaction['ci95_low']:+.3f}, {interaction['ci95_high']:+.3f}]。",
        "- 结论：宏观移动显著有利；当前自主门控残差显著劣于强制纯 SCA，是完整方法的主要瓶颈。",
        "",
        "队列、完成率和完整逐种子统计见 `report.json`；宏观物理链见 "
        "`mechanism_report.json`；2×2 归因见 `layer_attribution_report.json`。",
    ])
    (output / "gate_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_layer_summary(report: dict, output: Path) -> None:
    effects = report["paired_effects"]
    lines = [
        "# v52 宏微观层同 checkpoint 事后归因",
        "",
        "> 零训练只读诊断；不改变冻结的 v52 seed-1 FAIL 判决。",
        "",
    ]
    labels = {
        "macro_effect_with_gated_sac": "门控SAC下：学习轨迹 - 动态质心",
        "macro_effect_with_forced_sca": "纯SCA下：学习轨迹 - 动态质心",
        "micro_gate_effect_with_learned_macro": "学习轨迹下：门控SAC - 纯SCA",
        "micro_gate_effect_with_centroid": "动态质心下：门控SAC - 纯SCA",
    }
    for key, label in labels.items():
        item = effects[key]["task_delay_weighted_ms"]
        lines.append(
            f"- {label} = {item['mean']:+.3f} ms，95% CI "
            f"[{item['ci95_low']:+.3f}, {item['ci95_high']:+.3f}]。"
        )
    interaction = report["interaction"]["task_delay_weighted_ms"]
    lines.append(
        f"- 宏观×微观交互项 = {interaction['mean']:+.3f} ms，95% CI "
        f"[{interaction['ci95_low']:+.3f}, {interaction['ci95_high']:+.3f}]。"
    )
    (output / "layer_attribution.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="formal")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8300-8319")
    parser.add_argument("--trajectory-seed", type=int, default=8300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    seeds = v45.parse_seed_spec(args.eval_seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = load_policies(args.stage, args.train_seed, args.device)

    raw_path = output / "per_seed_raw.csv"
    rows = normalized_rows(read_csv(raw_path)) if args.reuse else []
    existing = {(int(row["seed"]), row["method"]) for row in rows}
    for seed in seeds:
        for method in METHODS:
            if (seed, method) in existing:
                continue
            trajectory_path = (
                output / "trajectories" / f"{method}_seed{seed}.npz"
                if int(seed) == int(args.trajectory_seed)
                else None
            )
            row = evaluate_method(
                policies[method], method, seed, trajectory_path
            )
            rows.append(row)
            existing.add((seed, method))
            rows.sort(key=lambda item: (int(item["seed"]), METHODS.index(item["method"])))
            write_csv(raw_path, rows)
            print(
                f"eval={seed} method={method} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms "
                f"queue={row['mean_queue_mbits']:.3f}Mbit "
                f"completion={row['task_completion_rate']:.6f}",
                flush=True,
            )

    report = build_report(rows, seeds, policies)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mechanism_path = output / "mechanism_per_seed_raw.csv"
    mechanism_rows = normalized_rows(read_csv(mechanism_path)) if args.reuse else []
    mechanism_existing = {
        (int(row["seed"]), row["method"]) for row in mechanism_rows
    }
    complete = policies[MAIN_METHOD]
    for seed in seeds:
        for arm in MECHANISM_ARMS:
            if (seed, arm) in mechanism_existing:
                continue
            macro_policy, metadata = mechanism_policy(complete, arm)
            row = evaluate_policy_arm(
                complete,
                arm,
                seed,
                macro_policy,
                metadata,
            )
            mechanism_rows.append(row)
            mechanism_existing.add((seed, arm))
            mechanism_rows.sort(
                key=lambda item: (
                    int(item["seed"]),
                    MECHANISM_ARMS.index(item["method"]),
                )
            )
            write_csv(mechanism_path, mechanism_rows)
            print(
                f"mechanism={seed} arm={arm} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms",
                flush=True,
            )

    mechanism = build_mechanism_report(mechanism_rows, seeds)
    (output / "mechanism_report.json").write_text(
        json.dumps(mechanism, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    layer_path = output / "layer_attribution_per_seed_raw.csv"
    layer_rows = normalized_rows(read_csv(layer_path)) if args.reuse else []
    layer_existing = {(int(row["seed"]), row["method"]) for row in layer_rows}
    source_arms = {
        "learned_gated": "learned_macro",
        "centroid_gated": "dynamic_centroid",
    }
    for seed in seeds:
        for arm, source in source_arms.items():
            if (seed, arm) in layer_existing:
                continue
            source_row = next(
                row
                for row in mechanism_rows
                if int(row["seed"]) == int(seed) and row["method"] == source
            )
            copied = dict(source_row)
            copied["method"] = arm
            layer_rows.append(copied)
            layer_existing.add((seed, arm))

    forced_sca = dict(complete)
    forced_sca.update(
        name="complete_checkpoint_forced_sca",
        micro_policy=forced_sca_micro_policy(complete["args"]),
        micro_solver="sca",
    )
    forced_arms = {
        "learned_forced_sca": "learned_macro",
        "centroid_forced_sca": "dynamic_centroid",
    }
    for seed in seeds:
        for arm, macro_arm in forced_arms.items():
            if (seed, arm) in layer_existing:
                continue
            macro_policy, metadata = mechanism_policy(complete, macro_arm)
            row = evaluate_policy_arm(
                forced_sca,
                arm,
                seed,
                macro_policy,
                metadata,
            )
            layer_rows.append(row)
            layer_existing.add((seed, arm))
            layer_rows.sort(
                key=lambda item: (
                    int(item["seed"]),
                    LAYER_ARMS.index(item["method"]),
                )
            )
            write_csv(layer_path, layer_rows)
            print(
                f"layer={seed} arm={arm} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms",
                flush=True,
            )

    layer = build_layer_report(layer_rows, seeds)
    (output / "layer_attribution_report.json").write_text(
        json.dumps(layer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_verdict(report, mechanism, layer, output)
    write_layer_summary(layer, output)
    plot_method_summary(report, output)
    plot_paired_delay(report, output)
    plot_trajectories(output, int(args.trajectory_seed))
    plot_mechanism(mechanism, output)
    plot_convergence(args.stage, args.train_seed, output)
    plot_layer_attribution(layer, output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
