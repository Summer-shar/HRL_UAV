"""Evaluate the frozen seed-1 v51 autonomous residual-gate screen."""

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
from train_centralized_macro import zero_macro_policy  # noqa: E402


RESULTS_VERSION = "tcom_v51_actor_autonomous_gate"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v51_actor_autonomous_gate_seed1"
METHODS = (
    "cppo_sca",
    "joint_ppo_sac",
    "centroid_gated_sac",
    "joint_ppo_gated_sac",
)
MAIN_METHOD = "joint_ppo_gated_sac"
LABELS = {
    "cppo_sca": "PPO-SCA",
    "joint_ppo_sac": "PPO-原始残差SAC",
    "centroid_gated_sac": "动态质心-自主门控SAC",
    "joint_ppo_gated_sac": "PPO-自主门控SAC（本文）",
}
COLORS = {
    "cppo_sca": "#7A65A8",
    "joint_ppo_sac": "#E07B39",
    "centroid_gated_sac": "#2E8B57",
    "joint_ppo_gated_sac": "#2878B5",
}
EVAL_FILES = {
    "cppo_sca": "eval_central_sca.csv",
    "joint_ppo_sac": "eval_central_macro.csv",
    "centroid_gated_sac": "eval_central_macro.csv",
    "joint_ppo_gated_sac": "eval_central_macro.csv",
}
TRAIN_FILES = {method: "train_centralized.csv" for method in METHODS}
SAC_METHODS = (
    "joint_ppo_sac",
    "centroid_gated_sac",
    "joint_ppo_gated_sac",
)
DIAGNOSTIC_METHODS = (
    "complete_checkpoint_centroid_macro",
    "complete_checkpoint_forced_sca",
    "complete_checkpoint_centroid_macro_forced_sca",
)
DIAGNOSTIC_LABELS = {
    "complete_checkpoint_centroid_macro": "同一完整checkpoint-动态质心宏观",
    "complete_checkpoint_forced_sca": "同一完整checkpoint-强制纯SCA微观",
    "complete_checkpoint_centroid_macro_forced_sca": (
        "同一完整checkpoint-动态质心宏观+强制纯SCA微观"
    ),
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


def load_policies(stage: str, train_seed: int, device: str) -> dict[str, dict]:
    policies = {}
    for method in METHODS:
        path = checkpoint(stage, train_seed, method)
        if not path.is_file():
            raise FileNotFoundError(path)
        policy = v45.load_method_policy(method, path, device)
        if method == "centroid_gated_sac":
            policy["macro_policy"] = zero_macro_policy
        policies[method] = policy

    reference = v45.scenario_signature(policies[METHODS[0]]["args"])
    for method in METHODS[1:]:
        if v45.scenario_signature(policies[method]["args"]) != reference:
            raise RuntimeError(f"Scenario mismatch: {method}")
    return policies


def forced_sca_micro_policy(args):
    """Execute the complete checkpoint with an exact zero residual (g=0)."""
    base_policy = v45.sca_micro_policy(v45.solve_sca_micro_action, args)

    def _policy(env, eval_mode):
        _policy.residual_gate_values.append(0.0)
        return base_policy(env, eval_mode)

    _policy.residual_gate_values = []
    return _policy


def component_policies(policies: dict[str, dict]) -> dict[str, dict]:
    complete = policies[MAIN_METHOD]
    centroid = dict(complete)
    centroid.update(
        name="complete_checkpoint_centroid_macro",
        macro_policy=zero_macro_policy,
    )
    forced_sca = dict(complete)
    forced_sca.update(
        name="complete_checkpoint_forced_sca",
        micro_policy=forced_sca_micro_policy(complete["args"]),
        micro_solver="sca",
    )
    centroid_forced_sca = dict(complete)
    centroid_forced_sca.update(
        name="complete_checkpoint_centroid_macro_forced_sca",
        macro_policy=zero_macro_policy,
        micro_policy=forced_sca_micro_policy(complete["args"]),
        micro_solver="sca",
    )
    return {
        centroid["name"]: centroid,
        forced_sca["name"]: forced_sca,
        centroid_forced_sca["name"]: centroid_forced_sca,
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_difference(
    rows: list[dict],
    left: str,
    right: str,
    field: str,
) -> dict:
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
        **v45.mean_ci95(deltas),
        "left": left,
        "right": right,
        "field": field,
        "per_seed": {str(seed): delta for seed, delta in zip(seeds, deltas)},
    }


def factorial_effects(rows: list[dict], field: str) -> dict:
    """Compute paired 2x2 effects from the complete checkpoint only."""
    cells = {
        "ppo_gate": MAIN_METHOD,
        "ppo_sca": "complete_checkpoint_forced_sca",
        "centroid_gate": "complete_checkpoint_centroid_macro",
        "centroid_sca": "complete_checkpoint_centroid_macro_forced_sca",
    }
    values = {
        cell: {
            int(row["seed"]): float(row[field])
            for row in rows
            if row["method"] == method
        }
        for cell, method in cells.items()
    }
    seeds = sorted(set.intersection(*(set(item) for item in values.values())))
    if not seeds:
        raise ValueError(f"No complete paired 2x2 data for field {field}")

    contrasts = {
        "gate_effect_under_ppo": lambda item: item["ppo_gate"] - item["ppo_sca"],
        "gate_effect_under_centroid": (
            lambda item: item["centroid_gate"] - item["centroid_sca"]
        ),
        "macro_effect_with_gate": (
            lambda item: item["ppo_gate"] - item["centroid_gate"]
        ),
        "macro_effect_with_sca": lambda item: item["ppo_sca"] - item["centroid_sca"],
        "interaction": lambda item: (
            item["ppo_gate"]
            + item["centroid_sca"]
            - item["ppo_sca"]
            - item["centroid_gate"]
        ),
    }
    output = {
        "field": field,
        "cell_methods": cells,
        "paired_seeds": seeds,
    }
    for name, contrast in contrasts.items():
        per_seed = {}
        for seed in seeds:
            item = {cell: cell_values[seed] for cell, cell_values in values.items()}
            per_seed[str(seed)] = float(contrast(item))
        output[name] = {
            **v45.mean_ci95(list(per_seed.values())),
            "per_seed": per_seed,
        }
    return output


def method_summaries(rows: list[dict]) -> dict:
    fields = (
        "task_delay_weighted_ms",
        "queue_total_mbits",
        "soft_objective_j",
        "average_power_w",
        "coop_ratio",
        "u2g_distance_m",
        "residual_gate_mean",
        "residual_gate_std",
    )
    return {
        method: {
            field: v45.mean_ci95(
                [
                    float(row[field])
                    for row in rows
                    if row["method"] == method
                ]
            )
            for field in fields
        }
        for method in METHODS
    }


def component_diagnostic_report(
    formal_rows: list[dict],
    diagnostic_rows: list[dict],
) -> dict:
    rows = formal_rows + diagnostic_rows
    fields = (
        "task_delay_weighted_ms",
        "queue_total_mbits",
        "soft_objective_j",
        "average_power_w",
        "coop_ratio",
        "u2g_distance_m",
        "residual_gate_mean",
    )
    methods = (MAIN_METHOD,) + DIAGNOSTIC_METHODS
    summaries = {
        method: {
            field: v45.mean_ci95(
                [
                    float(row[field])
                    for row in rows
                    if row["method"] == method
                ]
            )
            for field in fields
        }
        for method in methods
    }
    factorial_fields = (
        "task_delay_weighted_ms",
        "queue_total_mbits",
        "soft_objective_j",
        "average_power_w",
        "coop_ratio",
    )
    return {
        "scope": (
            "Read-only counterfactual replays of the final complete-method "
            "checkpoint; no retraining and no effect on the screening verdict."
        ),
        "method_summaries": summaries,
        "complete_minus_same_checkpoint_centroid_macro_delay_ms": paired_difference(
            rows,
            MAIN_METHOD,
            "complete_checkpoint_centroid_macro",
            "task_delay_weighted_ms",
        ),
        "complete_minus_same_macro_forced_sca_delay_ms": paired_difference(
            rows,
            MAIN_METHOD,
            "complete_checkpoint_forced_sca",
            "task_delay_weighted_ms",
        ),
        "centroid_gate_minus_centroid_sca_delay_ms": paired_difference(
            rows,
            "complete_checkpoint_centroid_macro",
            "complete_checkpoint_centroid_macro_forced_sca",
            "task_delay_weighted_ms",
        ),
        "ppo_sca_minus_centroid_sca_delay_ms": paired_difference(
            rows,
            "complete_checkpoint_forced_sca",
            "complete_checkpoint_centroid_macro_forced_sca",
            "task_delay_weighted_ms",
        ),
        "factorial_2x2": {
            field: factorial_effects(rows, field) for field in factorial_fields
        },
    }


def finite_values(rows: list[dict], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def training_health(stage: str, train_seed: int) -> dict:
    path = (
        ROOT
        / "results"
        / RESULTS_VERSION
        / stage
        / f"seed{train_seed}"
        / MAIN_METHOD
        / TRAIN_FILES[MAIN_METHOD]
    )
    rows = read_csv(path)
    final_queue = finite_values(rows, "micro_q_total_final")
    overflow = finite_values(rows, "micro_overflow_cost_mean")
    previous = final_queue[-200:-100]
    terminal = final_queue[-100:]
    previous_mean = float(np.mean(previous)) if previous.size else None
    terminal_mean = float(np.mean(terminal)) if terminal.size else None
    queue_ratio = (
        terminal_mean / previous_mean
        if previous_mean is not None
        and terminal_mean is not None
        and previous_mean > 0.0
        else None
    )
    queue_stable = bool(
        previous.size == 100
        and terminal.size == 100
        and queue_ratio is not None
        and np.isfinite(queue_ratio)
        and queue_ratio <= 1.10
        and (not overflow.size or float(np.max(overflow)) <= 1e-9)
    )
    ppo_losses = finite_values(rows, "ppo_actor_loss")
    sac_actor_updates = finite_values(rows, "sac_actor_updates_total")
    sac_critic_updates = finite_values(rows, "sac_critic_updates_total")
    parameters_updated = bool(
        ppo_losses.size > 0
        and sac_actor_updates.size > 0
        and sac_critic_updates.size > 0
        and float(sac_actor_updates[-1]) > 0
        and float(sac_critic_updates[-1]) > 0
    )
    return {
        "train_csv": str(path),
        "episode_count": len(rows),
        "queue_previous_100_mean_mbits": previous_mean,
        "queue_terminal_100_mean_mbits": terminal_mean,
        "queue_terminal_to_previous_ratio": queue_ratio,
        "overflow_cost_max": float(np.max(overflow)) if overflow.size else 0.0,
        "queue_stable": queue_stable,
        "ppo_update_rows": int(ppo_losses.size),
        "sac_actor_updates_final": (
            float(sac_actor_updates[-1]) if sac_actor_updates.size else 0.0
        ),
        "sac_critic_updates_final": (
            float(sac_critic_updates[-1]) if sac_critic_updates.size else 0.0
        ),
        "parameters_updated": parameters_updated,
    }


def build_report(rows: list[dict], stage: str, train_seed: int) -> dict:
    summaries = method_summaries(rows)
    comparisons = {
        baseline: paired_difference(
            rows,
            MAIN_METHOD,
            baseline,
            "task_delay_weighted_ms",
        )
        for baseline in METHODS
        if baseline != MAIN_METHOD
    }
    criteria = {
        "better_than_ppo_sca": bool(
            comparisons["cppo_sca"]["ci95_high"] < 0.0
        ),
        "macro_gain_over_centroid_gt_3ms": bool(
            comparisons["centroid_gated_sac"]["ci95_high"] < -3.0
        ),
        "better_than_original_residual_sac": bool(
            comparisons["joint_ppo_sac"]["ci95_high"] < 0.0
        ),
    }
    health = training_health(stage, train_seed)
    criteria["queue_stable"] = health["queue_stable"]
    criteria["parameters_updated"] = health["parameters_updated"]
    if all(criteria.values()):
        verdict = "PASS_PENDING_TRAJECTORY_REVIEW"
    elif any(criteria.values()):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    ranking = sorted(
        METHODS,
        key=lambda method: summaries[method]["task_delay_weighted_ms"]["mean"],
    )
    return {
        "results_version": RESULTS_VERSION,
        "stage": stage,
        "train_seed": train_seed,
        "evaluation_seed_count": len(rows) // len(METHODS),
        "statistics_scope": (
            "One training seed with paired environment seeds; screening only."
        ),
        "method_summaries": summaries,
        "complete_minus_baseline_delay_ms": comparisons,
        "delay_ranking": ranking,
        "training_health": health,
        "criteria": criteria,
        "verdict": verdict,
    }


def configure_plotting() -> None:
    v45.configure_plotting()
    plt.rcParams.update({"axes.titleweight": "bold"})


def plot_method_summary(report: dict, output: Path) -> None:
    configure_plotting()
    x = np.arange(len(METHODS))
    panels = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 时延"),
        ("queue_total_mbits", "平均队列积压 / Mbit", "(b) 队列"),
        ("soft_objective_j", "软目标 J（越小越好）", "(c) 综合目标"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    for axis, (field, ylabel, title) in zip(axes, panels):
        stats = [report["method_summaries"][m][field] for m in METHODS]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=[COLORS[m] for m in METHODS],
            width=0.68,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3)
        axis.set_xticks(x, [LABELS[m] for m in METHODS], rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.24)
    fig.suptitle("v51 四种方法独立训练后的统一评估")
    fig.tight_layout()
    fig.savefig(output / "01_方法时延队列目标对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def curve(stage: str, train_seed: int, method: str, field: str):
    base = ROOT / "results" / RESULTS_VERSION / stage / f"seed{train_seed}"
    x_parts = []
    y_parts = []
    if method in SAC_METHODS:
        pretrain_path = base / f"micro_pretrain_{method}" / "eval_central_macro.csv"
        pretrain = read_csv(pretrain_path)
        x_parts.extend(int(float(row["episode"])) for row in pretrain)
        y_parts.extend(float(row[field]) for row in pretrain)
    joint = read_csv(base / method / EVAL_FILES[method])
    x_parts.extend(200 + int(float(row["episode"])) for row in joint)
    y_parts.extend(float(row[field]) for row in joint)
    return np.asarray(x_parts), np.asarray(y_parts)


def plot_convergence(stage: str, train_seed: int, output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2))
    for method in METHODS:
        x, y = curve(stage, train_seed, method, "task_delay_weighted_ms")
        axes[0].plot(x, y, marker="o", markersize=3, linewidth=2, color=COLORS[method], label=LABELS[method])
    for method in ("centroid_gated_sac", "joint_ppo_gated_sac"):
        x, y = curve(stage, train_seed, method, "residual_gate_mean")
        axes[1].plot(x, y, marker="o", markersize=3, linewidth=2, color=COLORS[method], label=LABELS[method])
    for axis in axes:
        axis.axvline(200, color="#333333", linestyle="--", linewidth=1.1)
        axis.grid(linestyle="--", alpha=0.24)
        axis.set_xlabel("累计训练轮次")
        axis.legend(fontsize=9)
    axes[0].set_ylabel("固定评估端到端时延 / ms")
    axes[0].set_title("(a) 真实评估收敛曲线")
    axes[1].set_ylabel("微观残差门控均值 g")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("(b) Actor 自主门控演化")
    fig.suptitle("前200轮训练微观层，后800轮上下层同步训练")
    fig.tight_layout()
    fig.savefig(output / "02_完整训练与门控曲线.png", dpi=300, facecolor="white")
    plt.close(fig)


def trajectory_signature(trace: dict) -> str:
    return hashlib.sha256(np.asarray(trace["uavs"], dtype=np.float32).tobytes()).hexdigest()


def plot_trajectories(policies: dict, seed: int, output: Path) -> dict:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.4), sharex=True, sharey=True)
    zone_colors = ("#2878B5", "#E67E22", "#2E8B57")
    signatures = {}
    for axis, method in zip(axes.flat, METHODS):
        result = v45.evaluate_one(policies[method], seed, record_trajectory=True)
        trace = result.pop("trajectory")
        signatures[method] = trajectory_signature(trace)
        users = trace["users"]
        allocations = np.asarray(trace["user_allocations"], dtype=np.int64)
        for user in range(users.shape[1]):
            color = zone_colors[int(allocations[user])]
            axis.plot(users[:, user, 0], users[:, user, 1], color=color, alpha=0.25, linewidth=0.7)
        for uav, color in enumerate(zone_colors):
            axis.plot(trace["centroids"][:, uav, 0], trace["centroids"][:, uav, 1], linestyle="--", color=color, linewidth=1.2)
            axis.plot(trace["uavs"][:, uav, 0], trace["uavs"][:, uav, 1], color=color, linewidth=2.5)
            axis.scatter(*trace["uavs"][0, uav], marker="s", s=34, facecolor="white", edgecolor=color, zorder=4)
            axis.scatter(*trace["uavs"][-1, uav], marker="*", s=90, facecolor=color, edgecolor="#222222", zorder=5)
        axis.set_title(f"{LABELS[method]}\n时延 {result['task_delay_weighted_ms']:.1f} ms")
        axis.set_xlabel("X坐标 / m")
        axis.set_ylabel("Y坐标 / m")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(linestyle="--", alpha=0.2)
    fig.suptitle("最终策略轨迹：细线为用户，虚线为业务质心，粗线为UAV")
    fig.tight_layout()
    fig.savefig(output / "03_四种方法轨迹对比.png", dpi=300, facecolor="white")
    plt.close(fig)
    return {"eval_seed": seed, "signatures": signatures}


def plot_queue_health(stage: str, train_seed: int, output: Path) -> None:
    configure_plotting()
    path = ROOT / "results" / RESULTS_VERSION / stage / f"seed{train_seed}" / MAIN_METHOD / TRAIN_FILES[MAIN_METHOD]
    rows = read_csv(path)
    queue = finite_values(rows, "micro_q_total_final")
    if not queue.size:
        return
    window = 50
    smooth = np.asarray([np.mean(queue[max(0, i - window + 1) : i + 1]) for i in range(queue.size)])
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.plot(np.arange(1, queue.size + 1), queue, color="#AFC8E6", linewidth=0.7, label="逐轮最终队列")
    axis.plot(np.arange(1, queue.size + 1), smooth, color=COLORS[MAIN_METHOD], linewidth=2.2, label="50轮滑动平均")
    axis.set_xlabel("联合训练轮次")
    axis.set_ylabel("每轮末总队列 / Mbit")
    axis.set_title("完整方法的队列稳定性")
    axis.grid(linestyle="--", alpha=0.24)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "04_队列稳定性.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_factorial_diagnostics(report: dict, output: Path) -> None:
    diagnostics = report["component_diagnostics"]
    factor = diagnostics["factorial_2x2"]["task_delay_weighted_ms"]
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    cells = (
        (MAIN_METHOD, "PPO + 门控"),
        ("complete_checkpoint_forced_sca", "PPO + 纯 SCA"),
        ("complete_checkpoint_centroid_macro", "质心 + 门控"),
        (
            "complete_checkpoint_centroid_macro_forced_sca",
            "质心 + 纯 SCA",
        ),
    )
    means = [
        diagnostics["method_summaries"][method]["task_delay_weighted_ms"]["mean"]
        for method, _ in cells
    ]
    errors = [
        diagnostics["method_summaries"][method]["task_delay_weighted_ms"][
            "ci95_high"
        ]
        - mean
        for (method, _), mean in zip(cells, means)
    ]
    x = np.arange(len(cells))
    bars = axes[0].bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=("#2878B5", "#A9C7E3", "#2E8B57", "#9CC8B1"),
        width=0.68,
    )
    axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3)
    axes[0].set_xticks(x, [label for _, label in cells], rotation=16, ha="right")
    axes[0].set_ylabel("优先级加权端到端时延 / ms")
    axes[0].set_title("(a) 同一 checkpoint 四格重放")
    axes[0].grid(axis="y", linestyle="--", alpha=0.24)

    contrasts = (
        ("gate_effect_under_ppo", "门控效应\n(PPO 下)"),
        ("gate_effect_under_centroid", "门控效应\n(质心下)"),
        ("macro_effect_with_gate", "宏观效应\n(门控下)"),
        ("macro_effect_with_sca", "宏观效应\n(SCA 下)"),
        ("interaction", "交互项"),
    )
    y = np.arange(len(contrasts))
    values = np.asarray([factor[key]["mean"] for key, _ in contrasts])
    low = np.asarray([factor[key]["ci95_low"] for key, _ in contrasts])
    high = np.asarray([factor[key]["ci95_high"] for key, _ in contrasts])
    axes[1].errorbar(
        values,
        y,
        xerr=np.vstack((values - low, high - values)),
        fmt="o",
        color="#333333",
        ecolor="#5B7FA3",
        capsize=4,
        markersize=6,
    )
    axes[1].axvline(0.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[1].set_yticks(y, [label for _, label in contrasts])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("配对时延差 / ms（负数更好）")
    axes[1].set_title("(b) 主效应与交互项（95% CI）")
    axes[1].grid(axis="x", linestyle="--", alpha=0.24)
    fig.suptitle("v51 宏观轨迹与微观门控的同 checkpoint 2×2 分解")
    fig.tight_layout()
    fig.savefig(output / "05_同checkpoint四格贡献分解.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_verdict(report: dict, output: Path) -> None:
    lines = [
        "# v51 seed-1 筛查判决",
        "",
        f"**判决：{report['verdict']}**",
        "",
        "> 统计范围：1 个训练种子、20 个共同环境评估种子；不能替代多训练种子结论。",
        "",
        "## 方法排名",
        "",
        "| 排名 | 方法 | 时延/ms | 队列/Mbit | 软目标J | 门控均值 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, method in enumerate(report["delay_ranking"], start=1):
        stats = report["method_summaries"][method]
        lines.append(
            f"| {rank} | {LABELS[method]} | {stats['task_delay_weighted_ms']['mean']:.2f} | "
            f"{stats['queue_total_mbits']['mean']:.2f} | {stats['soft_objective_j']['mean']:.4f} | "
            f"{stats['residual_gate_mean']['mean']:.3f} |"
        )
    lines.extend(["", "## 配对差（完整方法减基线，负数更好）", ""])
    for baseline, stats in report["complete_minus_baseline_delay_ms"].items():
        lines.append(
            f"- 对 {LABELS[baseline]}：{stats['mean']:+.2f} ms，95% CI "
            f"[{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}]。"
        )
    diagnostics = report.get("component_diagnostics")
    if diagnostics:
        lines.extend(["", "## 同一完整 checkpoint 的只读组件重放", ""])
        macro = diagnostics[
            "complete_minus_same_checkpoint_centroid_macro_delay_ms"
        ]
        micro = diagnostics["complete_minus_same_macro_forced_sca_delay_ms"]
        lines.append(
            "- 宏观轨迹效应（完整方法 − 同微观动态质心）："
            f"{macro['mean']:+.2f} ms，95% CI "
            f"[{macro['ci95_low']:+.2f}, {macro['ci95_high']:+.2f}]。"
        )
        lines.append(
            "- 微观门控效应（完整方法 − 同宏观强制纯 SCA）："
            f"{micro['mean']:+.2f} ms，95% CI "
            f"[{micro['ci95_low']:+.2f}, {micro['ci95_high']:+.2f}]。"
        )
        fourth = diagnostics["method_summaries"][
            "complete_checkpoint_centroid_macro_forced_sca"
        ]["task_delay_weighted_ms"]
        factorial = diagnostics["factorial_2x2"]["task_delay_weighted_ms"]
        interaction = factorial["interaction"]
        lines.append(
            "- 第四格（同 checkpoint 动态质心 + 强制纯 SCA）："
            f"{fourth['mean']:.2f} ms，95% CI "
            f"[{fourth['ci95_low']:.2f}, {fourth['ci95_high']:.2f}]。"
        )
        lines.append(
            "- 四格交互项 `A + D - B - C`："
            f"{interaction['mean']:+.2f} ms，95% CI "
            f"[{interaction['ci95_low']:+.2f}, {interaction['ci95_high']:+.2f}]。"
        )
    lines.extend(["", "## 预注册判据", ""])
    for name, passed in report["criteria"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "轨迹方向需结合 `03_四种方法轨迹对比.png` 人工核对；程序不以距离或图形美观替代性能判据。",
        ]
    )
    (output / "gate_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--trajectory-seed", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Reuse per_seed_raw.csv and run only the frozen component replays.",
    )
    args = parser.parse_args()

    seeds = v45.parse_seed_spec(args.eval_seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = load_policies(args.stage, args.train_seed, args.device)
    if args.diagnostics_only:
        rows = read_csv(output / "per_seed_raw.csv")
    else:
        rows = []
        for seed in seeds:
            for method in METHODS:
                row = v45.evaluate_one(policies[method], seed)
                rows.append(row)
                print(
                    f"eval={seed} {method}: delay={row['task_delay_weighted_ms']:.2f}ms "
                    f"queue={row['queue_total_mbits']:.2f}Mbit gate={row['residual_gate_mean']:.3f}",
                    flush=True,
                )
        write_csv(output / "per_seed_raw.csv", rows)

    diagnostics = component_policies(policies)
    diagnostic_path = output / "component_diagnostics.csv"
    diagnostic_rows = []
    if args.diagnostics_only and diagnostic_path.is_file():
        requested_seeds = set(seeds)
        diagnostic_rows = [
            row
            for row in read_csv(diagnostic_path)
            if int(row["seed"]) in requested_seeds
            and row["method"] in DIAGNOSTIC_METHODS
        ]
    existing = {
        (int(row["seed"]), row["method"]) for row in diagnostic_rows
    }
    for seed in seeds:
        for method in DIAGNOSTIC_METHODS:
            if (seed, method) in existing:
                continue
            row = v45.evaluate_one(diagnostics[method], seed)
            diagnostic_rows.append(row)
            print(
                f"diagnostic={seed} {method}: "
                f"delay={row['task_delay_weighted_ms']:.2f}ms "
                f"queue={row['queue_total_mbits']:.2f}Mbit",
                flush=True,
            )
    method_order = {method: index for index, method in enumerate(DIAGNOSTIC_METHODS)}
    diagnostic_rows.sort(
        key=lambda row: (int(row["seed"]), method_order[row["method"]])
    )
    write_csv(diagnostic_path, diagnostic_rows)
    report = build_report(rows, args.stage, args.train_seed)
    report["component_diagnostics"] = component_diagnostic_report(
        rows,
        diagnostic_rows,
    )
    trajectory = plot_trajectories(policies, args.trajectory_seed, output)
    report["trajectory_audit"] = trajectory
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_verdict(report, output)
    plot_method_summary(report, output)
    plot_convergence(args.stage, args.train_seed, output)
    plot_queue_health(args.stage, args.train_seed, output)
    plot_factorial_diagnostics(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
