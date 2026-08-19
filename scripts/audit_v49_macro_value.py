"""Paired macro-trajectory value audit for the v49 residual HRL policy.

The micro policy and every environment setting are held fixed.  Only the macro
policy changes between the learned trajectory, dynamic-centroid tracking, and
holding the episode-start deployment.  This isolates the value of learned UAV
motion from lower-level routing quality.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v48_priority_gated as v48  # noqa: E402
import evaluate_v49_support_residual as v49  # noqa: E402
from compare_sca import solve_sca_micro_action  # noqa: E402
from train_centralized_macro import fixed_macro_policy  # noqa: E402
from train_centralized_macro import sca_micro_policy  # noqa: E402
from train_centralized_macro import zero_macro_policy  # noqa: E402


DEFAULT_OUTPUT = ROOT / "result_archive" / "v49_macro_value_audit_seed1"
MACRO_VARIANTS = ("learned", "centroid", "fixed")
VARIANT_ORDER = MACRO_VARIANTS + ("learned_sca", "centroid_sca")
VARIANT_LABELS = {
    "learned": "学习轨迹",
    "centroid": "动态质心跟踪",
    "fixed": "固定初始位置",
    "learned_sca": "学习轨迹+纯SCA",
    "centroid_sca": "质心轨迹+纯SCA",
}


def trajectory_metrics(trace: dict) -> dict[str, float]:
    uavs = np.asarray(trace["uavs"], dtype=np.float64)
    centroids = np.asarray(trace["centroids"], dtype=np.float64)
    steps = min(uavs.shape[0], centroids.shape[0])
    uavs = uavs[:steps]
    centroids = centroids[:steps]
    segment_lengths = np.linalg.norm(np.diff(uavs, axis=0), axis=-1)
    centroid_distances = np.linalg.norm(uavs - centroids, axis=-1)
    return {
        "uav_path_length_mean_m": float(np.mean(np.sum(segment_lengths, axis=0))),
        "uav_net_displacement_mean_m": float(
            np.mean(np.linalg.norm(uavs[-1] - uavs[0], axis=-1))
        ),
        "centroid_distance_mean_m": float(np.mean(centroid_distances)),
        "centroid_distance_p95_m": float(np.percentile(centroid_distances, 95)),
    }


def evaluate_variant(policy: dict, seed: int, *, trajectory: bool = True) -> tuple[dict, dict]:
    row, _, _, trace = v48.evaluate_detailed(policy, seed, trajectory=trajectory)
    if trace is None:
        raise RuntimeError("Trajectory recording was requested but no trace was returned.")
    return {**row, **trajectory_metrics(trace)}, trace


def paired_summary(rows: list[dict]) -> dict:
    fields = tuple(
        key for key in rows[0] if key not in {"seed", "variant"}
    )
    by_variant = {
        variant: {
            int(row["seed"]): row
            for row in rows
            if row["variant"] == variant
        }
        for variant in VARIANT_ORDER
    }
    seeds = sorted(set.intersection(*(set(values) for values in by_variant.values())))
    summaries = {
        variant: {
            field: v45.mean_ci95(
                [float(by_variant[variant][seed][field]) for seed in seeds]
            )
            for field in fields
        }
        for variant in VARIANT_ORDER
    }
    paired_learned_minus_baseline = {
        baseline: {
            field: v45.mean_ci95(
                [
                    float(by_variant["learned"][seed][field])
                    - float(by_variant[baseline][seed][field])
                    for seed in seeds
                ]
            )
            for field in fields
        }
        for baseline in ("centroid", "fixed")
    }
    paired_residual_minus_sca = {
        "learned_macro": {
            field: v45.mean_ci95(
                [
                    float(by_variant["learned"][seed][field])
                    - float(by_variant["learned_sca"][seed][field])
                    for seed in seeds
                ]
            )
            for field in fields
        },
        "centroid_macro": {
            field: v45.mean_ci95(
                [
                    float(by_variant["centroid"][seed][field])
                    - float(by_variant["centroid_sca"][seed][field])
                    for seed in seeds
                ]
            )
            for field in fields
        },
    }
    interaction = {
        field: v45.mean_ci95(
            [
                (
                    float(by_variant["learned"][seed][field])
                    - float(by_variant["centroid"][seed][field])
                )
                - (
                    float(by_variant["learned_sca"][seed][field])
                    - float(by_variant["centroid_sca"][seed][field])
                )
                for seed in seeds
            ]
        )
        for field in fields
    }
    delay_vs_centroid = paired_learned_minus_baseline["centroid"][
        "task_delay_weighted_ms"
    ]
    objective_vs_centroid = paired_learned_minus_baseline["centroid"][
        "soft_objective_j"
    ]
    delay_vs_fixed = paired_learned_minus_baseline["fixed"][
        "task_delay_weighted_ms"
    ]
    verdict = {
        "learned_better_than_centroid_delay": bool(delay_vs_centroid["ci95_high"] < 0.0),
        "learned_better_than_centroid_objective": bool(
            objective_vs_centroid["ci95_high"] < 0.0
        ),
        "learned_better_than_fixed_delay": bool(delay_vs_fixed["ci95_high"] < 0.0),
        "residual_better_than_sca_with_learned_macro": bool(
            paired_residual_minus_sca["learned_macro"][
                "task_delay_weighted_ms"
            ]["ci95_high"]
            < 0.0
        ),
    }
    verdict["macro_value_gate_passed"] = bool(
        verdict["learned_better_than_centroid_delay"]
        and verdict["learned_better_than_centroid_objective"]
        and verdict["learned_better_than_fixed_delay"]
    )
    verdict["micro_value_gate_passed"] = bool(
        verdict["residual_better_than_sca_with_learned_macro"]
    )
    verdict["full_hierarchy_gate_passed"] = bool(
        verdict["macro_value_gate_passed"]
        and verdict["micro_value_gate_passed"]
    )
    return {
        "evaluation_seeds": seeds,
        "summaries": summaries,
        "paired_learned_minus_baseline": paired_learned_minus_baseline,
        "paired_residual_minus_sca": paired_residual_minus_sca,
        "factorial_interaction": interaction,
        "difference_definition": "learned minus baseline; negative means learned is better",
        "verdict": verdict,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_performance(rows: list[dict], output: Path) -> None:
    v45.configure_plotting()
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 时延"),
        ("queue_total_mbits", "平均队列积压 / Mbit", "(b) 队列"),
        ("soft_objective_j", "软目标 J", "(c) 综合目标"),
        ("centroid_distance_mean_m", "UAV-质心平均距离 / m", "(d) 位置偏离"),
    )
    colors = ("#2878B5", "#E67E22", "#7F7F7F")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    x = np.arange(len(MACRO_VARIANTS))
    for axis, (field, ylabel, title) in zip(axes.flat, fields):
        values_by_variant = [
            np.asarray(
                [float(row[field]) for row in rows if row["variant"] == variant],
                dtype=np.float64,
            )
            for variant in MACRO_VARIANTS
        ]
        means = [float(np.mean(values)) for values in values_by_variant]
        cis = [v45.mean_ci95(values) for values in values_by_variant]
        lower = [mean - ci["ci95_low"] for mean, ci in zip(means, cis)]
        upper = [ci["ci95_high"] - mean for mean, ci in zip(means, cis)]
        axis.bar(x, means, color=colors, width=0.62)
        axis.errorbar(x, means, yerr=[lower, upper], fmt="none", color="#222222", capsize=4)
        for index, values in enumerate(values_by_variant):
            jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
            axis.scatter(
                index + jitter,
                values,
                s=22,
                facecolors="white",
                edgecolors=colors[index],
                zorder=3,
            )
        axis.set_xticks(x, [VARIANT_LABELS[name] for name in MACRO_VARIANTS])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.24)
    fig.suptitle("同一微观策略下的宏观轨迹净收益")
    fig.tight_layout()
    fig.savefig(output / "01_轨迹策略性能对照.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectories(traces: dict[str, dict], delays: dict[str, float], output: Path) -> None:
    v45.configure_plotting()
    colors = ("#2878B5", "#E67E22", "#2E8B57")
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.8), sharex=True, sharey=True)
    all_points = []
    for axis, variant in zip(axes, MACRO_VARIANTS):
        trace = traces[variant]
        users = np.asarray(trace["users"])
        allocations = np.asarray(trace["user_allocations"], dtype=np.int64)
        for user in range(users.shape[1]):
            zone = int(allocations[user])
            axis.plot(
                users[:, user, 0],
                users[:, user, 1],
                color=colors[zone],
                alpha=0.12,
                linewidth=0.55,
            )
        for index, color in enumerate(colors):
            axis.plot(
                trace["centroids"][:, index, 0],
                trace["centroids"][:, index, 1],
                linestyle="--",
                color=color,
                linewidth=1.3,
                alpha=0.85,
            )
            axis.plot(
                trace["uavs"][:, index, 0],
                trace["uavs"][:, index, 1],
                color=color,
                linewidth=2.6,
                label=f"UAV {index + 1}",
            )
            axis.scatter(
                *trace["uavs"][0, index],
                marker="s",
                s=38,
                facecolor="white",
                edgecolor=color,
                zorder=4,
            )
            axis.scatter(
                *trace["uavs"][-1, index],
                marker="*",
                s=105,
                facecolor=color,
                edgecolor="#222222",
                zorder=5,
            )
        all_points.extend((users.reshape(-1, 2), trace["uavs"].reshape(-1, 2)))
        axis.set_title(f"{VARIANT_LABELS[variant]}\n{delays[variant]:.1f} ms")
        axis.set_xlabel("X坐标 / m")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(linestyle="--", alpha=0.2)
    combined = np.concatenate(all_points, axis=0)
    padding = 25.0
    axes[0].set_xlim(float(np.min(combined[:, 0]) - padding), float(np.max(combined[:, 0]) + padding))
    axes[0].set_ylim(float(np.min(combined[:, 1]) - padding), float(np.max(combined[:, 1]) + padding))
    axes[0].set_ylabel("Y坐标 / m")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("共同随机种子下的用户、质心与 UAV 轨迹")
    fig.tight_layout()
    fig.savefig(output / "02_三种宏观轨迹对照.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_factorial(rows: list[dict], output: Path) -> None:
    """Show macro and micro contributions without conflating the two layers."""
    v45.configure_plotting()
    groups = (
        ("centroid_sca", "centroid"),
        ("learned_sca", "learned"),
    )
    group_labels = ("动态质心轨迹", "学习轨迹")
    micro_labels = ("纯 SCA", "支撑集残差 SAC")
    micro_colors = ("#8C8C8C", "#2878B5")
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 时延"),
        ("soft_objective_j", "软目标 J", "(b) 综合目标"),
    )
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for axis, (field, ylabel, title) in zip(axes, fields):
        for micro_index in range(2):
            variants = [group[micro_index] for group in groups]
            values = [
                np.asarray(
                    [float(row[field]) for row in rows if row["variant"] == variant],
                    dtype=np.float64,
                )
                for variant in variants
            ]
            stats = [v45.mean_ci95(item) for item in values]
            means = np.asarray([item["mean"] for item in stats])
            lower = np.asarray([item["mean"] - item["ci95_low"] for item in stats])
            upper = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
            positions = x + (micro_index - 0.5) * width
            axis.bar(
                positions,
                means,
                width=width,
                color=micro_colors[micro_index],
                label=micro_labels[micro_index],
            )
            axis.errorbar(
                positions,
                means,
                yerr=[lower, upper],
                fmt="none",
                color="#222222",
                capsize=4,
            )
        axis.set_xticks(x, group_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.24)
        axis.legend()
    fig.suptitle("宏观轨迹与微观路由的 2×2 控制变量分解")
    fig.tight_layout()
    fig.savefig(output / "03_上下层贡献分解.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_verdict(path: Path, report: dict) -> None:
    verdict = report["verdict"]
    paired = report["paired_learned_minus_baseline"]
    lines = [
        "# v49 宏观轨迹价值判决",
        "",
        f"- 上层轨迹门禁：{'通过' if verdict['macro_value_gate_passed'] else '未通过'}",
        f"- 下层残差门禁：{'通过' if verdict['micro_value_gate_passed'] else '未通过'}",
        f"- 完整分层门禁：{'通过' if verdict['full_hierarchy_gate_passed'] else '未通过'}",
        "- 差值口径：学习轨迹减基线；负值表示学习轨迹更优。",
    ]
    for baseline in ("centroid", "fixed"):
        delay = paired[baseline]["task_delay_weighted_ms"]
        objective = paired[baseline]["soft_objective_j"]
        lines.extend(
            [
                "",
                f"## 相对{VARIANT_LABELS[baseline]}",
                (
                    f"- 时延差：{delay['mean']:.3f} ms，95% CI "
                    f"[{delay['ci95_low']:.3f}, {delay['ci95_high']:.3f}]。"
                ),
                (
                    f"- J 差：{objective['mean']:.6f}，95% CI "
                    f"[{objective['ci95_low']:.6f}, {objective['ci95_high']:.6f}]。"
                ),
            ]
        )
    residual = report["paired_residual_minus_sca"]["learned_macro"]
    delay = residual["task_delay_weighted_ms"]
    objective = residual["soft_objective_j"]
    lines.extend(
        [
            "",
            "## 相对纯 SCA 下层",
            (
                f"- 时延差：{delay['mean']:.3f} ms，95% CI "
                f"[{delay['ci95_low']:.3f}, {delay['ci95_high']:.3f}]。"
            ),
            (
                f"- J 差：{objective['mean']:.6f}，95% CI "
                f"[{objective['ci95_low']:.6f}, {objective['ci95_high']:.6f}]。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("formal", "diagnostic", "smoke"),
        default="formal",
    )
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--trajectory-seed", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = v49.checkpoint(args.stage, args.train_seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    learned = v45.load_method_policy("learned", checkpoint, args.device)
    sca_args = argparse.Namespace(**vars(learned["args"]))
    sca_args.sca_update_interval = int(
        getattr(sca_args, "micro_sca_residual_update_interval", 1)
    )
    sca_args.sca_max_iter = int(getattr(sca_args, "micro_sca_residual_max_iter", 3))
    sca_args.sca_tol = float(getattr(sca_args, "micro_sca_residual_tol", 1e-3))
    sca_args.sca_solver = str(
        getattr(sca_args, "micro_sca_residual_solver", "CLARABEL")
    )
    sca_args.sca_delay_weight = float(getattr(sca_args, "delay_objective_weight", 0.9))
    sca_args.sca_energy_weight = float(getattr(sca_args, "energy_objective_weight", 0.1))
    sca_args.sca_proximal_weight = float(
        getattr(sca_args, "micro_sca_residual_proximal_weight", 1e-3)
    )
    pure_sca_policy = sca_micro_policy(solve_sca_micro_action, sca_args)
    policies = {
        "learned": learned,
        "centroid": {**learned, "name": "centroid", "macro_policy": zero_macro_policy},
        "fixed": {**learned, "name": "fixed", "macro_policy": fixed_macro_policy},
        "learned_sca": {
            **learned,
            "name": "learned_sca",
            "micro_policy": pure_sca_policy,
            "micro_solver": "sca",
        },
        "centroid_sca": {
            **learned,
            "name": "centroid_sca",
            "macro_policy": zero_macro_policy,
            "micro_policy": pure_sca_policy,
            "micro_solver": "sca",
        },
    }
    seeds = v45.parse_seed_spec(args.eval_seeds)
    rows: list[dict] = []
    traces: dict[str, dict] = {}
    trajectory_delays: dict[str, float] = {}
    for seed in seeds:
        for variant in VARIANT_ORDER:
            row, trace = evaluate_variant(policies[variant], seed)
            rows.append({"variant": variant, **row})
            if seed == int(args.trajectory_seed):
                traces[variant] = trace
                trajectory_delays[variant] = float(row["task_delay_weighted_ms"])
            print(
                f"seed={seed} {variant}: delay={row['task_delay_weighted_ms']:.2f}ms "
                f"queue={row['queue_total_mbits']:.2f}Mbit "
                f"centroid={row['centroid_distance_mean_m']:.1f}m",
                flush=True,
            )
    if not set(MACRO_VARIANTS).issubset(traces):
        raise ValueError("trajectory-seed must be included in eval-seeds")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_seed_raw.csv", rows)
    report = paired_summary(rows)
    report["checkpoint"] = str(checkpoint)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_verdict(output / "gate_verdict.md", report)
    plot_performance(rows, output)
    plot_trajectories(traces, trajectory_delays, output)
    plot_factorial(rows, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
