"""Audit whether the v51 micro policy needs a fresh action every 0.5 s.

This is a read-only same-checkpoint counterfactual. The physical environment
continues to advance at 0.5 s per resource frame in both arms. Only the
frequency at which the deterministic micro policy is queried changes:

* ``every_frame``: query on every active service frame (current behavior);
* ``hold_two_frames``: query once, then reuse the executed action for one frame.

The macro policy is fixed to dynamic-centroid tracking in both arms so that the
comparison isolates micro action update frequency from learned macro motion.
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
from hrl_uav.load_switching import (  # noqa: E402
    build_load_switch_schedule,
    parse_load_switch_phases,
)
from run_load_switch_gate import run_gate_episode  # noqa: E402
from run_v34_cooperative_mobility_gate import (  # noqa: E402
    queue_metrics,
    soft_objective_metrics,
)
from train_centralized_macro import zero_macro_policy  # noqa: E402


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
DEFAULT_OUTPUT = ROOT / "result_archive" / "v51_micro_decision_frequency_audit"

VARIANTS = {
    "every_frame": 1,
    "hold_two_frames": 2,
}
LABELS = {
    "every_frame": "每0.5 s更新（当前）",
    "hold_two_frames": "每1.0 s更新（动作保持2帧）",
}
COLORS = {
    "every_frame": "#2878B5",
    "hold_two_frames": "#E07B39",
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
    "soft_objective_j",
    "average_power_w",
    "u2g_active_time_s",
    "u2u_active_time_s",
    "coop_ratio",
    "coop_flow_total_mbits",
    "cross_flow_mbits",
    "task_completion_rate",
    "base_action_updates",
)


class ActionHoldPolicy:
    """Reuse an already-executed micro action for a fixed frame count."""

    def __init__(self, base_policy, hold_frames: int):
        hold_frames = int(hold_frames)
        if hold_frames <= 0:
            raise ValueError("hold_frames must be positive")
        self.base_policy = base_policy
        self.hold_frames = hold_frames
        self.frame_count = 0
        self.base_action_updates = 0
        self._cached_action = None

    def __call__(self, env, eval_mode):
        if self._cached_action is None or self.frame_count % self.hold_frames == 0:
            self._cached_action = np.asarray(
                self.base_policy(env, eval_mode), dtype=np.float32
            ).reshape(-1).copy()
            self.base_action_updates += 1
        self.frame_count += 1
        return self._cached_action.copy()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_schedule(args, seed: int):
    phases = parse_load_switch_phases(
        getattr(args, "load_switch_phases", ""),
        int(args.num_agents),
    )
    return build_load_switch_schedule(
        mode=getattr(args, "load_switch_mode", "off"),
        phases_mbps=phases,
        macro_steps=int(args.macro_steps),
        episode_seed=int(seed),
        dwell_min=int(getattr(args, "load_switch_dwell_min", 8)),
        dwell_max=int(getattr(args, "load_switch_dwell_max", 12)),
    )


def evaluate_variant(policy: dict, variant: str, seed: int) -> dict:
    args = policy["args"]
    env = v45.make_eval_env(args, record_trajectory=True)
    held_policy = ActionHoldPolicy(policy["micro_policy"], VARIANTS[variant])
    result, trace = run_gate_episode(
        env,
        args,
        held_policy,
        variant,
        seed,
        macro_steps=int(args.macro_steps),
        load_switch_schedule=make_schedule(args, seed),
        macro_policy=zero_macro_policy,
    )
    result.update(queue_metrics(trace, float(env.env.macro_dt)))
    result.update(
        soft_objective_metrics(
            trace,
            float(getattr(args, "micro_reward_scale", 1.0)),
        )
    )
    arrived = max(int(result["task_arrived_count"]), 1)
    result.update(
        action_hold_frames=int(VARIANTS[variant]),
        base_action_updates=int(held_policy.base_action_updates),
        task_completion_rate=float(result["task_completed_count"] / arrived),
    )
    return result


def paired_summary(rows: list[dict], field: str) -> dict:
    values = {
        variant: {
            int(row["seed"]): float(row[field])
            for row in rows
            if row["policy"] == variant
        }
        for variant in VARIANTS
    }
    seeds = sorted(set(values["every_frame"]) & set(values["hold_two_frames"]))
    deltas = [
        values["hold_two_frames"][seed] - values["every_frame"][seed]
        for seed in seeds
    ]
    return {
        **v45.mean_ci95(deltas),
        "contrast": "hold_two_frames_minus_every_frame",
        "field": field,
        "per_seed": {str(seed): value for seed, value in zip(seeds, deltas)},
    }


def method_summaries(rows: list[dict]) -> dict:
    return {
        variant: {
            field: v45.mean_ci95(
                [
                    float(row[field])
                    for row in rows
                    if row["policy"] == variant
                ]
            )
            for field in SUMMARY_FIELDS
        }
        for variant in VARIANTS
    }


def crn_report(rows: list[dict]) -> dict:
    hashes = {}
    for row in rows:
        hashes.setdefault(int(row["seed"]), {})[str(row["policy"])] = str(
            row["external_trace_hash"]
        )
    mismatches = {
        str(seed): values
        for seed, values in hashes.items()
        if len(values) != len(VARIANTS) or len(set(values.values())) != 1
    }
    return {
        "all_external_trace_hashes_match": not mismatches,
        "mismatches": mismatches,
    }


def build_report(rows: list[dict], checkpoint: Path, seeds: list[int]) -> dict:
    paired = {field: paired_summary(rows, field) for field in SUMMARY_FIELDS}
    delay = paired["task_delay_weighted_ms"]
    if delay["ci95_low"] > 0.0:
        conclusion = "SLOWER_UPDATE_HARMS_DELAY"
    elif delay["ci95_high"] < 0.0:
        conclusion = "SLOWER_UPDATE_IMPROVES_DELAY"
    elif delay["ci95_low"] >= -1.0 and delay["ci95_high"] <= 1.0:
        conclusion = "NO_EVIDENCE_OF_MORE_THAN_1MS_EFFECT"
    else:
        conclusion = "INCONCLUSIVE_WITHIN_20_EVAL_SEEDS"
    return {
        "scope": (
            "Read-only same-checkpoint CRN audit. Both arms keep the physical "
            "resource frame at 0.5 s and use dynamic-centroid macro control."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "evaluation_seeds": seeds,
        "variants": VARIANTS,
        "method_summaries": method_summaries(rows),
        "paired_hold_two_minus_every_frame": paired,
        "crn": crn_report(rows),
        "screening_conclusion": conclusion,
        "interpretation_boundary": (
            "This comparison tests whether slowing action refresh from 0.5 s "
            "to 1.0 s changes performance. It does not directly prove whether "
            "a faster 0.25 s physical frame would help."
        ),
    }


def plot_report(report: dict, output: Path) -> None:
    v45.configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    variants = tuple(VARIANTS)
    x = np.arange(len(variants))

    for axis, field, ylabel, title in (
        (
            axes[0],
            "task_delay_weighted_ms",
            "优先级加权端到端时延 / ms",
            "(a) 端到端时延",
        ),
        (
            axes[1],
            "mean_queue_mbits",
            "平均总队列 / Mbit",
            "(b) 队列积压",
        ),
    ):
        stats = [report["method_summaries"][item][field] for item in variants]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=[COLORS[item] for item in variants],
            width=0.62,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3)
        axis.set_xticks(x, [LABELS[item] for item in variants])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)

    delay = report["paired_hold_two_minus_every_frame"][
        "task_delay_weighted_ms"
    ]
    seeds = [int(seed) for seed in delay["per_seed"]]
    deltas = np.asarray([delay["per_seed"][str(seed)] for seed in seeds])
    axes[2].scatter(
        seeds,
        deltas,
        color="#4B5563",
        s=32,
        label="逐种子配对差",
    )
    axes[2].axhline(0.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[2].axhspan(-1.0, 1.0, color="#DDE8D5", alpha=0.6, label="±1 ms 描述带")
    axes[2].errorbar(
        np.mean(seeds),
        delay["mean"],
        yerr=[[delay["mean"] - delay["ci95_low"]], [delay["ci95_high"] - delay["mean"]]],
        fmt="D",
        color="#E07B39",
        ecolor="#E07B39",
        capsize=5,
        label="均值与95% CI",
    )
    axes[2].set_xlabel("共同评估种子")
    axes[2].set_ylabel("1.0 s更新 − 0.5 s更新 / ms")
    axes[2].set_title("(c) 时延配对差")
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle("v51 微观动作更新频率审计（物理帧均为0.5 s）")
    fig.tight_layout()
    fig.savefig(output / "01_微观动作更新频率审计.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_result(report: dict, output: Path) -> None:
    delay = report["paired_hold_two_minus_every_frame"][
        "task_delay_weighted_ms"
    ]
    queue = report["paired_hold_two_minus_every_frame"]["mean_queue_mbits"]
    completion = report["paired_hold_two_minus_every_frame"][
        "task_completion_rate"
    ]
    lines = [
        "# v51 微观动作更新频率审计",
        "",
        f"**筛查结论：`{report['screening_conclusion']}`**",
        "",
        "两臂的物理资源帧均为 0.5 s，宏观层均使用动态业务质心；唯一差异是微观动作每帧更新，还是连续保持两帧。",
        "",
        "## 配对结果（1.0 s 更新减 0.5 s 更新）",
        "",
        f"- 端到端时延：{delay['mean']:+.3f} ms，95% CI [{delay['ci95_low']:+.3f}, {delay['ci95_high']:+.3f}]。",
        f"- 平均总队列：{queue['mean']:+.3f} Mbit，95% CI [{queue['ci95_low']:+.3f}, {queue['ci95_high']:+.3f}]。",
        f"- 任务完成率：{completion['mean']:+.6f}，95% CI [{completion['ci95_low']:+.6f}, {completion['ci95_high']:+.6f}]。",
        f"- 外生随机轨迹哈希全部一致：{report['crn']['all_external_trace_hashes_match']}。",
        "",
        "## 解释边界",
        "",
        "本实验回答的是把动作更新从 0.5 s 放慢到 1.0 s 是否损失性能；它不直接证明把物理帧缩短到 0.25 s 一定无效。若放慢后几乎无损，只能说明当前控制频率可能已有冗余，0.25 s 的优先级应降低。",
    ]
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    seeds = v45.parse_seed_spec(args.eval_seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    policy = v45.load_method_policy(
        "joint_ppo_gated_sac",
        args.checkpoint,
        args.device,
    )
    rows = []
    for seed in seeds:
        for variant in VARIANTS:
            row = evaluate_variant(policy, variant, seed)
            rows.append(row)
            print(
                f"seed={seed} {variant}: "
                f"delay={row['task_delay_weighted_ms']:.3f}ms "
                f"queue={row['mean_queue_mbits']:.3f}Mbit "
                f"updates={row['base_action_updates']}",
                flush=True,
            )

    write_csv(output / "per_seed_raw.csv", rows)
    report = build_report(rows, args.checkpoint, seeds)
    if not report["crn"]["all_external_trace_hashes_match"]:
        raise RuntimeError("CRN external trace hashes differ between variants")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_report(report, output)
    write_result(report, output)
    print(json.dumps({
        "output": str(output),
        "screening_conclusion": report["screening_conclusion"],
        "delay_delta": report["paired_hold_two_minus_every_frame"]["task_delay_weighted_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
