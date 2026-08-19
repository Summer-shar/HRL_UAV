"""Evaluate and archive the frozen v40 micro-SAC warm-start experiment."""

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

import audit_v39_micro_regression as v39_audit


CONFIG = ROOT / "configs" / "tcom_v40_micro_warmstart.json"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v40_micro_warmstart_audit"
CHECKPOINTS = {
    "v33_final": (
        ROOT
        / "results"
        / "tcom_v33_soft_energy_random_mobility"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "checkpoints"
        / "final.pt"
    ),
    "v39_final": (
        ROOT
        / "results"
        / "tcom_v39_calibrated_soft_objective"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "checkpoints"
        / "final.pt"
    ),
    "v40_final": (
        ROOT
        / "results"
        / "tcom_v40_micro_warmstart"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "checkpoints"
        / "final.pt"
    ),
}
RUNS = {
    "v33": ROOT / "results" / "tcom_v33_soft_energy_random_mobility",
    "v39": ROOT / "results" / "tcom_v39_calibrated_soft_objective",
    "v40": ROOT / "results" / "tcom_v40_micro_warmstart",
}
COLORS = {"v33": "#2878B5", "v39": "#D9534F", "v40": "#2E8B57"}
LABELS = {
    "v33": "v33 旧策略",
    "v39": "v39 冷启动",
    "v40": "v40 暖启动",
    "v33_final": "v33 旧策略",
    "v39_final": "v39 冷启动 final",
    "v40_final": "v40 暖启动 final",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="6000-6019")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def numeric(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) if row.get(field, "") else np.nan for row in rows],
        dtype=np.float64,
    )


def run_paths(name: str) -> tuple[Path, Path]:
    base = RUNS[name] / "formal" / "seed1" / "micro_pretrain"
    return base / "train_centralized.csv", base / "eval_central_macro.csv"


def plot_convergence(output: Path) -> None:
    v39_audit.configure_chinese_font()
    logs = {}
    evaluations = {}
    for name in RUNS:
        train_path, eval_path = run_paths(name)
        logs[name] = read_csv(train_path)
        evaluations[name] = read_csv(eval_path)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2))
    ax_full, ax_tail, ax_queue, ax_critic = axes.ravel()
    for name, rows in evaluations.items():
        episodes = numeric(rows, "episode")
        delays = numeric(rows, "task_delay_weighted_ms")
        queues = numeric(rows, "q_high") + numeric(rows, "q_low")
        ax_full.plot(
            episodes,
            delays,
            marker="o",
            markersize=3.2,
            linewidth=1.8,
            color=COLORS[name],
            label=LABELS[name],
        )
        mask = episodes >= 140
        ax_tail.plot(
            episodes[mask],
            delays[mask],
            marker="o",
            markersize=3.2,
            linewidth=1.8,
            color=COLORS[name],
            label=LABELS[name],
        )
        ax_queue.plot(
            episodes,
            queues,
            marker="o",
            markersize=3.0,
            linewidth=1.6,
            color=COLORS[name],
            label=LABELS[name],
        )

    for name, rows in logs.items():
        episodes = numeric(rows, "episode")
        losses = numeric(rows, "sac_critic_loss")
        ax_critic.plot(
            episodes,
            v39_audit.rolling_mean(losses, window=20),
            linewidth=1.7,
            color=COLORS[name],
            label=LABELS[name],
        )

    for axis in (ax_full, ax_queue, ax_critic):
        axis.axvspan(1, 20, color="#D9EAF7", alpha=0.42, linewidth=0)
        axis.axvspan(21, 50, color="#F5E6C8", alpha=0.42, linewidth=0)
        axis.axvline(50, color="#666666", linestyle="--", linewidth=1.0)

    ax_full.set_title("(a) 固定评估端到端时延")
    ax_full.set_ylabel("优先级加权端到端时延 (ms)")
    ax_full.set_ylim(bottom=0)
    ax_tail.set_title("(b) 后半程时延稳定性")
    ax_tail.set_ylabel("优先级加权端到端时延 (ms)")
    ax_tail.set_xlim(140, 300)
    ax_tail.set_ylim(600, 1150)
    ax_queue.set_title("(c) 固定评估队列积压")
    ax_queue.set_ylabel("高低优先级队列总量 (Mbit)")
    ax_queue.set_ylim(bottom=0)
    ax_critic.set_title("(d) SAC Critic 损失（20轮滑动均值）")
    ax_critic.set_ylabel("Critic loss")
    ax_critic.set_yscale("log")
    for axis in axes.ravel():
        axis.set_xlabel("微观训练轮次")
        axis.grid(alpha=0.25, linestyle="--")
    ax_full.legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("微观 SAC 冷启动与 Actor-only 暖启动对比", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "01_微观SAC暖启动收敛对比.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(rows: list[dict], summary: dict, output: Path) -> None:
    v39_audit.configure_chinese_font()
    order = list(CHECKPOINTS)
    colors = [COLORS[name.split("_")[0]] for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    metrics = (
        ("delay_ms", "最终策略端到端时延", "时延 (ms)"),
        ("mean_queue_mbits", "最终策略平均队列", "平均队列 (Mbit)"),
    )
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        means = [summary[name][metric]["mean"] for name in order]
        lows = [summary[name][metric]["ci_low"] for name in order]
        highs = [summary[name][metric]["ci_high"] for name in order]
        x = np.arange(len(order))
        axis.bar(x, means, color=colors, alpha=0.82, width=0.62)
        axis.errorbar(
            x,
            means,
            yerr=[
                np.asarray(means) - np.asarray(lows),
                np.asarray(highs) - np.asarray(means),
            ],
            fmt="none",
            ecolor="#222222",
            capsize=4,
            linewidth=1.1,
        )
        for index, name in enumerate(order):
            values = [
                float(row["task_delay_weighted_ms"] if metric == "delay_ms" else row["mean_queue_mbits"])
                for row in rows
                if row["checkpoint"] == name
            ]
            jitter = np.linspace(-0.16, 0.16, len(values))
            axis.scatter(
                index + jitter,
                values,
                s=16,
                facecolors="white",
                edgecolors="#333333",
                linewidths=0.6,
                zorder=3,
            )
        axis.set_xticks(x, [LABELS[name] for name in order], rotation=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output / "02_最终策略时延与队列对比.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def paired_summary(rows: list[dict], reference: str) -> dict:
    by_name = {
        name: {
            int(row["seed"]): row
            for row in rows
            if row["checkpoint"] == name
        }
        for name in CHECKPOINTS
    }
    treatment = by_name["v40_final"]
    baseline = by_name[reference]
    metrics = {
        "delay_ms": "task_delay_weighted_ms",
        "mean_queue_mbits": "mean_queue_mbits",
        "terminal_queue_mbits": "terminal_queue_mbits",
        "average_power_w": "average_power_w",
    }
    result = {}
    for output_name, field in metrics.items():
        deltas = np.asarray(
            [
                float(treatment[seed][field]) - float(baseline[seed][field])
                for seed in sorted(treatment)
            ],
            dtype=np.float64,
        )
        result[output_name] = {
            **v39_audit.mean_ci(deltas),
            "negative_count": int(np.sum(deltas < 0.0)),
            "positive_count": int(np.sum(deltas > 0.0)),
        }
    return result


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = v39_audit.parse_seed_list(args.seeds)
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    v39_audit.CHECKPOINTS = CHECKPOINTS
    rows = v39_audit.evaluate_checkpoints(config, seeds, args.device)
    summary = v39_audit.summarize_evaluation(rows)
    v39_audit.write_rows(output / "per_seed_checkpoint_comparison.csv", rows)

    v40_train = v39_audit.read_training_log(run_paths("v40")[0])
    v40_eval_rows = read_csv(run_paths("v40")[1])
    tail_delays = np.asarray(
        [
            float(row["task_delay_weighted_ms"])
            for row in v40_eval_rows
            if 240 <= int(float(row["episode"])) <= 300
        ],
        dtype=np.float64,
    )
    delta = summary["v40_final"]["delay_delta_vs_v33_ms"]
    report = {
        "protocol": "v40 actor-only micro warm start",
        "config": str(CONFIG),
        "seeds": seeds,
        "checkpoints": {name: str(path) for name, path in CHECKPOINTS.items()},
        "initialization_audit": str(
            RUNS["v40"]
            / "formal"
            / "seed1"
            / "micro_pretrain"
            / "micro_warmstart_initialization.json"
        ),
        "checkpoint_evaluation": summary,
        "paired_v40_minus_v33": paired_summary(rows, "v33_final"),
        "paired_v40_minus_v39": paired_summary(rows, "v39_final"),
        "v40_training_diagnostics": v39_audit.training_summary(v40_train),
        "tail_eval_delays_ms": tail_delays.tolist(),
        "tail_eval_range_ms": float(np.max(tail_delays) - np.min(tail_delays)),
        "quality_pass": bool(delta["ci_high"] <= 0.0),
        "stability_pass": bool(
            float(np.max(tail_delays) - np.min(tail_delays)) < 20.0
        ),
    }
    report["verdict"] = (
        "PASS"
        if report["quality_pass"] and report["stability_pass"]
        else "PARTIAL_PASS_QUALITY_ONLY"
        if report["quality_pass"]
        else "FAIL"
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_convergence(output)
    plot_final_comparison(rows, summary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
