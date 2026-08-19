"""Audit the frozen 500-episode v41 micro-SAC warm-start repeat."""

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

import audit_v39_micro_regression as shared


CONFIG = ROOT / "configs" / "tcom_v41_micro_warmstart_500ep.json"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v41_micro_warmstart_500ep_audit"
CHECKPOINTS = {
    "v33_final": ROOT
    / "results/tcom_v33_soft_energy_random_mobility/formal/seed1/micro_pretrain/checkpoints/final.pt",
    "v40_final": ROOT
    / "results/tcom_v40_micro_warmstart/formal/seed1/micro_pretrain/checkpoints/final.pt",
    "v41_final": ROOT
    / "results/tcom_v41_micro_warmstart_500ep/formal/seed1/micro_pretrain/checkpoints/final.pt",
}
RUNS = {
    "v40": ROOT / "results/tcom_v40_micro_warmstart/formal/seed1/micro_pretrain",
    "v41": ROOT / "results/tcom_v41_micro_warmstart_500ep/formal/seed1/micro_pretrain",
}
COLORS = {"v40": "#2878B5", "v41": "#D9534F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="6000-6019")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def eval_path(name: str) -> Path:
    return RUNS[name] / "eval_central_macro.csv"


def train_path(name: str) -> Path:
    return RUNS[name] / "train_centralized.csv"


def paired_summary(rows: list[dict], treatment: str, baseline: str) -> dict:
    grouped = {
        name: {
            int(row["seed"]): row
            for row in rows
            if row["checkpoint"] == name
        }
        for name in CHECKPOINTS
    }
    fields = {
        "delay_ms": "task_delay_weighted_ms",
        "mean_queue_mbits": "mean_queue_mbits",
        "terminal_queue_mbits": "terminal_queue_mbits",
        "average_power_w": "average_power_w",
    }
    result = {}
    for label, field in fields.items():
        common = sorted(set(grouped[treatment]) & set(grouped[baseline]))
        deltas = np.asarray(
            [
                float(grouped[treatment][seed][field])
                - float(grouped[baseline][seed][field])
                for seed in common
            ],
            dtype=np.float64,
        )
        result[label] = {
            **shared.mean_ci(deltas),
            "negative_count": int(np.sum(deltas < 0.0)),
            "positive_count": int(np.sum(deltas > 0.0)),
        }
    return result


def aligned_eval_prefix() -> dict:
    rows40 = {
        int(float(row["episode"])): row
        for row in read_csv(eval_path("v40"))
        if int(float(row["episode"])) <= 300
    }
    rows41 = {
        int(float(row["episode"])): row
        for row in read_csv(eval_path("v41"))
        if int(float(row["episode"])) <= 300
    }
    episodes = sorted(set(rows40) & set(rows41))
    delay40 = np.asarray(
        [float(rows40[episode]["task_delay_weighted_ms"]) for episode in episodes]
    )
    delay41 = np.asarray(
        [float(rows41[episode]["task_delay_weighted_ms"]) for episode in episodes]
    )
    delta = delay41 - delay40
    return {
        "episodes": episodes,
        "v40_delay_ms": delay40.tolist(),
        "v41_delay_ms": delay41.tolist(),
        "delta_ms": delta.tolist(),
        "max_abs_delta_ms": float(np.max(np.abs(delta))),
        "exact_within_1e_6_ms": bool(np.max(np.abs(delta)) <= 1e-6),
        "correlation": float(np.corrcoef(delay40, delay41)[0, 1]),
    }


def plot_training(output: Path) -> None:
    shared.configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    for name in ("v40", "v41"):
        rows = read_csv(eval_path(name))
        episodes = np.asarray([int(float(row["episode"])) for row in rows])
        delays = np.asarray(
            [float(row["task_delay_weighted_ms"]) for row in rows]
        )
        axes[0].plot(
            episodes,
            delays,
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=COLORS[name],
            label=f"{name} 固定评估",
        )
        mask = episodes >= 200
        axes[1].plot(
            episodes[mask],
            delays[mask],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=COLORS[name],
            label=name,
        )
    axes[0].axvspan(1, 20, color="#D9EAF7", alpha=0.4, linewidth=0)
    axes[0].axvspan(21, 50, color="#F5E6C8", alpha=0.4, linewidth=0)
    axes[0].set_title("(a) 300轮与500轮完整训练轨迹")
    axes[0].set_ylabel("优先级加权端到端时延 (ms)")
    axes[0].set_ylim(bottom=0)
    axes[1].axvspan(440, 500, color="#E8F3E8", alpha=0.55, linewidth=0)
    axes[1].set_title("(b) 后半程与冻结尾段")
    axes[1].set_ylabel("优先级加权端到端时延 (ms)")
    for axis in axes:
        axis.set_xlabel("微观训练轮次")
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "01_v40与v41训练稳定性对比.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = shared.parse_seed_list(args.seeds)
    shared.CHECKPOINTS = CHECKPOINTS
    rows = shared.evaluate_checkpoints(config, seeds, args.device)
    summary = shared.summarize_evaluation(rows)
    shared.write_rows(output / "per_seed_checkpoint_comparison.csv", rows)

    v41_eval = read_csv(eval_path("v41"))
    tail = np.asarray(
        [
            float(row["task_delay_weighted_ms"])
            for row in v41_eval
            if int(float(row["episode"])) in {440, 460, 480, 500}
        ],
        dtype=np.float64,
    )
    if tail.size != 4:
        raise RuntimeError(f"Expected four frozen tail evaluations, got {tail.size}")
    versus_v33 = paired_summary(rows, "v41_final", "v33_final")
    versus_v40 = paired_summary(rows, "v41_final", "v40_final")
    quality_pass = bool(versus_v33["delay_ms"]["ci_high"] <= 0.0)
    stability_range = float(np.max(tail) - np.min(tail))
    stability_pass = bool(stability_range < 20.0)
    report = {
        "protocol": "v41 500-episode actor-only warm-start repeat",
        "config": str(CONFIG),
        "seeds": seeds,
        "checkpoints": {name: str(path) for name, path in CHECKPOINTS.items()},
        "checkpoint_evaluation": summary,
        "paired_v41_minus_v33": versus_v33,
        "paired_v41_minus_v40": versus_v40,
        "prefix_reproduction": aligned_eval_prefix(),
        "v41_training_diagnostics": shared.training_summary(
            shared.read_training_log(train_path("v41"))
        ),
        "tail_eval_episodes": [440, 460, 480, 500],
        "tail_eval_delays_ms": tail.tolist(),
        "tail_eval_range_ms": stability_range,
        "quality_pass": quality_pass,
        "stability_pass": stability_pass,
        "verdict": (
            "PASS"
            if quality_pass and stability_pass
            else "PARTIAL_PASS_QUALITY_ONLY"
            if quality_pass
            else "FAIL"
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_training(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
