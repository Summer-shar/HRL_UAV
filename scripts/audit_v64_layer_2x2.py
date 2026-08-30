"""Complete PPO/centroid x residual/QA layer attribution for v64."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v64_causal_fair as v64  # noqa: E402


ARCHIVE = ROOT / "result_archive" / "v64_causal_fair_hrl"
SEEDS = list(range(8000, 8020))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float]) -> dict:
    return v45.mean_ci95(values)


def main() -> None:
    learned = v45.load_method_policy(
        "ppo_residual",
        v64.checkpoint("joint_ppo_sac"),
        "cuda",
    )
    ppo_qa_rows: list[dict] = []
    for seed in SEEDS:
        qa = v64.static_policy("qa_psca_causal", learned["args"], seed)
        policy = dict(learned)
        policy["name"] = "ppo_qa"
        policy["args"] = qa["args"]
        policy["micro_policy"] = qa["micro_policy"]
        policy["micro_solver"] = "qa_psca_causal"
        row, _, _, _ = v64.evaluate(policy, seed)
        ppo_qa_rows.append(row)
        print(
            f"seed={seed} method=ppo_qa "
            f"delay={row['task_delay_weighted_ms']:.3f}ms",
            flush=True,
        )

    main_rows = read_csv(ARCHIVE / "per_seed_raw.csv")
    macro_rows = read_csv(ARCHIVE / "macro_counterfactual_per_seed.csv")
    rows: list[dict] = []
    for row in main_rows:
        if row["method"] == "hrl_residual":
            copied = dict(row)
            copied["method"] = "ppo_residual"
            rows.append(copied)
        elif row["method"] == "qa_psca_causal":
            copied = dict(row)
            copied["method"] = "centroid_qa"
            rows.append(copied)
    for row in macro_rows:
        if row["method"] == "centroid_residual":
            rows.append(dict(row))
    rows.extend(ppo_qa_rows)

    indexed = {
        (str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    methods = ("ppo_residual", "centroid_residual", "ppo_qa", "centroid_qa")
    metrics = [key for key in rows[0] if key not in {"seed", "method"}]
    summaries = {
        method: {
            metric: mean_ci([
                float(indexed[(method, seed)][metric])
                for seed in SEEDS
            ])
            for metric in metrics
        }
        for method in methods
    }

    def contrast(a: str, b: str, metric: str) -> dict:
        return mean_ci([
            float(indexed[(a, seed)][metric])
            - float(indexed[(b, seed)][metric])
            for seed in SEEDS
        ])

    effects = {
        "macro_effect_with_residual_ppo_minus_centroid": {
            metric: contrast("ppo_residual", "centroid_residual", metric)
            for metric in metrics
        },
        "macro_effect_with_qa_ppo_minus_centroid": {
            metric: contrast("ppo_qa", "centroid_qa", metric)
            for metric in metrics
        },
        "residual_effect_with_ppo_residual_minus_qa": {
            metric: contrast("ppo_residual", "ppo_qa", metric)
            for metric in metrics
        },
        "residual_effect_with_centroid_residual_minus_qa": {
            metric: contrast("centroid_residual", "centroid_qa", metric)
            for metric in metrics
        },
    }
    interaction = {
        metric: mean_ci([
            (
                float(indexed[("ppo_residual", seed)][metric])
                - float(indexed[("centroid_residual", seed)][metric])
                - float(indexed[("ppo_qa", seed)][metric])
                + float(indexed[("centroid_qa", seed)][metric])
            )
            for seed in SEEDS
        ])
        for metric in metrics
    }
    report = {
        "scope": "Same learned PPO checkpoint, same residual SAC checkpoint, and paired CRN seeds.",
        "method_summaries": summaries,
        "effects": effects,
        "interaction": interaction,
    }
    write_csv(ARCHIVE / "layer_2x2_per_seed.csv", rows)
    (ARCHIVE / "layer_2x2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    v64.configure_plotting()
    labels = ("PPO+残差", "质心+残差", "PPO+纯QA", "质心+纯QA")
    delay = [summaries[method]["task_delay_weighted_ms"]["mean"] for method in methods]
    queue = [summaries[method]["queue_total_mbits"]["mean"] for method in methods]
    colors = ("#D1495B", "#E78C99", "#247B70", "#2A9D8F")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, values, ylabel, title in (
        (axes[0], delay, "优先级加权端到端时延 / ms", "(a) 时延"),
        (axes[1], queue, "平均队列 / Mbit", "(b) 队列"),
    ):
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    residual_effect = effects["residual_effect_with_ppo_residual_minus_qa"]["task_delay_weighted_ms"]
    macro_effect = effects["macro_effect_with_residual_ppo_minus_centroid"]["task_delay_weighted_ms"]
    fig.suptitle(
        "v64分层2×2归因："
        f"SAC残差修正 {residual_effect['mean']:+.3f} ms，"
        f"PPO轨迹 {macro_effect['mean']:+.3f} ms（正值为变慢）"
    )
    fig.tight_layout()
    fig.savefig(ARCHIVE / "08_分层2x2归因.png", dpi=300, facecolor="white")
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
