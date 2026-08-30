"""Measure the v64 macro-policy contribution with the final residual SAC fixed."""

from __future__ import annotations

import argparse
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "result_archive" / "v64_causal_fair_hrl",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = v45.parse_seed_spec(args.eval_seeds)

    learned = v45.load_method_policy(
        "hrl_residual",
        v64.checkpoint("joint_ppo_sac"),
        args.device,
    )
    centroid = dict(learned)
    centroid["name"] = "centroid_residual"
    centroid["macro_policy"] = v64.zero_macro_policy
    centroid["macro_solver"] = "centroid"

    policies = (learned, centroid)
    rows: list[dict] = []
    for seed in seeds:
        for policy in policies:
            row, _, _, _ = v64.evaluate(policy, seed)
            rows.append(row)
            print(
                f"seed={seed} method={policy['name']} "
                f"delay={row['task_delay_weighted_ms']:.3f}ms "
                f"distance={row['u2g_distance_m']:.3f}m",
                flush=True,
            )

    indexed = {
        (str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    metrics = [key for key in rows[0] if key not in {"seed", "method"}]
    methods = ("hrl_residual", "centroid_residual")
    summaries = {
        method: {
            metric: v45.mean_ci95([
                float(indexed[(method, seed)][metric])
                for seed in seeds
            ])
            for metric in metrics
        }
        for method in methods
    }
    paired = {
        metric: v45.mean_ci95([
            float(indexed[("hrl_residual", seed)][metric])
            - float(indexed[("centroid_residual", seed)][metric])
            for seed in seeds
        ])
        for metric in metrics
    }
    report = {
        "scope": (
            "Same final residual-SAC checkpoint and CRN seeds. The only "
            "difference is learned PPO macro actions versus zero residual "
            "dynamic-centroid tracking."
        ),
        "training_seed": 1,
        "evaluation_seeds": seeds,
        "method_summaries": summaries,
        "paired_hrl_minus_centroid": paired,
    }
    write_csv(output / "macro_counterfactual_per_seed.csv", rows)
    (output / "macro_counterfactual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    configure = getattr(v64, "configure_plotting")
    configure()
    deltas = np.asarray([
        float(indexed[("hrl_residual", seed)]["task_delay_weighted_ms"])
        - float(indexed[("centroid_residual", seed)]["task_delay_weighted_ms"])
        for seed in seeds
    ])
    stats = paired["task_delay_weighted_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].bar(
        np.arange(len(seeds)),
        deltas,
        color=np.where(deltas <= 0.0, "#2A9D8F", "#D1495B"),
    )
    axes[0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[0].set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds], rotation=60)
    axes[0].set_ylabel("PPO轨迹 - 动态质心时延 / ms")
    axes[0].set_title("(a) 逐环境种子配对差")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    labels = ("PPO+残差SAC", "质心+同一残差SAC")
    values = [
        summaries["hrl_residual"]["task_delay_weighted_ms"]["mean"],
        summaries["centroid_residual"]["task_delay_weighted_ms"]["mean"],
    ]
    errors = [
        summaries[method]["task_delay_weighted_ms"]["ci95_high"]
        - summaries[method]["task_delay_weighted_ms"]["mean"]
        for method in methods
    ]
    axes[1].bar(labels, values, yerr=errors, capsize=4, color=("#D1495B", "#2A9D8F"))
    axes[1].set_ylabel("优先级加权端到端时延 / ms")
    axes[1].set_title(
        "(b) 同checkpoint宏观贡献\n"
        f"配对均值 {stats['mean']:+.3f} ms "
        f"[{stats['ci95_low']:+.3f}, {stats['ci95_high']:+.3f}]"
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "04_同checkpoint宏观轨迹贡献.png", dpi=300, facecolor="white")
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
