"""Cross-replay v45 macro and micro policies to isolate where gains originate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import evaluate_v45_fair_screen as v45


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "result_archive" / "v45_cross_policy_matrix"
TD3_CHECKPOINT = (
    ROOT / "results/tcom_v45_fair_screen_stage2/formal/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
PPO_CHECKPOINT = (
    ROOT / "results/tcom_v45_fair_screen_stage2/formal/seed1/"
    "joint_ppo_sac/checkpoints/final.pt"
)

MACRO_LABELS = {"td3": "TD3宏观", "ppo": "PPO宏观", "centroid": "动态质心"}
MICRO_LABELS = {"td3": "TD3训练出的SAC", "ppo": "PPO训练出的SAC"}


def combination_name(macro: str, micro: str) -> str:
    return f"{macro}_macro__{micro}_micro"


def paired(rows: list[dict], left: str, right: str, field: str) -> dict:
    left_values = {
        int(row["seed"]): float(row[field]) for row in rows if row["method"] == left
    }
    right_values = {
        int(row["seed"]): float(row[field]) for row in rows if row["method"] == right
    }
    seeds = sorted(set(left_values) & set(right_values))
    return v45.mean_ci95([left_values[seed] - right_values[seed] for seed in seeds])


def build_report(rows: list[dict]) -> dict:
    fields = ("task_delay_weighted_ms", "queue_total_mbits", "average_power_w", "soft_objective_j")
    methods = sorted({row["method"] for row in rows})
    summaries = {
        method: {
            field: v45.mean_ci95(
                [float(row[field]) for row in rows if row["method"] == method]
            )
            for field in fields
        }
        for method in methods
    }
    comparisons = {
        "macro_td3_minus_ppo_with_td3_micro": paired(
            rows, combination_name("td3", "td3"), combination_name("ppo", "td3"),
            "task_delay_weighted_ms",
        ),
        "macro_td3_minus_ppo_with_ppo_micro": paired(
            rows, combination_name("td3", "ppo"), combination_name("ppo", "ppo"),
            "task_delay_weighted_ms",
        ),
        "micro_ppo_minus_td3_with_td3_macro": paired(
            rows, combination_name("td3", "ppo"), combination_name("td3", "td3"),
            "task_delay_weighted_ms",
        ),
        "micro_ppo_minus_td3_with_ppo_macro": paired(
            rows, combination_name("ppo", "ppo"), combination_name("ppo", "td3"),
            "task_delay_weighted_ms",
        ),
    }
    for micro in ("td3", "ppo"):
        for macro in ("td3", "ppo"):
            comparisons[f"{macro}_macro_minus_centroid_with_{micro}_micro"] = paired(
                rows,
                combination_name(macro, micro),
                combination_name("centroid", micro),
                "task_delay_weighted_ms",
            )
    return {"summaries": summaries, "paired_delay_differences_ms": comparisons}


def write_report(report: dict, output: Path) -> None:
    lines = [
        "# v45 宏观-微观交叉重放",
        "",
        "> 零训练；3种宏观策略 × 2个SAC微观策略；20个共同随机评估种子。",
        "> 差值统一为“左侧组合减右侧组合”，负值表示左侧时延更低。",
        "",
        "| 宏观策略 | 微观策略 | 端到端时延/ms | 95% CI | 队列/Mbit | 功率/W |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for macro in ("td3", "ppo", "centroid"):
        for micro in ("td3", "ppo"):
            stats = report["summaries"][combination_name(macro, micro)]
            delay = stats["task_delay_weighted_ms"]
            lines.append(
                f"| {MACRO_LABELS[macro]} | {MICRO_LABELS[micro]} | {delay['mean']:.2f} | "
                f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
                f"{stats['queue_total_mbits']['mean']:.2f} | {stats['average_power_w']['mean']:.2f} |"
            )
    lines.extend(["", "## 配对差值", ""])
    for name, stats in report["paired_delay_differences_ms"].items():
        lines.append(
            f"- `{name}`：{stats['mean']:+.2f} ms，95% CI "
            f"[{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}]。"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_matrix(report: dict, output: Path) -> None:
    v45.configure_plotting()
    macros = ("td3", "ppo", "centroid")
    micros = ("td3", "ppo")
    x = np.arange(len(macros))
    width = 0.34
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    for offset, micro, color in ((-width / 2, "td3", "#2878B5"), (width / 2, "ppo", "#E07B39")):
        stats = [
            report["summaries"][combination_name(macro, micro)]["task_delay_weighted_ms"]
            for macro in macros
        ]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x + offset, means, width, yerr=errors, capsize=4,
            color=color, label=MICRO_LABELS[micro],
        )
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=4)
    axis.set_xticks(x, [MACRO_LABELS[item] for item in macros])
    axis.set_ylabel("优先级加权端到端时延 / ms")
    axis.set_title("宏观策略与微观SAC的交叉重放（20个共同种子）")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "01_宏观微观交叉矩阵.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    seeds = v45.parse_seed_spec(args.seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source = {
        "td3": v45.load_method_policy("td3", TD3_CHECKPOINT, args.device),
        "ppo": v45.load_method_policy("ppo", PPO_CHECKPOINT, args.device),
    }
    if v45.scenario_signature(source["td3"]["args"]) != v45.scenario_signature(source["ppo"]["args"]):
        raise RuntimeError("TD3 and PPO checkpoints do not share the same scenario")
    base_args = source["td3"]["args"]
    centroid_policy = v45.make_centroid_offset_policy(np.zeros((3, 2)))
    policies = {}
    for macro in ("td3", "ppo", "centroid"):
        for micro in ("td3", "ppo"):
            macro_policy = centroid_policy if macro == "centroid" else source[macro]["macro_policy"]
            policies[combination_name(macro, micro)] = {
                "name": combination_name(macro, micro),
                "args": base_args,
                "macro_policy": macro_policy,
                "micro_policy": source[micro]["micro_policy"],
                "micro_solver": "sac",
            }

    rows = []
    for seed in seeds:
        for policy in policies.values():
            row = v45.evaluate_one(policy, seed)
            rows.append(row)
            print(
                f"seed={seed} method={row['method']} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms",
                flush=True,
            )
    v45.write_csv(output / "per_seed_raw.csv", rows)
    report = build_report(rows)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(report, output)
    plot_matrix(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
