"""Evaluate the v65 cross-scene clone-to-SAC micro gate with CRN seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v48_priority_gated as v48  # noqa: E402
import evaluate_v64_causal_fair as v64  # noqa: E402
from hrl_uav.agents.sac_agent import Actor as SACActor  # noqa: E402
import hrl_uav.agents.sac_agent as sac_agent_mod  # noqa: E402
from train_centralized_macro import sac_micro_policy  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v65_clone_finetune_micro_gate"
    / "formal"
    / "seed1"
    / "micro_pretrain_joint_ppo_direct_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_CLONE = (
    ROOT
    / "result_archive"
    / "v56_direct_sac_sca_distillation"
    / "clone_actor.pt"
)
DEFAULT_TRAIN_EVAL = (
    ROOT
    / "results"
    / "tcom_v65_clone_finetune_micro_gate"
    / "formal"
    / "seed1"
    / "micro_pretrain_joint_ppo_direct_sac"
    / "eval_central_macro.csv"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v65_clone_finetune_micro_gate"

LABELS = {
    "clone_frozen": "v56克隆（未微调）",
    "clone_finetuned": "v65克隆+在线微调",
    "qa_psca": "因果QA-PSCA",
}
COLORS = {
    "clone_frozen": "#6A4C93",
    "clone_finetuned": "#D1495B",
    "qa_psca": "#2A9D8F",
}


def zero_macro_policy(env, _local_obs, _deterministic=False):
    return np.zeros((env.N, 2), dtype=np.float32)


def frozen_clone_policy(base: dict, clone_path: Path, device_name: str) -> dict:
    args = argparse.Namespace(**vars(base["args"]))
    use_cuda = device_name == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sac_agent_mod.device = device
    probe_env = v45.make_eval_env(args, record_trajectory=False)
    action_dim = int(probe_env.micro_action_space.shape[0])
    actor = SACActor(int(probe_env.sac_obs_dim), action_dim)
    payload = torch.load(clone_path, map_location=device, weights_only=False)
    actor.action_net.load_state_dict(payload["sac_actor"])
    actor.action_net.eval()
    return {
        "name": "clone_frozen",
        "path": clone_path,
        "args": args,
        "macro_policy": zero_macro_policy,
        "micro_policy": sac_micro_policy(actor, args),
        "micro_solver": "sac",
        "macro_solver": "centroid",
    }


def learned_policy(checkpoint: Path, device_name: str) -> dict:
    policy = v45.load_method_policy("clone_finetuned", checkpoint, device_name)
    policy["macro_policy"] = zero_macro_policy
    policy["macro_solver"] = "centroid"
    return policy


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], methods: list[str], seeds: list[int]) -> dict:
    metrics = [key for key in rows[0] if key not in {"seed", "method"}]
    indexed = {
        (str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    method_summaries = {
        method: {
            metric: v45.mean_ci95([
                float(indexed[(method, seed)][metric]) for seed in seeds
            ])
            for metric in metrics
        }
        for method in methods
    }
    paired = {}
    for reference in ("clone_frozen", "qa_psca"):
        paired[f"clone_finetuned_minus_{reference}"] = {
            metric: v45.mean_ci95([
                float(indexed[("clone_finetuned", seed)][metric])
                - float(indexed[(reference, seed)][metric])
                for seed in seeds
            ])
            for metric in metrics
        }
    final_delay = method_summaries["clone_finetuned"]["task_delay_weighted_ms"]
    final_queue = method_summaries["clone_finetuned"]["queue_total_mbits"]
    checks = {
        "mean_delay_below_700_ms": bool(final_delay["mean"] < 700.0),
        "delay_ci95_high_below_720_ms": bool(final_delay["ci95_high"] < 720.0),
        "mean_queue_below_15_mbits": bool(final_queue["mean"] < 15.0),
    }
    return {
        "experiment": "v65 cross-scene clone Actor online micro fine-tuning gate",
        "training_seed": 1,
        "evaluation_seeds": seeds,
        "scope": (
            "The v56 clone was trained in the ultra-imbalanced load scene, then "
            "evaluated and fine-tuned in the v64 moderate-load scene. Macro motion "
            "is fixed to the dynamic centroid for every learned arm."
        ),
        "method_summaries": method_summaries,
        "paired_contrasts": paired,
        "preregistered_checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def plot_results(
    rows: list[dict],
    report: dict,
    methods: list[str],
    train_eval_path: Path,
    output: Path,
) -> None:
    v45.configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    with train_eval_path.open("r", encoding="utf-8-sig", newline="") as handle:
        curve = list(csv.DictReader(handle))
    episodes = [int(float(row["episode"])) for row in curve]
    delays = [float(row["task_delay_weighted_ms"]) for row in curve]
    axes[0].plot(episodes, delays, color=COLORS["clone_finetuned"], marker="o")
    axes[0].axhline(700.0, color="#333333", linestyle="--", linewidth=1.1)
    axes[0].set_title("(a) 在线微调过程")
    axes[0].set_xlabel("微观训练轮次")
    axes[0].set_ylabel("加权端到端时延 / ms")
    axes[0].grid(alpha=0.25)

    x = np.arange(len(methods))
    for axis, metric, title, ylabel in (
        (axes[1], "task_delay_weighted_ms", "(b) 20种子时延", "时延 / ms"),
        (axes[2], "queue_total_mbits", "(c) 20种子队列", "队列 / Mbit"),
    ):
        stats = [report["method_summaries"][m][metric] for m in methods]
        means = [item["mean"] for item in stats]
        errors = [
            [item["mean"] - item["ci95_low"] for item in stats],
            [item["ci95_high"] - item["mean"] for item in stats],
        ]
        axis.bar(
            x,
            means,
            yerr=np.asarray(errors),
            color=[COLORS[m] for m in methods],
            capsize=4,
        )
        axis.set_xticks(x, [LABELS[m] for m in methods], rotation=15, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_cn(report: dict, output: Path) -> None:
    summaries = report["method_summaries"]
    lines = [
        "# v65 克隆迁移微调门禁结论",
        "",
        f"**判决：{report['verdict']}。**",
        "",
        "本实验只验证一个问题：v56 在超不均衡场景训练的克隆 Actor，能否作为 v64 "
        "中等负载场景的初始化，并通过纯微观 SAC 在线微调达到可用水平。宏观始终固定为动态质心。",
        "",
        "| 方法 | 加权端到端时延 / ms | 队列 / Mbit |",
        "|---|---:|---:|",
    ]
    for method in ("clone_frozen", "clone_finetuned", "qa_psca"):
        delay = summaries[method]["task_delay_weighted_ms"]
        queue = summaries[method]["queue_total_mbits"]
        lines.append(
            f"| {LABELS[method]} | {delay['mean']:.2f} "
            f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
            f"{queue['mean']:.2f} [{queue['ci95_low']:.2f}, {queue['ci95_high']:.2f}] |"
        )
    lines.extend([
        "",
        "## 结论边界",
        "",
        "- 失败的是跨场景迁移方案，不是对 SAC 在同场景内可学习性的最终否定。",
        "- v56 克隆携带超不均衡场景的路由先验；负载分布改变后，它不是可靠的初始化。",
        "- 在线微调没有在冻结预算内修复该偏差，因此不进入位置门禁或 PPO 联合训练。",
        "- 若继续教师路线，教师数据必须在最终冻结场景内生成，并在论文方法中明确披露。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    parser.add_argument("--train-eval", type=Path, default=DEFAULT_TRAIN_EVAL)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Regenerate reports from an existing per_seed_raw.csv without simulation.",
    )
    args = parser.parse_args()

    for required in (args.checkpoint, args.clone, args.train_eval):
        if not required.is_file():
            raise FileNotFoundError(required)
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = v45.parse_seed_spec(args.eval_seeds)

    methods = ["clone_frozen", "clone_finetuned", "qa_psca"]
    raw_path = args.output / "per_seed_raw.csv"
    if args.reuse_raw:
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = []
            for source in csv.DictReader(handle):
                row = {
                    key: value if key == "method" else (
                        int(value) if key == "seed" else float(value)
                    )
                    for key, value in source.items()
                }
                rows.append(row)
    else:
        final = learned_policy(args.checkpoint, args.device)
        frozen = frozen_clone_policy(final, args.clone, args.device)
        rows = []
        for seed in seeds:
            policies = {
                "clone_frozen": frozen,
                "clone_finetuned": final,
                "qa_psca": v64.static_policy("qa_psca_causal", final["args"], seed),
            }
            for method in methods:
                row, _, _, _ = v48.evaluate_detailed(policies[method], seed)
                row["method"] = method
                rows.append(row)
                print(
                    f"seed={seed} method={method} "
                    f"delay={row['task_delay_weighted_ms']:.3f}ms "
                    f"queue={row['queue_total_mbits']:.3f}Mbit"
                )

    report = summarize(rows, methods, seeds)
    write_csv(raw_path, rows)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_results(
        rows,
        report,
        methods,
        args.train_eval,
        args.output / "01_v65微观迁移门禁.png",
    )
    write_cn(report, args.output / "RESULTS_CN.md")
    print(json.dumps(report["preregistered_checks"], ensure_ascii=False))
    print(f"verdict={report['verdict']}")


if __name__ == "__main__":
    main()
