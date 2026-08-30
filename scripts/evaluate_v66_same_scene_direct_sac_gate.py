"""Evaluate the v66 same-scene teacher-initialized Direct SAC micro gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from compare_sca import solve_sca_micro_action  # noqa: E402
from hrl_uav.agents.sac_agent import Actor as SACActor  # noqa: E402
import hrl_uav.agents.sac_agent as sac_agent_mod  # noqa: E402
from train_centralized_macro import sac_micro_policy, sca_micro_policy  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v66_same_scene_direct_sac_gate"
    / "formal"
    / "seed1"
    / "micro_pretrain_joint_ppo_direct_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_CLONE = (
    ROOT
    / "result_archive"
    / "v66_same_scene_direct_sac_distillation"
    / "clone_actor.pt"
)
DEFAULT_TRAIN_EVAL = (
    ROOT
    / "results"
    / "tcom_v66_same_scene_direct_sac_gate"
    / "formal"
    / "seed1"
    / "micro_pretrain_joint_ppo_direct_sac"
    / "eval_central_macro.csv"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v66_same_scene_direct_sac_gate"

METHODS = ("clone_frozen", "direct_sac_final", "qa_expected")
LABELS = {
    "clone_frozen": "同场景克隆Actor",
    "direct_sac_final": "Direct SAC final",
    "qa_expected": "期望负载QA教师",
}
COLORS = {
    "clone_frozen": "#6A4C93",
    "direct_sac_final": "#D1495B",
    "qa_expected": "#2A9D8F",
}


def zero_macro_policy(env, _local_obs, _deterministic=False):
    return np.zeros((env.N, 2), dtype=np.float32)


def clone_args(args):
    return argparse.Namespace(**vars(args))


def load_final(checkpoint: Path, device_name: str) -> dict:
    policy = v45.load_method_policy("direct_sac_final", checkpoint, device_name)
    policy["macro_policy"] = zero_macro_policy
    policy["macro_solver"] = "centroid"
    return policy


def load_frozen_clone(base: dict, clone_path: Path, device_name: str) -> dict:
    args = clone_args(base["args"])
    use_cuda = device_name == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sac_agent_mod.device = device
    probe = v45.make_eval_env(args, record_trajectory=False)
    actor = SACActor(int(probe.sac_obs_dim), int(probe.micro_action_space.shape[0]))
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


def expected_teacher(base_args) -> dict:
    args = clone_args(base_args)
    args.sca_load_source = "expected_model"
    args.sca_update_interval = 1
    args.sca_max_iter = 3
    args.sca_tol = 1e-3
    args.sca_solver = "CLARABEL"
    args.sca_delay_weight = float(args.delay_objective_weight)
    args.sca_energy_weight = float(args.energy_objective_weight)
    args.sca_proximal_weight = 1e-3
    return {
        "name": "qa_expected",
        "path": Path("<fixed-policy>"),
        "args": args,
        "macro_policy": zero_macro_policy,
        "micro_policy": sca_micro_policy(solve_sca_micro_action, args),
        "micro_solver": "qa_expected",
        "macro_solver": "centroid",
    }


def state_dict_digest(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def actor_change(clone_path: Path, checkpoint_path: Path) -> dict:
    initial = torch.load(clone_path, map_location="cpu", weights_only=False)["sac_actor"]
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["sac_actor"]
    keys_match = set(initial) == set(final)
    if not keys_match:
        raise RuntimeError("Initial and final actor state-dict keys differ")
    differences = [
        (final[key].double() - initial[key].double()).reshape(-1)
        for key in sorted(initial)
    ]
    vector = torch.cat(differences)
    return {
        "initial_sha256": state_dict_digest(initial),
        "final_sha256": state_dict_digest(final),
        "parameters_changed": bool(torch.any(vector != 0).item()),
        "l2_parameter_change": float(torch.linalg.vector_norm(vector).item()),
        "max_abs_parameter_change": float(torch.max(torch.abs(vector)).item()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                key: value if key == "method" else (
                    int(value) if key == "seed" else float(value)
                )
                for key, value in source.items()
            }
            for source in csv.DictReader(handle)
        ]


def summarize(rows: list[dict], seeds: list[int], change: dict) -> dict:
    metrics = [key for key in rows[0] if key not in {"seed", "method"}]
    indexed = {(str(row["method"]), int(row["seed"])): row for row in rows}
    method_summaries = {
        method: {
            metric: v45.mean_ci95([
                float(indexed[(method, seed)][metric]) for seed in seeds
            ])
            for metric in metrics
        }
        for method in METHODS
    }
    paired = {
        f"direct_sac_final_minus_{reference}": {
            metric: v45.mean_ci95([
                float(indexed[("direct_sac_final", seed)][metric])
                - float(indexed[(reference, seed)][metric])
                for seed in seeds
            ])
            for metric in metrics
        }
        for reference in ("clone_frozen", "qa_expected")
    }
    final_delay = method_summaries["direct_sac_final"]["task_delay_weighted_ms"]
    final_queue = method_summaries["direct_sac_final"]["queue_total_mbits"]
    completion = method_summaries["direct_sac_final"]["task_completion_rate"]
    checks = {
        "mean_delay_below_700_ms": bool(final_delay["mean"] < 700.0),
        "delay_ci95_high_below_720_ms": bool(final_delay["ci95_high"] < 720.0),
        "mean_queue_below_15_mbits": bool(final_queue["mean"] < 15.0),
        "mean_completion_above_0_995": bool(completion["mean"] > 0.995),
        "sac_actor_parameters_changed": bool(change["parameters_changed"]),
    }
    return {
        "experiment": "v66 same-scene teacher-initialized Direct SAC micro gate",
        "training_seed": 1,
        "evaluation_seeds": seeds,
        "deployment_uses_online_optimizer": False,
        "training_uses_offline_teacher_initialization": True,
        "teacher_information_boundary": "expected_model (no realized-arrival lookahead)",
        "actor_change": change,
        "method_summaries": method_summaries,
        "paired_contrasts": paired,
        "preregistered_checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def plot_results(report: dict, train_eval: Path, output: Path) -> None:
    v45.configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    with train_eval.open("r", encoding="utf-8-sig", newline="") as handle:
        curve = list(csv.DictReader(handle))
    episodes = [int(float(row["episode"])) for row in curve]
    delays = [float(row["task_delay_weighted_ms"]) for row in curve]
    axes[0].plot(episodes, delays, color=COLORS["direct_sac_final"], marker="o")
    axes[0].axvline(50, color="#555555", linestyle="--", linewidth=1.1)
    axes[0].axhline(700, color="#222222", linestyle=":", linewidth=1.1)
    axes[0].set_title("(a) SAC在线微调")
    axes[0].set_xlabel("微观训练轮次")
    axes[0].set_ylabel("加权端到端时延 / ms")
    axes[0].grid(alpha=0.25)

    x = np.arange(len(METHODS))
    for axis, metric, title, ylabel in (
        (axes[1], "task_delay_weighted_ms", "(b) 20种子时延", "时延 / ms"),
        (axes[2], "queue_total_mbits", "(c) 20种子队列", "队列 / Mbit"),
    ):
        stats = [report["method_summaries"][m][metric] for m in METHODS]
        means = [item["mean"] for item in stats]
        yerr = np.asarray([
            [item["mean"] - item["ci95_low"] for item in stats],
            [item["ci95_high"] - item["mean"] for item in stats],
        ])
        axis.bar(
            x,
            means,
            yerr=yerr,
            color=[COLORS[m] for m in METHODS],
            capsize=4,
        )
        axis.set_xticks(x, [LABELS[m] for m in METHODS], rotation=12, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_cn(report: dict, output: Path) -> None:
    summaries = report["method_summaries"]
    versus_clone = report["paired_contrasts"][
        "direct_sac_final_minus_clone_frozen"
    ]["task_delay_weighted_ms"]
    versus_teacher = report["paired_contrasts"][
        "direct_sac_final_minus_qa_expected"
    ]["task_delay_weighted_ms"]
    lines = [
        "# v66 同场景 Direct SAC 微观门禁",
        "",
        f"**判决：{report['verdict']}。**",
        "",
        "部署时只运行 60维状态到18维完整路由动作的 Direct SAC Actor，不调用 QA、SCA 或 CVX。"
        "训练使用同场景、同可观测信息边界的离线教师初始化，之后清空 replay、重建 Critic，并执行在线 SAC 更新。",
        "",
        "| 方法 | 加权端到端时延 / ms | 队列 / Mbit | 完成率 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        delay = summaries[method]["task_delay_weighted_ms"]
        queue = summaries[method]["queue_total_mbits"]
        completion = summaries[method]["task_completion_rate"]
        lines.append(
            f"| {LABELS[method]} | {delay['mean']:.2f} "
            f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
            f"{queue['mean']:.2f} [{queue['ci95_low']:.2f}, {queue['ci95_high']:.2f}] | "
            f"{completion['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## 配对结论",
        "",
        f"- 在线 SAC 微调相对冻结克隆 Actor 改善 {-versus_clone['mean']:.2f} ms，"
        f"95% CI [{-versus_clone['ci95_high']:.2f}, {-versus_clone['ci95_low']:.2f}]；"
        "区间排除 0。",
        f"- Direct SAC final 相对离线 QA 教师仍慢 {versus_teacher['mean']:.2f} ms，"
        f"95% CI [{versus_teacher['ci95_low']:.2f}, {versus_teacher['ci95_high']:.2f}]。",
    ])
    change = report["actor_change"]
    lines.extend([
        "",
        "## 解释边界",
        "",
        f"- SAC Actor 参数已改变：L2={change['l2_parameter_change']:.6g}，"
        f"最大绝对变化={change['max_abs_parameter_change']:.6g}。",
        "- 这是教师初始化的 Direct SAC，不是从随机初始化训练出的纯 SAC。",
        "- 教师只在离线训练期使用期望负载，不读取当帧随机到达；部署不使用优化器。",
        "- 当前只完成一个训练初始化；20个 CRN 种子验证的是环境稳健性，不是训练种子稳健性。",
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
    parser.add_argument("--reuse-raw", action="store_true")
    cli = parser.parse_args()
    for required in (cli.checkpoint, cli.clone, cli.train_eval):
        if not required.is_file():
            raise FileNotFoundError(required)
    cli.output.mkdir(parents=True, exist_ok=True)
    seeds = v45.parse_seed_spec(cli.eval_seeds)
    raw_path = cli.output / "per_seed_raw.csv"

    if cli.reuse_raw:
        rows = read_csv(raw_path)
    else:
        final = load_final(cli.checkpoint, cli.device)
        policies = {
            "clone_frozen": load_frozen_clone(final, cli.clone, cli.device),
            "direct_sac_final": final,
            "qa_expected": expected_teacher(final["args"]),
        }
        rows = []
        for seed in seeds:
            for method in METHODS:
                row, _, _, _ = v48.evaluate_detailed(policies[method], seed)
                row["method"] = method
                rows.append(row)
                print(
                    f"seed={seed} method={method} "
                    f"delay={row['task_delay_weighted_ms']:.3f}ms "
                    f"queue={row['queue_total_mbits']:.3f}Mbit",
                    flush=True,
                )
        write_csv(raw_path, rows)

    change = actor_change(cli.clone, cli.checkpoint)
    report = summarize(rows, seeds, change)
    (cli.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_results(report, cli.train_eval, cli.output / "01_v66纯SAC微观门禁.png")
    write_cn(report, cli.output / "RESULTS_CN.md")
    print(json.dumps(report["preregistered_checks"], ensure_ascii=False))
    print(f"verdict={report['verdict']}")


if __name__ == "__main__":
    main()
