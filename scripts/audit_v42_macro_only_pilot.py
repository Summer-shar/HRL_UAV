"""Evaluate the frozen-micro v42 TD3 final policy against matched centroid tracking."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hrl_uav.agents.sac_agent as sac_agent_mod
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
from compare_common import evaluate_policy, make_env_from_args
from run_v34_cooperative_mobility_gate import V34MacroPolicy
from train_centralized_macro import configure_macro_objective, sac_micro_policy
from train_centralized_td3 import (
    MacroActor,
    td3_macro_policy,
    torch_load_checkpoint,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v42_macro_only_pilot"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_TRAIN_CSV = DEFAULT_CHECKPOINT.parents[1] / "train_td3.csv"
DEFAULT_EVAL_CSV = DEFAULT_CHECKPOINT.parents[1] / "eval_central_td3.csv"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v42_macro_only_pilot_audit"
DEFAULT_SEEDS = tuple(range(7000, 7020))


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    return seeds


def mean_ci95(values: list[float] | np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty sample")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    critical = 2.093 if values.size == 20 else 1.96
    half = critical * std / np.sqrt(values.size) if values.size > 1 else 0.0
    return {
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "std": std,
        "values": values.tolist(),
    }


def load_policies(checkpoint_path: Path, requested_device: str):
    checkpoint = torch_load_checkpoint(str(checkpoint_path))
    args = argparse.Namespace(**checkpoint["args"])
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sac_agent_mod.device = device

    probe_env = make_env_from_args(args, record_trajectory=False)
    _, global_state = probe_env.reset(seed=int(args.seed))[0]
    macro_actor = MacroActor(
        int(np.asarray(global_state).size),
        int(args.num_agents) * 2,
        hidden_dim=int(args.td3_hidden_dim),
    ).to(device)
    macro_actor.load_state_dict(checkpoint["macro_actor"])
    macro_actor.eval()

    micro_actor = SAC_Actor(
        int(probe_env.sac_obs_dim),
        int(probe_env.micro_action_space.shape[0]),
    )
    micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    micro_actor.action_net.eval()
    return args, td3_macro_policy(macro_actor), sac_micro_policy(micro_actor, args)


def evaluate_one(args, macro_policy, micro_policy, seed: int) -> dict:
    env = make_env_from_args(args, record_trajectory=False)
    configure_macro_objective(env, args, getattr(args, "macro_shaping_final", 0.0))
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(env, eval_args, macro_policy, micro_policy)
    return {
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "soft_objective_j": float(metrics["J"]),
        "mean_queue_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "u2g_distance_m": float(metrics["u2g"]),
        "coop_ratio": float(metrics["coop_ratio"]),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_results(
    eval_csv: Path,
    paired: dict,
    output: Path,
) -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
    })
    eval_rows = read_csv(eval_csv)
    episodes = np.asarray([int(float(row["episode"])) for row in eval_rows])
    delays = np.asarray([float(row["task_delay_weighted_ms"]) for row in eval_rows])
    powers = np.asarray([float(row["average_power_w"]) for row in eval_rows])
    delay_delta = np.asarray(paired["delay_ms"]["values"], dtype=np.float64)
    objective_delta = np.asarray(paired["soft_objective_j"]["values"], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    axes[0].plot(episodes, delays, color="#2878B5", marker="o", linewidth=2.2)
    axes[0].set_title("(a) 宏观 TD3 固定评估时延")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("优先级加权端到端时延 / ms")
    axes[0].grid(linestyle="--", alpha=0.3)

    axes[1].plot(episodes, powers, color="#E45756", marker="o", linewidth=2.2)
    axes[1].set_title("(b) 确定性策略平均功率")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("系统平均功率 / W")
    axes[1].grid(linestyle="--", alpha=0.3)

    x = np.arange(delay_delta.size)
    colors = np.where(delay_delta < 0.0, "#2E7D6E", "#E45756")
    axes[2].bar(x, delay_delta, color=colors, alpha=0.9)
    axes[2].axhline(0.0, color="#333333", linewidth=1.0)
    axes[2].set_title("(c) TD3 相对质心的逐种子时延差")
    axes[2].set_xlabel("共同随机种子序号")
    axes[2].set_ylabel("TD3 - 质心 / ms（负值更优）")
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)
    axes[2].text(
        0.02,
        0.96,
        f"均值 {np.mean(delay_delta):+.2f} ms\n软目标差 {np.mean(objective_delta):+.4f}",
        transform=axes[2].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output / "01_宏观TD3训练与配对结果.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="7000-7019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse output/per_seed_raw.csv instead of repeating environment replay.",
    )
    args_cli = parser.parse_args()

    checkpoint_path = args_cli.checkpoint.resolve()
    eval_csv = args_cli.eval_csv.resolve()
    output = args_cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_spec(args_cli.seeds)
    raw_path = output / "per_seed_raw.csv"
    if args_cli.reuse_raw:
        rows = read_csv(raw_path)
        observed_seeds = sorted({int(row["seed"]) for row in rows})
        if observed_seeds != seeds or len(rows) != 2 * len(seeds):
            raise ValueError(
                "Existing per_seed_raw.csv does not match the requested seeds"
            )
    else:
        args, td3_policy, micro_policy = load_policies(
            checkpoint_path, args_cli.device
        )
        rows: list[dict] = []
        for seed in seeds:
            policies = {
                "centroid": V34MacroPolicy("centroid", 0.0, 0.5),
                "td3": td3_policy,
            }
            for name, policy in policies.items():
                metrics = evaluate_one(args, policy, micro_policy, seed)
                rows.append({"seed": seed, "policy": name, **metrics})
                print(
                    f"seed={seed} policy={name} "
                    f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
                    f"J={metrics['soft_objective_j']:.4f}"
                )
        write_rows(raw_path, rows)

    by_policy = {
        name: {int(row["seed"]): row for row in rows if row["policy"] == name}
        for name in ("centroid", "td3")
    }
    metric_fields = (
        "task_delay_weighted_ms",
        "soft_objective_j",
        "mean_queue_mbits",
        "average_power_w",
        "u2g_distance_m",
        "coop_ratio",
    )
    policy_means = {
        name: {
            field: float(np.mean([
                float(by_policy[name][seed][field]) for seed in seeds
            ]))
            for field in metric_fields
        }
        for name in ("centroid", "td3")
    }
    paired: dict[str, dict] = {}
    for report_name, field in (
        ("delay_ms", "task_delay_weighted_ms"),
        ("soft_objective_j", "soft_objective_j"),
        ("mean_queue_mbits", "mean_queue_mbits"),
        ("average_power_w", "average_power_w"),
        ("u2g_distance_m", "u2g_distance_m"),
    ):
        deltas = [
            float(by_policy["td3"][seed][field])
            - float(by_policy["centroid"][seed][field])
            for seed in seeds
        ]
        paired[report_name] = mean_ci95(deltas)
        paired[report_name]["negative_count"] = int(np.sum(np.asarray(deltas) < 0.0))
        paired[report_name]["positive_count"] = int(np.sum(np.asarray(deltas) > 0.0))

    paired_rows = []
    for seed in seeds:
        paired_rows.append({
            "seed": seed,
            **{
                f"td3_minus_centroid_{field}": (
                    float(by_policy["td3"][seed][field])
                    - float(by_policy["centroid"][seed][field])
                )
                for field in metric_fields
            },
        })
    write_rows(output / "paired_deltas.csv", paired_rows)

    centroid_queue = float(np.mean([
        float(by_policy["centroid"][seed]["mean_queue_mbits"]) for seed in seeds
    ]))
    checks = {
        "delay_ci_upper_lt_zero": bool(paired["delay_ms"]["ci_high"] < 0.0),
        "soft_objective_ci_upper_lt_zero": bool(
            paired["soft_objective_j"]["ci_high"] < 0.0
        ),
        "queue_noninferior": bool(
            paired["mean_queue_mbits"]["ci_high"] < 0.05 * centroid_queue
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checkpoint": str(checkpoint_path),
        "seeds": seeds,
        "policy_delta_definition": "td3_minus_centroid",
        "policy_means": policy_means,
        "paired": paired,
        "centroid_mean_queue_mbits": centroid_queue,
        "queue_noninferiority_margin_mbits": 0.05 * centroid_queue,
        "checks": checks,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(eval_csv, paired, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
