import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = Path(bootstrap())

from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import resolve_project_path
from compare_common import torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SACActor
import hrl_uav.agents.sac_agent as sac_agent_mod
from search_position_oracle import make_capacity_ordered_policy
from search_position_oracle import make_centroid_offset_policy
from search_position_oracle import make_target_policy
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import sac_micro_policy


METRICS = (
    "delay_objective",
    "task_delay_weighted_ms",
    "average_power_w",
    "u2g",
    "coop_ratio",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the macro-position reward signal with paired seeds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-base-seed", type=int, default=31000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def configure_chinese_font():
    installed = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "STSong"):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_micro_policy(checkpoint_path, requested_device):
    checkpoint_path = resolve_project_path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sac_agent_mod.device = device

    env = make_env_from_args(checkpoint_args, record_trajectory=False)
    configure_macro_objective(
        env,
        checkpoint_args,
        checkpoint_args.macro_shaping_final,
    )
    saved_duals = getattr(
        checkpoint_args,
        "energy_lagrange_multipliers_current",
        None,
    )
    if saved_duals is not None:
        env.set_energy_lagrange_multipliers(saved_duals)

    micro_actor = SACActor(
        int(env.sac_obs_dim),
        int(env.micro_action_space.shape[0]),
    )
    micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    micro_actor.action_net.eval()
    return checkpoint_path, checkpoint_args, env, sac_micro_policy(micro_actor)


def compact_metrics(metrics):
    compact = {}
    for key in METRICS:
        value = np.asarray(metrics[key], dtype=np.float64)
        compact[key] = float(value.reshape(-1)[0])
    compact["battery_power_per_uav_w"] = np.asarray(
        metrics["average_battery_power_per_uav_w"],
        dtype=np.float64,
    ).tolist()
    compact["battery_power_budget_per_uav_w"] = np.asarray(
        metrics["uav_power_budgets_w"],
        dtype=np.float64,
    ).tolist()
    return compact


def paired_summary(records, reference):
    summary = {}
    count = len(records)
    critical = (
        float(student_t.ppf(0.975, count - 1))
        if count > 1
        else float("inf")
    )
    for key in METRICS:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        reference_values = np.asarray(
            [record[key] for record in reference],
            dtype=np.float64,
        )
        differences = values - reference_values
        std = float(np.std(values, ddof=1)) if count > 1 else 0.0
        diff_std = (
            float(np.std(differences, ddof=1))
            if count > 1
            else 0.0
        )
        diff_mean = float(np.mean(differences))
        half_width = (
            critical * diff_std / np.sqrt(count)
            if count > 1
            else float("inf")
        )
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": std,
            "paired_difference_vs_anchor": diff_mean,
            "paired_ci95_low": diff_mean - half_width,
            "paired_ci95_high": diff_mean + half_width,
            "paired_difference_std": diff_std,
            "paired_signal_to_noise": (
                abs(diff_mean) / max(diff_std, 1e-12)
            ),
        }

    power = np.asarray(
        [record["battery_power_per_uav_w"] for record in records],
        dtype=np.float64,
    )
    budgets = np.asarray(
        records[0]["battery_power_budget_per_uav_w"],
        dtype=np.float64,
    )
    if count > 1:
        sem = np.std(power, axis=0, ddof=1) / np.sqrt(count)
        ucb95 = np.mean(power, axis=0) + float(
            student_t.ppf(0.95, count - 1)
        ) * sem
    else:
        ucb95 = np.full_like(budgets, np.inf)
    summary["battery_power_mean_per_uav_w"] = np.mean(power, axis=0).tolist()
    summary["battery_power_ucb95_per_uav_w"] = ucb95.tolist()
    summary["battery_power_feasible_ucb95"] = bool(np.all(ucb95 <= budgets))
    return summary


def evaluate_policies(
    env,
    checkpoint_args,
    policies,
    micro_policy,
    episodes,
    eval_base_seed,
):
    records = {name: [] for name in policies}
    for eval_index in range(int(episodes)):
        seed = int(eval_base_seed) + eval_index
        for name, policy in policies.items():
            eval_args = argparse.Namespace(**vars(checkpoint_args))
            eval_args.eval_episodes = 1
            eval_args.eval_base_seed = seed
            metrics = evaluate_policy(
                env,
                eval_args,
                policy,
                micro_policy,
            )
            record = compact_metrics(metrics)
            record["eval_seed"] = seed
            records[name].append(record)
            print(
                f"[{eval_index + 1:02d}/{episodes:02d}] {name}: "
                f"delay={record['task_delay_weighted_ms']:.1f}ms "
                f"Jd={record['delay_objective']:.4f} "
                f"P={record['average_power_w']:.1f}W "
                f"U2G={record['u2g']:.1f}m"
            )
    reference = records["anchor"]
    summaries = {
        name: paired_summary(policy_records, reference)
        for name, policy_records in records.items()
    }
    return records, summaries


def save_csv(path, records):
    fieldnames = ["policy", "eval_seed", *METRICS]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, policy_records in records.items():
            for record in policy_records:
                writer.writerow(
                    {
                        "policy": name,
                        "eval_seed": record["eval_seed"],
                        **{key: record[key] for key in METRICS},
                    }
                )


def plot_summary(path, summaries):
    configure_chinese_font()
    names = list(summaries)
    labels = {
        "anchor": "区域锚点",
        "centroid": "追踪本区质心",
        "toward_20_40": "轻载机靠近20/40m",
        "toward_40_80": "轻载机靠近40/80m",
        "toward_60_120": "轻载机靠近60/120m",
    }
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    delay = [
        summaries[name]["task_delay_weighted_ms"]["mean"]
        for name in names
    ]
    delay_std = [
        summaries[name]["task_delay_weighted_ms"]["std"]
        for name in names
    ]
    axes[0].bar(x, delay, yerr=delay_std, capsize=3, color="#2878B5")
    axes[0].set_ylabel("加权任务时延（ms）")
    axes[0].set_title("(a) 真实端到端时延")

    delay_delta = [
        summaries[name]["task_delay_weighted_ms"][
            "paired_difference_vs_anchor"
        ]
        for name in names
    ]
    delay_ci = [
        max(
            summaries[name]["task_delay_weighted_ms"]["paired_ci95_high"]
            - delay_delta[index],
            0.0,
        )
        for index, name in enumerate(names)
    ]
    axes[1].bar(
        x,
        delay_delta,
        yerr=delay_ci,
        capsize=3,
        color="#D9534F",
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("相对锚点的配对时延差（ms）")
    axes[1].set_title("(b) 位置动作净收益")

    u2g = [summaries[name]["u2g"]["mean"] for name in names]
    coop = [summaries[name]["coop_ratio"]["mean"] for name in names]
    axes[2].plot(x, u2g, "o-", color="#2CA02C", label="U2G距离")
    axes[2].set_ylabel("平均U2G距离（m）", color="#2CA02C")
    coop_axis = axes[2].twinx()
    coop_axis.plot(x, coop, "s-", color="#9467BD", label="协作比例")
    coop_axis.set_ylabel("协作比例", color="#9467BD")
    axes[2].set_title("(c) 上行与协作变化")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(
            [labels.get(name, name) for name in names],
            rotation=22,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("固定区域接入下的宏观位置奖励信号")
    figure.tight_layout()
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    checkpoint_path, checkpoint_args, env, micro_policy = load_micro_policy(
        args.checkpoint,
        args.device,
    )
    anchors = np.asarray(env.env.anchor_points, dtype=np.float64)
    policies = {
        "anchor": make_target_policy(anchors),
        "centroid": make_centroid_offset_policy(np.zeros_like(anchors)),
        "toward_20_40": make_capacity_ordered_policy(
            medium_offset_m=20.0,
            light_offset_m=40.0,
        ),
        "toward_40_80": make_capacity_ordered_policy(
            medium_offset_m=40.0,
            light_offset_m=80.0,
        ),
        "toward_60_120": make_capacity_ordered_policy(
            medium_offset_m=60.0,
            light_offset_m=120.0,
        ),
    }
    records, summaries = evaluate_policies(
        env,
        checkpoint_args,
        policies,
        micro_policy,
        episodes=args.episodes,
        eval_base_seed=args.eval_base_seed,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "macro_reward_signal_raw.csv"
    json_path = output_dir / "macro_reward_signal_summary.json"
    figure_path = output_dir / "macro_reward_signal.png"
    save_csv(csv_path, records)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "episodes": int(args.episodes),
                "eval_base_seed": int(args.eval_base_seed),
                "association_policy": env.env.u2g_association_policy,
                "summaries": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    plot_summary(figure_path, summaries)
    print(f"[DONE] {json_path}")
    print(f"[DONE] {figure_path}")


if __name__ == "__main__":
    main()
