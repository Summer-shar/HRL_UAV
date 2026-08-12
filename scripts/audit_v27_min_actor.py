import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from audit_dual_pilot_trajectories import replay_checkpoint
from audit_dual_pilot_trajectories import write_csv
from compare_common import torch_load_checkpoint
import hrl_uav.agents.sac_agent as sac_agent_mod


DEFAULT_CHECKPOINT = (
    "results/tcom_dynamic_u2u_sparse_v27_min_actor/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_TRAIN_LOG = (
    "results/tcom_dynamic_u2u_sparse_v27_min_actor/pilot/seed1/"
    "joint_td3_sac/train_td3.csv"
)
DEFAULT_CONTROL_ROWS = (
    "result_archive/v24_system_dual_pilot_trajectory_audit/per_seed_raw.csv"
)
DEFAULT_OUTPUT = "result_archive/v27_min_actor_pilot_audit"

METRICS = (
    "J",
    "task_delay_weighted_ms",
    "average_power_w",
    "propulsion_energy_j",
    "compute_energy_j",
    "mean_centroid_distance_m",
    "p95_centroid_distance_m",
    "path_total_m",
    "net_total_m",
)
CHECKPOINT_EPISODES = (20, 40, 60, 80, 100)
POWER_BUDGET_W = 840.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pre-registered v27 min-Actor pilot against the v24 "
            "Q1-Actor control and a same-checkpoint centroid reference."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-log", default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--control-rows", default=DEFAULT_CONTROL_ROWS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_arm_rows(path, arm, seeds):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        source = [
            row
            for row in csv.DictReader(handle)
            if row["arm"] == arm and int(row["seed"]) in seeds
        ]
    rows = []
    for row in source:
        parsed = {"arm": arm, "seed": int(row["seed"])}
        for key, value in row.items():
            if key in ("arm", "seed", "centroid_trace_hash", "users_trace_hash"):
                continue
            parsed[key] = float(value)
        parsed["centroid_trace_hash"] = row["centroid_trace_hash"]
        parsed["users_trace_hash"] = row["users_trace_hash"]
        rows.append(parsed)
    if {row["seed"] for row in rows} != set(seeds):
        raise RuntimeError(f"{path} is missing requested {arm} seeds.")
    return sorted(rows, key=lambda row: row["seed"])


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean, mean
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return mean, mean - critical * sem, mean + critical * sem


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return float(np.mean(values))
    return float(
        np.mean(values)
        + stats.t.ppf(0.95, values.size - 1) * stats.sem(values)
    )


def paired_statistics(left_rows, right_rows, left_label, right_label):
    left = {row["seed"]: row for row in left_rows}
    right = {row["seed"]: row for row in right_rows}
    if set(left) != set(right):
        raise RuntimeError("Paired seed sets do not match.")

    per_seed = []
    summary = {}
    for seed in sorted(left):
        left_row = left[seed]
        right_row = right[seed]
        trace_match = (
            left_row["centroid_trace_hash"]
            == right_row["centroid_trace_hash"]
            and left_row["users_trace_hash"] == right_row["users_trace_hash"]
        )
        if not trace_match:
            raise RuntimeError(f"External trajectory mismatch for seed {seed}.")
        row = {"seed": seed, "external_trajectory_match": True}
        for metric in METRICS:
            row[f"{metric}_{left_label}"] = left_row[metric]
            row[f"{metric}_{right_label}"] = right_row[metric]
            row[f"{metric}_{left_label}_minus_{right_label}"] = (
                left_row[metric] - right_row[metric]
            )
        per_seed.append(row)

    for metric in METRICS:
        left_values = np.asarray(
            [left[seed][metric] for seed in sorted(left)], dtype=np.float64
        )
        right_values = np.asarray(
            [right[seed][metric] for seed in sorted(right)], dtype=np.float64
        )
        deltas = left_values - right_values
        mean_delta, ci_low, ci_high = mean_ci95(deltas)
        summary[metric] = {
            "n": int(deltas.size),
            f"{left_label}_mean": float(np.mean(left_values)),
            f"{right_label}_mean": float(np.mean(right_values)),
            f"{left_label}_minus_{right_label}_mean": mean_delta,
            f"{left_label}_minus_{right_label}_ci95_low": ci_low,
            f"{left_label}_minus_{right_label}_ci95_high": ci_high,
            "paired_p_value": float(
                stats.ttest_rel(left_values, right_values).pvalue
            )
            if deltas.size > 1
            else None,
        }
    return per_seed, summary


def read_training_progression(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = {int(row["episode"]): row for row in csv.DictReader(handle)}
    missing = sorted(set(CHECKPOINT_EPISODES) - set(rows))
    if missing:
        raise RuntimeError(f"Training log is missing episodes: {missing}")
    fields = (
        "td3_delay_q1_minus_q2_actor_mean",
        "td3_delay_q1_minus_q2_zero_mean",
        "td3_delay_q_twin_gap_actor_mean",
        "td3_delay_q_twin_gap_zero_mean",
        "td3_delay_q_actor_minus_zero_mean",
        "td3_delay_q_actor_preference_fraction",
        "energy_dual_after",
        "average_power_w",
    )
    progression = []
    for episode in CHECKPOINT_EPISODES:
        row = {"episode": episode}
        for field in fields:
            row[field] = float(rows[episode][field])
        progression.append(row)
    return progression


def draw_audit_figure(paired_control, v27_rows, progression, output_dir):
    delay_delta = np.asarray(
        [row["task_delay_weighted_ms_v27_minus_v24"] for row in paired_control]
    )
    distance_delta = np.asarray(
        [row["mean_centroid_distance_m_v27_minus_v24"] for row in paired_control]
    )
    power = np.asarray([row["average_power_w"] for row in v27_rows])
    episode = np.asarray([row["episode"] for row in progression])

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.2), constrained_layout=True)
    panels = (
        (axes[0], delay_delta, "Delay: v27 - v24", "ms", "#C82423"),
        (
            axes[1],
            distance_delta,
            "Centroid distance: v27 - v24",
            "m",
            "#2878B5",
        ),
    )
    for axis, values, title, unit, color in panels:
        jitter = np.linspace(-0.08, 0.08, values.size)
        axis.scatter(jitter, values, s=30, alpha=0.75, color=color)
        mean, low, high = mean_ci95(values)
        axis.errorbar(
            [0.0], [mean], yerr=[[mean - low], [high - mean]],
            fmt="D", color="black", capsize=5, label="Mean and 95% CI"
        )
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_xlim(-0.25, 0.25)
        axis.set_xticks([])
        axis.set_ylabel(unit)
        axis.set_title(title)

    mean_power, low_power, _ = mean_ci95(power)
    ucb = one_sided_ucb95(power)
    axes[2].scatter(np.arange(power.size), power, s=25, alpha=0.72, color="#2E8B57")
    axes[2].axhline(POWER_BUDGET_W, color="#C82423", linestyle="--", label="840 W")
    axes[2].axhline(mean_power, color="black", linewidth=1.0, label="Mean")
    axes[2].axhline(ucb, color="#7A5195", linewidth=1.2, label="UCB95")
    axes[2].fill_between(
        [-0.5, power.size - 0.5], low_power, ucb, color="#2E8B57", alpha=0.08
    )
    axes[2].set_xlabel("Evaluation seed index")
    axes[2].set_ylabel("W")
    axes[2].set_title("v27 power feasibility")
    axes[2].legend(frameon=False, fontsize=8)

    axes[3].plot(
        episode,
        [row["td3_delay_q1_minus_q2_actor_mean"] for row in progression],
        marker="o",
        label="Actor action",
        color="#C82423",
    )
    axes[3].plot(
        episode,
        [row["td3_delay_q1_minus_q2_zero_mean"] for row in progression],
        marker="s",
        label="Zero action",
        color="#2878B5",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_xlabel("Training episode")
    axes[3].set_ylabel("Q1 - Q2")
    axes[3].set_title("Signed twin-head disagreement")
    axes[3].legend(frameon=False, fontsize=8)

    fig.savefig(output_dir / "min_actor_pilot_audit.png", dpi=300)
    fig.savefig(output_dir / "min_actor_pilot_audit.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    device = torch.device(args.device)
    sac_agent_mod.device = device
    checkpoint = project_path(args.checkpoint)
    payload = torch_load_checkpoint(str(checkpoint))
    if payload.get("td3_actor_delay_aggregation") != "min":
        raise RuntimeError("The v27 audit requires a min-Actor checkpoint.")
    base_seed = int(payload["args"]["eval_base_seed"])
    seeds = list(range(base_seed, base_seed + int(args.episodes)))

    control_rows = read_arm_rows(
        project_path(args.control_rows), "dual_on", seeds
    )
    v27_rows = replay_checkpoint("v27_min_actor", str(checkpoint), seeds, device)
    centroid_rows = replay_checkpoint(
        "v27_centroid_reference",
        str(checkpoint),
        seeds,
        device,
        macro_policy_kind="centroid_zero",
    )
    paired_control, versus_control = paired_statistics(
        v27_rows, control_rows, "v27", "v24"
    )
    paired_centroid, versus_centroid = paired_statistics(
        v27_rows, centroid_rows, "v27", "centroid"
    )
    progression = read_training_progression(project_path(args.train_log))

    power_mean = float(np.mean([row["average_power_w"] for row in v27_rows]))
    power_ucb95 = one_sided_ucb95(
        [row["average_power_w"] for row in v27_rows]
    )
    delay_ci_high = versus_control["task_delay_weighted_ms"][
        "v27_minus_v24_ci95_high"
    ]
    distance_ci_high = versus_control["mean_centroid_distance_m"][
        "v27_minus_v24_ci95_high"
    ]
    behavior_criteria = {
        "delay_v27_minus_v24_ci95_high_below_zero": bool(delay_ci_high < 0.0),
        "centroid_distance_v27_minus_v24_ci95_high_below_zero": bool(
            distance_ci_high < 0.0
        ),
        "power_ucb95_within_840_w": bool(power_ucb95 <= POWER_BUDGET_W),
    }
    behavior_verdict = "PASS" if all(behavior_criteria.values()) else "FAIL"

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_raw.csv", v27_rows + centroid_rows)
    write_csv(output_dir / "paired_vs_v24.csv", paired_control)
    write_csv(output_dir / "paired_vs_centroid.csv", paired_centroid)
    write_csv(output_dir / "training_progression.csv", progression)
    result = {
        "status": "PRE_REGISTERED_BEHAVIOR_AUDIT",
        "behavior_verdict": behavior_verdict,
        "checkpoint": str(checkpoint),
        "checkpoint_episode": 100,
        "eval_seeds": seeds,
        "single_scientific_change": "TD3 Actor delay aggregation: Q1 -> min(Q1,Q2)",
        "external_trajectory_match_all": True,
        "power_budget_w": POWER_BUDGET_W,
        "observed": {
            "delay_ms": float(
                versus_control["task_delay_weighted_ms"]["v27_mean"]
            ),
            "mean_centroid_distance_m": float(
                versus_control["mean_centroid_distance_m"]["v27_mean"]
            ),
            "average_power_w": power_mean,
            "power_ucb95_w": power_ucb95,
        },
        "behavior_criteria": behavior_criteria,
        "versus_v24_q1_actor": versus_control,
        "versus_v27_centroid_reference": versus_centroid,
        "training_progression": progression,
        "mechanism_verdict": "PENDING_FIXED_PROBE",
        "overall_verdict": "PENDING_FIXED_PROBE",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    draw_audit_figure(paired_control, v27_rows, progression, output_dir)

    print("\nv27 min-Actor behavior audit")
    print(f"  verdict: {behavior_verdict}")
    print(
        "  delay v27-v24: "
        f"{versus_control['task_delay_weighted_ms']['v27_minus_v24_mean']:.6f} "
        "ms, 95% CI ["
        f"{versus_control['task_delay_weighted_ms']['v27_minus_v24_ci95_low']:.6f}, "
        f"{delay_ci_high:.6f}]"
    )
    print(
        "  centroid v27-v24: "
        f"{versus_control['mean_centroid_distance_m']['v27_minus_v24_mean']:.6f} "
        "m, 95% CI ["
        f"{versus_control['mean_centroid_distance_m']['v27_minus_v24_ci95_low']:.6f}, "
        f"{distance_ci_high:.6f}]"
    )
    print(f"  power: mean={power_mean:.6f} W, UCB95={power_ucb95:.6f} W")
    print(f"  criteria: {behavior_criteria}")
    print(f"Saved audit -> {output_dir}")


if __name__ == "__main__":
    main()
