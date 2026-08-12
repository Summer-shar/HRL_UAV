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
import hrl_uav.agents.sac_agent as sac_agent_mod


DEFAULT_HISTORICAL = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_CONTROL = (
    "results/tcom_dynamic_u2u_sparse_v29_control_n1/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_TREATMENT = (
    "results/tcom_dynamic_u2u_sparse_v29_treatment_n5/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_OUTPUT = "result_archive/v29_online_nstep_pilot/final_crn_evaluation"
EVAL_SEEDS = tuple(range(1000, 1020))
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
EQUIVALENCE_MARGIN_MS = 3.0
POWER_BUDGET_W = 840.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen v29 final CRN and control-reproduction audit."
    )
    parser.add_argument("--historical", default=DEFAULT_HISTORICAL)
    parser.add_argument("--control", default=DEFAULT_CONTROL)
    parser.add_argument("--treatment", default=DEFAULT_TREATMENT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return [mean, mean, mean]
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return [mean, mean - critical * sem, mean + critical * sem]


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return float(np.mean(values))
    return float(
        np.mean(values)
        + stats.t.ppf(0.95, values.size - 1) * stats.sem(values)
    )


def pair_rows(left_rows, right_rows, left_label, right_label):
    left = {int(row["seed"]): row for row in left_rows}
    right = {int(row["seed"]): row for row in right_rows}
    if set(left) != set(right):
        raise RuntimeError("Paired seed sets do not match.")
    per_seed = []
    summary = {}
    for seed in sorted(left):
        trace_match = (
            left[seed]["centroid_trace_hash"]
            == right[seed]["centroid_trace_hash"]
            and left[seed]["users_trace_hash"]
            == right[seed]["users_trace_hash"]
        )
        if not trace_match:
            raise RuntimeError(f"External trajectory mismatch for seed {seed}.")
        row = {"seed": seed, "external_trajectory_match": True}
        for metric in METRICS:
            row[f"{metric}_{left_label}"] = left[seed][metric]
            row[f"{metric}_{right_label}"] = right[seed][metric]
            row[f"{metric}_{left_label}_minus_{right_label}"] = (
                left[seed][metric] - right[seed][metric]
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
        summary[metric] = {
            f"{left_label}_mean": float(np.mean(left_values)),
            f"{right_label}_mean": float(np.mean(right_values)),
            f"{left_label}_minus_{right_label}_ci95": mean_ci95(deltas),
            "paired_p_value": float(stats.ttest_rel(left_values, right_values).pvalue),
        }
    return per_seed, summary


def draw_figure(treatment_rows, control_rows, paired_rows, output_dir):
    treatment = {int(row["seed"]): row for row in treatment_rows}
    control = {int(row["seed"]): row for row in control_rows}
    delay_delta = np.asarray(
        [
            treatment[seed]["task_delay_weighted_ms"]
            - control[seed]["task_delay_weighted_ms"]
            for seed in sorted(control)
        ]
    )
    centroid_delta = np.asarray(
        [
            treatment[seed]["mean_centroid_distance_m"]
            - control[seed]["mean_centroid_distance_m"]
            for seed in sorted(control)
        ]
    )
    path_delta = np.asarray(
        [
            treatment[seed]["path_total_m"] - control[seed]["path_total_m"]
            for seed in sorted(control)
        ]
    )
    power_control = np.asarray(
        [control[seed]["average_power_w"] for seed in sorted(control)]
    )
    power_treatment = np.asarray(
        [treatment[seed]["average_power_w"] for seed in sorted(control)]
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.2), constrained_layout=True)
    for axis, values, title, ylabel, color in (
        (axes[0], delay_delta, "Delay: n=5 - n=1", "ms", "#C82423"),
        (
            axes[1],
            centroid_delta,
            "Centroid distance: n=5 - n=1",
            "m",
            "#2878B5",
        ),
        (axes[2], path_delta, "Path length: n=5 - n=1", "m", "#7A5195"),
    ):
        jitter = np.linspace(-0.08, 0.08, values.size)
        axis.scatter(jitter, values, color=color, alpha=0.72, s=28)
        mean, low, high = mean_ci95(values)
        axis.errorbar(
            [0], [mean], yerr=[[mean - low], [high - mean]],
            fmt="D", color="black", capsize=5
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlim(-0.25, 0.25)
        axis.set_xticks([])
        axis.set_ylabel(ylabel)
        axis.set_title(title)

    labels = ("n=1", "n=5")
    values = (float(np.mean(power_control)), float(np.mean(power_treatment)))
    ucb = (one_sided_ucb95(power_control), one_sided_ucb95(power_treatment))
    axes[3].bar(labels, values, color=("#7A7A7A", "#2878B5"), alpha=0.88)
    axes[3].scatter(
        np.zeros_like(power_control), power_control, color="black", alpha=0.45, s=16
    )
    axes[3].scatter(
        np.ones_like(power_treatment), power_treatment, color="black", alpha=0.45, s=16
    )
    axes[3].scatter((0, 1), ucb, marker="D", color="#C82423", label="UCB95")
    axes[3].axhline(POWER_BUDGET_W, color="black", linestyle="--", linewidth=0.9)
    axes[3].set_ylabel("Average power (W)")
    axes[3].set_title("Constraint feasibility")
    axes[3].legend(frameon=False)
    fig.savefig(output_dir / "v29_final_crn_comparison.png", dpi=300)
    fig.savefig(output_dir / "v29_final_crn_comparison.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)
    sac_agent_mod.device = device
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_rows = replay_checkpoint(
        "historical_v24", project_path(args.historical), EVAL_SEEDS, device
    )
    control_rows = replay_checkpoint(
        "control_n1", project_path(args.control), EVAL_SEEDS, device
    )
    treatment_rows = replay_checkpoint(
        "treatment_n5", project_path(args.treatment), EVAL_SEEDS, device
    )
    reproduction_rows, reproduction = pair_rows(
        control_rows, historical_rows, "control_n1", "historical_v24"
    )
    comparison_rows, comparison = pair_rows(
        treatment_rows, control_rows, "treatment_n5", "control_n1"
    )

    reproduction_ci = reproduction["task_delay_weighted_ms"][
        "control_n1_minus_historical_v24_ci95"
    ]
    control_power_ucb = one_sided_ucb95(
        [row["average_power_w"] for row in control_rows]
    )
    treatment_power_ucb = one_sided_ucb95(
        [row["average_power_w"] for row in treatment_rows]
    )
    delay_ci = comparison["task_delay_weighted_ms"][
        "treatment_n5_minus_control_n1_ci95"
    ]
    reproduction_pass = bool(
        reproduction_ci[0] >= -EQUIVALENCE_MARGIN_MS
        and reproduction_ci[2] <= EQUIVALENCE_MARGIN_MS
        and control_power_ucb <= POWER_BUDGET_W
    )
    delay_pass = bool(delay_ci[2] < 0.0)
    constraint_pass = bool(treatment_power_ucb <= POWER_BUDGET_W)
    payload = {
        "eval_seeds": list(EVAL_SEEDS),
        "equivalence_margin_ms": EQUIVALENCE_MARGIN_MS,
        "control_reproduction": reproduction,
        "control_power_ucb95_w": control_power_ucb,
        "control_reproduced": reproduction_pass,
        "n5_minus_n1": comparison,
        "treatment_power_ucb95_w": treatment_power_ucb,
        "delay_improvement_pass": delay_pass,
        "constraint_pass": constraint_pass,
        "final_evaluation_pass": bool(delay_pass and constraint_pass),
    }
    write_csv(output_dir / "per_seed_raw.csv", historical_rows + control_rows + treatment_rows)
    write_csv(output_dir / "control_reproduction_paired.csv", reproduction_rows)
    write_csv(output_dir / "n5_minus_n1_paired.csv", comparison_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    draw_figure(treatment_rows, control_rows, comparison_rows, output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
