import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import t as student_t


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize support-aware same-state TD3 probes."
    )
    parser.add_argument("--branch-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row, field):
    return float(row[field])


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return [mean, None, None]
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    margin = float(student_t.ppf(0.975, values.size - 1) * standard_error)
    return [mean, mean - margin, mean + margin]


def safe_correlation(function, x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size < 3 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return None
    result = function(x_values, y_values)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def enrich_rows(rows):
    for row in rows:
        q_difference = as_float(row, "q_actor_minus_zero")
        return_difference = as_float(row, "return_actor_minus_zero")
        actor_conditional = as_float(
            row,
            "actor_support_conditional_action_nn_distance",
        )
        zero_conditional = as_float(
            row,
            "zero_support_conditional_action_nn_distance",
        )
        row["q_minus_return_error"] = q_difference - return_difference
        row["q1_actor_minus_zero"] = (
            as_float(row, "q1_actor") - as_float(row, "q1_zero")
        )
        row["q2_actor_minus_zero"] = (
            as_float(row, "q2_actor") - as_float(row, "q2_zero")
        )
        row["absolute_q_minus_return_error"] = abs(
            q_difference - return_difference
        )
        row["zero_minus_actor_conditional_support_distance"] = (
            zero_conditional - actor_conditional
        )
        row["zero_minus_actor_joint_support_distance"] = (
            as_float(row, "zero_support_joint_nn_distance")
            - as_float(row, "actor_support_joint_nn_distance")
        )
        row["ranking_reversed"] = float(
            np.sign(q_difference) != np.sign(return_difference)
        )
    return rows


def aggregate_episode(rows):
    grouped = {}
    for row in rows:
        key = (row["snapshot_policy"], int(row["episode_index"]))
        grouped.setdefault(key, []).append(row)
    result = []
    fields = (
        "q_actor_minus_zero",
        "q1_actor_minus_zero",
        "q2_actor_minus_zero",
        "return_actor_minus_zero",
        "q_minus_return_error",
        "absolute_q_minus_return_error",
        "zero_minus_actor_conditional_support_distance",
        "zero_minus_actor_joint_support_distance",
        "actor_support_conditional_action_nn_distance",
        "zero_support_conditional_action_nn_distance",
        "actor_support_joint_nn_distance",
        "zero_support_joint_nn_distance",
        "ranking_agreement",
        "ranking_reversed",
    )
    for (policy, episode_index), episode_rows in sorted(grouped.items()):
        aggregate = {
            "snapshot_policy": policy,
            "episode_index": episode_index,
            "seed": int(episode_rows[0]["seed"]),
            "branch_count": len(episode_rows),
        }
        for field in fields:
            aggregate[f"{field}_mean"] = float(
                np.mean([as_float(row, field) for row in episode_rows])
            )
        result.append(aggregate)
    return result


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, episode_rows):
    result = {
        "statistical_unit_for_ci": "episode",
        "branch_rows_used_for_correlations_only": True,
        "by_snapshot_policy": {},
    }
    for policy in sorted({row["snapshot_policy"] for row in rows}):
        raw = [row for row in rows if row["snapshot_policy"] == policy]
        episodes = [
            row for row in episode_rows if row["snapshot_policy"] == policy
        ]
        q_error = [as_float(row, "q_minus_return_error") for row in raw]
        absolute_error = [
            as_float(row, "absolute_q_minus_return_error") for row in raw
        ]
        support_gap = [
            as_float(
                row,
                "zero_minus_actor_joint_support_distance",
            )
            for row in raw
        ]
        result["by_snapshot_policy"][policy] = {
            "episodes": len(episodes),
            "branches": len(raw),
            "q_actor_minus_zero_episode_mean_ci95": mean_ci95(
                [row["q_actor_minus_zero_mean"] for row in episodes]
            ),
            "q1_actor_minus_zero_episode_mean_ci95": mean_ci95(
                [row["q1_actor_minus_zero_mean"] for row in episodes]
            ),
            "q2_actor_minus_zero_episode_mean_ci95": mean_ci95(
                [row["q2_actor_minus_zero_mean"] for row in episodes]
            ),
            "return_actor_minus_zero_episode_mean_ci95": mean_ci95(
                [row["return_actor_minus_zero_mean"] for row in episodes]
            ),
            "ranking_agreement_episode_mean_ci95": mean_ci95(
                [row["ranking_agreement_mean"] for row in episodes]
            ),
            "zero_minus_actor_support_distance_episode_mean_ci95": mean_ci95(
                [
                    row[
                        "zero_minus_actor_joint_support_distance_mean"
                    ]
                    for row in episodes
                ]
            ),
            "q_error_episode_mean_ci95": mean_ci95(
                [row["q_minus_return_error_mean"] for row in episodes]
            ),
            "support_gap_vs_signed_q_error_pearson_branch_level": (
                safe_correlation(pearsonr, support_gap, q_error)
            ),
            "support_gap_vs_abs_q_error_spearman_branch_level": (
                safe_correlation(spearmanr, support_gap, absolute_error)
            ),
        }
    return result


def draw_figure(rows, output_dir):
    policies = sorted({row["snapshot_policy"] for row in rows})
    colors = {"centroid": "#2878B5", "actor": "#C82423"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    for policy in policies:
        selected = [row for row in rows if row["snapshot_policy"] == policy]
        color = colors.get(policy, None)
        axes[0].scatter(
            [as_float(row, "return_actor_minus_zero") for row in selected],
            [as_float(row, "q_actor_minus_zero") for row in selected],
            label=policy,
            alpha=0.7,
            s=24,
            color=color,
        )
        axes[1].scatter(
            [
                as_float(
                    row,
                    "actor_support_joint_nn_distance",
                )
                for row in selected
            ],
            [
                as_float(
                    row,
                    "zero_support_joint_nn_distance",
                )
                for row in selected
            ],
            label=policy,
            alpha=0.7,
            s=24,
            color=color,
        )
        axes[2].scatter(
            [
                as_float(
                    row,
                    "zero_minus_actor_joint_support_distance",
                )
                for row in selected
            ],
            [as_float(row, "q_minus_return_error") for row in selected],
            label=policy,
            alpha=0.7,
            s=24,
            color=color,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("True return difference: actor - zero")
    axes[0].set_ylabel("Critic Q difference: actor - zero")
    axes[0].set_title("Critic ranking versus same-state return")

    limits = axes[1].get_xlim()
    combined_limits = (
        min(limits[0], axes[1].get_ylim()[0]),
        max(limits[1], axes[1].get_ylim()[1]),
    )
    axes[1].plot(combined_limits, combined_limits, "k--", linewidth=0.8)
    axes[1].set_xlim(combined_limits)
    axes[1].set_ylim(combined_limits)
    axes[1].set_xlabel("Actor query joint-support distance")
    axes[1].set_ylabel("Zero query joint-support distance")
    axes[1].set_title("Standardized joint state-action support")

    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Zero joint distance - actor joint distance")
    axes[2].set_ylabel("Critic difference - true return difference")
    axes[2].set_title("Support mismatch versus signed Q error")
    axes[0].legend(frameon=False)
    fig.savefig(os.path.join(output_dir, "support_probe_diagnosis.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "support_probe_diagnosis.pdf"))
    plt.close(fig)


def draw_critic_head_figure(episode_rows, output_dir):
    policies = [
        policy
        for policy in ("actor", "centroid")
        if any(row["snapshot_policy"] == policy for row in episode_rows)
    ]
    series = (
        ("Q1", "q1_actor_minus_zero_mean", "#C82423"),
        ("Q2", "q2_actor_minus_zero_mean", "#2878B5"),
        ("True return", "return_actor_minus_zero_mean", "#2E8B57"),
    )
    x = np.arange(len(policies), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for series_index, (label, field, color) in enumerate(series):
        means = []
        lower_errors = []
        upper_errors = []
        for policy in policies:
            values = [
                row[field]
                for row in episode_rows
                if row["snapshot_policy"] == policy
            ]
            mean, lower, upper = mean_ci95(values)
            means.append(mean)
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)
        axis.bar(
            x + (series_index - 1) * width,
            means,
            width,
            yerr=np.asarray([lower_errors, upper_errors]),
            capsize=4,
            label=label,
            color=color,
            alpha=0.88,
        )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(x, policies)
    axis.set_ylabel("Actor action - zero action")
    axis.set_title("Twin-Critic head ranking versus same-state return")
    axis.legend(frameon=False)
    fig.savefig(os.path.join(output_dir, "critic_head_comparison.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "critic_head_comparison.pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = enrich_rows(read_rows(args.branch_csv))
    episode_rows = aggregate_episode(rows)
    write_csv(os.path.join(args.output_dir, "branch_support_enriched.csv"), rows)
    write_csv(os.path.join(args.output_dir, "episode_support_summary.csv"), episode_rows)
    summary = summarize(rows, episode_rows)
    with open(
        os.path.join(args.output_dir, "support_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    draw_figure(rows, args.output_dir)
    draw_critic_head_figure(episode_rows, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
