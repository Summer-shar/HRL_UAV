import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_OUTPUT = Path(
    "result_archive/v29_online_nstep_pilot/staleness_audit"
)
DEFAULT_INPUTS = {
    "control_n1": Path(
        "results/tcom_dynamic_u2u_sparse_v29_control_n1/pilot/seed1/"
        "joint_td3_sac/nstep_replay_diagnostics.csv"
    ),
    "treatment_n5": Path(
        "results/tcom_dynamic_u2u_sparse_v29_treatment_n5/pilot/seed1/"
        "joint_td3_sac/nstep_replay_diagnostics.csv"
    ),
}
AGE_BUCKETS = ("0-4", "5-14", "15-29", ">=30")
FIELDS = (
    "y5_minus_y1_mean",
    "abs_y5_minus_y1_mean",
    "q_y5_gap_abs_mean",
    "policy_mismatch_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the post-hoc v29 replay-age diagnostics."
    )
    parser.add_argument("--control", default=str(DEFAULT_INPUTS["control_n1"]))
    parser.add_argument(
        "--treatment", default=str(DEFAULT_INPUTS["treatment_n5"])
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--episode-start", type=int, default=60)
    return parser.parse_args()


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    mean = float(np.mean(values))
    if values.size == 1:
        return {"n": 1, "mean": mean, "low": mean, "high": mean}
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return {
        "n": int(values.size),
        "mean": mean,
        "low": mean - critical * sem,
        "high": mean + critical * sem,
    }


def load_rows(path, arm, episode_start):
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            episode = int(raw["episode"])
            if episode < episode_start or raw["age_bucket"] not in AGE_BUCKETS:
                continue
            row = {
                "arm": arm,
                "episode": episode,
                "age_bucket": raw["age_bucket"],
                "samples": int(raw["samples"]),
            }
            for field in FIELDS:
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def summarize_arm(rows):
    keyed = {(row["episode"], row["age_bucket"]): row for row in rows}
    episodes = sorted(
        episode
        for episode, bucket in keyed
        if bucket == AGE_BUCKETS[0]
        and all((episode, candidate) in keyed for candidate in AGE_BUCKETS)
    )
    bucket_summary = {}
    for bucket in AGE_BUCKETS:
        bucket_summary[bucket] = {
            field: mean_ci95([keyed[(episode, bucket)][field] for episode in episodes])
            for field in FIELDS
        }
    old_minus_young = {
        field: mean_ci95(
            [
                keyed[(episode, AGE_BUCKETS[-1])][field]
                - keyed[(episode, AGE_BUCKETS[0])][field]
                for episode in episodes
            ]
        )
        for field in FIELDS
    }
    return {
        "episode_start": min(episodes) if episodes else None,
        "episode_end": max(episodes) if episodes else None,
        "episode_count": len(episodes),
        "bucket_summary": bucket_summary,
        "old_ge30_minus_young_0_4": old_minus_young,
        "registered_staleness_readout": {
            "older_samples_have_larger_abs_y5_minus_y1": bool(
                old_minus_young["abs_y5_minus_y1_mean"]["low"] is not None
                and old_minus_young["abs_y5_minus_y1_mean"]["low"] > 0.0
            ),
            "older_samples_have_larger_policy_mismatch": bool(
                old_minus_young["policy_mismatch_mean"]["low"] is not None
                and old_minus_young["policy_mismatch_mean"]["low"] > 0.0
            ),
            "older_samples_have_larger_q_y5_diagnostic_gap": bool(
                old_minus_young["q_y5_gap_abs_mean"]["low"] is not None
                and old_minus_young["q_y5_gap_abs_mean"]["low"] > 0.0
            ),
        },
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_profiles(summary, output_dir):
    colors = {"control_n1": "#2F6690", "treatment_n5": "#C14953"}
    panels = (
        ("abs_y5_minus_y1_mean", r"$|Y_5-Y_1|$"),
        ("q_y5_gap_abs_mean", r"$|Q-Y_5|$ diagnostic gap"),
        ("policy_mismatch_mean", "Behavior-target action distance"),
    )
    x = np.arange(len(AGE_BUCKETS))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    for axis, (field, title) in zip(axes, panels):
        for arm in ("control_n1", "treatment_n5"):
            stats_by_bucket = summary["arms"][arm]["bucket_summary"]
            means = [stats_by_bucket[bucket][field]["mean"] for bucket in AGE_BUCKETS]
            lows = [stats_by_bucket[bucket][field]["low"] for bucket in AGE_BUCKETS]
            highs = [stats_by_bucket[bucket][field]["high"] for bucket in AGE_BUCKETS]
            errors = np.vstack(
                (np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means))
            )
            axis.errorbar(
                x,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[arm],
                label=arm.replace("_", " "),
            )
        axis.set_title(title)
        axis.set_xticks(x, AGE_BUCKETS)
        axis.set_xlabel("Replay age (episodes)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Episode-level mean")
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "v29_replay_age_profiles.png", dpi=300)
    fig.savefig(output_dir / "v29_replay_age_profiles.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "control_n1": Path(args.control),
        "treatment_n5": Path(args.treatment),
    }
    all_rows = []
    summary = {
        "status": "POST_HOC_READOUT_ONLY",
        "statistical_unit": "training episode",
        "age_contrast": ">=30 episodes minus 0-4 episodes",
        "q_y5_gap_semantics": (
            "Absolute gap between the current conservative Critic and Y5 before "
            "the update; it is not a ground-truth TD residual."
        ),
        "arms": {},
    }
    for arm, path in paths.items():
        rows = load_rows(path, arm, args.episode_start)
        all_rows.extend(rows)
        summary["arms"][arm] = summarize_arm(rows)
    write_csv(output_dir / "per_episode_age_bucket.csv", all_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plot_profiles(summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
