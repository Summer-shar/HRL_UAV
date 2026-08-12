import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


METRICS = (
    "return_actor_minus_zero",
    "q1_actor_minus_zero",
    "q2_actor_minus_zero",
    "qmin_actor_minus_zero",
    "ranking_agreement",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize same-state Critic probes across checkpoints."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", default="20,40,60,80,100")
    return parser.parse_args()


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    margin = float(
        student_t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / np.sqrt(values.size)
    )
    return mean, mean - margin, mean + margin


def read_checkpoint_rows(input_root, checkpoint_episode):
    path = os.path.join(
        input_root,
        f"ep_{checkpoint_episode:04d}",
        "branch_raw.csv",
    )
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No probe rows found in {path}")
    return rows


def aggregate_episodes(rows):
    grouped = {}
    for row in rows:
        key = (row["snapshot_policy"], int(row["episode_index"]))
        grouped.setdefault(key, []).append(row)

    result = []
    for (policy, episode_index), episode_rows in sorted(grouped.items()):
        q1 = np.asarray(
            [float(row["q1_actor"]) - float(row["q1_zero"]) for row in episode_rows]
        )
        q2 = np.asarray(
            [float(row["q2_actor"]) - float(row["q2_zero"]) for row in episode_rows]
        )
        aggregate = {
            "snapshot_policy": policy,
            "episode_index": episode_index,
            "return_actor_minus_zero": float(
                np.mean([float(row["return_actor_minus_zero"]) for row in episode_rows])
            ),
            "q1_actor_minus_zero": float(np.mean(q1)),
            "q2_actor_minus_zero": float(np.mean(q2)),
            "qmin_actor_minus_zero": float(
                np.mean([float(row["q_actor_minus_zero"]) for row in episode_rows])
            ),
            "ranking_agreement": float(
                np.mean([float(row["ranking_agreement"]) for row in episode_rows])
            ),
        }
        result.append(aggregate)
    return result


def summarize(input_root, checkpoint_episodes):
    raw_summary = []
    for checkpoint_episode in checkpoint_episodes:
        episode_rows = aggregate_episodes(
            read_checkpoint_rows(input_root, checkpoint_episode)
        )
        policies = sorted({row["snapshot_policy"] for row in episode_rows})
        if policies != ["centroid"]:
            raise ValueError(
                f"Checkpoint {checkpoint_episode} must contain centroid probes only; "
                f"found {policies}"
            )
        row = {
            "checkpoint_episode": checkpoint_episode,
            "probe_episodes": len(episode_rows),
        }
        for metric in METRICS:
            mean, low, high = mean_ci95([item[metric] for item in episode_rows])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        raw_summary.append(row)
    return raw_summary


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows, output_dir):
    episodes = np.asarray([row["checkpoint_episode"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    styles = (
        ("return_actor_minus_zero", "True return", "#202020", "o"),
        ("q1_actor_minus_zero", "Q1", "#C82423", "s"),
        ("q2_actor_minus_zero", "Q2", "#2878B5", "^"),
        ("qmin_actor_minus_zero", "min(Q1,Q2)", "#3B9B6D", "D"),
    )
    for metric, label, color, marker in styles:
        means = np.asarray([row[f"{metric}_mean"] for row in rows])
        lows = np.asarray([row[f"{metric}_ci95_low"] for row in rows])
        highs = np.asarray([row[f"{metric}_ci95_high"] for row in rows])
        axes[0].plot(episodes, means, color=color, marker=marker, label=label)
        axes[0].fill_between(episodes, lows, highs, color=color, alpha=0.12)
    axes[0].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Actor action minus zero action")
    axes[0].set_title("Same-state Critic ranking")
    axes[0].legend(frameon=False)

    agreement = np.asarray(
        [row["ranking_agreement_mean"] for row in rows]
    )
    agreement_low = np.asarray(
        [row["ranking_agreement_ci95_low"] for row in rows]
    )
    agreement_high = np.asarray(
        [row["ranking_agreement_ci95_high"] for row in rows]
    )
    axes[1].plot(episodes, agreement * 100.0, color="#6F4E7C", marker="o")
    axes[1].fill_between(
        episodes,
        agreement_low * 100.0,
        agreement_high * 100.0,
        color="#6F4E7C",
        alpha=0.15,
    )
    axes[1].axhline(50.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Ranking agreement (%)")
    axes[1].set_title("Critic agreement with true return")

    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(output_dir, f"critic_checkpoint_progression.{extension}"),
            dpi=300,
        )
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint_episodes = [
        int(value.strip()) for value in args.episodes.split(",") if value.strip()
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    rows = summarize(args.input_root, checkpoint_episodes)
    write_csv(os.path.join(args.output_dir, "checkpoint_progression.csv"), rows)
    with open(
        os.path.join(args.output_dir, "checkpoint_progression.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    draw_figure(rows, args.output_dir)


if __name__ == "__main__":
    main()
