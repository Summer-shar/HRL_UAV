"""Scan v33 macro actors while holding the final v33 micro SAC fixed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_joint_hrl_v1_matched import mean_ci95, replay
from audit_v42_macro_only_pilot import load_policies
from search_position_oracle import make_centroid_offset_policy


DEFAULT_ROOT = (
    ROOT
    / "results"
    / "tcom_v33_soft_energy_random_mobility"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
)
DEFAULT_FINAL = DEFAULT_ROOT / "checkpoints" / "final.pt"
DEFAULT_CANDIDATES = DEFAULT_ROOT / "checkpoints" / "eval_candidates"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v33_macro_checkpoint_scan"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_actor(
    checkpoint_args,
    macro_policy,
    micro_policy,
    centroid_by_seed: dict[int, dict],
    seeds: list[int],
) -> tuple[list[dict], dict]:
    rows = []
    for seed in seeds:
        metrics = replay(checkpoint_args, macro_policy, micro_policy, seed)
        baseline = centroid_by_seed[seed]
        rows.append({
            "seed": seed,
            **metrics,
            "delay_delta_ms": (
                metrics["task_delay_weighted_ms"]
                - baseline["task_delay_weighted_ms"]
            ),
            "J_delta": metrics["J"] - baseline["J"],
            "queue_delta_mbits": metrics["queue_mbits"] - baseline["queue_mbits"],
        })
    summary = {
        "delay": mean_ci95([row["delay_delta_ms"] for row in rows]),
        "J": mean_ci95([row["J_delta"] for row in rows]),
        "queue": mean_ci95([row["queue_delta_mbits"] for row in rows]),
        "mean_delay_ms": float(np.mean([
            row["task_delay_weighted_ms"] for row in rows
        ])),
        "mean_u2g_distance_m": float(np.mean([
            row["u2g_distance_m"] for row in rows
        ])),
        "faster_count": int(np.sum([
            row["delay_delta_ms"] < 0.0 for row in rows
        ])),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-checkpoint", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episodes",
        default="50,100,200,300,400,500,600,650,700,800,900,1000",
    )
    parser.add_argument("--seeds", default="6000,6001,6002")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cli = parser.parse_args()

    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = parse_int_list(cli.episodes)
    seeds = parse_int_list(cli.seeds)
    final_args, _, final_micro = load_policies(
        cli.final_checkpoint.resolve(), cli.device
    )
    centroid_policy = make_centroid_offset_policy(np.zeros((3, 2)))
    centroid_by_seed = {
        seed: replay(final_args, centroid_policy, final_micro, seed)
        for seed in seeds
    }

    raw_rows: list[dict] = []
    summaries: list[dict] = []
    for episode in episodes:
        checkpoint = (
            cli.candidate_dir.resolve() / f"eval_ep_{episode:04d}.pt"
        )
        candidate_args, macro_policy, _ = load_policies(checkpoint, cli.device)
        if int(candidate_args.num_agents) != int(final_args.num_agents):
            raise ValueError("Candidate and final checkpoints disagree on UAV count")
        rows, summary = evaluate_actor(
            final_args,
            macro_policy,
            final_micro,
            centroid_by_seed,
            seeds,
        )
        raw_rows.extend({"episode": episode, **row} for row in rows)
        summaries.append({
            "episode": episode,
            "mean_delay_ms": summary["mean_delay_ms"],
            "mean_delay_delta_ms": summary["delay"]["mean"],
            "delay_ci95_low": summary["delay"]["ci95_low"],
            "delay_ci95_high": summary["delay"]["ci95_high"],
            "mean_J_delta": summary["J"]["mean"],
            "mean_queue_delta_mbits": summary["queue"]["mean"],
            "mean_u2g_distance_m": summary["mean_u2g_distance_m"],
            "faster_count": summary["faster_count"],
            "seed_count": len(seeds),
        })
        print(
            f"episode={episode:04d} delay={summary['mean_delay_ms']:.2f}ms "
            f"delta={summary['delay']['mean']:+.2f}ms "
            f"faster={summary['faster_count']}/{len(seeds)} "
            f"u2g={summary['mean_u2g_distance_m']:.1f}m"
        )

    summaries.sort(key=lambda row: row["mean_delay_delta_ms"])
    write_csv(output / "search_raw.csv", raw_rows)
    write_csv(output / "search_summary.csv", summaries)
    report = {
        "final_micro_checkpoint": str(cli.final_checkpoint.resolve()),
        "search_seeds": seeds,
        "centroid_mean_delay_ms": float(np.mean([
            metrics["task_delay_weighted_ms"]
            for metrics in centroid_by_seed.values()
        ])),
        "ranked_candidates": summaries,
        "best_episode": int(summaries[0]["episode"]),
    }
    (output / "search_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
