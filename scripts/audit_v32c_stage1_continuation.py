import argparse
import csv
import json
import math
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

from audit_v32b_actor_energy_lock import evaluate_action_values
from audit_v32b_actor_energy_lock import load_models
from compare_common import make_episode_load_switch_schedule
from run_load_switch_gate import run_gate_episode
from search_position_oracle import load_checkpoint_components
from train_centralized_td3 import td3_macro_policy


ARMS = ("ACTOR", "ZERO_ONCE", "RETURN")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit one-step versus sustained centroid correction."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=(
            "results/tcom_dynamic_load_switch_v32b_formal/formal/seed1/"
            "joint_td3_sac/checkpoints/eval_candidates"
        ),
    )
    parser.add_argument("--episodes", default="100,150,300,450")
    parser.add_argument("--base-seed", type=int, default=41000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--drift-threshold-m", type=float, default=50.0)
    parser.add_argument("--fork-min", type=int, default=1)
    parser.add_argument("--fork-max", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="result_archive/v32c_stage1_continuation_audit",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_episode_list(value):
    episodes = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not episodes:
        raise ValueError("At least one checkpoint episode is required.")
    return episodes


def checkpoint_path(checkpoint_dir, episode):
    return checkpoint_dir / f"eval_ep_{int(episode):04d}.pt"


def centroid_distance_series(trace):
    uav = np.asarray(trace["uav_trajectory_m"], dtype=np.float64)
    centroid = np.asarray(trace["centroid_trajectory_m"], dtype=np.float64)
    return np.mean(np.linalg.norm(uav - centroid, axis=2), axis=1)


def select_fork_index(trace, threshold_m, fork_min, fork_max):
    distances = centroid_distance_series(trace)
    upper = min(int(fork_max), len(distances) - 2)
    for index in range(max(int(fork_min), 0), upper + 1):
        if float(distances[index]) >= float(threshold_m):
            return int(index)
    return None


def discounted_suffix(records, fork_index, gamma):
    rewards = [
        float(record["td3_replay_reward"])
        for record in records[int(fork_index) :]
    ]
    return float(
        sum((float(gamma) ** offset) * reward for offset, reward in enumerate(rewards))
    )


def recovery_period(trace, fork_index, tolerance_m=20.0):
    distances = centroid_distance_series(trace)
    for index in range(int(fork_index), len(distances)):
        if float(distances[index]) <= float(tolerance_m):
            return float(index - int(fork_index))
    return float("nan")


class BranchPolicy:
    def __init__(self, actor_policy, fork_index, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown continuation arm: {arm}")
        self.actor_policy = actor_policy
        self.fork_index = int(fork_index)
        self.arm = arm
        self.macro_index = 0
        self.fork_state = None
        self.fork_actor_action = None

    def __call__(self, env, local_obs_batch, deterministic):
        actor_action = np.asarray(
            self.actor_policy(env, local_obs_batch, deterministic),
            dtype=np.float32,
        )
        if self.macro_index == self.fork_index:
            _, global_state = env.get_macro_state()
            self.fork_state = np.asarray(global_state, dtype=np.float32).reshape(-1)
            self.fork_actor_action = actor_action.copy().reshape(-1)

        use_zero = (
            self.arm == "ZERO_ONCE" and self.macro_index == self.fork_index
        ) or (self.arm == "RETURN" and self.macro_index >= self.fork_index)
        action = np.zeros_like(actor_action) if use_zero else actor_action
        self.macro_index += 1
        return action


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def paired_stats(rows, field, left_arm, right_arm):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), {})[row["arm"]] = float(row[field])
    values = np.asarray(
        [
            grouped[seed][left_arm] - grouped[seed][right_arm]
            for seed in sorted(grouped)
            if left_arm in grouped[seed] and right_arm in grouped[seed]
        ],
        dtype=np.float64,
    )
    mean, low, high = mean_ci95(values)
    return {
        "left": left_arm,
        "right": right_arm,
        "field": field,
        "count": int(values.size),
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "values": values.tolist(),
    }


def classify_checkpoint(summary):
    q = summary["q_actor_minus_zero"]
    one = summary["return_zero_once_minus_actor"]
    sustained = summary["return_sustained_minus_actor"]
    delay = summary["delay_sustained_minus_actor_ms"]
    q_actor = q["ci95_low"] > 0.0
    one_actor = one["ci95_high"] < 0.0
    one_zero = one["ci95_low"] > 0.0
    sustained_better = (
        sustained["ci95_low"] > 0.0 and delay["ci95_high"] < 0.0
    )
    if q_actor and one_zero:
        return "ONE_STEP_CRITIC_RANKING_ERROR"
    if q_actor and one_actor and sustained_better:
        return "ON_POLICY_SELF_CONSISTENCY_TRAP"
    if sustained["ci95_high"] <= 0.0 and delay["ci95_low"] >= 0.0:
        return "SUSTAINED_RECOVERY_NOT_VALUABLE"
    return "INCONCLUSIVE_OR_MIXED"


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_checkpoint(path, args):
    (
        resolved,
        checkpoint_args,
        env,
        actor,
        micro_policy,
        device,
    ) = load_checkpoint_components(path, args.device)
    actor_policy = td3_macro_policy(actor)
    gamma = float(getattr(checkpoint_args, "td3_gamma", 0.99))
    raw_checkpoint = torch.load(resolved, map_location=device, weights_only=False)
    _, initial_global_state = env.get_macro_state()
    state_dim = int(np.asarray(initial_global_state).size)
    action_dim = int(env.N) * 2
    _, delay_critic, energy_critic = load_models(
        raw_checkpoint,
        state_dim,
        action_dim,
        device,
    )

    rows = []
    probe_rows = []
    excluded = []
    for seed in range(int(args.base_seed), int(args.base_seed) + int(args.eval_episodes)):
        schedule = make_episode_load_switch_schedule(
            checkpoint_args,
            seed,
            int(env.N),
        )
        baseline_row, baseline_trace = run_gate_episode(
            env,
            checkpoint_args,
            micro_policy,
            "ACTOR_SCAN",
            seed,
            macro_steps=int(checkpoint_args.macro_steps),
            load_switch_schedule=schedule,
            macro_policy=actor_policy,
        )
        fork_index = select_fork_index(
            baseline_trace,
            args.drift_threshold_m,
            args.fork_min,
            args.fork_max,
        )
        if fork_index is None:
            excluded.append(int(seed))
            continue

        arm_payload = {}
        for arm in ARMS:
            policy = BranchPolicy(actor_policy, fork_index, arm)
            row, trace = run_gate_episode(
                env,
                checkpoint_args,
                micro_policy,
                arm,
                seed,
                macro_steps=int(checkpoint_args.macro_steps),
                load_switch_schedule=schedule,
                macro_policy=policy,
            )
            if policy.fork_state is None or policy.fork_actor_action is None:
                raise RuntimeError("Branch policy did not capture its fork state.")
            arm_payload[arm] = (row, trace, policy)

        hashes = {arm: arm_payload[arm][0]["external_trace_hash"] for arm in ARMS}
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"CRN mismatch for seed {seed}: {hashes}")
        actor_replay = arm_payload["ACTOR"][0]
        if not math.isclose(
            float(actor_replay["task_delay_weighted_ms"]),
            float(baseline_row["task_delay_weighted_ms"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("ACTOR branch failed to reproduce the scan rollout.")

        fork_state = arm_payload["ACTOR"][2].fork_state
        actor_action = arm_payload["ACTOR"][2].fork_actor_action
        zero_action = np.zeros_like(actor_action)
        if fork_state.size != state_dim or actor_action.size != action_dim:
            raise RuntimeError("Fork state/action dimensions changed within replay.")
        q_delay, _ = evaluate_action_values(
            np.stack([fork_state, fork_state]),
            np.stack([actor_action, zero_action]),
            delay_critic,
            energy_critic,
            str(getattr(checkpoint_args, "td3_actor_delay_aggregation", "q1")),
            device,
            2,
        )
        q_actor = float(q_delay[0])
        q_zero = float(q_delay[1])
        checkpoint_episode = int(path.stem.rsplit("_", 1)[-1])

        for arm in ARMS:
            row, trace, _policy = arm_payload[arm]
            rows.append(
                {
                    "checkpoint_episode": checkpoint_episode,
                    "seed": int(seed),
                    "fork_index": int(fork_index),
                    "fork_distance_m": float(
                        centroid_distance_series(trace)[fork_index]
                    ),
                    "arm": arm,
                    "discounted_replay_return": discounted_suffix(
                        trace["macro_records"], fork_index, gamma
                    ),
                    "task_delay_weighted_ms": float(
                        row["task_delay_weighted_ms"]
                    ),
                    "average_power_w": float(row["average_power_w"]),
                    "final_centroid_distance_m": float(
                        centroid_distance_series(trace)[-1]
                    ),
                    "recovery_periods_to_20m": recovery_period(
                        trace, fork_index
                    ),
                    "external_trace_hash": row["external_trace_hash"],
                }
            )
        actor_return = discounted_suffix(
            arm_payload["ACTOR"][1]["macro_records"], fork_index, gamma
        )
        zero_once_return = discounted_suffix(
            arm_payload["ZERO_ONCE"][1]["macro_records"], fork_index, gamma
        )
        sustained_return = discounted_suffix(
            arm_payload["RETURN"][1]["macro_records"], fork_index, gamma
        )
        probe_rows.append(
            {
                "checkpoint_episode": checkpoint_episode,
                "seed": int(seed),
                "fork_index": int(fork_index),
                "q_actor": q_actor,
                "q_zero": q_zero,
                "q_actor_minus_zero": q_actor - q_zero,
                "return_actor": actor_return,
                "return_zero_once": zero_once_return,
                "return_sustained": sustained_return,
                "return_zero_once_minus_actor": zero_once_return - actor_return,
                "return_sustained_minus_actor": sustained_return - actor_return,
                "delay_sustained_minus_actor_ms": float(
                    arm_payload["RETURN"][0]["task_delay_weighted_ms"]
                    - arm_payload["ACTOR"][0]["task_delay_weighted_ms"]
                ),
                "q_one_step_sign_agreement": int(
                    np.sign(q_actor - q_zero)
                    == np.sign(actor_return - zero_once_return)
                ),
            }
        )

    if not probe_rows:
        raise RuntimeError(f"No eligible drift forks for checkpoint {path}.")
    q_stats = mean_ci95([row["q_actor_minus_zero"] for row in probe_rows])
    one_stats = mean_ci95(
        [row["return_zero_once_minus_actor"] for row in probe_rows]
    )
    sustained_stats = mean_ci95(
        [row["return_sustained_minus_actor"] for row in probe_rows]
    )
    delay_stats = mean_ci95(
        [row["delay_sustained_minus_actor_ms"] for row in probe_rows]
    )
    summary = {
        "checkpoint": str(resolved),
        "checkpoint_episode": int(path.stem.rsplit("_", 1)[-1]),
        "eligible_seed_count": len(probe_rows),
        "excluded_seeds": excluded,
        "q_actor_minus_zero": dict(zip(("mean", "ci95_low", "ci95_high"), q_stats)),
        "return_zero_once_minus_actor": dict(
            zip(("mean", "ci95_low", "ci95_high"), one_stats)
        ),
        "return_sustained_minus_actor": dict(
            zip(("mean", "ci95_low", "ci95_high"), sustained_stats)
        ),
        "delay_sustained_minus_actor_ms": dict(
            zip(("mean", "ci95_low", "ci95_high"), delay_stats)
        ),
        "q_one_step_sign_agreement": float(
            np.mean([row["q_one_step_sign_agreement"] for row in probe_rows])
        ),
    }
    summary["verdict"] = classify_checkpoint(summary)
    return rows, probe_rows, summary


def draw_summary(summaries, output_dir):
    episodes = [item["checkpoint_episode"] for item in summaries]
    predicted_zero = [
        -item["q_actor_minus_zero"]["mean"] for item in summaries
    ]
    predicted_zero_err = np.asarray(
        [
            [
                item["q_actor_minus_zero"]["ci95_high"]
                - item["q_actor_minus_zero"]["mean"]
                for item in summaries
            ],
            [
                item["q_actor_minus_zero"]["mean"]
                - item["q_actor_minus_zero"]["ci95_low"]
                for item in summaries
            ],
        ],
        dtype=np.float64,
    )
    realized_zero = [
        item["return_zero_once_minus_actor"]["mean"] for item in summaries
    ]
    realized_zero_err = np.asarray(
        [
            [
                item["return_zero_once_minus_actor"]["mean"]
                - item["return_zero_once_minus_actor"]["ci95_low"]
                for item in summaries
            ],
            [
                item["return_zero_once_minus_actor"]["ci95_high"]
                - item["return_zero_once_minus_actor"]["mean"]
                for item in summaries
            ],
        ],
        dtype=np.float64,
    )
    sustained = [item["return_sustained_minus_actor"]["mean"] for item in summaries]
    sustained_err = np.asarray(
        [
            [
                item["return_sustained_minus_actor"]["mean"]
                - item["return_sustained_minus_actor"]["ci95_low"]
                for item in summaries
            ],
            [
                item["return_sustained_minus_actor"]["ci95_high"]
                - item["return_sustained_minus_actor"]["mean"]
                for item in summaries
            ],
        ],
        dtype=np.float64,
    )
    delay = [item["delay_sustained_minus_actor_ms"]["mean"] for item in summaries]
    delay_err = np.asarray(
        [
            [
                item["delay_sustained_minus_actor_ms"]["mean"]
                - item["delay_sustained_minus_actor_ms"]["ci95_low"]
                for item in summaries
            ],
            [
                item["delay_sustained_minus_actor_ms"]["ci95_high"]
                - item["delay_sustained_minus_actor_ms"]["mean"]
                for item in summaries
            ],
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    axes[0].errorbar(
        episodes,
        predicted_zero,
        yerr=predicted_zero_err,
        marker="o",
        capsize=3,
        color="#D55E00",
        label="Critic prediction",
    )
    axes[0].errorbar(
        episodes,
        realized_zero,
        yerr=realized_zero_err,
        marker="s",
        capsize=3,
        color="#2878B5",
        label="Realized return",
    )
    axes[0].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Zero-action advantage")
    axes[0].set_xlabel("Training episode")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(
        episodes,
        sustained,
        yerr=sustained_err,
        marker="o",
        capsize=3,
        color="#2E8B57",
    )
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_ylabel("RETURN - ACTOR\ndiscounted replay return")
    axes[1].set_xlabel("Training episode")
    axes[2].errorbar(
        episodes,
        delay,
        yerr=delay_err,
        marker="o",
        capsize=3,
        color="#A23B72",
    )
    axes[2].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_ylabel("RETURN - ACTOR delay (ms)")
    axes[2].set_xlabel("Training episode")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_continuation_audit.png", dpi=240)
    fig.savefig(output_dir / "stage1_continuation_audit.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = project_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = project_path(args.checkpoint_dir).resolve()
    all_rows = []
    all_probes = []
    summaries = []
    for episode in parse_episode_list(args.episodes):
        path = checkpoint_path(checkpoint_dir, episode)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, probes, summary = run_checkpoint(path, args)
        all_rows.extend(rows)
        all_probes.extend(probes)
        summaries.append(summary)
        print(
            f"[ep {episode}] {summary['verdict']} "
            f"eligible={summary['eligible_seed_count']}"
        )
    write_csv(output_dir / "per_seed_branches.csv", all_rows)
    write_csv(output_dir / "per_fork_q_and_returns.csv", all_probes)
    (output_dir / "checkpoint_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    draw_summary(summaries, output_dir)
    verdict_lines = ["# v32c Stage-1 Continuation Audit", ""]
    for summary in summaries:
        verdict_lines.append(
            f"- ep{summary['checkpoint_episode']}: "
            f"**{summary['verdict']}** "
            f"(eligible {summary['eligible_seed_count']}/"
            f"{args.eval_episodes})"
        )
    (output_dir / "verdict.md").write_text(
        "\n".join(verdict_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
