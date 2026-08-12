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
from audit_v29_fixed_probe import evaluate_checkpoint
from audit_v29_fixed_probe import frozen_probe_rows
from audit_v29_fixed_probe import load_critic
import hrl_uav.agents.sac_agent as sac_agent_mod


TRAIN_SEEDS = (1, 42, 123)
EVAL_SEEDS = tuple(range(1000, 1020))
PROBE_CHECKPOINTS = (60, 70, 80, 90, 100)
CONTROL_VERSION = "tcom_dynamic_u2u_sparse_v32a_control_n1"
TREATMENT_VERSION = "tcom_dynamic_u2u_sparse_v32a_treatment_n5"
POWER_BUDGET_W = 840.0
BOOTSTRAP_REPETITIONS = 20000


def parse_args():
    parser = argparse.ArgumentParser(description="Audit the frozen v32a arms.")
    parser.add_argument(
        "--output-dir",
        default="result_archive/v32a_clean_nstep",
    )
    parser.add_argument(
        "--dataset",
        default="result_archive/v28_td_vs_mc_counterfactual/test_transitions.npz",
    )
    parser.add_argument(
        "--returns",
        default="result_archive/v28_td_vs_mc_counterfactual/test_paired_returns.csv",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def checkpoint_path(version, seed, episode=None):
    root = (
        PROJECT_ROOT
        / "results"
        / version
        / "pilot"
        / f"seed{int(seed)}"
        / "joint_td3_sac"
        / "checkpoints"
    )
    if episode is None:
        return root / "final.pt"
    return root / "eval_candidates" / f"eval_ep_{int(episode):04d}.pt"


def diagnostics_path(version, seed):
    return (
        PROJECT_ROOT
        / "results"
        / version
        / "pilot"
        / f"seed{int(seed)}"
        / "joint_td3_sac"
        / "nstep_replay_diagnostics.csv"
    )


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return float(np.mean(values))
    return float(
        np.mean(values)
        + stats.t.ppf(0.95, values.size - 1) * stats.sem(values)
    )


def hierarchical_interval(values_by_seed, rng_seed=32032):
    keys = sorted(values_by_seed)
    rng = np.random.default_rng(rng_seed)
    draws = np.zeros(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        selected_keys = rng.choice(keys, size=len(keys), replace=True)
        seed_means = []
        for key in selected_keys:
            values = np.asarray(values_by_seed[int(key)], dtype=np.float64)
            sampled = rng.choice(values, size=values.size, replace=True)
            seed_means.append(float(np.mean(sampled)))
        draws[index] = float(np.mean(seed_means))
    point = float(
        np.mean([np.mean(values_by_seed[key]) for key in keys])
    )
    return [
        point,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def hash_rows(path):
    rows = read_csv(path)
    result = {}
    for row in rows:
        episode = int(row["episode"])
        value = (
            row["sample_start_sha256"],
            row["target_noise_sha256"],
        )
        if episode in result and result[episode] != value:
            raise RuntimeError(f"Inconsistent hashes inside episode {episode}: {path}")
        result[episode] = value
    return result


def audit_hashes():
    rows = []
    all_match = True
    for seed in TRAIN_SEEDS:
        control = hash_rows(diagnostics_path(CONTROL_VERSION, seed))
        treatment = hash_rows(diagnostics_path(TREATMENT_VERSION, seed))
        if set(control) != set(treatment):
            raise RuntimeError(f"Diagnostic episode sets differ for seed {seed}.")
        for episode in sorted(control):
            start_match = control[episode][0] == treatment[episode][0]
            noise_match = control[episode][1] == treatment[episode][1]
            all_match = all_match and start_match and noise_match
            rows.append(
                {
                    "training_seed": seed,
                    "episode": episode,
                    "sample_start_hash_match": start_match,
                    "target_noise_hash_match": noise_match,
                    "control_sample_start_sha256": control[episode][0],
                    "treatment_sample_start_sha256": treatment[episode][0],
                    "control_target_noise_sha256": control[episode][1],
                    "treatment_target_noise_sha256": treatment[episode][1],
                }
            )
    return rows, bool(all_match)


def audit_fixed_probe(dataset_path, returns_path, device):
    probe = frozen_probe_rows(dataset_path, returns_path)
    first = next(iter(probe.values()))
    state_dim = int(first["state"].size)
    action_dim = int(first["action_actor"].size)
    raw = []
    for training_seed in TRAIN_SEEDS:
        for arm, version in (
            ("control_n1", CONTROL_VERSION),
            ("treatment_n5", TREATMENT_VERSION),
        ):
            for episode in PROBE_CHECKPOINTS:
                critic = load_critic(
                    checkpoint_path(version, training_seed, episode),
                    state_dim,
                    action_dim,
                    device,
                )
                rows = evaluate_checkpoint(
                    critic,
                    probe,
                    arm,
                    episode,
                    device,
                )
                for row in rows:
                    row["training_seed"] = training_seed
                raw.extend(rows)

    grouped = {}
    for row in raw:
        if row["snapshot_policy"] != "centroid":
            continue
        key = (
            int(row["training_seed"]),
            row["arm"],
            int(row["episode_seed"]),
        )
        grouped.setdefault(key, []).append(float(row["q1_agreement"]))

    paired = []
    values_by_seed = {}
    seed_points = {}
    for training_seed in TRAIN_SEEDS:
        deltas = []
        for eval_seed in sorted(
            key[2]
            for key in grouped
            if key[0] == training_seed and key[1] == "control_n1"
        ):
            control = grouped[(training_seed, "control_n1", eval_seed)]
            treatment = grouped[(training_seed, "treatment_n5", eval_seed)]
            if len(control) != 30 or len(treatment) != 30:
                raise RuntimeError("Expected five checkpoints by six fork points.")
            delta = float(np.mean(treatment) - np.mean(control))
            deltas.append(delta)
            paired.append(
                {
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "q1_agreement_control_n1": float(np.mean(control)),
                    "q1_agreement_treatment_n5": float(np.mean(treatment)),
                    "q1_agreement_n5_minus_n1": delta,
                }
            )
        values_by_seed[training_seed] = np.asarray(deltas)
        seed_points[training_seed] = float(np.mean(deltas))
    interval = hierarchical_interval(values_by_seed, rng_seed=32033)
    passed = bool(
        all(value > 0.0 for value in seed_points.values())
        and interval[1] > 0.0
    )
    summary = {
        "seed_point_estimates": seed_points,
        "hierarchical_ci95": interval,
        "all_seed_points_positive": all(
            value > 0.0 for value in seed_points.values()
        ),
        "pass": passed,
    }
    return raw, paired, summary


def check_trace_pair(control, treatment, training_seed, eval_seed):
    fields = ("centroid_trace_hash", "users_trace_hash")
    for field in fields:
        if control[field] != treatment[field]:
            raise RuntimeError(
                f"CRN mismatch: train={training_seed}, eval={eval_seed}, {field}"
            )


def audit_final(device):
    raw = []
    paired = []
    delay_by_seed = {}
    seed_delay_points = {}
    power_ucb = {}
    for training_seed in TRAIN_SEEDS:
        arm_rows = {}
        for arm, version in (
            ("control_n1", CONTROL_VERSION),
            ("treatment_n5", TREATMENT_VERSION),
        ):
            rows = replay_checkpoint(
                arm,
                checkpoint_path(version, training_seed),
                EVAL_SEEDS,
                device,
            )
            for row in rows:
                row["training_seed"] = training_seed
            raw.extend(rows)
            arm_rows[arm] = {int(row["seed"]): row for row in rows}
        deltas = []
        for eval_seed in EVAL_SEEDS:
            control = arm_rows["control_n1"][eval_seed]
            treatment = arm_rows["treatment_n5"][eval_seed]
            check_trace_pair(control, treatment, training_seed, eval_seed)
            delta = float(
                treatment["task_delay_weighted_ms"]
                - control["task_delay_weighted_ms"]
            )
            deltas.append(delta)
            paired.append(
                {
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "delay_control_n1_ms": control["task_delay_weighted_ms"],
                    "delay_treatment_n5_ms": treatment["task_delay_weighted_ms"],
                    "delay_n5_minus_n1_ms": delta,
                    "power_control_n1_w": control["average_power_w"],
                    "power_treatment_n5_w": treatment["average_power_w"],
                    "centroid_distance_control_n1_m": control[
                        "mean_centroid_distance_m"
                    ],
                    "centroid_distance_treatment_n5_m": treatment[
                        "mean_centroid_distance_m"
                    ],
                    "path_control_n1_m": control["path_total_m"],
                    "path_treatment_n5_m": treatment["path_total_m"],
                }
            )
        delay_by_seed[training_seed] = np.asarray(deltas)
        seed_delay_points[training_seed] = float(np.mean(deltas))
        power_ucb[training_seed] = one_sided_ucb95(
            [
                row["average_power_w"]
                for row in arm_rows["treatment_n5"].values()
            ]
        )
    delay_interval = hierarchical_interval(delay_by_seed, rng_seed=32034)
    delay_pass = bool(
        all(value < 0.0 for value in seed_delay_points.values())
        and delay_interval[2] < 0.0
    )
    constraint_pass = bool(
        all(value <= POWER_BUDGET_W for value in power_ucb.values())
    )
    summary = {
        "seed_delay_point_estimates_ms": seed_delay_points,
        "delay_n5_minus_n1_hierarchical_ci95_ms": delay_interval,
        "all_seed_delay_points_negative": all(
            value < 0.0 for value in seed_delay_points.values()
        ),
        "delay_pass": delay_pass,
        "treatment_power_ucb95_w_by_seed": power_ucb,
        "constraint_pass": constraint_pass,
    }
    return raw, paired, summary


def draw_summary(mechanism, final, output_dir):
    seeds = list(TRAIN_SEEDS)
    probe = [mechanism["seed_point_estimates"][seed] for seed in seeds]
    delay = [final["seed_delay_point_estimates_ms"][seed] for seed in seeds]
    power = [final["treatment_power_ucb95_w_by_seed"][seed] for seed in seeds]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    axes[0].bar([str(seed) for seed in seeds], probe, color="#2878B5")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Centroid probe: n=5 - n=1")
    axes[0].set_xlabel("Macro training seed")
    axes[0].set_ylabel("Q1 agreement difference")
    axes[1].bar([str(seed) for seed in seeds], delay, color="#C82423")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Final delay: n=5 - n=1")
    axes[1].set_xlabel("Macro training seed")
    axes[1].set_ylabel("ms")
    axes[2].bar([str(seed) for seed in seeds], power, color="#54A24B")
    axes[2].axhline(POWER_BUDGET_W, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("n=5 power feasibility")
    axes[2].set_xlabel("Macro training seed")
    axes[2].set_ylabel("UCB95 (W)")
    fig.savefig(output_dir / "v32a_summary.png", dpi=300)
    fig.savefig(output_dir / "v32a_summary.pdf")
    plt.close(fig)


def render_verdict(payload):
    mechanism = payload["fixed_probe"]
    final = payload["final_crn"]
    return "\n".join(
        (
            "# v32a clean online n-step verdict",
            "",
            f"- Frozen verdict: **{payload['verdict']}**",
            f"- Replay/target-noise hash gate: `{payload['hash_gate_pass']}`",
            "- Centroid-probe n5-n1 hierarchical 95% CI: "
            f"`{mechanism['hierarchical_ci95']}`",
            "- Delay n5-n1 hierarchical 95% CI (ms): "
            f"`{final['delay_n5_minus_n1_hierarchical_ci95_ms']}`",
            "- n=5 power UCB95 by training seed (W): "
            f"`{final['treatment_power_ucb95_w_by_seed']}`",
            "",
            "Only FULL_PASS authorizes n=5 in v32b. Every other valid verdict "
            "closes the n-step line and retains n=1.",
            "",
        )
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    sac_agent_mod.device = device
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixed_probe").mkdir(exist_ok=True)
    (output_dir / "final_crn").mkdir(exist_ok=True)

    pipeline_path = output_dir / "pipeline_gate.json"
    if not pipeline_path.is_file():
        raise FileNotFoundError(pipeline_path)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    hash_rows_output, hash_pass = audit_hashes()
    write_csv(output_dir / "replay_hash_audit.csv", hash_rows_output)

    probe_raw, probe_paired, mechanism = audit_fixed_probe(
        project_path(args.dataset),
        project_path(args.returns),
        device,
    )
    write_csv(output_dir / "fixed_probe" / "per_fork.csv", probe_raw)
    write_csv(output_dir / "fixed_probe" / "paired_episode.csv", probe_paired)
    (output_dir / "fixed_probe" / "summary.json").write_text(
        json.dumps(mechanism, indent=2), encoding="utf-8"
    )

    final_raw, final_paired, final = audit_final(device)
    write_csv(output_dir / "final_crn" / "per_episode.csv", final_raw)
    write_csv(output_dir / "final_crn" / "paired_episode.csv", final_paired)
    (output_dir / "final_crn" / "summary.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    engineering_pass = bool(pipeline["status"] == "PASS" and hash_pass)
    if not engineering_pass:
        verdict = "PIPELINE_INVALID"
    elif not final["constraint_pass"]:
        verdict = "CONSTRAINT_FAIL"
    elif mechanism["pass"] and final["delay_pass"]:
        verdict = "FULL_PASS"
    elif mechanism["pass"]:
        verdict = "MECHANISM_ONLY"
    else:
        verdict = "NSTEP_REJECTED"
    payload = {
        "verdict": verdict,
        "pipeline_gate": pipeline,
        "hash_gate_pass": hash_pass,
        "fixed_probe": mechanism,
        "final_crn": final,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "verdict.md").write_text(
        render_verdict(payload), encoding="utf-8"
    )
    draw_summary(mechanism, final, output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
