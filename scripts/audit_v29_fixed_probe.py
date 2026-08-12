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

from compare_common import torch_load_checkpoint
from train_centralized_td3 import TwinCritic


DEFAULT_CONTROL = (
    "results/tcom_dynamic_u2u_sparse_v29_control_n1/pilot/seed1/"
    "joint_td3_sac/checkpoints/eval_candidates"
)
DEFAULT_TREATMENT = (
    "results/tcom_dynamic_u2u_sparse_v29_treatment_n5/pilot/seed1/"
    "joint_td3_sac/checkpoints/eval_candidates"
)
DEFAULT_DATASET = (
    "result_archive/v28_td_vs_mc_counterfactual/test_transitions.npz"
)
DEFAULT_RETURNS = (
    "result_archive/v28_td_vs_mc_counterfactual/test_paired_returns.csv"
)
DEFAULT_OUTPUT = (
    "result_archive/v29_online_nstep_pilot/fixed_probe_progression"
)
CHECKPOINT_EPISODES = (20, 40, 60, 70, 80, 90, 100)
MAIN_WINDOW = (60, 70, 80, 90, 100)
ARMS = ("control_n1", "treatment_n5")
POLICIES = ("actor", "centroid")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate v29 checkpoints on the frozen v28 probe set."
    )
    parser.add_argument("--control-dir", default=DEFAULT_CONTROL)
    parser.add_argument("--treatment-dir", default=DEFAULT_TREATMENT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--returns", default=DEFAULT_RETURNS)
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


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return [mean, mean, mean]
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return [mean, mean - critical * sem, mean + critical * sem]


def checkpoint_path(directory, episode):
    return directory / f"eval_ep_{int(episode):04d}.pt"


def load_critic(path, state_dim, action_dim, device):
    checkpoint = torch_load_checkpoint(str(path))
    checkpoint_args = checkpoint["args"]
    critic = TwinCritic(
        state_dim,
        action_dim,
        hidden_dim=int(checkpoint_args["td3_hidden_dim"]),
    ).to(device)
    critic.load_state_dict(checkpoint["macro_delay_critic"])
    critic.eval()
    return critic


def frozen_probe_rows(dataset_path, returns_path):
    with np.load(dataset_path) as payload:
        data = {key: payload[key].copy() for key in payload.files}
    first = np.asarray(data["branch_step"], dtype=np.int64) == 0
    returns = {
        (
            int(row["episode_seed"]),
            row["snapshot_policy"],
            int(row["fork_macro_index"]),
        ): float(row["g30_actor_minus_zero"])
        for row in read_csv(returns_path)
    }
    rows = {}
    for index in np.flatnonzero(first):
        key = (
            int(data["episode_seed"][index]),
            str(data["snapshot_policy"][index]),
            int(data["fork_macro_index"][index]),
        )
        branch_action = str(data["branch_action"][index])
        rows.setdefault(
            key,
            {
                "state": np.asarray(data["state"][index], dtype=np.float32),
                "true_delta": returns[key],
            },
        )[f"action_{branch_action}"] = np.asarray(
            data["action"][index], dtype=np.float32
        )
    for key, row in rows.items():
        if "action_actor" not in row or "action_zero" not in row:
            raise RuntimeError(f"Frozen probe pair is incomplete: {key}")
    return rows


def sign_agreement(predicted, observed):
    return float(np.sign(predicted) == np.sign(observed))


def evaluate_checkpoint(critic, probe, arm, episode, device):
    keys = sorted(probe)
    states = torch.as_tensor(
        np.stack([probe[key]["state"] for key in keys]),
        dtype=torch.float32,
        device=device,
    )
    actor_actions = torch.as_tensor(
        np.stack([probe[key]["action_actor"] for key in keys]),
        dtype=torch.float32,
        device=device,
    )
    zero_actions = torch.as_tensor(
        np.stack([probe[key]["action_zero"] for key in keys]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        q1_actor, q2_actor = critic(states, actor_actions)
        q1_zero, q2_zero = critic(states, zero_actions)
    q1_delta = (q1_actor - q1_zero).squeeze(-1).cpu().numpy()
    q2_delta = (q2_actor - q2_zero).squeeze(-1).cpu().numpy()
    qmin_delta = (
        torch.minimum(q1_actor, q2_actor)
        - torch.minimum(q1_zero, q2_zero)
    ).squeeze(-1).cpu().numpy()
    rows = []
    for index, key in enumerate(keys):
        seed, policy, fork = key
        true_delta = float(probe[key]["true_delta"])
        rows.append(
            {
                "arm": arm,
                "checkpoint_episode": int(episode),
                "episode_seed": int(seed),
                "snapshot_policy": policy,
                "fork_macro_index": int(fork),
                "q1_delta": float(q1_delta[index]),
                "q2_delta": float(q2_delta[index]),
                "qmin_delta": float(qmin_delta[index]),
                "true_delta_g30": true_delta,
                "q1_agreement": sign_agreement(q1_delta[index], true_delta),
                "q2_agreement": sign_agreement(q2_delta[index], true_delta),
                "qmin_agreement": sign_agreement(qmin_delta[index], true_delta),
                "twin_gap_actor_minus_zero": float(
                    abs(q1_delta[index] - q2_delta[index])
                ),
            }
        )
    return rows


def aggregate_episode_checkpoint(rows):
    grouped = {}
    for row in rows:
        key = (
            row["arm"],
            int(row["checkpoint_episode"]),
            int(row["episode_seed"]),
            row["snapshot_policy"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    value_fields = (
        "q1_delta",
        "q2_delta",
        "qmin_delta",
        "true_delta_g30",
        "q1_agreement",
        "q2_agreement",
        "qmin_agreement",
        "twin_gap_actor_minus_zero",
    )
    for key, selected in sorted(grouped.items()):
        if len(selected) != 6:
            raise RuntimeError(f"Expected six fork points for {key}.")
        row = {
            "arm": key[0],
            "checkpoint_episode": key[1],
            "episode_seed": key[2],
            "snapshot_policy": key[3],
            "fork_count": len(selected),
        }
        for field in value_fields:
            row[f"{field}_mean"] = float(
                np.mean([item[field] for item in selected])
            )
        result.append(row)
    return result


def aggregate_main_window(episode_rows):
    grouped = {}
    for row in episode_rows:
        if int(row["checkpoint_episode"]) not in MAIN_WINDOW:
            continue
        key = (
            row["arm"],
            int(row["episode_seed"]),
            row["snapshot_policy"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    fields = (
        "q1_agreement_mean",
        "q2_agreement_mean",
        "qmin_agreement_mean",
        "q1_delta_mean",
        "q2_delta_mean",
        "qmin_delta_mean",
        "true_delta_g30_mean",
        "twin_gap_actor_minus_zero_mean",
    )
    for key, selected in sorted(grouped.items()):
        observed = {int(row["checkpoint_episode"]) for row in selected}
        if observed != set(MAIN_WINDOW):
            raise RuntimeError(f"Incomplete main probe window for {key}: {observed}")
        row = {
            "arm": key[0],
            "episode_seed": key[1],
            "snapshot_policy": key[2],
            "checkpoint_count": len(selected),
        }
        for field in fields:
            row[field] = float(np.mean([item[field] for item in selected]))
        result.append(row)
    return result


def paired_mechanism_summary(window_rows):
    keyed = {
        (row["arm"], row["episode_seed"], row["snapshot_policy"]): row
        for row in window_rows
    }
    paired = []
    summary = {}
    for policy in POLICIES:
        deltas = []
        seeds = sorted(
            row["episode_seed"]
            for row in window_rows
            if row["arm"] == "control_n1"
            and row["snapshot_policy"] == policy
        )
        for seed in seeds:
            control = keyed[("control_n1", seed, policy)]
            treatment = keyed[("treatment_n5", seed, policy)]
            delta = (
                treatment["q1_agreement_mean"]
                - control["q1_agreement_mean"]
            )
            deltas.append(delta)
            paired.append(
                {
                    "episode_seed": seed,
                    "snapshot_policy": policy,
                    "q1_agreement_control_n1": control[
                        "q1_agreement_mean"
                    ],
                    "q1_agreement_treatment_n5": treatment[
                        "q1_agreement_mean"
                    ],
                    "q1_agreement_n5_minus_n1": delta,
                }
            )
        summary[policy] = {
            "n_episode_seeds": len(seeds),
            "q1_agreement_n5_minus_n1_ci95": mean_ci95(deltas),
        }
    centroid_ci = summary["centroid"]["q1_agreement_n5_minus_n1_ci95"]
    summary["criterion"] = (
        "Centroid-state Q1 agreement n5-minus-n1 has a paired episode-seed "
        "95% CI lower bound above zero."
    )
    summary["mechanism_pass"] = bool(centroid_ci[1] > 0.0)
    summary["verdict"] = "PASS" if summary["mechanism_pass"] else "FAIL"
    return paired, summary


def draw_figure(episode_rows, paired_rows, output_dir):
    colors = {"control_n1": "#7A7A7A", "treatment_n5": "#2878B5"}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    for axis, policy in zip(axes[:2], POLICIES):
        for arm in ARMS:
            means = []
            lows = []
            highs = []
            for episode in CHECKPOINT_EPISODES:
                values = [
                    row["q1_agreement_mean"]
                    for row in episode_rows
                    if row["arm"] == arm
                    and row["snapshot_policy"] == policy
                    and int(row["checkpoint_episode"]) == episode
                ]
                mean, low, high = mean_ci95(values)
                means.append(mean)
                lows.append(low)
                highs.append(high)
            means = np.asarray(means)
            axis.plot(
                CHECKPOINT_EPISODES,
                means,
                marker="o",
                linewidth=2.0,
                color=colors[arm],
                label=arm,
            )
            axis.fill_between(
                CHECKPOINT_EPISODES,
                lows,
                highs,
                color=colors[arm],
                alpha=0.14,
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.axvspan(60, 100, color="#E8E8E8", alpha=0.35)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Q1 ranking agreement")
        axis.set_title(f"{policy.capitalize()}-state fixed probe")
        axis.legend(frameon=False)

    centroid = [
        row["q1_agreement_n5_minus_n1"]
        for row in paired_rows
        if row["snapshot_policy"] == "centroid"
    ]
    jitter = np.linspace(-0.08, 0.08, len(centroid))
    axes[2].scatter(jitter, centroid, color="#2878B5", alpha=0.72, s=28)
    mean, low, high = mean_ci95(centroid)
    axes[2].errorbar(
        [0.0],
        [mean],
        yerr=[[mean - low], [high - mean]],
        fmt="D",
        color="black",
        capsize=5,
    )
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_xlim(-0.25, 0.25)
    axes[2].set_xticks([])
    axes[2].set_ylabel("Agreement: n=5 minus n=1")
    axes[2].set_title("Paired main-window effect")
    fig.savefig(output_dir / "v29_fixed_probe_progression.png", dpi=300)
    fig.savefig(output_dir / "v29_fixed_probe_progression.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)
    control_dir = project_path(args.control_dir)
    treatment_dir = project_path(args.treatment_dir)
    dataset_path = project_path(args.dataset)
    returns_path = project_path(args.returns)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = frozen_probe_rows(dataset_path, returns_path)
    first = next(iter(probe.values()))
    state_dim = int(first["state"].size)
    action_dim = int(first["action_actor"].size)

    raw_rows = []
    directories = {
        "control_n1": control_dir,
        "treatment_n5": treatment_dir,
    }
    for arm in ARMS:
        for episode in CHECKPOINT_EPISODES:
            path = checkpoint_path(directories[arm], episode)
            if not path.is_file():
                raise FileNotFoundError(path)
            critic = load_critic(path, state_dim, action_dim, device)
            raw_rows.extend(
                evaluate_checkpoint(critic, probe, arm, episode, device)
            )
            print(f"{arm}: fixed probe checkpoint {episode}")

    episode_rows = aggregate_episode_checkpoint(raw_rows)
    window_rows = aggregate_main_window(episode_rows)
    paired_rows, summary = paired_mechanism_summary(window_rows)
    write_csv(output_dir / "per_fork_predictions.csv", raw_rows)
    write_csv(output_dir / "episode_checkpoint_summary.csv", episode_rows)
    write_csv(output_dir / "main_window_episode_summary.csv", window_rows)
    write_csv(output_dir / "paired_n5_minus_n1.csv", paired_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    draw_figure(episode_rows, paired_rows, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
