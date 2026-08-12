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
from train_centralized_td3 import MacroActor
from train_centralized_td3 import TwinCritic


DEFAULT_CHECKPOINT = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_DATASET = (
    "result_archive/v28_td_vs_mc_counterfactual/test_transitions.npz"
)
DEFAULT_RETURNS = (
    "result_archive/v28_td_vs_mc_counterfactual/test_paired_returns.csv"
)
DEFAULT_OUTPUT = (
    "result_archive/v28_td_vs_mc_counterfactual/v28a_target_horizon"
)
HORIZONS = (1, 5, 15)
GAMMA = 0.99


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen v28a forward-only target-horizon decomposition."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
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


def load_frozen_models(checkpoint_path, device):
    payload = torch_load_checkpoint(str(checkpoint_path))
    args = payload["args"]
    state_dim = int(payload["macro_actor"]["net.0.weight"].shape[1])
    action_dim = int(payload["macro_actor"]["net.4.weight"].shape[0])
    hidden_dim = int(args["td3_hidden_dim"])
    actor = MacroActor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(payload["macro_actor"])
    actor.eval()
    critic = TwinCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    critic.load_state_dict(payload["macro_delay_critic_target"])
    critic.eval()
    return actor, critic, state_dim, action_dim


def target_tail_values(actor, critic, states, device, batch_size=512):
    states = np.asarray(states, dtype=np.float32)
    values = []
    with torch.no_grad():
        for start in range(0, states.shape[0], int(batch_size)):
            state_t = torch.as_tensor(
                states[start : start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            action_t = actor(state_t)
            q1, q2 = critic(state_t, action_t)
            values.append(torch.minimum(q1, q2).squeeze(1).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def branch_groups(payload):
    groups = {}
    for index, branch_id in enumerate(payload["branch_id"].astype(str)):
        groups.setdefault(branch_id, []).append(index)
    result = {}
    for branch_id, indices in groups.items():
        ordered = sorted(indices, key=lambda idx: int(payload["branch_step"][idx]))
        result[branch_id] = np.asarray(ordered, dtype=np.int64)
    return result


def build_branch_values(payload, actor, critic, device):
    groups = branch_groups(payload)
    tail_queries = []
    query_keys = []
    for branch_id, indices in groups.items():
        if indices.size < 60:
            raise RuntimeError(f"Test branch {branch_id} has fewer than 60 steps.")
        for horizon in HORIZONS:
            tail_queries.append(payload["next_state"][indices[horizon - 1]])
            query_keys.append((branch_id, horizon))
    tail_values = target_tail_values(actor, critic, tail_queries, device)
    tails = {key: float(value) for key, value in zip(query_keys, tail_values)}

    result = {}
    for branch_id, indices in groups.items():
        rewards = np.asarray(payload["reward"][indices], dtype=np.float64)
        values = {
            "g30": float(
                np.sum(np.power(GAMMA, np.arange(30)) * rewards[:30])
            ),
            "g60": float(
                np.sum(np.power(GAMMA, np.arange(60)) * rewards[:60])
            ),
        }
        for horizon in HORIZONS:
            prefix = float(
                np.sum(
                    np.power(GAMMA, np.arange(horizon)) * rewards[:horizon]
                )
            )
            values[f"y{horizon}"] = (
                prefix + (GAMMA ** horizon) * tails[(branch_id, horizon)]
            )
        result[branch_id] = values
    return result


def paired_rows(return_rows, branch_values):
    result = []
    for row in return_rows:
        fork_state_id = row["fork_state_id"]
        actor = branch_values[f"{fork_state_id}:actor"]
        zero = branch_values[f"{fork_state_id}:zero"]
        observed_g30 = float(row["g30_actor_minus_zero"])
        observed_g60 = float(row["g60_actor_minus_zero"])
        if not np.isclose(
            actor["g30"] - zero["g30"], observed_g30, atol=1e-7
        ):
            raise RuntimeError(f"G30 mismatch for {fork_state_id}.")
        output = {
            "episode_seed": int(row["episode_seed"]),
            "snapshot_policy": row["snapshot_policy"],
            "fork_macro_index": int(row["fork_macro_index"]),
            "fork_state_id": fork_state_id,
            "g30_actor_minus_zero": observed_g30,
            "g60_actor_minus_zero": observed_g60,
            "g30_g60_sign_match": float(
                np.sign(observed_g30) == np.sign(observed_g60)
            ),
        }
        for horizon in HORIZONS:
            delta = actor[f"y{horizon}"] - zero[f"y{horizon}"]
            output[f"y{horizon}_actor_minus_zero"] = delta
            output[f"y{horizon}_ranking_agreement_g30"] = float(
                np.sign(delta) == np.sign(observed_g30)
            )
        result.append(output)
    return result


def aggregate_episode(rows):
    grouped = {}
    for row in rows:
        key = (int(row["episode_seed"]), row["snapshot_policy"])
        grouped.setdefault(key, []).append(row)
    result = []
    fields = ["g30_g60_sign_match"] + [
        f"y{horizon}_ranking_agreement_g30" for horizon in HORIZONS
    ]
    for (seed, policy), episode_rows in sorted(grouped.items()):
        if len(episode_rows) != 6:
            raise RuntimeError(f"Expected six fork states for {seed}/{policy}.")
        output = {
            "episode_seed": seed,
            "snapshot_policy": policy,
            "fork_count": len(episode_rows),
        }
        for field in fields:
            output[f"{field}_mean"] = float(
                np.mean([float(row[field]) for row in episode_rows])
            )
        result.append(output)
    return result


def summarize(episode_rows):
    result = {
        "statistical_unit": "episode",
        "forks_aggregated_within_episode": True,
        "by_snapshot_policy": {},
    }
    for policy in ("actor", "centroid"):
        selected = [row for row in episode_rows if row["snapshot_policy"] == policy]
        policy_result = {
            "episodes": len(selected),
            "h30_h60_sign_agreement_ci95": mean_ci95(
                [row["g30_g60_sign_match_mean"] for row in selected]
            ),
        }
        for horizon in HORIZONS:
            field = f"y{horizon}_ranking_agreement_g30_mean"
            policy_result[f"y{horizon}_ranking_agreement_ci95"] = mean_ci95(
                [row[field] for row in selected]
            )
        result["by_snapshot_policy"][policy] = policy_result
    horizon_gate = all(
        result["by_snapshot_policy"][policy][
            "h30_h60_sign_agreement_ci95"
        ][0]
        >= 0.90
        for policy in ("actor", "centroid")
    )
    result["horizon_stability_gate_pass"] = bool(horizon_gate)
    result["verdict"] = "VALID_TARGET_AUDIT" if horizon_gate else "HORIZON_SENSITIVE"
    return result


def draw_figure(summary, output_dir):
    policies = ("actor", "centroid")
    colors = ("#C82423", "#F39B7F", "#2878B5")
    x = np.arange(len(policies), dtype=np.float64)
    width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for index, (horizon, color) in enumerate(zip(HORIZONS, colors)):
        means = []
        errors = []
        for policy in policies:
            mean, low, high = summary["by_snapshot_policy"][policy][
                f"y{horizon}_ranking_agreement_ci95"
            ]
            means.append(mean)
            errors.append([mean - low, high - mean])
        axes[0].bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=np.asarray(errors).T,
            capsize=4,
            label=f"n={horizon}",
            color=color,
            alpha=0.9,
        )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, policies)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Agreement with G30 ranking")
    axes[0].set_title("Frozen target-Critic horizon audit")
    axes[0].legend(frameon=False)

    means = []
    errors = []
    for policy in policies:
        mean, low, high = summary["by_snapshot_policy"][policy][
            "h30_h60_sign_agreement_ci95"
        ]
        means.append(mean)
        errors.append([mean - low, high - mean])
    axes[1].bar(x, means, yerr=np.asarray(errors).T, capsize=4, color=("#4DBBD5", "#00A087"))
    axes[1].axhline(0.9, color="#C82423", linestyle="--", label="90% gate")
    axes[1].set_xticks(x, policies)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("G30/G60 sign agreement")
    axes[1].set_title("Truncation-horizon stability")
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "v28a_target_horizon.png", dpi=300)
    fig.savefig(output_dir / "v28a_target_horizon.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    dataset_path = project_path(args.dataset)
    returns_path = project_path(args.returns)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actor, critic, state_dim, _action_dim = load_frozen_models(
        checkpoint_path, device
    )
    with np.load(dataset_path) as payload_file:
        payload = {key: payload_file[key].copy() for key in payload_file.files}
    if int(payload["state"].shape[1]) != state_dim:
        raise RuntimeError("Dataset/checkpoint state dimensions do not match.")
    return_rows = read_csv(returns_path)
    branch_values = build_branch_values(payload, actor, critic, device)
    state_rows = paired_rows(return_rows, branch_values)
    episode_rows = aggregate_episode(state_rows)
    summary = summarize(episode_rows)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_path),
            "horizons": list(HORIZONS),
            "reference_return": "G30",
            "target_policy": "frozen deployed v24 macro_actor",
            "target_smoothing": False,
        }
    )
    write_csv(output_dir / "per_state_target_horizon.csv", state_rows)
    write_csv(output_dir / "episode_target_horizon.csv", episode_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    draw_figure(summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
