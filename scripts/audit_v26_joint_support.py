import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import torch_load_checkpoint
from train_centralized_td3 import MacroActor
from train_centralized_td3 import TwinCritic


STATE_LABELS = ("centroid <=20m", "transition 20-50m", "drift >50m")
ACTION_LABELS = ("near-zero <=0.10", "medium 0.10-0.40", "large >0.40")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit state-action support and deterministic Bellman residuals "
            "from an exported TD3 macro replay."
        )
    )
    parser.add_argument("--replay", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def state_distance_m(states, num_agents):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] % int(num_agents) != 0:
        raise ValueError("Global state cannot be split evenly across UAVs.")
    local_dim = states.shape[1] // int(num_agents)
    local_states = states.reshape(-1, int(num_agents), local_dim)
    delta_centroid = local_states[:, :, :2]
    return 1000.0 * np.mean(
        np.linalg.norm(delta_centroid, axis=2),
        axis=1,
    )


def action_norm(actions, num_agents):
    actions = np.asarray(actions, dtype=np.float32)
    return np.mean(
        np.linalg.norm(actions.reshape(-1, int(num_agents), 2), axis=2),
        axis=1,
    )


def state_bins(distances_m):
    epsilon = 1e-7
    return np.where(
        distances_m <= 20.0 + epsilon,
        0,
        np.where(distances_m <= 50.0 + epsilon, 1, 2),
    ).astype(np.int64)


def action_bins(norms):
    epsilon = 1e-7
    return np.where(
        norms <= 0.10 + epsilon,
        0,
        np.where(norms <= 0.40 + epsilon, 1, 2),
    ).astype(np.int64)


def load_models(checkpoint, state_dim, action_dim, num_agents, device):
    args = checkpoint["args"]
    hidden_dim = int(args.get("td3_hidden_dim", 256))
    actor = MacroActor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(checkpoint["macro_actor"])
    actor.eval()

    target_actor = MacroActor(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    target_actor.load_state_dict(
        checkpoint.get("macro_actor_target", checkpoint["macro_actor"])
    )
    target_actor.eval()

    critic = TwinCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    critic.load_state_dict(
        checkpoint.get("macro_delay_critic", checkpoint["macro_critic"])
    )
    critic.eval()

    target_critic = TwinCritic(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    target_critic.load_state_dict(
        checkpoint.get(
            "macro_delay_critic_target",
            checkpoint.get("macro_critic_target", checkpoint["macro_critic"]),
        )
    )
    target_critic.eval()
    return actor, target_actor, critic, target_critic


def critic_diagnostics(
    replay,
    checkpoint,
    num_agents,
    device,
    batch_size,
):
    states = replay["state"]
    actions = replay["action"]
    rewards = replay["reward"]
    next_states = replay["next_state"]
    dones = replay["done"]
    actor, target_actor, critic, target_critic = load_models(
        checkpoint,
        states.shape[1],
        actions.shape[1],
        num_agents,
        device,
    )
    gamma = float(checkpoint["args"].get("td3_gamma", 0.99))

    outputs = {
        "q1": [],
        "q2": [],
        "target": [],
        "actor_action": [],
        "q_actor_minus_zero": [],
        "actor_twin_gap": [],
        "zero_twin_gap": [],
    }
    for start in range(0, states.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), states.shape[0])
        state_t = torch.as_tensor(states[start:stop], device=device)
        action_t = torch.as_tensor(actions[start:stop], device=device)
        reward_t = torch.as_tensor(rewards[start:stop], device=device)
        next_state_t = torch.as_tensor(next_states[start:stop], device=device)
        done_t = torch.as_tensor(dones[start:stop], device=device)
        with torch.no_grad():
            q1, q2 = critic(state_t, action_t)
            next_action = target_actor(next_state_t)
            target_q1, target_q2 = target_critic(next_state_t, next_action)
            target = reward_t + (1.0 - done_t) * gamma * torch.minimum(
                target_q1,
                target_q2,
            )
            actor_action_t = actor(state_t)
            zero_action_t = torch.zeros_like(actor_action_t)
            actor_q1, actor_q2 = critic(state_t, actor_action_t)
            zero_q1, zero_q2 = critic(state_t, zero_action_t)
            actor_advantage = torch.minimum(
                actor_q1,
                actor_q2,
            ) - torch.minimum(zero_q1, zero_q2)
        for key, value in (
            ("q1", q1),
            ("q2", q2),
            ("target", target),
            ("actor_action", actor_action_t),
            ("q_actor_minus_zero", actor_advantage),
            ("actor_twin_gap", torch.abs(actor_q1 - actor_q2)),
            ("zero_twin_gap", torch.abs(zero_q1 - zero_q2)),
        ):
            outputs[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else None


def cell_rows(replay, diagnostics, state_index, action_index):
    total = int(len(state_index))
    q1 = diagnostics["q1"].reshape(-1)
    q2 = diagnostics["q2"].reshape(-1)
    target = diagnostics["target"].reshape(-1)
    q_min = np.minimum(q1, q2)
    rows = []
    for state_bin in range(3):
        for action_bin in range(3):
            mask = (state_index == state_bin) & (action_index == action_bin)
            rows.append(
                {
                    "state_region": STATE_LABELS[state_bin],
                    "action_region": ACTION_LABELS[action_bin],
                    "count": int(np.sum(mask)),
                    "fraction": float(np.mean(mask)),
                    "neutral_episode_fraction": safe_mean(
                        replay["neutral_episode"][mask]
                    ),
                    "twin_gap_mean": safe_mean(np.abs(q1[mask] - q2[mask])),
                    "bellman_residual_signed_mean": safe_mean(
                        q_min[mask] - target[mask]
                    ),
                    "bellman_residual_abs_mean": safe_mean(
                        np.abs(q_min[mask] - target[mask])
                    ),
                    "q_actor_minus_zero_mean": safe_mean(
                        diagnostics["q_actor_minus_zero"].reshape(-1)[mask]
                    ),
                    "total_replay_size": total,
                }
            )
    return rows


def matrix_from_rows(rows, field):
    values = np.full((3, 3), np.nan, dtype=np.float64)
    for state_bin in range(3):
        for action_bin in range(3):
            value = rows[state_bin * 3 + action_bin][field]
            if value is not None:
                values[state_bin, action_bin] = float(value)
    return values


def draw_heatmaps(rows, output_dir):
    panels = (
        ("count", "Replay sample count", "Blues", ".0f"),
        ("twin_gap_mean", "Mean twin-Critic gap", "Reds", ".4f"),
        (
            "bellman_residual_abs_mean",
            "Mean absolute Bellman residual",
            "Oranges",
            ".4f",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for axis, (field, title, cmap, value_format) in zip(axes, panels):
        matrix = matrix_from_rows(rows, field)
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(3), ACTION_LABELS, rotation=24, ha="right")
        axis.set_yticks(range(3), STATE_LABELS)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                label = "NA" if np.isnan(value) else format(value, value_format)
                axis.text(column, row, label, ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(output_dir, "joint_support_heatmaps.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "joint_support_heatmaps.pdf"))
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with np.load(args.replay) as payload:
        replay = {key: payload[key].copy() for key in payload.files}
    checkpoint = torch_load_checkpoint(args.checkpoint)
    num_agents = int(checkpoint["args"].get("num_agents", 3))
    distances_m = state_distance_m(replay["state"], num_agents)
    norms = action_norm(replay["action"], num_agents)
    state_index = state_bins(distances_m)
    action_index = action_bins(norms)
    diagnostics = critic_diagnostics(
        replay,
        checkpoint,
        num_agents,
        device,
        args.batch_size,
    )
    rows = cell_rows(replay, diagnostics, state_index, action_index)
    write_csv(os.path.join(args.output_dir, "joint_support_cells.csv"), rows)
    draw_heatmaps(rows, args.output_dir)

    summary = {
        "replay": os.path.abspath(args.replay),
        "checkpoint": os.path.abspath(args.checkpoint),
        "replay_size": int(replay["state"].shape[0]),
        "state_dim": int(replay["state"].shape[1]),
        "action_dim": int(replay["action"].shape[1]),
        "episode_range": [
            int(np.min(replay["episode_index"])),
            int(np.max(replay["episode_index"])),
        ],
        "neutral_transition_fraction": float(
            np.mean(replay["neutral_episode"])
        ),
        "state_distance_m_mean": float(np.mean(distances_m)),
        "action_norm_mean": float(np.mean(norms)),
        "actor_action_norm_mean": float(
            np.mean(action_norm(diagnostics["actor_action"], num_agents))
        ),
        "twin_gap_replay_action_mean": float(
            np.mean(np.abs(diagnostics["q1"] - diagnostics["q2"]))
        ),
        "bellman_residual_abs_mean": float(
            np.mean(
                np.abs(
                    np.minimum(diagnostics["q1"], diagnostics["q2"])
                    - diagnostics["target"]
                )
            )
        ),
        "q_actor_minus_zero_mean": float(
            np.mean(diagnostics["q_actor_minus_zero"])
        ),
        "actor_twin_gap_mean": float(
            np.mean(diagnostics["actor_twin_gap"])
        ),
        "zero_twin_gap_mean": float(
            np.mean(diagnostics["zero_twin_gap"])
        ),
        "state_regions": list(STATE_LABELS),
        "action_regions": list(ACTION_LABELS),
        "bellman_residual_note": (
            "Read-only deterministic target without TD3 target smoothing; "
            "this is not a Monte Carlo Q error."
        ),
    }
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
