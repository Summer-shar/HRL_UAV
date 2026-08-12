import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import torch_load_checkpoint
from train_centralized_td3 import MacroActor
from train_centralized_td3 import TwinCritic


DEFAULT_RESULTS = (
    "results/tcom_dynamic_load_switch_v32b_formal/formal/seed1/"
    "joint_td3_sac"
)
DEFAULT_OUTPUT = "result_archive/v32b_seed1_actor_energy_lock"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the frozen TD3 Actor objective on common drift-state "
            "probes for Actor, centroid-return, and current-position hover "
            "actions. This is a read-only checkpoint audit."
        )
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-drift-m", type=float, default=100.0)
    parser.add_argument("--max-probes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def project_unit_disk_np(actions, num_agents):
    actions = np.asarray(actions, dtype=np.float32)
    original_shape = actions.shape
    pairs = actions.reshape(-1, int(num_agents), 2)
    norms = np.linalg.norm(pairs, axis=2, keepdims=True)
    projected = pairs / np.maximum(norms, 1.0)
    return projected.reshape(original_shape).astype(np.float32)


def split_local_states(states, num_agents):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] % int(num_agents) != 0:
        raise ValueError("Global states cannot be split evenly across UAVs.")
    return states.reshape(states.shape[0], int(num_agents), -1)


def centroid_delta_m(states, num_agents):
    """Return centroid minus UAV displacement encoded in the macro state."""
    return 1000.0 * split_local_states(states, num_agents)[:, :, :2]


def state_mean_centroid_distance_m(states, num_agents):
    return np.mean(
        np.linalg.norm(centroid_delta_m(states, num_agents), axis=2),
        axis=1,
    )


def hover_actions_from_states(states, num_agents, waypoint_radius_m):
    """Residual that places each centroid-relative waypoint at the current UAV."""
    raw = -centroid_delta_m(states, num_agents) / float(waypoint_radius_m)
    raw_norm = np.linalg.norm(raw, axis=2)
    projected = project_unit_disk_np(raw, num_agents)
    exact = np.all(raw_norm <= 1.0 + 1e-7, axis=1)
    return projected.reshape(len(states), -1), raw_norm, exact


def predicted_movement_m(
    states,
    actions,
    num_agents,
    waypoint_radius_m,
    max_displacement_m,
):
    delta_m = centroid_delta_m(states, num_agents)
    action_pairs = np.asarray(actions, dtype=np.float32).reshape(
        -1, int(num_agents), 2
    )
    to_target = delta_m + action_pairs * float(waypoint_radius_m)
    return np.minimum(
        np.linalg.norm(to_target, axis=2),
        float(max_displacement_m),
    )


def mean_action_norm(actions, num_agents):
    pairs = np.asarray(actions, dtype=np.float32).reshape(
        -1, int(num_agents), 2
    )
    return np.mean(np.linalg.norm(pairs, axis=2), axis=1)


def dual_state_local_index(args):
    index = 2 + 5
    if bool(args.get("macro_flow_state", False)):
        index += 2
    if bool(args.get("macro_load_state", False)):
        index += 1
    if bool(args.get("macro_capacity_state", False)):
        index += 1
    scope = str(args.get("power_constraint_scope", "system"))
    has_dual = scope == "per_uav" or bool(args.get("system_dual_state", False))
    return index if has_dual else None


def patch_system_dual_state(states, args, multiplier, num_agents):
    patched = np.asarray(states, dtype=np.float32).copy()
    index = dual_state_local_index(args)
    if index is None:
        return patched
    local = split_local_states(patched, num_agents)
    if index >= local.shape[2]:
        raise ValueError("Derived dual-state index exceeds local state size.")
    norm = max(float(args.get("energy_dual_state_norm", 10.0)), 1e-8)
    local[:, :, index] = np.clip(float(multiplier) / norm, 0.0, 1.0)
    return local.reshape(patched.shape)


def actor_loss_components(delay_value, energy_value, multiplier):
    delay_loss = -np.asarray(delay_value, dtype=np.float64)
    energy_loss = float(multiplier) * np.asarray(energy_value, dtype=np.float64)
    return delay_loss, energy_loss, delay_loss + energy_loss


def load_models(checkpoint, state_dim, action_dim, device):
    args = checkpoint["args"]
    hidden_dim = int(args.get("td3_hidden_dim", 256))
    actor = MacroActor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(checkpoint["macro_actor"])
    actor.eval()

    delay = TwinCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    delay_state = checkpoint.get("macro_delay_critic")
    if delay_state is None:
        delay_state = checkpoint.get("macro_critic")
    if delay_state is None:
        raise KeyError("Checkpoint does not contain a macro delay Critic.")
    delay.load_state_dict(delay_state)
    delay.eval()

    energy_dim = (
        1
        if str(args.get("power_constraint_scope", "system")) == "system"
        else int(args["num_agents"])
    )
    energy = TwinCritic(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        output_dim=energy_dim,
    ).to(device)
    energy.load_state_dict(checkpoint["macro_energy_critic"])
    energy.eval()
    return actor, delay, energy


def evaluate_action_values(
    states,
    actions,
    delay_critic,
    energy_critic,
    delay_aggregation,
    device,
    batch_size,
):
    delay_values = []
    energy_values = []
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        state_t = torch.as_tensor(states[start:stop], device=device)
        action_t = torch.as_tensor(actions[start:stop], device=device)
        with torch.no_grad():
            delay_q1, delay_q2 = delay_critic(state_t, action_t)
            energy_q1, energy_q2 = energy_critic(state_t, action_t)
            if delay_aggregation == "q1":
                delay_q = delay_q1
            elif delay_aggregation == "min":
                delay_q = torch.minimum(delay_q1, delay_q2)
            else:
                raise ValueError(f"Unsupported delay aggregation: {delay_aggregation}")
            energy_q = torch.maximum(energy_q1, energy_q2).sum(dim=1, keepdim=True)
        delay_values.append(delay_q.cpu().numpy().reshape(-1))
        energy_values.append(energy_q.cpu().numpy().reshape(-1))
    return np.concatenate(delay_values), np.concatenate(energy_values)


def checkpoint_episode(path):
    return int(path.stem.rsplit("_", 1)[-1])


def choose_probe_states(replay_path, num_agents, min_drift_m, max_probes):
    with np.load(replay_path) as payload:
        states = np.asarray(payload["state"], dtype=np.float32).copy()
    distance = state_mean_centroid_distance_m(states, num_agents)
    eligible = np.flatnonzero(distance >= float(min_drift_m))
    if eligible.size == 0:
        raise RuntimeError("Macro replay contains no states above the drift threshold.")
    if eligible.size > int(max_probes):
        positions = np.linspace(0, eligible.size - 1, int(max_probes))
        eligible = eligible[np.round(positions).astype(np.int64)]
    return states[eligible], distance[eligible], eligible


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_checkpoint(
    checkpoint_path,
    base_states,
    base_distance,
    device,
    batch_size,
):
    checkpoint = torch_load_checkpoint(checkpoint_path)
    args = checkpoint["args"]
    num_agents = int(args["num_agents"])
    action_dim = 2 * num_agents
    multiplier = float(args.get("energy_lagrange_multiplier_current", 0.0))
    states = patch_system_dual_state(
        base_states,
        args,
        multiplier,
        num_agents,
    )
    actor, delay_critic, energy_critic = load_models(
        checkpoint,
        states.shape[1],
        action_dim,
        device,
    )
    actor_actions = []
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        with torch.no_grad():
            actor_actions.append(
                actor(torch.as_tensor(states[start:stop], device=device))
                .cpu()
                .numpy()
            )
    actions = {
        "actor": np.concatenate(actor_actions).astype(np.float32),
        "return": np.zeros((len(states), action_dim), dtype=np.float32),
    }
    actions["hover"], hover_raw_norm, hover_exact = hover_actions_from_states(
        states,
        num_agents,
        float(args["macro_waypoint_radius_m"]),
    )

    values = {}
    movement = {}
    action_norm = {}
    for label, action in actions.items():
        delay_q, energy_q = evaluate_action_values(
            states,
            action,
            delay_critic,
            energy_critic,
            str(args.get("td3_actor_delay_aggregation", "q1")),
            device,
            batch_size,
        )
        delay_loss, energy_loss, total_loss = actor_loss_components(
            delay_q,
            energy_q,
            multiplier,
        )
        values[label] = {
            "delay_q": delay_q,
            "energy_q": energy_q,
            "delay_loss": delay_loss,
            "energy_loss": energy_loss,
            "total_loss": total_loss,
        }
        movement[label] = predicted_movement_m(
            states,
            action,
            num_agents,
            float(args["macro_waypoint_radius_m"]),
            float(args.get("max_macro_displacement", 20.0)),
        ).mean(axis=1)
        action_norm[label] = mean_action_norm(action, num_agents)

    delay_return_better = (
        values["return"]["delay_loss"] < values["hover"]["delay_loss"]
    )
    total_hover_better = (
        values["hover"]["total_loss"] < values["return"]["total_loss"]
    )
    energy_overturn = delay_return_better & total_hover_better
    actor_hover_distance = np.linalg.norm(
        actions["actor"] - actions["hover"], axis=1
    )
    actor_return_distance = np.linalg.norm(actions["actor"], axis=1)

    row = {
        "episode": checkpoint_episode(Path(checkpoint_path)),
        "lambda": multiplier,
        "probe_count": int(len(states)),
        "probe_mean_centroid_distance_m": float(np.mean(base_distance)),
        "hover_exact_fraction": float(np.mean(hover_exact)),
        "hover_raw_norm_max": float(np.max(hover_raw_norm)),
        "delay_return_better_fraction": float(np.mean(delay_return_better)),
        "total_hover_better_fraction": float(np.mean(total_hover_better)),
        "energy_overturn_fraction": float(np.mean(energy_overturn)),
        "actor_closer_to_hover_fraction": float(
            np.mean(actor_hover_distance < actor_return_distance)
        ),
    }
    for label in ("actor", "return", "hover"):
        row[f"{label}_action_norm_mean"] = float(np.mean(action_norm[label]))
        row[f"{label}_predicted_move_m_mean"] = float(np.mean(movement[label]))
        for component in ("delay_loss", "energy_loss", "total_loss"):
            row[f"{label}_{component}_mean"] = float(
                np.mean(values[label][component])
            )
    for component in ("delay_loss", "energy_loss", "total_loss"):
        row[f"return_minus_hover_{component}_mean"] = float(
            np.mean(values["return"][component] - values["hover"][component])
        )

    detail = []
    for index in range(len(states)):
        item = {
            "episode": row["episode"],
            "probe_index": index,
            "centroid_distance_m": float(base_distance[index]),
            "hover_exact": bool(hover_exact[index]),
            "energy_overturn": bool(energy_overturn[index]),
        }
        for label in ("actor", "return", "hover"):
            item[f"{label}_action_norm"] = float(action_norm[label][index])
            item[f"{label}_predicted_move_m"] = float(movement[label][index])
            for component in ("delay_loss", "energy_loss", "total_loss"):
                item[f"{label}_{component}"] = float(
                    values[label][component][index]
                )
        detail.append(item)
    return row, detail


def draw_summary(rows, output_dir):
    episode = np.asarray([row["episode"] for row in rows])
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.2), constrained_layout=True)

    axes[0].plot(episode, [row["lambda"] for row in rows], marker="o")
    axes[0].set_title("Episode-level multiplier")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("lambda")

    for field, label, color in (
        ("return_minus_hover_delay_loss_mean", "Delay component", "#2878B5"),
        ("return_minus_hover_energy_loss_mean", "Energy component", "#E07B39"),
        ("return_minus_hover_total_loss_mean", "Total Actor loss", "#C82423"),
    ):
        axes[1].plot(episode, [row[field] for row in rows], marker="o", label=label, color=color)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Return minus hover")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Loss difference")
    axes[1].legend(frameon=False, fontsize=8)

    for action, color in (("actor", "#C82423"), ("return", "#2878B5"), ("hover", "#2E8B57")):
        axes[2].plot(
            episode,
            [row[f"{action}_predicted_move_m_mean"] for row in rows],
            marker="o",
            label=action,
            color=color,
        )
    axes[2].set_title("Predicted first-period movement")
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("m per UAV")
    axes[2].legend(frameon=False, fontsize=8)

    axes[3].plot(
        episode,
        [row["delay_return_better_fraction"] for row in rows],
        marker="o",
        label="Delay favors return",
    )
    axes[3].plot(
        episode,
        [row["energy_overturn_fraction"] for row in rows],
        marker="s",
        label="Energy overturns delay",
    )
    axes[3].plot(
        episode,
        [row["actor_closer_to_hover_fraction"] for row in rows],
        marker="^",
        label="Actor closer to hover",
    )
    axes[3].set_ylim(-0.03, 1.03)
    axes[3].set_title("Mechanism fractions")
    axes[3].set_xlabel("Training episode")
    axes[3].legend(frameon=False, fontsize=8)

    fig.savefig(output_dir / "actor_energy_lock_audit.png", dpi=300)
    fig.savefig(output_dir / "actor_energy_lock_audit.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = results_dir / "checkpoints" / "eval_candidates"
    checkpoints = sorted(checkpoint_dir.glob("eval_ep_*.pt"))
    replay_path = results_dir / "checkpoints" / "macro_replay.npz"
    if not checkpoints:
        raise FileNotFoundError(f"No evaluation checkpoints found in {checkpoint_dir}")
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"Final macro replay is not available yet: {replay_path}"
        )

    first = torch_load_checkpoint(checkpoints[0])
    num_agents = int(first["args"]["num_agents"])
    base_states, base_distance, replay_indices = choose_probe_states(
        replay_path,
        num_agents,
        args.min_drift_m,
        args.max_probes,
    )
    np.savez_compressed(
        output_dir / "fixed_drift_probes.npz",
        replay_indices=replay_indices,
        states=base_states,
        centroid_distance_m=base_distance,
    )

    device = torch.device(args.device)
    rows = []
    detail = []
    for checkpoint_path in checkpoints:
        row, checkpoint_detail = audit_checkpoint(
            checkpoint_path,
            base_states,
            base_distance,
            device,
            args.batch_size,
        )
        rows.append(row)
        detail.extend(checkpoint_detail)
        print(
            f"ep={row['episode']:04d} lambda={row['lambda']:.4f} "
            f"delay_return={row['delay_return_better_fraction']:.3f} "
            f"overturn={row['energy_overturn_fraction']:.3f}"
        )

    write_csv(output_dir / "checkpoint_summary.csv", rows)
    write_csv(output_dir / "per_probe_components.csv", detail)
    draw_summary(rows, output_dir)
    manifest = {
        "results_dir": str(results_dir),
        "replay_path": str(replay_path),
        "checkpoints": [str(path) for path in checkpoints],
        "probe_count": int(len(base_states)),
        "minimum_drift_m": float(args.min_drift_m),
        "probe_mean_centroid_distance_m": float(np.mean(base_distance)),
        "interpretation": (
            "Energy-lock support requires the delay component to prefer "
            "centroid return while the lambda-weighted total prefers hover. "
            "If the delay component also prefers hover, delay-Critic "
            "misranking remains and the lock mechanism is not isolated."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved v32b Actor-energy lock audit -> {output_dir}")


if __name__ == "__main__":
    main()
