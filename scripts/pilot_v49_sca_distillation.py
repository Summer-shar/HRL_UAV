"""Test whether the v48 SAC policy can represent the SCA routing policy.

This is a diagnostic, not a paper experiment.  It trains a fresh SAC actor by
supervised imitation and never updates the environment, reward, or checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v48_priority_gated as v48  # noqa: E402
from compare_sca import solve_sca_micro_action  # noqa: E402
from hrl_uav.agents.sac_agent import Actor as SACActor  # noqa: E402
import hrl_uav.agents.sac_agent as sac_agent_mod  # noqa: E402
from train_centralized_macro import configure_sca_wrapper  # noqa: E402
from train_centralized_macro import zero_macro_policy  # noqa: E402


DEFAULT_OUTPUT = ROOT / "result_archive" / "v49_sca_distillation_pilot"


def configure_sca_args(args) -> None:
    defaults = {
        "sca_update_interval": 1,
        "sca_max_iter": 3,
        "sca_tol": 1e-3,
        "sca_solver": "CLARABEL",
        "sca_delay_weight": 0.9,
        "sca_energy_weight": 0.1,
        "sca_proximal_weight": 1e-3,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def routing_to_priority_latent(matrix, env) -> np.ndarray:
    """Invert the v48 priority-gated transform for a three-UAV routing matrix."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(2, env.N, env.N)
    if int(env.N) != 3:
        raise ValueError("The diagnostic inverse currently requires three UAVs.")
    scale = float(env.micro_stick_breaking_scale)
    biases = np.asarray(
        [np.log(float(env.micro_priority_weight_ratio)), 0.0],
        dtype=np.float64,
    )
    latent = np.zeros((2, env.N, env.N - 1), dtype=np.float64)
    for priority in range(2):
        for source in range(env.N):
            local = float(np.clip(matrix[priority, source, source], 1e-5, 1.0 - 1e-5))
            latent[priority, source, 0] = (
                float(logit(np.asarray(local))) - biases[priority]
            ) / scale
            peers = [destination for destination in range(env.N) if destination != source]
            remote = np.maximum(matrix[priority, source, peers], 0.0)
            remote_total = float(np.sum(remote))
            first_share = 0.5 if remote_total <= 1e-12 else float(remote[0] / remote_total)
            latent[priority, source, 1] = float(logit(np.asarray(first_share))) / scale
    return np.clip(latent.reshape(-1), -0.999, 0.999).astype(np.float32)


def sca_action(env, args) -> np.ndarray:
    configure_sca_wrapper(env, args)
    return np.asarray(
        solve_sca_micro_action(env, deterministic=True),
        dtype=np.float32,
    )


def collect_dataset(args, seeds, actor=None):
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_metrics: list[dict] = []
    for seed in seeds:
        env = v45.make_eval_env(args, record_trajectory=False)
        eval_args = argparse.Namespace(**vars(args))
        eval_args.eval_episodes = 1
        eval_args.eval_base_seed = int(seed)

        def policy(current_env, deterministic):
            state = np.asarray(current_env.get_sac_state(), dtype=np.float32)
            teacher_full = sca_action(current_env, args)
            states.append(state)
            targets.append(routing_to_priority_latent(teacher_full, current_env))
            if actor is None:
                return teacher_full
            return actor.choose_action(state, deterministic=True)

        metrics = v45.evaluate_policy(
            env,
            eval_args,
            zero_macro_policy,
            policy,
        )
        episode_metrics.append(
            {
                "seed": int(seed),
                "delay_ms": float(metrics["task_delay_weighted_ms"]),
                "queue_mbits": float(metrics["q_high"] + metrics["q_low"]),
                "power_w": float(metrics["average_power_w"]),
            }
        )
        mode = "teacher" if actor is None else "clone"
        print(
            f"collect {mode} seed={seed}: delay={episode_metrics[-1]['delay_ms']:.2f}ms",
            flush=True,
        )
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        episode_metrics,
    )


def train_actor(actor, states, targets, *, updates, batch_size, lr, seed):
    actor.optimizer = torch.optim.Adam(actor.action_net.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))
    device = sac_agent_mod.device
    history = []
    actor.action_net.train()
    for update in range(1, int(updates) + 1):
        indices = rng.integers(0, states.shape[0], size=int(batch_size))
        batch_states = torch.as_tensor(states[indices], dtype=torch.float32, device=device)
        batch_targets = torch.as_tensor(targets[indices], dtype=torch.float32, device=device)
        mean, _ = actor.action_net(batch_states)
        prediction = torch.tanh(mean)
        loss = F.mse_loss(prediction, batch_targets)
        actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.action_net.parameters(), 1.0)
        actor.optimizer.step()
        if update == 1 or update % 250 == 0 or update == int(updates):
            history.append({"update": update, "mse": float(loss.item())})
            print(f"BC update={update}: mse={loss.item():.6f}", flush=True)
    actor.action_net.eval()
    return history


def summarize_metrics(rows):
    return {
        field: v45.mean_ci95([float(row[field]) for row in rows])
        for field in ("delay_ms", "queue_mbits", "power_w")
    }


def latent_error(actor, states, targets):
    device = sac_agent_mod.device
    with torch.no_grad():
        tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        mean, _ = actor.action_net(tensor)
        prediction = torch.tanh(mean).cpu().numpy()
    error = prediction - targets
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train-seeds", default="7100-7102")
    parser.add_argument("--dagger-seeds", default="7200-7201")
    parser.add_argument("--test-seeds", default="8000-8004")
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--dagger-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli = parser.parse_args()

    use_cuda = cli.device == "cuda" and torch.cuda.is_available()
    sac_agent_mod.device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    np.random.seed(1)

    checkpoint = v48.checkpoint("formal", 1)
    loaded = v45.load_method_policy("joint_td3_sac", checkpoint, cli.device)
    args = argparse.Namespace(**vars(loaded["args"]))
    configure_sca_args(args)
    probe_env = v45.make_eval_env(args, record_trajectory=False)
    actor = SACActor(
        int(probe_env.sac_obs_dim),
        int(probe_env.micro_action_space.shape[0]),
    )

    train_states, train_targets, teacher_train_metrics = collect_dataset(
        args,
        v45.parse_seed_spec(cli.train_seeds),
    )
    history = train_actor(
        actor,
        train_states,
        train_targets,
        updates=cli.updates,
        batch_size=cli.batch_size,
        lr=cli.lr,
        seed=1,
    )

    dagger_states, dagger_targets, clone_dagger_metrics = collect_dataset(
        args,
        v45.parse_seed_spec(cli.dagger_seeds),
        actor=actor,
    )
    all_states = np.concatenate([train_states, dagger_states], axis=0)
    all_targets = np.concatenate([train_targets, dagger_targets], axis=0)
    history.extend(
        train_actor(
            actor,
            all_states,
            all_targets,
            updates=cli.dagger_updates,
            batch_size=cli.batch_size,
            lr=cli.lr,
            seed=2,
        )
    )

    test_states, test_targets, teacher_test_metrics = collect_dataset(
        args,
        v45.parse_seed_spec(cli.test_seeds),
    )
    _, _, clone_test_metrics = collect_dataset(
        args,
        v45.parse_seed_spec(cli.test_seeds),
        actor=actor,
    )
    report = {
        "status": "diagnostic_only",
        "train_samples": int(all_states.shape[0]),
        "test_samples": int(test_states.shape[0]),
        "latent_test_error": latent_error(actor, test_states, test_targets),
        "teacher_train": summarize_metrics(teacher_train_metrics),
        "clone_dagger_before_second_fit": summarize_metrics(clone_dagger_metrics),
        "teacher_test": summarize_metrics(teacher_test_metrics),
        "clone_test": summarize_metrics(clone_test_metrics),
        "training_history": history,
    }
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(actor.action_net.state_dict(), output / "clone_actor.pt")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
