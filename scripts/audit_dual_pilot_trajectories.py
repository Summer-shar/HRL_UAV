import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = bootstrap()

from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import resolve_project_path
from compare_common import torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SACActor
import hrl_uav.agents.sac_agent as sac_agent_mod
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import sac_micro_policy
from train_centralized_td3 import MacroActor
from train_centralized_td3 import td3_macro_policy


DEFAULT_ON = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_OFF = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_off/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_OUTPUT = "result_archive/v24_system_dual_pilot_trajectory_audit"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay dual-on/off TD3 checkpoints under common random seeds and "
            "audit centroid distance and cumulative movement."
        )
    )
    parser.add_argument("--dual-on", default=DEFAULT_ON)
    parser.add_argument("--dual-off", default=DEFAULT_OFF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-base-seed", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def array_hash(value):
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_policy(checkpoint_path, device):
    checkpoint = torch_load_checkpoint(resolve_project_path(checkpoint_path))
    if "macro_actor" not in checkpoint or "args" not in checkpoint:
        raise KeyError(f"Invalid TD3 checkpoint: {checkpoint_path}")
    if str(checkpoint.get("micro_solver", "")) != "sac":
        raise ValueError("The trajectory audit requires a SAC micro policy.")

    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    env = make_env_from_args(checkpoint_args, record_trajectory=True)
    configure_macro_objective(
        env,
        checkpoint_args,
        float(getattr(checkpoint_args, "macro_shaping_final", 0.0)),
    )
    _, global_state = env.reset(seed=int(checkpoint_args.seed))[0]

    actor = MacroActor(
        int(np.asarray(global_state).size),
        int(checkpoint_args.num_agents) * 2,
        hidden_dim=int(checkpoint_args.td3_hidden_dim),
    ).to(device)
    actor.load_state_dict(checkpoint["macro_actor"])
    actor.eval()

    micro_actor = SACActor(
        int(env.sac_obs_dim),
        int(env.micro_action_space.shape[0]),
    )
    micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    micro_actor.action_net.eval()
    return checkpoint_args, env, actor, micro_actor


def replay_checkpoint(
    label,
    checkpoint_path,
    seeds,
    device,
    macro_policy_kind="actor",
):
    checkpoint_args, env, actor, micro_actor = load_policy(
        checkpoint_path,
        device,
    )
    if macro_policy_kind == "actor":
        macro_policy = td3_macro_policy(actor)
    elif macro_policy_kind == "centroid_zero":
        macro_policy = lambda policy_env, _local_obs, _eval_mode: np.zeros(
            (policy_env.N, 2), dtype=np.float32
        )
    else:
        raise ValueError(f"Unsupported macro policy kind: {macro_policy_kind}")
    micro_policy = sac_micro_policy(micro_actor)
    rows = []

    for seed in seeds:
        eval_args = argparse.Namespace(**vars(checkpoint_args))
        eval_args.eval_base_seed = int(seed)
        eval_args.eval_episodes = 1
        metrics = evaluate_policy(env, eval_args, macro_policy, micro_policy)

        raw_env = getattr(env, "env", env)
        uav = np.asarray(raw_env.uav_trajectory, dtype=np.float64)
        centroid = np.asarray(raw_env.centroid_trajectory, dtype=np.float64)
        users = np.asarray(raw_env.users_trajectory, dtype=np.float64)
        if uav.shape != centroid.shape or uav.ndim != 3:
            raise RuntimeError(
                f"Unexpected trajectory shapes: UAV={uav.shape}, "
                f"centroid={centroid.shape}."
            )

        centroid_distance = np.linalg.norm(uav - centroid, axis=2)
        path_per_uav = np.linalg.norm(np.diff(uav, axis=0), axis=2).sum(axis=0)
        net_per_uav = np.linalg.norm(uav[-1] - uav[0], axis=1)
        row = {
            "arm": label,
            "seed": int(seed),
            "J": float(metrics["J"]),
            "task_delay_weighted_ms": float(
                metrics["task_delay_weighted_ms"]
            ),
            "average_power_w": float(metrics["average_power_w"]),
            "propulsion_energy_j": float(metrics["energy_propulsion_episode"]),
            "compute_energy_j": float(metrics["energy_compute_episode"]),
            "mean_centroid_distance_m": float(np.mean(centroid_distance)),
            "p95_centroid_distance_m": float(
                np.percentile(centroid_distance, 95)
            ),
            "path_total_m": float(np.sum(path_per_uav)),
            "net_total_m": float(np.sum(net_per_uav)),
            "centroid_trace_hash": array_hash(centroid),
            "users_trace_hash": array_hash(users),
        }
        for index in range(uav.shape[1]):
            row[f"uav{index + 1}_mean_centroid_distance_m"] = float(
                np.mean(centroid_distance[:, index])
            )
            row[f"uav{index + 1}_path_m"] = float(path_per_uav[index])
            row[f"uav{index + 1}_net_m"] = float(net_per_uav[index])
        rows.append(row)
        print(
            f"{label} seed={seed}: delay="
            f"{row['task_delay_weighted_ms']:.3f} ms, "
            f"power={row['average_power_w']:.3f} W, "
            f"centroid={row['mean_centroid_distance_m']:.3f} m, "
            f"path={row['path_total_m']:.3f} m"
        )
    return rows


def paired_summary(on_rows, off_rows):
    on_by_seed = {row["seed"]: row for row in on_rows}
    off_by_seed = {row["seed"]: row for row in off_rows}
    if set(on_by_seed) != set(off_by_seed):
        raise RuntimeError("Dual-on/off seed sets do not match.")

    metrics = (
        "J",
        "task_delay_weighted_ms",
        "average_power_w",
        "propulsion_energy_j",
        "compute_energy_j",
        "mean_centroid_distance_m",
        "p95_centroid_distance_m",
        "path_total_m",
        "net_total_m",
    )
    summary = {}
    seed_rows = []
    for seed in sorted(on_by_seed):
        on = on_by_seed[seed]
        off = off_by_seed[seed]
        external_match = (
            on["centroid_trace_hash"] == off["centroid_trace_hash"]
            and on["users_trace_hash"] == off["users_trace_hash"]
        )
        if not external_match:
            raise RuntimeError(
                f"External trajectory mismatch for common seed {seed}."
            )
        paired = {"seed": seed, "external_trajectory_match": True}
        for metric in metrics:
            paired[f"{metric}_dual_on"] = on[metric]
            paired[f"{metric}_dual_off"] = off[metric]
            paired[f"{metric}_on_minus_off"] = on[metric] - off[metric]
        seed_rows.append(paired)

    for metric in metrics:
        on_values = np.asarray(
            [on_by_seed[seed][metric] for seed in sorted(on_by_seed)],
            dtype=np.float64,
        )
        off_values = np.asarray(
            [off_by_seed[seed][metric] for seed in sorted(off_by_seed)],
            dtype=np.float64,
        )
        deltas = on_values - off_values
        n = int(deltas.size)
        mean_delta = float(np.mean(deltas))
        sem = float(stats.sem(deltas)) if n > 1 else 0.0
        critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        summary[metric] = {
            "n": n,
            "dual_on_mean": float(np.mean(on_values)),
            "dual_off_mean": float(np.mean(off_values)),
            "on_minus_off_mean": mean_delta,
            "on_minus_off_ci95_low": mean_delta - critical * sem,
            "on_minus_off_ci95_high": mean_delta + critical * sem,
            "paired_p_value": float(stats.ttest_rel(on_values, off_values).pvalue)
            if n > 1
            else None,
        }
    return seed_rows, summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    device = torch.device(args.device)
    sac_agent_mod.device = device

    on_checkpoint = resolve_project_path(args.dual_on)
    off_checkpoint = resolve_project_path(args.dual_off)
    on_payload = torch_load_checkpoint(on_checkpoint)
    off_payload = torch_load_checkpoint(off_checkpoint)
    on_args = on_payload["args"]
    off_args = off_payload["args"]
    base_seed = (
        int(args.eval_base_seed)
        if args.eval_base_seed is not None
        else int(on_args["eval_base_seed"])
    )
    if int(off_args["eval_base_seed"]) != int(on_args["eval_base_seed"]):
        raise RuntimeError("Checkpoint evaluation seed bases do not match.")
    seeds = list(range(base_seed, base_seed + int(args.episodes)))

    on_rows = replay_checkpoint("dual_on", on_checkpoint, seeds, device)
    off_rows = replay_checkpoint("dual_off", off_checkpoint, seeds, device)
    paired_rows, summary = paired_summary(on_rows, off_rows)

    output_dir = Path(resolve_project_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_raw.csv", on_rows + off_rows)
    write_csv(output_dir / "paired_deltas.csv", paired_rows)
    payload = {
        "dual_on_checkpoint": str(on_checkpoint),
        "dual_off_checkpoint": str(off_checkpoint),
        "eval_seeds": seeds,
        "external_trajectory_match_all": True,
        "metrics": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\nPaired dual_on - dual_off summary")
    for metric, values in summary.items():
        print(
            f"  {metric}: {values['on_minus_off_mean']:.6f} "
            f"[{values['on_minus_off_ci95_low']:.6f}, "
            f"{values['on_minus_off_ci95_high']:.6f}], "
            f"p={values['paired_p_value']:.6g}"
        )
    print(f"Saved audit -> {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
