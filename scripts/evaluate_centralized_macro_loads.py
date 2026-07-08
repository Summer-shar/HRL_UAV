import argparse
import csv
from pathlib import Path

import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

from compare_common import evaluate_policy
from compare_common import make_env_from_args
from train_centralized_macro import PPO_Actor
from train_centralized_macro import SAC_Actor
from train_centralized_macro import centralized_macro_policy
from train_centralized_macro import sac_micro_policy
import hrl_uav.agents.ppo_agent as ppo_agent_mod
import hrl_uav.agents.sac_agent as sac_agent_mod


def parse_loads(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_float_array(value):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def _safe_rate(completed, arrived):
    completed = np.asarray(completed, dtype=np.float64)
    arrived = np.asarray(arrived, dtype=np.float64)
    return np.divide(
        completed,
        np.maximum(arrived, 1e-12),
        out=np.zeros_like(completed, dtype=np.float64),
        where=arrived > 0,
    )


def _add_completion_summary_fields(row):
    row = dict(row)
    high_arrived = _to_float_array(row.get("task_arrived_source_high_count"))
    high_completed = _to_float_array(row.get("task_completed_source_high_count"))
    low_arrived = _to_float_array(row.get("task_arrived_source_low_count"))
    low_completed = _to_float_array(row.get("task_completed_source_low_count"))
    total_arrived = _to_float_array(row.get("task_arrived_source_count"))
    total_completed = _to_float_array(row.get("task_completed_source_count"))

    priority_pairs = [
        ("high", high_completed, high_arrived),
        ("low", low_completed, low_arrived),
        ("all", total_completed, total_arrived),
    ]
    for name, completed, arrived in priority_pairs:
        if completed is None or arrived is None:
            continue
        row[f"task_arrived_{name}_count"] = float(np.sum(arrived))
        row[f"task_completed_{name}_count"] = float(np.sum(completed))
        row[f"task_completion_{name}_rate"] = float(
            np.sum(completed) / max(np.sum(arrived), 1e-12)
        )

    zone_pairs = [
        ("high", high_completed, high_arrived),
        ("low", low_completed, low_arrived),
        ("all", total_completed, total_arrived),
    ]
    for name, completed, arrived in zone_pairs:
        if completed is None or arrived is None or completed.ndim != 1:
            continue
        rates = _safe_rate(completed, arrived)
        for idx, value in enumerate(rates):
            row[f"task_completion_source_{name}_rate_zone{idx + 1}"] = float(value)

    return row


def _flatten_row(row):
    flat = {}
    for key, value in _add_completion_summary_fields(row).items():
        array = _to_float_array(value)
        if array is None:
            flat[key] = value
        elif array.ndim == 0:
            flat[key] = float(array)
        elif array.ndim == 1:
            for idx, item in enumerate(array):
                flat[f"{key}_{idx + 1}"] = float(item)
        elif array.ndim == 2:
            for row_idx in range(array.shape[0]):
                for col_idx in range(array.shape[1]):
                    flat[f"{key}_{row_idx + 1}_{col_idx + 1}"] = float(
                        array[row_idx, col_idx]
                    )
    return flat


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_fields = [
        "cpu_ghz",
        "load_scale",
        "zone_loads",
        "score",
        "J",
        "legacy_score",
        "u2g",
        "coop_ratio",
        "task_delay_high_ms",
        "task_delay_low_ms",
        "task_delay_weighted_ms",
        "energy",
        "energy_total_episode",
        "cpu_saturation",
        "task_completion_rate",
        "task_qos_violation_rate",
        "task_arrived_high_count",
        "task_completed_high_count",
        "task_completion_high_rate",
        "task_arrived_low_count",
        "task_completed_low_count",
        "task_completion_low_rate",
        "task_arrived_all_count",
        "task_completed_all_count",
        "task_completion_all_rate",
    ]
    flat_rows = [_flatten_row(row) for row in rows]
    extra_fields = sorted(
        {
            field
            for row in flat_rows
            for field in row.keys()
            if field not in preferred_fields
        }
    )
    fields = preferred_fields + extra_fields
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--loads", default="0.7,0.8,0.9,1.0,1.1,1.2,1.3")
    parser.add_argument(
        "--cpu_ghz_values",
        default="",
        help="Optional comma-separated CPU GHz values to override checkpoint CPU",
    )
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--eval_base_seed", type=int, default=2000)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args_cli = parser.parse_args()

    device = torch.device(
        "cuda" if args_cli.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    ppo_agent_mod.device = device
    sac_agent_mod.device = device

    ckpt_path = Path(args_cli.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    checkpoint = torch_load(ckpt_path)
    ckpt_args = argparse.Namespace(**checkpoint["args"])
    ckpt_args.eval_episodes = int(args_cli.eval_episodes)
    ckpt_args.eval_base_seed = int(args_cli.eval_base_seed)
    ckpt_args.trajectory_plot_interval = 0

    build_env = make_env_from_args(ckpt_args, record_trajectory=False)
    _, global_state0 = build_env.reset(seed=ckpt_args.seed)[0]
    macro_actor = PPO_Actor(int(np.asarray(global_state0).size), int(ckpt_args.num_agents) * 2)
    macro_actor.old_pi.load_state_dict(checkpoint["macro_actor"])
    macro_actor.new_pi.load_state_dict(checkpoint["macro_actor"])
    macro_actor.old_pi.eval()
    macro_actor.new_pi.eval()

    sac_actor = SAC_Actor(build_env.sac_obs_dim, build_env.micro_action_space.shape[0])
    sac_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    sac_actor.action_net.eval()

    cpu_values = (
        parse_loads(args_cli.cpu_ghz_values)
        if args_cli.cpu_ghz_values.strip()
        else [float(getattr(ckpt_args, "cpu_ghz", 16.0))]
    )

    rows = []
    for cpu_ghz in cpu_values:
        eval_args_base = argparse.Namespace(**vars(ckpt_args))
        eval_args_base.cpu_ghz = float(cpu_ghz)
        build_env = make_env_from_args(eval_args_base, record_trajectory=False)
        _, global_state0 = build_env.reset(seed=ckpt_args.seed)[0]
        if int(np.asarray(global_state0).size) != int(
            macro_actor.old_pi.in_to_y1.in_features
        ):
            raise ValueError(
                "Checkpoint macro state dimension does not match evaluation "
                f"environment for cpu_ghz={cpu_ghz}."
            )

        for load_scale in parse_loads(args_cli.loads):
            eval_args = argparse.Namespace(**vars(ckpt_args))
            eval_args.cpu_ghz = float(cpu_ghz)
            eval_args.load_scale = float(load_scale)
            eval_args.eval_episodes = int(args_cli.eval_episodes)
            eval_args.eval_base_seed = int(args_cli.eval_base_seed)
            eval_env = make_env_from_args(eval_args, record_trajectory=False)
            metrics = evaluate_policy(
                eval_env,
                eval_args,
                centralized_macro_policy(macro_actor, deterministic=True),
                sac_micro_policy(sac_actor),
            )
            row = {
                "cpu_ghz": float(cpu_ghz),
                "load_scale": float(load_scale),
                "zone_loads": getattr(eval_args, "zone_loads", ""),
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"cpu={cpu_ghz:.1f} load={load_scale:.2f} "
                f"J={metrics['J']:.4f} "
                f"high={metrics.get('task_delay_high_ms', 0.0):.1f}ms "
                f"low={metrics.get('task_delay_low_ms', 0.0):.1f}ms "
                f"energy={metrics.get('energy', 0.0):.2f} "
                f"coop={metrics.get('coop_ratio', 0.0):.3f}"
            )

    if args_cli.output_csv:
        out = Path(args_cli.output_csv)
    else:
        out = ckpt_path.parents[1] / "cross_load_eval.csv"
    write_csv(out, rows)
    print(f"Saved cross-load eval -> {out}")


if __name__ == "__main__":
    main()
