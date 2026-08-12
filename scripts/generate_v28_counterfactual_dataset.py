import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import torch_load_checkpoint
from diagnose_td3_counterfactual import actor_action
from diagnose_td3_counterfactual import build_env
from diagnose_td3_counterfactual import execute_macro_interval
from diagnose_td3_counterfactual import load_models
import hrl_uav.agents.sac_agent as sac_agent_mod


DEFAULT_CHECKPOINT = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_OUTPUT = "result_archive/v28_td_vs_mc_counterfactual"
TRAIN_SEEDS = tuple(range(63000, 63040))
TEST_SEEDS = tuple(range(64000, 64020))
BRANCH_POINTS = (0, 5, 10, 15, 20, 25)
SNAPSHOT_POLICIES = ("actor", "centroid")
BRANCH_ACTIONS = ("actor", "zero")
GAMMA = 0.99


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen v28 same-state counterfactual transition "
            "dataset with episode-level train/test separation."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "test", "both"), default="test")
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated worker subset; must belong to the frozen split.",
    )
    parser.add_argument("--no-combine", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_worker_seeds(raw_value, allowed_seeds):
    if not str(raw_value).strip():
        return tuple(allowed_seeds)
    requested = tuple(
        int(value.strip())
        for value in str(raw_value).split(",")
        if value.strip()
    )
    invalid = sorted(set(requested) - set(allowed_seeds))
    if invalid:
        raise ValueError(f"Worker seeds are outside the frozen split: {invalid}")
    if len(requested) != len(set(requested)) or not requested:
        raise ValueError("Worker seed list must be non-empty and unique.")
    return requested


def discounted_return(rewards, horizon, gamma=GAMMA):
    values = np.asarray(rewards[: int(horizon)], dtype=np.float64)
    weights = np.power(float(gamma), np.arange(values.size, dtype=np.float64))
    return float(np.sum(weights * values))


def digest_macro_trace(trace):
    hasher = hashlib.sha256()
    for value in trace:
        hasher.update(str(value).encode("ascii"))
    return hasher.hexdigest()


def collect_snapshots(env, eval_args, actor, micro_policy, snapshot_policy):
    snapshots = {}
    for macro_index in range(int(eval_args.macro_steps)):
        if macro_index in BRANCH_POINTS:
            snapshots[macro_index] = copy.deepcopy(env)
        if snapshot_policy == "actor":
            action = actor_action(actor, env)
        elif snapshot_policy == "centroid":
            action = np.zeros((env.N, 2), dtype=np.float32)
        else:
            raise ValueError(f"Unknown snapshot policy: {snapshot_policy}")
        _, done, _ = execute_macro_interval(
            env,
            eval_args,
            action,
            micro_policy,
        )
        if done:
            raise RuntimeError("v28 reference rollout terminated unexpectedly.")
    if set(snapshots) != set(BRANCH_POINTS):
        raise RuntimeError("Reference rollout did not produce all frozen snapshots.")
    return snapshots


def rollout_branch(snapshot, eval_args, actor, micro_policy, first_action, horizon):
    records = []
    traces = []
    for branch_step in range(int(horizon)):
        _, state = snapshot.get_macro_state()
        if branch_step == 0 and first_action == "zero":
            action = np.zeros((snapshot.N, 2), dtype=np.float32)
        else:
            action = actor_action(actor, snapshot)
        reward, done, macro_trace = execute_macro_interval(
            snapshot,
            eval_args,
            action,
            micro_policy,
        )
        _, next_state = snapshot.get_macro_state()
        if done:
            raise RuntimeError("v28 fixed-horizon branch terminated unexpectedly.")
        trace_hash = digest_macro_trace(macro_trace)
        traces.append(trace_hash)
        records.append(
            {
                "state": np.asarray(state, dtype=np.float32).reshape(-1),
                "action": np.asarray(action, dtype=np.float32).reshape(-1),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32).reshape(-1),
                "external_trace_hash": trace_hash,
            }
        )
    return records, traces


def branch_to_arrays(
    records,
    split,
    episode_seed,
    snapshot_policy,
    fork_macro_index,
    branch_action,
):
    count = len(records)
    fork_state_id = f"{split}:{episode_seed}:{snapshot_policy}:{fork_macro_index}"
    branch_id = f"{fork_state_id}:{branch_action}"
    return {
        "split": np.full(count, split),
        "episode_seed": np.full(count, int(episode_seed), dtype=np.int32),
        "snapshot_policy": np.full(count, snapshot_policy),
        "fork_macro_index": np.full(count, int(fork_macro_index), dtype=np.int16),
        "fork_state_id": np.full(count, fork_state_id),
        "branch_action": np.full(count, branch_action),
        "branch_id": np.full(count, branch_id),
        "branch_step": np.arange(count, dtype=np.int16),
        "state": np.stack([row["state"] for row in records]),
        "action": np.stack([row["action"] for row in records]),
        "reward": np.asarray([row["reward"] for row in records], dtype=np.float32),
        "next_state": np.stack([row["next_state"] for row in records]),
        "external_trace_hash": np.asarray(
            [row["external_trace_hash"] for row in records]
        ),
    }


def concatenate_arrays(chunks):
    keys = tuple(chunks[0])
    if any(tuple(chunk) != keys for chunk in chunks):
        raise RuntimeError("Transition chunk schemas do not match.")
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}


def save_seed_dataset(
    path,
    split,
    seed,
    horizon,
    eval_args,
    actor,
    micro_policy,
):
    chunks = []
    return_rows = []
    for snapshot_policy in SNAPSHOT_POLICIES:
        env = build_env(eval_args)
        env.reset(seed=int(seed))
        snapshots = collect_snapshots(
            env,
            eval_args,
            actor,
            micro_policy,
            snapshot_policy,
        )
        for fork_macro_index in BRANCH_POINTS:
            branch_results = {}
            for branch_action in BRANCH_ACTIONS:
                records, traces = rollout_branch(
                    copy.deepcopy(snapshots[fork_macro_index]),
                    eval_args,
                    actor,
                    micro_policy,
                    branch_action,
                    horizon,
                )
                branch_results[branch_action] = (records, traces)
                chunks.append(
                    branch_to_arrays(
                        records,
                        split,
                        seed,
                        snapshot_policy,
                        fork_macro_index,
                        branch_action,
                    )
                )
            actor_records, actor_traces = branch_results["actor"]
            zero_records, zero_traces = branch_results["zero"]
            if actor_traces != zero_traces:
                raise RuntimeError(
                    "External CRN trace mismatch for "
                    f"seed={seed}, policy={snapshot_policy}, "
                    f"fork={fork_macro_index}."
                )
            actor_rewards = [row["reward"] for row in actor_records]
            zero_rewards = [row["reward"] for row in zero_records]
            row = {
                "split": split,
                "episode_seed": int(seed),
                "snapshot_policy": snapshot_policy,
                "fork_macro_index": int(fork_macro_index),
                "fork_state_id": (
                    f"{split}:{seed}:{snapshot_policy}:{fork_macro_index}"
                ),
                "g30_actor": discounted_return(actor_rewards, 30),
                "g30_zero": discounted_return(zero_rewards, 30),
                "g30_actor_minus_zero": (
                    discounted_return(actor_rewards, 30)
                    - discounted_return(zero_rewards, 30)
                ),
                "external_trace_match": True,
            }
            if int(horizon) >= 60:
                row.update(
                    {
                        "g60_actor": discounted_return(actor_rewards, 60),
                        "g60_zero": discounted_return(zero_rewards, 60),
                        "g60_actor_minus_zero": (
                            discounted_return(actor_rewards, 60)
                            - discounted_return(zero_rewards, 60)
                        ),
                    }
                )
            return_rows.append(row)
    arrays = concatenate_arrays(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return return_rows, int(arrays["reward"].size), int(arrays["state"].shape[1])


def read_seed_returns(path):
    returns_path = path.with_suffix(".returns.json")
    with returns_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def combine_seed_files(split, seeds, transition_dir, output_dir):
    chunks = []
    return_rows = []
    for seed in seeds:
        path = transition_dir / f"{split}_seed_{seed}.npz"
        with np.load(path) as payload:
            chunks.append({key: payload[key].copy() for key in payload.files})
        return_rows.extend(read_seed_returns(path))
    arrays = concatenate_arrays(chunks)
    np.savez_compressed(output_dir / f"{split}_transitions.npz", **arrays)
    write_csv(output_dir / f"{split}_paired_returns.csv", return_rows)
    return arrays, return_rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    sac_agent_mod.device = device
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch_load_checkpoint(str(checkpoint_path))
    eval_args = argparse.Namespace(**checkpoint["args"])
    eval_args.device = str(device)
    expected_dual = 0.038131859182497896
    observed_dual = float(eval_args.energy_lagrange_multiplier_current)
    if not np.isclose(observed_dual, expected_dual, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Frozen v24 lambda mismatch: {observed_dual} != {expected_dual}."
        )
    probe_env = build_env(eval_args)
    probe_env.reset(seed=TEST_SEEDS[0])
    actor, _critic, micro_policy = load_models(
        checkpoint,
        probe_env,
        eval_args,
        device,
    )

    output_dir = project_path(args.output_dir)
    transition_dir = output_dir / "branch_transitions"
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = (
        (("train", TRAIN_SEEDS, 30), ("test", TEST_SEEDS, 60))
        if args.split == "both"
        else (
            ("train", TRAIN_SEEDS, 30)
            if args.split == "train"
            else ("test", TEST_SEEDS, 60)
        ,)
    )
    split_summaries = {}
    for split, frozen_seeds, horizon in requested:
        worker_seeds = parse_worker_seeds(args.seeds, frozen_seeds)
        if not args.combine_only:
            for index, seed in enumerate(worker_seeds, start=1):
                path = transition_dir / f"{split}_seed_{seed}.npz"
                returns_path = path.with_suffix(".returns.json")
                if path.exists() and returns_path.exists() and not args.overwrite:
                    rows = read_seed_returns(path)
                    print(
                        f"[{split} {index}/{len(worker_seeds)}] seed={seed} resume "
                        f"({len(rows)} forks)"
                    )
                    continue
                rows, transition_count, state_dim = save_seed_dataset(
                    path,
                    split,
                    seed,
                    horizon,
                    eval_args,
                    actor,
                    micro_policy,
                )
                with returns_path.open("w", encoding="utf-8") as handle:
                    json.dump(rows, handle, ensure_ascii=False, indent=2)
                print(
                    f"[{split} {index}/{len(worker_seeds)}] seed={seed}: "
                    f"forks={len(rows)}, transitions={transition_count}, "
                    f"state_dim={state_dim}"
                )
        if args.no_combine:
            continue
        missing = [
            seed
            for seed in frozen_seeds
            if not (transition_dir / f"{split}_seed_{seed}.npz").is_file()
            or not (
                transition_dir / f"{split}_seed_{seed}.returns.json"
            ).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Cannot combine incomplete frozen {split} split; missing {missing}."
            )
        arrays, return_rows = combine_seed_files(
            split,
            frozen_seeds,
            transition_dir,
            output_dir,
        )
        split_summaries[split] = {
            "episode_seeds": list(frozen_seeds),
            "episodes": len(frozen_seeds),
            "horizon_from_fork": horizon,
            "forks": len(return_rows),
            "transitions": int(arrays["reward"].size),
            "state_dim": int(arrays["state"].shape[1]),
        }

    if args.no_combine:
        print("Worker generation complete; combine deferred.")
        return

    manifest = {
        "status": "FROZEN_V28_DATASET",
        "checkpoint": str(checkpoint_path),
        "lambda": observed_dual,
        "gamma": GAMMA,
        "branch_points": list(BRANCH_POINTS),
        "snapshot_policies": list(SNAPSHOT_POLICIES),
        "branch_actions": list(BRANCH_ACTIONS),
        "episode_seed_split_isolation": True,
        "crn_role": "variance_reduction_only",
        "splits_generated": split_summaries,
    }
    with (output_dir / "dataset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
