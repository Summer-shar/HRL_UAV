import argparse
import copy
import csv
import hashlib
import json
import os

import numpy as np
import torch
from scipy.stats import t as student_t

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import make_env_from_args
from compare_common import torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
import hrl_uav.agents.sac_agent as sac_agent_mod
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import project_unit_disk_np
from train_centralized_macro import sac_micro_policy
from train_centralized_td3 import MacroActor
from train_centralized_td3 import TwinCritic
from train_centralized_td3 import macro_power_violation_penalty


DEFAULT_BRANCH_POINTS = (0, 5, 10, 15, 20, 25)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare TD3 Critic rankings with same-state counterfactual "
            "returns. The first macro action differs; both branches follow "
            "the same deterministic Actor thereafter."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-base-seed", type=int, default=61000)
    parser.add_argument(
        "--branch-points",
        default="0,5,10,15,20,25",
        help="Comma-separated zero-based macro indices.",
    )
    parser.add_argument(
        "--snapshot-policy",
        choices=["actor", "centroid", "both"],
        default="centroid",
        help=(
            "Policy used only to collect branch states. Both counterfactual "
            "branches use the Actor after their different first actions."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def parse_branch_points(raw_value, macro_steps):
    points = sorted(
        {
            int(value.strip())
            for value in str(raw_value).split(",")
            if value.strip()
        }
    )
    invalid = [point for point in points if point < 0 or point >= macro_steps]
    if invalid:
        raise ValueError(
            f"Branch points must be in [0, {macro_steps - 1}]: {invalid}"
        )
    if not points:
        raise ValueError("At least one branch point is required.")
    return tuple(points)


def build_env(eval_args):
    env = make_env_from_args(eval_args, record_trajectory=False)
    configure_macro_objective(
        env,
        eval_args,
        float(getattr(eval_args, "macro_shaping_final", 0.0)),
    )
    if str(getattr(eval_args, "power_constraint_scope", "system")) == "per_uav":
        multipliers = getattr(
            eval_args,
            "energy_lagrange_multipliers_current",
            None,
        )
        if multipliers is not None:
            env.set_energy_lagrange_multipliers(multipliers)
    else:
        env.energy_lagrange_multiplier = float(
            getattr(eval_args, "energy_lagrange_multiplier_current", 0.0)
        )
        env._sync_cpu_energy_prices()
    return env


def load_models(checkpoint, env, eval_args, device):
    _, global_state = env.get_macro_state()
    state_dim = int(np.asarray(global_state).size)
    action_dim = int(eval_args.num_agents) * 2

    actor = MacroActor(
        state_dim,
        action_dim,
        hidden_dim=int(eval_args.td3_hidden_dim),
    ).to(device)
    actor.load_state_dict(checkpoint["macro_actor"])
    actor.eval()

    critic = TwinCritic(
        state_dim,
        action_dim,
        hidden_dim=int(eval_args.td3_hidden_dim),
    ).to(device)
    critic_state = checkpoint.get(
        "macro_delay_critic",
        checkpoint.get("macro_critic"),
    )
    if critic_state is None:
        raise KeyError("Checkpoint does not contain a macro delay Critic.")
    critic.load_state_dict(critic_state)
    critic.eval()

    if "sac_actor" not in checkpoint:
        raise KeyError("Counterfactual diagnosis requires a SAC checkpoint.")
    micro_actor = SAC_Actor(
        int(env.sac_obs_dim),
        int(env.micro_action_space.shape[0]),
    )
    micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    micro_actor.action_net.eval()
    return actor, critic, sac_micro_policy(micro_actor)


def actor_action(actor, env):
    _, global_state = env.get_macro_state()
    action = actor.choose_action(global_state)
    return project_unit_disk_np(action.reshape(env.N, 2))


def critic_values(critic, state, actor_action_value, device):
    state_t = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    actor_t = torch.as_tensor(
        np.asarray(actor_action_value, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=device,
    )
    zero_t = torch.zeros_like(actor_t)
    with torch.no_grad():
        q1_actor, q2_actor = critic(state_t, actor_t)
        q1_zero, q2_zero = critic(state_t, zero_t)
    values = {
        "q1_actor": float(q1_actor.item()),
        "q2_actor": float(q2_actor.item()),
        "q1_zero": float(q1_zero.item()),
        "q2_zero": float(q2_zero.item()),
    }
    values["qmin_actor"] = min(values["q1_actor"], values["q2_actor"])
    values["qmin_zero"] = min(values["q1_zero"], values["q2_zero"])
    values["q_actor_minus_zero"] = (
        values["qmin_actor"] - values["qmin_zero"]
    )
    return values


def _hash_array(hasher, name, values, dtype):
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    hasher.update(name.encode("utf-8"))
    hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
    hasher.update(array.tobytes())


def exogenous_frame_digest(env, frame):
    hasher = hashlib.sha256()
    _hash_array(hasher, "user_coords", env.env.user_coords, "<f8")
    _hash_array(hasher, "user_velocities", env.env.user_velocities, "<f8")
    for index, info in enumerate(frame["infos"]):
        _hash_array(
            hasher,
            f"arrival_high_{index}",
            info.get("arrival_high_per_uav", np.zeros(env.N)),
            "<f8",
        )
        _hash_array(
            hasher,
            f"arrival_low_{index}",
            info.get("arrival_low_per_uav", np.zeros(env.N)),
            "<f8",
        )
    rng_state = env.env.np_random.get_state()
    hasher.update(str(rng_state[0]).encode("ascii"))
    _hash_array(hasher, "rng_keys", rng_state[1], "<u4")
    _hash_array(
        hasher,
        "rng_tail",
        [rng_state[2], rng_state[3], rng_state[4]],
        "<f8",
    )
    return hasher.hexdigest()


def replay_macro_reward(env, eval_args):
    macro_reward = float(env.get_macro_reward())
    reward_info = env.get_last_macro_reward_info()
    if str(eval_args.objective_mode) != "constrained_delay":
        return macro_reward * float(eval_args.td3_reward_scale)

    if str(getattr(eval_args, "power_constraint_scope", "system")) == "per_uav":
        energy_ratio = np.asarray(
            reward_info.get(
                "energy_constraint_ratio_per_uav_mean",
                np.zeros(env.N),
            ),
            dtype=np.float64,
        )
    else:
        energy_ratio = float(
            reward_info.get("energy_constraint_ratio_mean", 0.0)
        )
    penalty = macro_power_violation_penalty(
        energy_ratio,
        float(eval_args.macro_power_violation_penalty_weight),
    )
    primal_cost = float(reward_info.get("primal_cost_mean", 0.0)) + penalty
    return (
        -float(env.micro_reward_scale)
        * primal_cost
        * float(eval_args.td3_reward_scale)
    )


def execute_macro_interval(env, eval_args, action, micro_policy):
    env.macro_step(np.asarray(action, dtype=np.float32))
    done = False
    trace = []
    for _ in range(env.env.resource_frames_per_macro):
        service_paused = bool(
            getattr(env.env, "pause_service_during_flight", False)
            and env.env.macro_service_pause_steps_remaining > 0
        )
        if service_paused:
            micro_action = np.zeros(
                env.micro_action_space.shape[0],
                dtype=np.float32,
            )
        else:
            env.prepare_resource_frame()
            micro_action = np.asarray(
                micro_policy(env, True),
                dtype=np.float32,
            )
        frame = env.step_resource_frame(micro_action)
        trace.append(exogenous_frame_digest(env, frame))
        done = done or bool(frame["done"])
        if done:
            break
    reward = replay_macro_reward(env, eval_args)
    return reward, done, trace


def discounted_return(rewards, gamma):
    total = 0.0
    for reward in reversed(rewards):
        total = float(reward) + float(gamma) * total
    return total


def collect_reference_snapshots(
    env,
    eval_args,
    actor,
    critic,
    micro_policy,
    branch_points,
    device,
    snapshot_policy,
):
    snapshots = {}
    critic_records = {}
    done = False

    for macro_idx in range(int(eval_args.macro_steps)):
        _, state = env.get_macro_state()
        candidate_actor_action = actor_action(actor, env)
        if macro_idx in branch_points:
            snapshots[macro_idx] = copy.deepcopy(env)
            critic_records[macro_idx] = {
                **critic_values(
                    critic,
                    state,
                    candidate_actor_action,
                    device,
                ),
                "actor_action_norm_mean": float(
                    np.mean(np.linalg.norm(candidate_actor_action, axis=1))
                ),
                "actor_action_norm_max": float(
                    np.max(np.linalg.norm(candidate_actor_action, axis=1))
                ),
            }
        if snapshot_policy == "actor":
            reference_action = candidate_actor_action
        elif snapshot_policy == "centroid":
            reference_action = np.zeros((env.N, 2), dtype=np.float32)
        else:
            raise ValueError(f"Unknown snapshot policy: {snapshot_policy}")
        _, done, _ = execute_macro_interval(
            env,
            eval_args,
            reference_action,
            micro_policy,
        )
        if done:
            break
    return snapshots, critic_records


def run_first_action_branch(
    snapshot,
    start_macro,
    eval_args,
    actor,
    micro_policy,
    first_action,
):
    rewards = []
    traces = []
    done = False
    for macro_idx in range(start_macro, int(eval_args.macro_steps)):
        if macro_idx == start_macro:
            if first_action == "actor":
                action = actor_action(actor, snapshot)
            elif first_action == "zero":
                action = np.zeros((snapshot.N, 2), dtype=np.float32)
            else:
                raise ValueError(f"Unknown first action: {first_action}")
        else:
            action = actor_action(actor, snapshot)
        reward, done, macro_trace = execute_macro_interval(
            snapshot,
            eval_args,
            action,
            micro_policy,
        )
        rewards.append(reward)
        traces.append(macro_trace)
        if done:
            break
    return rewards, traces


def flatten_trace(trace_by_macro):
    return [digest for macro_trace in trace_by_macro for digest in macro_trace]


def summarize_episode_rows(rows):
    return {
        "episode_index": int(rows[0]["episode_index"]),
        "seed": int(rows[0]["seed"]),
        "snapshot_policy": str(rows[0]["snapshot_policy"]),
        "branch_count": len(rows),
        "trace_match": bool(all(row["trace_match"] for row in rows)),
        "q_actor_minus_zero_mean": float(
            np.mean([row["q_actor_minus_zero"] for row in rows])
        ),
        "return_actor_minus_zero_mean": float(
            np.mean([row["return_actor_minus_zero"] for row in rows])
        ),
        "ranking_agreement_fraction": float(
            np.mean([row["ranking_agreement"] for row in rows])
        ),
    }


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, None, None
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    margin = float(student_t.ppf(0.975, values.size - 1) * sem)
    return mean, mean - margin, mean + margin


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli_args = parse_args()
    requested_device = cli_args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    sac_agent_mod.device = device

    checkpoint_path = os.path.abspath(cli_args.checkpoint)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    eval_args = argparse.Namespace(**checkpoint["args"])
    eval_args.device = str(device)
    branch_points = parse_branch_points(
        cli_args.branch_points,
        int(eval_args.macro_steps),
    )
    output_dir = (
        os.path.abspath(cli_args.output_dir)
        if cli_args.output_dir
        else os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)),
            "counterfactual_diagnosis",
        )
    )

    probe_env = build_env(eval_args)
    probe_env.reset(seed=int(cli_args.eval_base_seed))
    actor, critic, micro_policy = load_models(
        checkpoint,
        probe_env,
        eval_args,
        device,
    )

    raw_rows = []
    episode_rows = []
    snapshot_policies = (
        ("actor", "centroid")
        if cli_args.snapshot_policy == "both"
        else (cli_args.snapshot_policy,)
    )
    for episode_index in range(int(cli_args.episodes)):
        seed = int(cli_args.eval_base_seed) + episode_index
        for snapshot_policy in snapshot_policies:
            env = build_env(eval_args)
            env.reset(seed=seed)
            snapshots, critic_records = collect_reference_snapshots(
                env,
                eval_args,
                actor,
                critic,
                micro_policy,
                branch_points,
                device,
                snapshot_policy,
            )
            current_episode_rows = []
            for branch_macro in branch_points:
                if branch_macro not in snapshots:
                    continue
                actor_rewards, actor_traces = run_first_action_branch(
                    copy.deepcopy(snapshots[branch_macro]),
                    branch_macro,
                    eval_args,
                    actor,
                    micro_policy,
                    first_action="actor",
                )
                zero_rewards, zero_traces = run_first_action_branch(
                    copy.deepcopy(snapshots[branch_macro]),
                    branch_macro,
                    eval_args,
                    actor,
                    micro_policy,
                    first_action="zero",
                )
                actor_return = discounted_return(
                    actor_rewards,
                    eval_args.td3_gamma,
                )
                zero_return = discounted_return(
                    zero_rewards,
                    eval_args.td3_gamma,
                )
                values = critic_records[branch_macro]
                q_difference = float(values["q_actor_minus_zero"])
                return_difference = float(actor_return - zero_return)
                row = {
                    "episode_index": episode_index,
                    "seed": seed,
                    "snapshot_policy": snapshot_policy,
                    "branch_macro": branch_macro,
                    **values,
                    "return_actor": actor_return,
                    "return_zero": zero_return,
                    "return_actor_minus_zero": return_difference,
                    "ranking_agreement": float(
                        np.sign(q_difference) == np.sign(return_difference)
                    ),
                    "trace_match": (
                        flatten_trace(actor_traces)
                        == flatten_trace(zero_traces)
                    ),
                }
                raw_rows.append(row)
                current_episode_rows.append(row)
            if not current_episode_rows:
                raise RuntimeError(
                    f"No valid branches were produced for seed {seed}."
                )
            episode_summary = summarize_episode_rows(current_episode_rows)
            episode_rows.append(episode_summary)
            print(json.dumps(episode_summary, ensure_ascii=False))

    if not all(row["trace_match"] for row in raw_rows):
        raise RuntimeError(
            "Exogenous traces differ between counterfactual branches."
        )

    policy_summaries = {}
    for snapshot_policy in snapshot_policies:
        policy_rows = [
            row
            for row in episode_rows
            if row["snapshot_policy"] == snapshot_policy
        ]
        return_differences = [
            row["return_actor_minus_zero_mean"] for row in policy_rows
        ]
        q_differences = [
            row["q_actor_minus_zero_mean"] for row in policy_rows
        ]
        return_mean, return_low, return_high = mean_ci95(
            return_differences
        )
        q_mean, q_low, q_high = mean_ci95(q_differences)
        policy_summaries[snapshot_policy] = {
            "episodes": len(policy_rows),
            "return_actor_minus_zero_mean": return_mean,
            "return_actor_minus_zero_ci95": [return_low, return_high],
            "q_actor_minus_zero_mean": q_mean,
            "q_actor_minus_zero_ci95": [q_low, q_high],
            "ranking_agreement_fraction_mean": float(
                np.mean(
                    [
                        row["ranking_agreement_fraction"]
                        for row in policy_rows
                    ]
                )
            ),
        }
    summary = {
        "checkpoint": checkpoint_path,
        "episodes_per_snapshot_policy": int(cli_args.episodes),
        "branch_points": list(branch_points),
        "statistical_unit": "episode",
        "exogenous_trace_match": True,
        "by_snapshot_policy": policy_summaries,
    }
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "branch_raw.csv"), raw_rows)
    write_csv(os.path.join(output_dir, "episode_summary.csv"), episode_rows)
    with open(
        os.path.join(output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
