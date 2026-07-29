import argparse
import json

import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
import hrl_uav.agents.sac_agent as sac_agent_mod
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import sac_micro_policy
from train_centralized_td3 import MacroActor
from train_centralized_td3 import td3_macro_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-evaluate old/new micro policies with centroid/learned "
            "macro policies under common random numbers."
        )
    )
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--pretrained-micro-checkpoint", required=True)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-base-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--per-uav-counterfactual",
        action="store_true",
        help="Also pin each learned UAV waypoint to its own centroid",
    )
    return parser.parse_args()


def load_micro_actor(checkpoint, env):
    actor = SAC_Actor(
        int(env.sac_obs_dim),
        int(env.micro_action_space.shape[0]),
    )
    actor.action_net.load_state_dict(checkpoint["sac_actor"])
    actor.action_net.eval()
    return actor


def centroid_macro_policy(env, _local_obs_batch, _eval_mode):
    return np.zeros((env.N, 2), dtype=np.float32)


def pin_uav_to_centroid(base_policy, uav_index):
    def policy(env, local_obs_batch, eval_mode):
        action = np.asarray(
            base_policy(env, local_obs_batch, eval_mode),
            dtype=np.float32,
        ).copy()
        action[int(uav_index)] = 0.0
        return action

    return policy


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


def select_metrics(metrics):
    keys = (
        "J",
        "delay_objective",
        "task_delay_weighted_ms",
        "task_delay_high_ms",
        "task_delay_low_ms",
        "average_power_w",
        "average_power_ucb95_w",
        "u2g",
        "coop_ratio",
        "cpu_utilization",
    )
    return {key: float(metrics.get(key, float("nan"))) for key in keys}


def main():
    cli_args = parse_args()
    requested_device = cli_args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    sac_agent_mod.device = device

    joint_checkpoint = torch_load_checkpoint(cli_args.joint_checkpoint)
    pretrained_checkpoint = torch_load_checkpoint(
        cli_args.pretrained_micro_checkpoint
    )
    eval_args = argparse.Namespace(**joint_checkpoint["args"])
    eval_args.eval_episodes = int(cli_args.eval_episodes)
    eval_args.eval_base_seed = int(cli_args.eval_base_seed)
    eval_args.device = str(device)

    probe_env = build_env(eval_args)
    _, global_state = probe_env.reset(seed=eval_args.eval_base_seed)[0]
    macro_actor = MacroActor(
        int(np.asarray(global_state).size),
        int(eval_args.num_agents) * 2,
        hidden_dim=int(eval_args.td3_hidden_dim),
    ).to(device)
    macro_actor.load_state_dict(joint_checkpoint["macro_actor"])
    macro_actor.eval()

    pretrained_micro = load_micro_actor(pretrained_checkpoint, probe_env)
    joint_micro = load_micro_actor(joint_checkpoint, probe_env)
    learned_macro = td3_macro_policy(macro_actor)
    policies = [
        (
            "pretrained_micro_centroid",
            centroid_macro_policy,
            sac_micro_policy(pretrained_micro),
        ),
        (
            "pretrained_micro_learned",
            learned_macro,
            sac_micro_policy(pretrained_micro),
        ),
        (
            "joint_micro_centroid",
            centroid_macro_policy,
            sac_micro_policy(joint_micro),
        ),
        (
            "joint_micro_learned",
            learned_macro,
            sac_micro_policy(joint_micro),
        ),
    ]
    if cli_args.per_uav_counterfactual:
        for uav_index in range(int(eval_args.num_agents)):
            policies.append(
                (
                    f"pretrained_micro_pin_uav{uav_index + 1}_centroid",
                    pin_uav_to_centroid(learned_macro, uav_index),
                    sac_micro_policy(pretrained_micro),
                )
            )

    results = {}
    for name, macro_policy, micro_policy in policies:
        env = build_env(eval_args)
        metrics = evaluate_policy(
            env,
            eval_args,
            macro_policy,
            micro_policy,
        )
        results[name] = select_metrics(metrics)
        print(json.dumps({name: results[name]}, ensure_ascii=False))

    print("SUMMARY=" + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
