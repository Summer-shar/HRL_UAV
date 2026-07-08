import argparse
import os

import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import EpisodeTrace
from compare_common import EvaluationHistory
from compare_common import add_common_args
from compare_common import build_local_only_micro_action
from compare_common import evaluate_policy
from compare_common import load_macro_checkpoint
from compare_common import maybe_append_final_evaluation
from compare_common import maybe_save_episode_snapshots
from compare_common import make_env_from_args
from compare_common import prepare_results_dirs
from compare_common import print_episode_summary
from compare_common import print_evaluation_summary
from compare_common import save_experiment_artifacts
from compare_common import save_scenario_metadata
from compare_common import set_global_seeds
from compare_macro_ppo import MacroPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Macro MAPPO + no-cooperation micro baseline",
    )
    add_common_args(parser, default_results_subdir="no_coop")
    return parser.parse_args()


def no_coop_micro_policy(env, deterministic=False):
    return build_local_only_micro_action(env.N)


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(results_dirs, args, "no_coop")

    env = make_env_from_args(
        args,
        record_trajectory=False,
    )
    eval_env = make_env_from_args(args, record_trajectory=False)

    macro_trainer = MacroPPOTrainer(
        local_state_dim=env.macro_local_obs_dim,
        global_state_dim=env.macro_global_obs_dim,
        num_agents=args.num_agents,
        macro_steps_per_episode=args.macro_steps,
    )
    loaded_macro_path = load_macro_checkpoint(
        macro_trainer.actors,
        macro_trainer.critic,
        args.macro_ckpt,
    )
    if loaded_macro_path is not None:
        print(f"[INFO] Loaded macro MAPPO checkpoint: {loaded_macro_path}")
    if args.freeze_macro and loaded_macro_path is None:
        raise ValueError("--freeze_macro requires --macro_ckpt.")

    history_avg_queue = []
    history_total_energy = []
    eval_history = EvaluationHistory()

    for episode in range(args.episodes):
        should_record_trajectory = (
            args.trajectory_plot_interval > 0
            and (episode + 1) % args.trajectory_plot_interval == 0
        )
        env.env.record_trajectory = should_record_trajectory
        env.reset(seed=args.seed + episode)
        trace = EpisodeTrace()
        episode_macro_reward = 0.0
        episode_micro_reward = 0.0
        done = False

        macro_trainer.start_episode()
        mappo_local_state, mappo_global_state = env.get_macro_state()

        for _ in range(args.macro_steps):
            local_obs_batch = np.vstack(mappo_local_state).astype(np.float32)
            if args.freeze_macro or args.macro_deterministic:
                mac_action = macro_trainer.act(
                    local_obs_batch,
                    deterministic=True,
                )
                mac_action_logprob = np.zeros(
                    (args.num_agents, 1),
                    dtype=np.float32,
                )
            else:
                mac_action, mac_action_logprob = macro_trainer.act(
                    local_obs_batch,
                    deterministic=False,
                )

            mac_action = np.asarray(mac_action, dtype=np.float32)
            mac_action_logprob = np.asarray(
                mac_action_logprob,
                dtype=np.float32,
            ).reshape(args.num_agents, 1)

            _, _, mac_info = env.macro_step(mac_action)
            trace.record_macro_info(mac_info)

            for _ in range(args.micro_steps):
                micro_action = no_coop_micro_policy(env, deterministic=False)
                _, mic_reward, done, _, mic_info = env.micro_step(micro_action)
                raw_state = env.env.get_raw_state()
                episode_micro_reward += float(mic_reward)
                trace.record_micro_step(raw_state, mic_info)
                if done:
                    break

            next_local_state, next_global_state = env.get_macro_state()
            macro_reward = env.get_macro_reward()
            episode_macro_reward += float(macro_reward)
            macro_reward_info = env.get_last_macro_reward_info()
            trace.record_macro_reward(macro_reward, macro_reward_info)

            reward_per_uav = np.asarray(
                macro_reward_info.get(
                    "macro_rewards_per_uav",
                    np.full(args.num_agents, macro_reward, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(args.num_agents)
            if not args.freeze_macro:
                macro_trainer.record_step(
                    local_obs_batch=local_obs_batch,
                    global_state=mappo_global_state,
                    action=mac_action,
                    action_logprob=mac_action_logprob,
                    reward_per_uav=reward_per_uav,
                )

            mappo_local_state = next_local_state
            mappo_global_state = next_global_state

            if done:
                break

        if not args.freeze_macro:
            macro_trainer.finish_episode(
                mappo_global_state,
                force_update=(episode == args.episodes - 1),
            )

        summary = trace.summary()
        history_avg_queue.append(summary["avg_queue"])
        history_total_energy.append(summary["total_energy"])

        print_episode_summary(
            experiment_label="NoCoop",
            episode_idx=episode,
            episode_macro_reward=episode_macro_reward,
            episode_micro_reward=episode_micro_reward,
            trace=trace,
        )

        maybe_save_episode_snapshots(
            env,
            results_dirs,
            args.trajectory_plot_interval,
            episode,
            trace,
        )

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_policy(
                eval_env,
                args,
                macro_trainer.macro_policy,
                no_coop_micro_policy,
            )
            eval_history.append(episode + 1, eval_metrics)
            print_evaluation_summary(episode + 1, eval_metrics)

    maybe_append_final_evaluation(
        eval_history,
        args,
        eval_env,
        macro_trainer.macro_policy,
        no_coop_micro_policy,
    )
    os.makedirs(results_dirs["ckpt_dir"], exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "macro_actors": [actor.old_pi.state_dict() for actor in macro_trainer.actors],
            "macro_critic": macro_trainer.critic.critic_v.state_dict(),
        },
        os.path.join(results_dirs["ckpt_dir"], "final.pt"),
    )
    save_experiment_artifacts(
        results_dirs["results_dir"],
        "eval_no_coop.csv",
        eval_history,
        history_avg_queue,
        history_total_energy,
    )


if __name__ == "__main__":
    main()
