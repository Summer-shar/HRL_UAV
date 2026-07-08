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
from hrl_uav.agents.ppo_micro_agent import MicroPPOAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Macro MAPPO + micro continuous-action PPO baseline",
    )
    add_common_args(parser, default_results_subdir="micro_ppo")
    parser.add_argument(
        "--micro_rollout_steps",
        type=int,
        default=0,
        help="Number of micro-step transitions per PPO update; 0 updates once per episode",
    )
    parser.add_argument(
        "--micro_update_epochs",
        type=int,
        default=4,
        help="Gradient epochs per micro PPO update",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1024,
        help="Mini-batch size for micro PPO",
    )
    parser.add_argument(
        "--micro_actor_lr",
        type=float,
        default=1e-4,
        help="Actor learning rate for micro PPO",
    )
    parser.add_argument(
        "--micro_critic_lr",
        type=float,
        default=5e-4,
        help="Critic learning rate for micro PPO",
    )
    parser.add_argument(
        "--micro_gamma",
        type=float,
        default=0.99,
        help="Discount factor for micro PPO",
    )
    parser.add_argument(
        "--micro_gae_lambda",
        type=float,
        default=0.95,
        help="GAE lambda for micro PPO",
    )
    parser.add_argument(
        "--micro_clip_epsilon",
        type=float,
        default=0.1,
        help="Clipping epsilon for micro PPO",
    )
    parser.add_argument(
        "--micro_entropy_coef",
        type=float,
        default=0.003,
        help="Entropy bonus coefficient for micro PPO",
    )
    parser.add_argument(
        "--micro_value_coef",
        type=float,
        default=0.5,
        help="Value loss coefficient for micro PPO",
    )
    return parser.parse_args()


def create_rollout_buffer():
    return {
        "states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "log_probs": [],
    }


def maybe_update_micro_agent(
    micro_agent,
    rollout_buffer,
    next_obs,
    done,
    args,
    force=False,
):
    rollout_size = len(rollout_buffer["states"])
    if rollout_size == 0:
        return None
    if not force:
        if args.micro_rollout_steps <= 0:
            return None
        if rollout_size < args.micro_rollout_steps:
            return None

    next_value = 0.0 if done else micro_agent.get_value(next_obs)
    advantages, returns = micro_agent.compute_gae(
        rollout_buffer["rewards"],
        rollout_buffer["values"],
        rollout_buffer["dones"],
        next_value,
    )
    actor_loss, critic_loss = micro_agent.update(
        rollout_buffer["states"],
        rollout_buffer["actions"],
        rollout_buffer["log_probs"],
        returns,
        advantages,
    )

    for key in rollout_buffer:
        rollout_buffer[key].clear()

    return actor_loss, critic_loss


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(results_dirs, args, "micro_ppo")

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

    micro_agent = MicroPPOAgent(
        state_dim=env.sac_obs_dim,
        action_dim=env.micro_action_space.shape[0],
        actor_lr=args.micro_actor_lr,
        critic_lr=args.micro_critic_lr,
        gamma=args.micro_gamma,
        gae_lambda=args.micro_gae_lambda,
        clip_epsilon=args.micro_clip_epsilon,
        update_epochs=args.micro_update_epochs,
        mini_batch_size=args.micro_batch_size,
        entropy_coef=args.micro_entropy_coef,
        value_coef=args.micro_value_coef,
    )

    history_avg_queue = []
    history_total_energy = []
    eval_history = EvaluationHistory()
    micro_rollout = create_rollout_buffer()
    micro_actor_losses = []
    micro_critic_losses = []

    def micro_policy(env_instance, deterministic=False):
        sac_obs = env_instance.get_sac_state()
        if deterministic:
            return micro_agent.choose_action_deterministic(sac_obs)
        action, _, _ = micro_agent.choose_action(sac_obs, deterministic=False)
        return action

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
        last_next_sac_obs = env.get_sac_state()

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
                sac_obs = env.get_sac_state()
                micro_action, log_prob, value = micro_agent.choose_action(
                    sac_obs,
                    deterministic=False,
                )

                next_sac_obs, mic_reward, done, _, mic_info = env.micro_step(micro_action)
                raw_state = env.env.get_raw_state()
                last_next_sac_obs = next_sac_obs
                episode_micro_reward += float(mic_reward)
                trace.record_micro_step(raw_state, mic_info)

                micro_rollout["states"].append(np.asarray(sac_obs, dtype=np.float32).copy())
                micro_rollout["actions"].append(np.asarray(micro_action, dtype=np.float32).copy())
                micro_rollout["rewards"].append(float(mic_reward))
                micro_rollout["dones"].append(float(done))
                micro_rollout["values"].append(float(value))
                micro_rollout["log_probs"].append(float(log_prob))

                update_info = maybe_update_micro_agent(
                    micro_agent,
                    micro_rollout,
                    next_sac_obs,
                    done,
                    args,
                    force=done,
                )
                if update_info is not None:
                    actor_loss, critic_loss = update_info
                    micro_actor_losses.append(actor_loss)
                    micro_critic_losses.append(critic_loss)

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

        update_info = maybe_update_micro_agent(
            micro_agent,
            micro_rollout,
            last_next_sac_obs,
            done,
            args,
            force=True,
        )
        if update_info is not None:
            actor_loss, critic_loss = update_info
            micro_actor_losses.append(actor_loss)
            micro_critic_losses.append(critic_loss)

        if not args.freeze_macro:
            macro_trainer.finish_episode(
                mappo_global_state,
                force_update=(episode == args.episodes - 1),
            )

        summary = trace.summary()
        history_avg_queue.append(summary["avg_queue"])
        history_total_energy.append(summary["total_energy"])

        print_episode_summary(
            experiment_label="MicroPPO",
            episode_idx=episode,
            episode_macro_reward=episode_macro_reward,
            episode_micro_reward=episode_micro_reward,
            trace=trace,
        )
        if micro_actor_losses and micro_critic_losses:
            print(
                f"  Micro PPO loss | Actor: {micro_actor_losses[-1]:.4f} | "
                f"Critic: {micro_critic_losses[-1]:.4f}"
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
                micro_policy,
            )
            eval_history.append(episode + 1, eval_metrics)
            print_evaluation_summary(episode + 1, eval_metrics)

    maybe_append_final_evaluation(
        eval_history,
        args,
        eval_env,
        macro_trainer.macro_policy,
        micro_policy,
    )
    os.makedirs(results_dirs["ckpt_dir"], exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "macro_actors": [actor.old_pi.state_dict() for actor in macro_trainer.actors],
            "macro_critic": macro_trainer.critic.critic_v.state_dict(),
            "micro_actor": micro_agent.actor.state_dict(),
            "micro_critic": micro_agent.critic.state_dict(),
            "micro_actor_optimizer": micro_agent.actor_optimizer.state_dict(),
            "micro_critic_optimizer": micro_agent.critic_optimizer.state_dict(),
        },
        os.path.join(results_dirs["ckpt_dir"], "final.pt"),
    )
    save_experiment_artifacts(
        results_dirs["results_dir"],
        "eval_micro_ppo.csv",
        eval_history,
        history_avg_queue,
        history_total_energy,
    )


if __name__ == "__main__":
    main()
