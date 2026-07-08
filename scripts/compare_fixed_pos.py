"""
Fixed-position baseline: UAVs stay at zone centroids (zero macro action),
while micro-layer SAC learns offloading/forwarding decisions.

This isolates the contribution of trajectory optimization.
"""
import argparse

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
from compare_common import make_env_from_args
from compare_common import maybe_append_final_evaluation
from compare_common import maybe_save_episode_snapshots
from compare_common import prepare_results_dirs
from compare_common import print_episode_summary
from compare_common import print_evaluation_summary
from compare_common import save_experiment_artifacts
from compare_common import save_scenario_metadata
from compare_common import set_global_seeds

import hrl_uav.agents.sac_agent as sac_agent_mod
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
from hrl_uav.agents.sac_agent import Critic as SAC_Critic
from hrl_uav.agents.sac_agent import Entroy as SAC_Entropy
from hrl_uav.agents.sac_agent import Memory as SAC_Memory

# SAC hyperparameters (same as train_hrl.py)
SAC_GAMMA = 0.99
SAC_BATCH_SIZE = 256
SAC_MEMORY_WARMUP = 1000
SAC_UPDATE_FREQ = 100
SAC_MEMORY_CAPACITY = 500000
SAC_GRADIENT_STEPS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-position + SAC micro baseline (no trajectory learning)",
    )
    add_common_args(parser, default_results_subdir="fixed_pos")
    return parser.parse_args()


def fixed_macro_policy(env, local_obs_batch, deterministic=False):
    """Zero displacement: UAVs stay at their current positions (anchors after reset)."""
    return np.zeros((env.N, 2), dtype=np.float32)


class FixedPosSACTrainer:
    """SAC micro-layer trainer with fixed (zero-action) macro layer."""

    def __init__(self, obs_dim, action_dim, num_agents):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.N = num_agents

        self.actor = SAC_Actor(obs_dim, action_dim)
        self.critic = SAC_Critic(obs_dim, action_dim)
        self.entropy = SAC_Entropy(action_dim)
        # Memory dims: state + action + reward + next_state
        mem_dims = obs_dim + action_dim + 1 + obs_dim
        self.memory = SAC_Memory(SAC_MEMORY_CAPACITY, mem_dims)
        self.global_step = 0

        import torch as _torch
        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    def select_action(self, state, deterministic=False):
        return self.actor.choose_action(state, deterministic=deterministic)

    def micro_policy(self, env, deterministic=False):
        raw_state = env.last_raw_state or env.env.get_raw_state()
        sac_state = env.get_sac_state(raw_state)
        return self.select_action(sac_state, deterministic=deterministic)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state)
        self.global_step += 1

    def maybe_update(self):
        if self.memory.memory_counter < SAC_MEMORY_WARMUP:
            return
        if self.global_step % SAC_UPDATE_FREQ != 0:
            return

        import torch as _torch

        for _ in range(SAC_GRADIENT_STEPS):
            b_M = self.memory.sample(SAC_BATCH_SIZE)
            b_s = b_M[:, :self.obs_dim]
            b_a = b_M[:, self.obs_dim:self.obs_dim + self.action_dim]
            b_r = b_M[:, self.obs_dim + self.action_dim:self.obs_dim + self.action_dim + 1]
            b_s_ = b_M[:, -self.obs_dim:]

            b_s = _torch.FloatTensor(b_s).to(self.device)
            b_a = _torch.FloatTensor(b_a).to(self.device)
            b_r = _torch.FloatTensor(b_r).to(self.device)
            b_s_ = _torch.FloatTensor(b_s_).to(self.device)

            # Target Q
            new_action, log_prob_ = self.actor.evaluate(b_s_)
            target_q1, target_q2 = self.critic.target_critic_v(b_s_, new_action)
            target_q = b_r + SAC_GAMMA * (
                _torch.min(target_q1, target_q2)
                - self.entropy.alpha.detach().to(self.device) * log_prob_
            )

            # Critic update
            current_q1, current_q2 = self.critic.get_v(b_s, b_a)
            self.critic.learn(current_q1, current_q2, target_q.detach())

            # Actor update
            a_eval, log_prob = self.actor.evaluate(b_s)
            q1_eval, q2_eval = self.critic.get_v(b_s, a_eval)
            q_eval = _torch.min(q1_eval, q2_eval)
            actor_loss = (
                self.entropy.alpha.detach().to(self.device) * log_prob - q_eval
            ).mean()
            self.actor.learn(actor_loss)

            # Entropy update
            alpha_loss = -(
                self.entropy.log_alpha.exp().to(self.device)
                * (log_prob + self.entropy.target_entropy).detach()
            ).mean()
            self.entropy.learn(alpha_loss)
            self.entropy.alpha = self.entropy.log_alpha.exp().to(self.device)


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(results_dirs, args, "FixedPos+SAC")

    env = make_env_from_args(args, record_trajectory=False)
    eval_env = make_env_from_args(args, record_trajectory=False)

    sac_trainer = FixedPosSACTrainer(
        obs_dim=env.sac_obs_dim,
        action_dim=2 * env.N * env.N,
        num_agents=args.num_agents,
    )

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

        mappo_local_state, _ = env.get_macro_state()

        for _ in range(args.macro_steps):
            # Fixed position: zero macro action
            macro_action = fixed_macro_policy(env, None)
            _, _, mac_info = env.macro_step(macro_action)
            trace.record_macro_info(mac_info)

            for _ in range(args.micro_steps):
                if env.env.macro_fly_steps_remaining > 0:
                    micro_action = np.zeros(
                        2 * env.N * env.N, dtype=np.float32
                    )
                    _, mic_reward, done, _, mic_info = env.micro_step(micro_action)
                    raw_state = env.env.get_raw_state()
                else:
                    raw_state_before = env.last_raw_state or env.env.get_raw_state()
                    sac_state = env.get_sac_state(raw_state_before)
                    micro_action = sac_trainer.select_action(sac_state, deterministic=False)
                    micro_action = np.asarray(micro_action, dtype=np.float32)
                    _, mic_reward, done, _, mic_info = env.micro_step(micro_action)
                    raw_state = env.env.get_raw_state()
                    sac_next_state = env.get_sac_state(raw_state)
                    sac_trainer.store_transition(
                        sac_state, micro_action, mic_reward, sac_next_state, done
                    )
                    sac_trainer.maybe_update()

                episode_micro_reward += float(mic_reward)
                trace.record_micro_step(raw_state, mic_info)
                if done:
                    break

            mappo_local_state, _ = env.get_macro_state()
            macro_reward = env.get_macro_reward()
            episode_macro_reward += float(macro_reward)
            trace.record_macro_reward(macro_reward, env.get_last_macro_reward_info())
            if done:
                break

        summary = trace.summary()
        history_avg_queue.append(summary["avg_queue"])
        history_total_energy.append(summary["total_energy"])

        print_episode_summary(
            experiment_label="FixedPos+SAC",
            episode_idx=episode,
            episode_macro_reward=episode_macro_reward,
            episode_micro_reward=episode_micro_reward,
            trace=trace,
        )

        maybe_save_episode_snapshots(
            env, results_dirs, args.trajectory_plot_interval, episode, trace
        )

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_policy(
                eval_env,
                args,
                fixed_macro_policy,
                sac_trainer.micro_policy,
            )
            eval_history.append(episode + 1, eval_metrics)
            print_evaluation_summary(episode + 1, eval_metrics)

    maybe_append_final_evaluation(
        eval_history, args, eval_env, fixed_macro_policy, sac_trainer.micro_policy
    )
    # Save SAC checkpoint
    import os
    os.makedirs(results_dirs["ckpt_dir"], exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "sac_actor": sac_trainer.actor.action_net.state_dict(),
            "sac_critic": sac_trainer.critic.critic_v.state_dict(),
        },
        os.path.join(results_dirs["ckpt_dir"], "final.pt"),
    )
    save_experiment_artifacts(
        results_dirs["results_dir"],
        "eval_fixed_pos.csv",
        eval_history,
        history_avg_queue,
        history_total_energy,
    )


if __name__ == "__main__":
    main()
