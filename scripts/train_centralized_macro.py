import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

from compare_common import EvaluationHistory
from compare_common import add_common_args
from compare_common import build_micro_step_metrics
from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import prepare_results_dirs
from compare_common import print_evaluation_summary
from compare_common import save_scenario_metadata
from hrl_uav.agents.ppo_agent import A_UPDATE_STEPS
from hrl_uav.agents.ppo_agent import CLIP_EPSILON
from hrl_uav.agents.ppo_agent import Actor as PPO_Actor
from hrl_uav.agents.ppo_agent import Critic as PPO_Critic
from hrl_uav.agents.ppo_agent import GAMMA as PPO_GAMMA
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
from hrl_uav.agents.sac_agent import Critic as SAC_Critic
from hrl_uav.agents.sac_agent import Entroy as SAC_Entropy
from hrl_uav.agents.sac_agent import Memory as SAC_Memory
import hrl_uav.agents.ppo_agent as ppo_agent_mod
import hrl_uav.agents.sac_agent as sac_agent_mod


PPO_LAMBDA = 0.95
SAC_GAMMA = 0.99
SAC_MEMORY_WARMUP = 1000
SAC_MEMORY_CAPACITY = 500000
SAC_RECENT_WINDOW = 500000
SAC_RECENT_RATIO = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized macro PPO + SAC micro training"
    )
    add_common_args(parser, default_results_subdir="central_macro")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--ppo_update_episode_window",
        type=int,
        default=20,
        help="Number of episodes worth of macro steps per PPO update",
    )
    parser.add_argument(
        "--advisor_weight",
        type=float,
        default=1.0,
        help="Initial imitation weight for the communication advisor",
    )
    parser.add_argument(
        "--advisor_final_weight",
        type=float,
        default=0.05,
        help="Final advisor imitation weight after annealing",
    )
    parser.add_argument(
        "--advisor_decay_episodes",
        type=int,
        default=200,
        help="Episodes over which advisor imitation weight is annealed",
    )
    parser.add_argument(
        "--advisor_steps",
        type=int,
        default=8,
        help="Projected gradient steps used by the communication advisor",
    )
    parser.add_argument(
        "--advisor_lr",
        type=float,
        default=0.35,
        help="Adam learning rate for advisor action optimization",
    )
    parser.add_argument(
        "--advisor_min_separation_m",
        type=float,
        default=80.0,
        help="Minimum UAV separation encouraged by advisor penalty",
    )
    parser.add_argument(
        "--advisor_move_penalty",
        type=float,
        default=0.02,
        help="Small advisor penalty on large macro displacements",
    )
    parser.add_argument(
        "--advisor_sep_penalty",
        type=float,
        default=10.0,
        help="Advisor penalty weight for violating UAV separation",
    )
    parser.add_argument(
        "--advisor_boundary_penalty",
        type=float,
        default=10.0,
        help="Advisor penalty weight for leaving the 1000m x 1000m area",
    )
    parser.add_argument(
        "--train_load_min",
        type=float,
        default=0.0,
        help="Minimum episode load_scale for randomized training; <=0 disables",
    )
    parser.add_argument(
        "--train_load_max",
        type=float,
        default=0.0,
        help="Maximum episode load_scale for randomized training; <=0 disables",
    )
    args = parser.parse_args()

    args.macro_action_mode = "delta"
    if "--macro_comm_weight" not in sys.argv:
        args.macro_comm_weight = 0.0
    return args


def set_wrapper_load_scale(env, load_scale):
    env.env.load_scale = float(load_scale)


def sample_training_load(args):
    lo = float(getattr(args, "train_load_min", 0.0))
    hi = float(getattr(args, "train_load_max", 0.0))
    if lo > 0.0 and hi >= lo:
        return float(np.random.uniform(lo, hi))
    return float(getattr(args, "load_scale", 1.0))


def advisor_weight_for_episode(args, episode):
    initial = float(args.advisor_weight)
    final = float(args.advisor_final_weight)
    decay = max(int(args.advisor_decay_episodes), 1)
    progress = min(max(float(episode) / float(decay), 0.0), 1.0)
    return final + (initial - final) * (1.0 - progress)


def project_unit_disk_np(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1, 2)
    norm = np.linalg.norm(action, axis=1, keepdims=True)
    scale = np.maximum(norm, 1.0)
    return (action / scale).astype(np.float32)


def centralized_macro_policy(actor, deterministic=False):
    def _policy(env, _local_obs_batch, eval_mode):
        _, global_state = env.get_macro_state()
        if deterministic or eval_mode:
            action = actor.choose_action_deterministic(global_state)
        else:
            action, _ = actor.choose_action(global_state)
        return project_unit_disk_np(action.reshape(env.N, 2))

    return _policy


def sac_micro_policy(actor):
    def _policy(env, eval_mode):
        return actor.choose_action(
            env.get_sac_state(),
            deterministic=bool(eval_mode),
        )

    return _policy


def save_trajectory_with_verification(env, save_path):
    traj_env = getattr(env, "env", env)
    uav_traj = np.asarray(getattr(traj_env, "uav_trajectory", []), dtype=np.float32)
    centroid_traj = np.asarray(
        getattr(traj_env, "centroid_trajectory", []), dtype=np.float32
    )
    users_traj = np.asarray(
        getattr(traj_env, "users_trajectory", []), dtype=np.float32
    )
    if uav_traj.size == 0 or centroid_traj.size == 0 or users_traj.size == 0:
        print(f"  [SKIP] No trajectory samples available for {save_path}")
        return

    anchor_points = np.asarray(
        getattr(traj_env, "anchor_points", []), dtype=np.float32
    )
    user_allocations = np.asarray(
        getattr(traj_env, "user_allocations", []), dtype=np.int32
    )
    zone_radius = float(getattr(traj_env, "user_zone_radius", 0.0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        uav_traj=uav_traj,
        centroid_traj=centroid_traj,
        users_traj=users_traj,
        anchor_points=anchor_points,
        user_allocations=user_allocations,
        zone_radius=np.asarray(zone_radius, dtype=np.float32),
    )

    if os.path.exists(save_path):
        file_size_kb = os.path.getsize(save_path) / 1024.0
        print(f"  Saved trajectory data -> {save_path} ({file_size_kb:.1f} KB)")
    else:
        print(f"  [WARN] Trajectory data was not found after saving: {save_path}")

    png_path = os.path.splitext(save_path)[0] + ".png"
    try:
        traj_env.render_trajectory(save_path=png_path, show=False)
    except Exception as exc:
        print(f"  [WARN] Failed to render trajectory PNG: {exc}")
        return

    if os.path.exists(png_path):
        file_size_kb = os.path.getsize(png_path) / 1024.0
        print(f"  Saved trajectory figure -> {png_path} ({file_size_kb:.1f} KB)")
    else:
        print(f"  [WARN] Trajectory figure was not found after saving: {png_path}")


def save_final_policy_trajectory(results_dirs, args, macro_actor, sac_actor):
    if int(getattr(args, "trajectory_plot_interval", 0)) <= 0:
        return

    traj_env = make_env_from_args(args, record_trajectory=True)
    traj_args = argparse.Namespace(**vars(args))
    traj_args.eval_episodes = 1
    evaluate_policy(
        traj_env,
        traj_args,
        centralized_macro_policy(macro_actor, deterministic=True),
        sac_micro_policy(sac_actor),
    )
    final_path = os.path.join(results_dirs["trajectory_dir"], "trajectory_final.npz")
    save_trajectory_with_verification(traj_env, final_path)


def compute_advisor_action(env, args):
    raw_state = env.env.get_raw_state()
    flow = env._estimate_macro_flow_reference(raw_state)

    device = ppo_agent_mod.device
    positions = torch.as_tensor(
        raw_state["uav_coords"],
        dtype=torch.float32,
        device=device,
    )
    user_coords = torch.as_tensor(
        raw_state["user_coords"],
        dtype=torch.float32,
        device=device,
    )
    user_alloc = torch.as_tensor(
        env.env.user_allocations,
        dtype=torch.long,
        device=device,
    ).clamp(0, env.N - 1)
    flow_t = torch.as_tensor(flow, dtype=torch.float32, device=device)

    action = torch.zeros((env.N, 2), dtype=torch.float32, device=device)
    action.requires_grad_(True)
    optimizer = torch.optim.Adam([action], lr=float(args.advisor_lr))

    max_disp = float(env.max_macro_displacement)
    macro_dt = max(float(env.env.macro_dt), 1e-8)
    min_sep = float(args.advisor_min_separation_m)

    for _ in range(max(int(args.advisor_steps), 1)):
        norm = torch.linalg.norm(action, dim=1, keepdim=True).clamp_min(1.0)
        action_proj = action / norm
        proposed = positions + action_proj * max_disp
        eval_pos = proposed.clamp(0.0, 1000.0)

        assigned_pos = eval_pos[user_alloc]
        diff_user = user_coords - assigned_pos
        dist_sq_user = (diff_user * diff_user).sum(dim=1) + float(env.env.H) ** 2
        gain_user = float(env.env.beta_0) / dist_sq_user.clamp_min(1e-8)
        user_bw_hz = float(env.env.B_gt_total) / max(int(env.env.M), 1)
        user_rate_mbps = (
            user_bw_hz
            * torch.log2(
                1.0
                + (float(env.env.p_gt) * gain_user)
                / float(env.env.noise_power)
            )
            / 1e6
        )
        zone_rates = torch.zeros(env.N, dtype=torch.float32, device=device)
        zone_rates.scatter_add_(0, user_alloc, user_rate_mbps)
        upload_volume = flow_t.sum(dim=1)
        uplink_time = (upload_volume / zone_rates.clamp_min(1e-6)).sum()

        active = flow_t > 1e-9
        active = active & (~torch.eye(env.N, dtype=torch.bool, device=device))
        active_count = int(active.sum().detach().cpu().item())
        if active_count > 0:
            diff_uav = eval_pos[:, None, :] - eval_pos[None, :, :]
            dist_sq_uav = (diff_uav * diff_uav).sum(dim=-1).clamp_min(1e-8)
            gain_uav = float(env.env.beta_0) / dist_sq_uav
            bw_u2u_hz = float(env.env.B_u2u_total) / float(active_count)
            rates_u2u_mbps = (
                bw_u2u_hz
                * torch.log2(
                    1.0
                    + (float(env.env.p_u2u) * gain_uav)
                    / float(env.env.noise_power)
                )
                / 1e6
            )
            u2u_time = (
                flow_t[active] / rates_u2u_mbps[active].clamp_min(1e-6)
            ).sum()
        else:
            u2u_time = torch.zeros((), dtype=torch.float32, device=device)

        pair_diff = proposed[:, None, :] - proposed[None, :, :]
        pair_dist = torch.linalg.norm(pair_diff + 1e-9, dim=-1)
        offdiag = ~torch.eye(env.N, dtype=torch.bool, device=device)
        sep_violation = F.relu(min_sep - pair_dist[offdiag]) / max(min_sep, 1e-8)
        sep_penalty = (sep_violation * sep_violation).mean()

        lower = F.relu(-proposed / 1000.0)
        upper = F.relu((proposed - 1000.0) / 1000.0)
        boundary_penalty = (lower * lower + upper * upper).mean()
        move_penalty = (action_proj * action_proj).sum(dim=1).mean()

        loss = (
            (uplink_time + u2u_time) / macro_dt
            + float(args.advisor_move_penalty) * move_penalty
            + float(args.advisor_sep_penalty) * sep_penalty
            + float(args.advisor_boundary_penalty) * boundary_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            action.clamp_(-1.0, 1.0)

    with torch.no_grad():
        norm = torch.linalg.norm(action, dim=1, keepdim=True).clamp_min(1.0)
        action_proj = action / norm
    return action_proj.detach().cpu().numpy().astype(np.float32)


def learn_central_actor(
    actor,
    states,
    actions,
    advantages,
    old_logprobs,
    advisor_actions,
    advisor_weights,
):
    device = ppo_agent_mod.device
    bs = torch.as_tensor(states, dtype=torch.float32, device=device)
    ba = torch.as_tensor(actions, dtype=torch.float32, device=device)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    old_lp = torch.as_tensor(old_logprobs, dtype=torch.float32, device=device)
    advisor = torch.as_tensor(advisor_actions, dtype=torch.float32, device=device)
    bc_w = torch.as_tensor(advisor_weights, dtype=torch.float32, device=device)

    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    else:
        adv = adv - adv.mean()

    last_loss = None
    last_ppo = None
    last_bc = None
    for _ in range(A_UPDATE_STEPS):
        mean, std = actor.new_pi(bs)
        dist = torch.distributions.Normal(mean, std)
        new_lp = dist.log_prob(ba).sum(dim=-1, keepdim=True)
        ratio = torch.exp(new_lp - old_lp.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(
            ratio,
            1.0 - CLIP_EPSILON,
            1.0 + CLIP_EPSILON,
        ) * adv
        ppo_loss = -torch.min(surr1, surr2).mean()

        bc_per_sample = ((mean - advisor) ** 2).mean(dim=1, keepdim=True)
        bc_loss = (bc_w * bc_per_sample).mean()
        loss = ppo_loss + bc_loss

        actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.new_pi.parameters(), 0.5)
        actor.optimizer.step()

        last_loss = loss.item()
        last_ppo = ppo_loss.item()
        last_bc = bc_loss.item()

    return float(last_loss), float(last_ppo), float(last_bc)


def update_sac(
    sac_actor,
    sac_critic,
    sac_entropy,
    sac_memory,
    sac_state_dim,
    sac_action_dim,
    args,
):
    device = sac_agent_mod.device
    actor_losses = []
    critic_losses = []
    for _ in range(args.sac_gradient_steps):
        batch = sac_memory.sample(args.sac_batch_size)
        b_s = torch.as_tensor(batch[:, :sac_state_dim], dtype=torch.float32, device=device)
        b_a = torch.as_tensor(
            batch[:, sac_state_dim:sac_state_dim + sac_action_dim],
            dtype=torch.float32,
            device=device,
        )
        b_r = torch.as_tensor(
            batch[
                :,
                sac_state_dim + sac_action_dim:
                sac_state_dim + sac_action_dim + 1,
            ],
            dtype=torch.float32,
            device=device,
        )
        b_s_ = torch.as_tensor(batch[:, -sac_state_dim:], dtype=torch.float32, device=device)

        new_action, log_prob_ = sac_actor.evaluate(b_s_)
        target_q1, target_q2 = sac_critic.target_critic_v(b_s_, new_action)
        target_q = b_r + SAC_GAMMA * (
            torch.min(target_q1, target_q2)
            - sac_entropy.alpha.detach().to(device) * log_prob_
        )

        current_q1, current_q2 = sac_critic.get_v(b_s, b_a)
        sac_critic.learn(current_q1, current_q2, target_q.detach())

        a_eval, log_prob = sac_actor.evaluate(b_s)
        q1_eval, q2_eval = sac_critic.get_v(b_s, a_eval)
        actor_loss = (
            sac_entropy.alpha.detach().to(device) * log_prob
            - torch.min(q1_eval, q2_eval)
        ).mean()
        sac_actor.learn(actor_loss)

        alpha_loss = -(
            sac_entropy.log_alpha.exp().to(device)
            * (log_prob + sac_entropy.target_entropy).detach()
        ).mean()
        sac_entropy.learn(alpha_loss)
        sac_entropy.alpha = sac_entropy.log_alpha.exp().to(device)
        sac_critic.soft_update()

        critic_loss = (
            F.mse_loss(current_q1, target_q.detach())
            + F.mse_loss(current_q2, target_q.detach())
        )
        actor_losses.append(float(actor_loss.item()))
        critic_losses.append(float(critic_loss.item()))

    return actor_losses, critic_losses


def main():
    args = parse_args()

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    device = torch.device(requested)
    ppo_agent_mod.device = device
    sac_agent_mod.device = device
    print(f"[INFO] Training on {str(device).upper()}")
    print("[INFO] Macro policy: centralized dx/dy actor")
    print(f"[INFO] macro_comm_weight={args.macro_comm_weight}")
    if float(args.train_load_min) > 0.0 and float(args.train_load_max) >= float(
        args.train_load_min
    ):
        print(
            f"[INFO] randomized train load_scale in "
            f"[{args.train_load_min}, {args.train_load_max}]"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = make_env_from_args(args, record_trajectory=False)
    eval_env = make_env_from_args(args, record_trajectory=False)
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(results_dirs, args, "CentralizedMacro")

    sac_state_dim = env.sac_obs_dim
    sac_action_dim = env.micro_action_space.shape[0]
    sac_actor = SAC_Actor(sac_state_dim, sac_action_dim)
    sac_critic = SAC_Critic(sac_state_dim, sac_action_dim)
    sac_entropy = SAC_Entropy(sac_action_dim)
    sac_memory = SAC_Memory(
        SAC_MEMORY_CAPACITY,
        2 * sac_state_dim + sac_action_dim + 1,
        recent_window=SAC_RECENT_WINDOW,
        recent_ratio=SAC_RECENT_RATIO,
    )

    _, global_state0 = env.reset(seed=args.seed)[0]
    global_state_dim = int(np.asarray(global_state0).size)
    central_action_dim = int(args.num_agents) * 2
    macro_actor = PPO_Actor(global_state_dim, central_action_dim)
    macro_critic = PPO_Critic(global_state_dim, 1)

    eval_history = EvaluationHistory()
    ppo_states = []
    ppo_actions = []
    ppo_old_logprobs = []
    ppo_returns = []
    ppo_advantages = []
    ppo_advisor_actions = []
    ppo_advisor_weights = []
    sac_trainable_micro_steps = 0

    best_eval_j = np.inf
    ckpt_dir = results_dirs["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for episode in range(args.episodes):
        should_record = (
            args.trajectory_plot_interval > 0
            and (episode + 1) % args.trajectory_plot_interval == 0
        )
        env.env.record_trajectory = should_record
        episode_load = sample_training_load(args)
        set_wrapper_load_scale(env, episode_load)
        _, info = env.reset(seed=args.seed + episode)
        _local_state, global_state = env.get_macro_state()

        ep_states = []
        ep_actions = []
        ep_logprobs = []
        ep_rewards = []
        ep_advisor_actions = []
        ep_advisor_weights = []
        ep_macro_reward = 0.0
        ep_micro_reward = 0.0
        ep_j = []
        ep_comm = []
        ep_u2g = []
        ep_move = []
        done = False

        bc_weight = advisor_weight_for_episode(args, episode)

        for _ in range(args.macro_steps):
            advisor_action = compute_advisor_action(env, args)
            if args.macro_deterministic:
                action_flat = macro_actor.choose_action_deterministic(global_state)
                logprob = np.zeros((1,), dtype=np.float32)
            else:
                action_flat, logprob = macro_actor.choose_action(global_state)
            action = project_unit_disk_np(action_flat.reshape(env.N, 2))
            action_flat = action.reshape(-1).astype(np.float32)

            _, _, mac_info = env.macro_step(action)
            if isinstance(mac_info, dict):
                ep_comm.append(float(mac_info.get("projected_macro_comm_cost", 0.0)))
                ep_u2g.append(float(mac_info.get("projected_mean_u2g_distance_m", 0.0)))
                ep_move.append(float(mac_info.get("macro_move_distance_mean", 0.0)))

            for _ in range(args.micro_steps):
                trainable = env.env.macro_fly_steps_remaining <= 0
                if trainable:
                    sac_obs = env.get_sac_state()
                    sac_action = sac_actor.choose_action(sac_obs)
                else:
                    sac_obs = None
                    sac_action = np.zeros(sac_action_dim, dtype=np.float32)

                next_sac_obs, mic_reward, done, _, mic_info = env.micro_step(sac_action)
                should_train_sac = trainable and not mic_info.get("flight_phase", False)
                if should_train_sac:
                    sac_trainable_micro_steps += 1
                    sac_memory.store_transition(
                        sac_obs,
                        sac_action,
                        mic_reward,
                        next_sac_obs,
                    )
                    if (
                        sac_memory.memory_counter > SAC_MEMORY_WARMUP
                        and sac_trainable_micro_steps % args.sac_update_freq == 0
                    ):
                        update_sac(
                            sac_actor,
                            sac_critic,
                            sac_entropy,
                            sac_memory,
                            sac_state_dim,
                            sac_action_dim,
                            args,
                        )

                ep_micro_reward += float(mic_reward)
                ep_j.append(float(mic_info.get("J_step", 0.0)))
                if done:
                    break

            next_local_state, next_global_state = env.get_macro_state()
            macro_reward = float(env.get_macro_reward())
            ep_macro_reward += macro_reward

            ep_states.append(np.asarray(global_state, dtype=np.float32))
            ep_actions.append(action_flat.copy())
            ep_logprobs.append(np.asarray(logprob, dtype=np.float32).reshape(1))
            ep_rewards.append(macro_reward)
            ep_advisor_actions.append(advisor_action.reshape(-1).astype(np.float32))
            ep_advisor_weights.append(np.asarray([bc_weight], dtype=np.float32))

            global_state = next_global_state
            if done:
                break

        if ep_states:
            T = len(ep_states)
            state_arr = np.vstack(ep_states).astype(np.float32)
            values = (
                macro_critic.get_v(state_arr)
                .detach()
                .cpu()
                .numpy()
                .reshape(T)
            )
            v_next = float(
                macro_critic.get_v(global_state)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)[0]
            )
            rewards = np.asarray(ep_rewards, dtype=np.float32).reshape(T)
            adv = np.zeros(T, dtype=np.float32)
            lastgae = 0.0
            for t in range(T - 1, -1, -1):
                next_v = v_next if t == T - 1 else values[t + 1]
                delta = rewards[t] + PPO_GAMMA * next_v - values[t]
                lastgae = delta + PPO_GAMMA * PPO_LAMBDA * lastgae
                adv[t] = lastgae
            returns = adv + values

            ppo_states.extend(ep_states)
            ppo_actions.extend(ep_actions)
            ppo_old_logprobs.extend(ep_logprobs)
            ppo_returns.extend(list(returns.reshape(-1, 1)))
            ppo_advantages.extend(list(adv.reshape(-1, 1)))
            ppo_advisor_actions.extend(ep_advisor_actions)
            ppo_advisor_weights.extend(ep_advisor_weights)

        update_threshold = args.macro_steps * max(
            int(args.ppo_update_episode_window),
            1,
        )
        if (
            len(ppo_states) >= update_threshold
            or (episode == args.episodes - 1 and ppo_states)
        ):
            bs = np.vstack(ppo_states).astype(np.float32)
            br = np.vstack(ppo_returns).astype(np.float32)
            ba = np.vstack(ppo_actions).astype(np.float32)
            bap = np.vstack(ppo_old_logprobs).astype(np.float32)
            badv = np.vstack(ppo_advantages).astype(np.float32)
            badvisor = np.vstack(ppo_advisor_actions).astype(np.float32)
            bw = np.vstack(ppo_advisor_weights).astype(np.float32)

            _, critic_loss = macro_critic.learn(bs, br)
            actor_loss, ppo_loss, bc_loss = learn_central_actor(
                macro_actor,
                bs,
                ba,
                badv,
                bap,
                badvisor,
                bw,
            )
            macro_actor.update_oldpi()

            print(
                f"  PPO update | actor={actor_loss:.4f} "
                f"ppo={ppo_loss:.4f} bc={bc_loss:.4f} critic={critic_loss:.4f}"
            )

            ppo_states.clear()
            ppo_actions.clear()
            ppo_old_logprobs.clear()
            ppo_returns.clear()
            ppo_advantages.clear()
            ppo_advisor_actions.clear()
            ppo_advisor_weights.clear()

        print(
            f"Episode {episode} | Macro R {ep_macro_reward:.2f} | "
            f"Micro R {ep_micro_reward:.2f} | "
            f"J {np.mean(ep_j) if ep_j else 0.0:.4f} | "
            f"Comm {np.mean(ep_comm) if ep_comm else 0.0:.3f} | "
            f"U2G {np.mean(ep_u2g) if ep_u2g else 0.0:.1f} | "
            f"Move {np.mean(ep_move) if ep_move else 0.0:.1f} | "
            f"Load {episode_load:.2f} | "
            f"BCw {bc_weight:.3f}"
        )

        if should_record:
            trajectory_path = os.path.join(
                results_dirs["trajectory_dir"],
                f"trajectory_ep_{episode + 1:04d}.npz",
            )
            save_trajectory_with_verification(env, trajectory_path)

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            metrics = evaluate_policy(
                eval_env,
                args,
                centralized_macro_policy(macro_actor, deterministic=True),
                sac_micro_policy(sac_actor),
            )
            eval_history.append(episode + 1, metrics)
            print_evaluation_summary(episode + 1, metrics)
            if metrics.get("J", np.inf) < best_eval_j:
                best_eval_j = metrics.get("J", np.inf)
                torch.save(
                    {
                        "macro_actor": macro_actor.old_pi.state_dict(),
                        "macro_critic": macro_critic.critic_v.state_dict(),
                        "sac_actor": sac_actor.action_net.state_dict(),
                        "sac_critic": sac_critic.critic_v.state_dict(),
                        "sac_target_critic": sac_critic.target_critic_v.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(ckpt_dir, "best.pt"),
                )

    if not eval_history.steps or eval_history.steps[-1] != args.episodes:
        metrics = evaluate_policy(
            eval_env,
            args,
            centralized_macro_policy(macro_actor, deterministic=True),
            sac_micro_policy(sac_actor),
        )
        eval_history.append(args.episodes, metrics)
        print_evaluation_summary(args.episodes, metrics)

    eval_csv = os.path.join(results_dirs["results_dir"], "eval_central_macro.csv")
    eval_history.save_csv(eval_csv)
    eval_history.save_detail_npz(
        os.path.join(results_dirs["results_dir"], "eval_central_macro_detail.npz")
    )
    torch.save(
        {
            "macro_actor": macro_actor.old_pi.state_dict(),
            "macro_critic": macro_critic.critic_v.state_dict(),
            "sac_actor": sac_actor.action_net.state_dict(),
            "sac_critic": sac_critic.critic_v.state_dict(),
            "sac_target_critic": sac_critic.target_critic_v.state_dict(),
            "args": vars(args),
        },
        os.path.join(ckpt_dir, "final.pt"),
    )
    save_final_policy_trajectory(results_dirs, args, macro_actor, sac_actor)
    print(f"Saved evaluation CSV -> {eval_csv}")
    print("Training complete.")


if __name__ == "__main__":
    main()
