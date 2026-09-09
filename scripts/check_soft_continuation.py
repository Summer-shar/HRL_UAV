"""Bounded read-only soft-return check, with paired stochastic continuation."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.agents.sac_agent import SQUASH_EPS
from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP
from scripts.audit_micro_feedback import file_hash, frame_step, forward_values


@torch.no_grad()
def sample_with_noise(actor, state, noise):
    states = tensor(state).reshape(1, -1)
    mean, std = actor.action_net(states)
    raw = mean + std * tensor(noise).reshape(-1, mean.shape[-1])
    action = raw.tanh()
    dist = torch.distributions.Normal(mean, std)
    logprob = (dist.log_prob(raw) - torch.log((1-action.square()).clamp(min=SQUASH_EPS))).sum(-1)
    return action, logprob


@torch.no_grad()
def soft_branch(snapshot, agent, *, bypass, noise_seed, alpha, horizons=(60, 120, 240)):
    mdp = copy.deepcopy(snapshot)
    gamma = mdp.config["training"]["gamma_per_resource_frame"]
    noise = np.random.default_rng(noise_seed)
    tail_noise = np.random.default_rng(noise_seed + 1000000)
    initial_action = agent.sac_actor.choose_action(mdp.prepare_service(), deterministic=True)
    if bypass:
        route = mdp.route(initial_action)
        route[1] = .5
        route[1, :, 1] = 0.
        initial_action = route.ravel()
    physical = entropy = 0.
    entropy_events = 0
    hasher = hashlib.sha256()
    records = {}
    for frame in range(max(horizons)):
        env = mdp.env
        if env.global_micro_step % env.micro_steps_per_macro == 0:
            if env.global_micro_step:
                mdp.advance_macro_clock()
            mdp.start_macro(np.zeros(mdp.macro_action_dim))
        if env.macro_service_pause_steps_remaining:
            scores = np.zeros(mdp.micro_action_dim)
        elif frame == 0:
            scores = initial_action
        else:
            scores, logprob = sample_with_noise(agent.sac_actor, mdp.prepare_service(),
                                               noise.standard_normal(mdp.micro_action_dim))
            entropy += gamma**frame * (-alpha * float(logprob[0]))
            entropy_events += 1
            scores = scores[0].cpu().numpy()
        for array in (env.user_coords, env.uav_coords, env.A_t_high, env.A_t_low):
            hasher.update(np.asarray(array, dtype=np.float64).tobytes())
        result = mdp.step_frame(scores, gamma)
        physical += gamma**frame * result["reward"]
        if frame+1 in horizons:
            # These horizons end at service decisions, never inside deployment.
            assert not env.macro_service_pause_steps_remaining
            assert env.global_micro_step % env.micro_steps_per_macro != 0
            state = mdp.prepare_service()
            actions, logprobs = sample_with_noise(agent.sac_actor, state,
                                                  tail_noise.standard_normal((16, mdp.micro_action_dim)))
            states = tensor(state)[None].expand(16, -1)
            q1, q2 = agent.sac_critic.target_critic_v(states, actions)
            tail = gamma**(frame+1) * float((torch.minimum(q1, q2).ravel()-alpha*logprobs).mean())
            records[str(frame+1)] = {
                "elapsed_s": (frame+1)*env.resource_frame_dt,
                "physical_prefix": physical, "entropy_prefix": entropy,
                "soft_prefix": physical+entropy, "target_tail": tail,
                "soft_with_target_tail": physical+entropy+tail,
                "future_service_entropy_events": entropy_events,
                "queue_compute_mbit": float(env.Q_loc_high.sum()+env.Q_loc_low.sum()),
                "exogenous_sha256": hasher.hexdigest(),
            }
    return records


def run_check(checkpoint_path, output):
    if (output/"report.json").exists():
        raise FileExistsError("Do not overwrite a previous audit")
    torch.set_num_threads(1)
    torch.manual_seed(94401)
    paths = [ROOT/p for p in ("configs/main.json", "hrl_uav/mdp.py", "hrl_uav/learning.py",
                              "hrl_uav/agents/sac_agent.py", "hrl_uav/env.py", "hrl_uav/routing.py")]
    before = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mdp = QueueMDP(checkpoint["config"])
    mdp.reset(33500, 90)
    agent = DirectHRL.from_mdp(mdp, checkpoint["config"])
    agent.sac_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    agent.sac_critic.critic_v.load_state_dict(checkpoint["sac_critic"])
    agent.sac_critic.target_critic_v.load_state_dict(checkpoint["sac_target"])
    for net in (agent.sac_actor.action_net, agent.sac_critic.critic_v, agent.sac_critic.target_critic_v):
        net.eval()
        net.requires_grad_(False)
    alpha = float(checkpoint["log_alpha"].exp())
    protocol = {
        "snapshot_seed": 33500, "snapshot_times_s": [31, 291],
        "continuation_noise_seeds": [94410, 94411, 94412, 94413],
        "horizons_s": [30, 60, 120], "tail_samples": 16, "alpha": alpha,
        "initial_action": "deterministic Actor vs only low-priority routes [0.5,0,0.5]",
        "continuation": "Same frozen stochastic SAC policy with paired Gaussian noise; centroid macro",
        "entropy": "Future service actions only; no entropy for initial conditioned action or deployment",
        "limits": "Exploratory two correlated states, four continuations. Not performance evidence or infinite-horizon truth. Target tail is learned, not independent ground truth.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    forks = []
    for _ in range(600):
        env = mdp.env
        if (mdp.macro_index in (3, 29) and
                env.global_micro_step % env.micro_steps_per_macro == env.deployment_steps):
            mdp.prepare_service()
            row = {"time_s": env._current_time(), "critic": forward_values(agent, mdp), "replicates": []}
            for noise_seed in protocol["continuation_noise_seeds"]:
                original = soft_branch(mdp, agent, bypass=False, noise_seed=noise_seed, alpha=alpha)
                bypass = soft_branch(mdp, agent, bypass=True, noise_seed=noise_seed, alpha=alpha)
                differences = {}
                for horizon, base in original.items():
                    assert base["exogenous_sha256"] == bypass[horizon]["exogenous_sha256"], "CRN mismatch"
                    differences[horizon] = {key: bypass[horizon][key]-base[key] for key in (
                        "physical_prefix", "entropy_prefix", "soft_prefix", "target_tail", "soft_with_target_tail")}
                row["replicates"].append({"noise_seed": noise_seed, "actor": original,
                                          "bypass": bypass, "bypass_minus_actor": differences, "crn_equal": True})
                print(f"soft check t={row['time_s']:.0f}s noise={noise_seed} done", flush=True)
            forks.append(row)
            if mdp.macro_index == 29:
                break
        frame_step(mdp, agent, "actor")
    assert len(forks) == 2
    result = {"protocol": protocol, "checkpoint_sha256": checkpoint_hash, "source_sha256": before,
              "script_sha256": file_hash(Path(__file__)), "forks": forks,
              "source_unchanged": before == {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
              "checkpoint_unchanged": file_hash(checkpoint_path) == checkpoint_hash}
    assert result["source_unchanged"] and result["checkpoint_unchanged"]
    (output/"report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in forks:
        summary = {h: {key: float(np.mean([r["bypass_minus_actor"][h][key] for r in row["replicates"]]))
                        for key in ("soft_prefix", "target_tail", "soft_with_target_tail")}
                   for h in ("60", "120", "240")}
        print(json.dumps({"time_s": row["time_s"], "critic_delta": row["critic"]["low_avoid_uav1_minus_actor_q"],
                          "bypass_minus_actor": summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"results/split_mdp_mainline_seed1/micro_pretrained.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"results/micro_soft_return_check")
    args = parser.parse_args()
    run_check(args.checkpoint, args.output)
