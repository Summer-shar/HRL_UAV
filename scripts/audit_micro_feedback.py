"""Read-only state/action/value audit; no optimizer steps or checkpoint writes."""

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

from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP
from hrl_uav.routing import project_rows_to_simplex_tensor


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def low_uniform_route(route):
    result = route.clone()
    result[..., 1, :, :] = 1.0 / route.shape[-1]
    return result


def low_avoid_uav1_route(route):
    """Diagnostic only: bypass the fixed destination identified as overloaded."""
    result = route.clone()
    result[..., 1, :, :] = 0.5
    result[..., 1, :, 1] = 0.0
    return result


def queue_group_preferences(checkpoint, agent, indices):
    """Prediction and behavior coverage by fixed-destination queue, not Q errors."""
    replay = np.asarray(checkpoint["replay"]["transitions"])
    n = checkpoint["micro_state_dim"]
    batch = replay[indices]
    states = tensor(batch[:, :n])
    q_start = checkpoint["observations"]["micro"]["slices"]["compute_queue"][0]
    queue = 10*np.expm1(batch[:, q_start+4].astype(np.float64))
    with torch.no_grad():
        mean, _ = agent.sac_actor.action_net(states)
        action = mean.tanh()
        route = project_rows_to_simplex_tensor(action.reshape(-1, 2, 3, 3))
        bypass = low_avoid_uav1_route(route).reshape_as(action)
        q1, q2 = agent.sac_critic.get_v(states, action)
        b1, b2 = agent.sac_critic.get_v(states, bypass)
        delta = (torch.minimum(b1, b2)-torch.minimum(q1,q2)).ravel().cpu().numpy()
    predicted_route = route[:, 1, :, 1].cpu().numpy()
    stored_route = batch[:, n+20:n+38].reshape(-1, 2, 3, 3)
    stored_low = stored_route[:, 1, :, 1]
    groups = []
    for label, mask in (
        ("low: [0,10) Mbit", queue < 10),
        ("middle: [10,100) Mbit", (queue >= 10) & (queue < 100)),
        ("high: [100,infinity) Mbit", queue >= 100),
    ):
        row = {"group": label, "states": int(mask.sum())}
        if mask.any():
            row.update(
                queue_min_mbit=float(queue[mask].min()), queue_max_mbit=float(queue[mask].max()),
                bypass_minus_actor_q_mean=float(delta[mask].mean()),
                bypass_q_preferred_states=int((delta[mask] > 1e-5).sum()),
                actor_mean_low_fraction_to_uav1=float(predicted_route[mask].mean()),
                recorded_low_rows_avoiding_uav1_fraction=float((stored_low[mask] <= 1e-6).mean()),
                recorded_all_low_rows_avoiding_uav1_states=int((stored_low[mask] <= 1e-6).all(-1).sum()),
            )
        groups.append(row)
    assert sum(g["states"] for g in groups) == len(indices)
    return {
        "scope": "512 chronological replay samples evaluated with final networks. No physical return labels for these states; not an error-rate or causal test.",
        "queue_definition": "Low-priority compute backlog at fixed destination UAV 1, not rotating heavy-arrival source role; excludes in-flight queues.",
        "cutoffs_mbit": [10,100], "cutoff_scope": "Descriptive fixed bins, not stability limits",
        "candidate": "Only low routes become [0.5,0,0.5]; high routes unchanged",
        "groups": groups,
        "per_state": [{"replay_index": int(index), "low_q_uav1_mbit": float(bits),
                       "bypass_minus_actor_q": float(diff)}
                      for index, bits, diff in zip(indices, queue, delta)],
        "coverage_limits": "Recorded zero proportions do not identify actual ready-task flow or matched counterfactual support. Low/high groups differ in other observations and behavior history.",
    }


def route_probe(network, states, scores, width=3):
    """Compare gradient before/after projection without changing learned weights."""
    scores = scores.detach().requires_grad_(True)
    routes = project_rows_to_simplex_tensor(scores.reshape(-1, 2, width, width))
    q1, q2 = network(states, scores)
    q = torch.minimum(q1, q2)
    score_grad, = torch.autograd.grad(q.sum(), scores)
    # The copy exposes the same learned q(s, rho), without a second projection.
    direct = copy.deepcopy(network)
    direct.route_width = 0
    independent_route = routes.detach().reshape_as(scores).requires_grad_(True)
    v1, v2 = direct(states, independent_route)
    route_grad, = torch.autograd.grad(torch.minimum(v1, v2).sum(), independent_route)
    torch.testing.assert_close(q1, v1, atol=2e-3, rtol=2e-5)
    torch.testing.assert_close(q2, v2, atol=2e-3, rtol=2e-5)
    candidate = low_uniform_route(routes.detach()).reshape_as(scores)
    with torch.no_grad():
        c1, c2 = network(states, candidate)
    active = (routes.detach() > 1e-6).sum(-1)
    frozen = score_grad.reshape(-1, 2, width, width).abs().amax(-1) < 1e-9
    low_rows = active[:, 1]
    delta = (torch.minimum(c1, c2) - q.detach()).ravel()
    gap = scores.detach().reshape(-1, 2, width, width).sort(-1, descending=True).values
    return {
        "states": len(states),
        "all_rows_vertex_fraction": float((active == 1).float().mean()),
        "low_rows_vertex_fraction": float((low_rows == 1).float().mean()),
        "low_rows_zero_q_score_gradient_fraction": float(frozen[:, 1].float().mean()),
        "low_route_gradient_abs_mean": float(route_grad.reshape(-1, 2, width, width)[:, 1].abs().mean()),
        "low_score_gradient_abs_mean": float(score_grad.reshape(-1, 2, width, width)[:, 1].abs().mean()),
        "low_score_top_gap_median": float((gap[:, 1, :, 0] - gap[:, 1, :, 1]).median()),
        "low_uniform_minus_actor_q_mean": float(delta.mean()),
        "low_uniform_q_preferred_fraction": float((delta > 1e-5).float().mean()),
        "mean_low_route": routes.detach()[:, 1].mean(0).cpu().tolist(),
    }


def replay_audit(checkpoint, agent, count=512):
    replay = np.asarray(checkpoint["replay"]["transitions"])
    n = checkpoint["micro_state_dim"]
    indices = np.linspace(0, len(replay)-1, min(count, len(replay)), dtype=int)
    states = tensor(replay[indices, :n])
    with torch.no_grad():
        mean, std = agent.sac_actor.action_net(states)
        deterministic = torch.tanh(mean)
    result = {"indices": indices.tolist(), "sampling": "Evenly spaced chronological replay states, evaluated with final networks"}
    network = agent.sac_critic.critic_v
    result["deterministic"] = route_probe(network, states, deterministic)
    torch.manual_seed(94401)
    draws = []
    for _ in range(4):
        with torch.no_grad():
            scores, _ = agent.sac_actor.evaluate(states)
        draws.append(route_probe(network, states, scores))
    result["stochastic_draws"] = draws
    result["log_std_mean"] = float(std.log().mean())
    result["alpha"] = float(checkpoint["log_alpha"].exp())
    # This is an input-sensitivity probe, not a physically consistent rollout.
    # Hold every other field fixed and vary only destination 1's low compute Q.
    q_slice = checkpoint["observations"]["micro"]["slices"]["compute_queue"]
    index = q_slice[0] + 3 + 1
    sensitivity = []
    for bits in (0., 10., 100., 1000.):
        altered = states.clone()
        altered[:, index] = float(np.log1p(bits/10.))
        with torch.no_grad():
            m, _ = agent.sac_actor.action_net(altered)
            route = project_rows_to_simplex_tensor(m.tanh().reshape(-1, 2, 3, 3))
        sensitivity.append({"synthetic_low_q_uav1_mbit": bits,
                            "mean_low_route_to_uav1": float(route[:, 1, :, 1].mean())})
    result["single_field_sensitivity_not_causal"] = sensitivity
    stored_action = replay[:, n:n+18]
    stored_route = replay[:, n+20:n+38]
    projected = project_rows_to_simplex_tensor(torch.from_numpy(stored_action.reshape(-1, 2, 3, 3))).numpy().reshape(-1, 18)
    result["stored_route_projection_max_error"] = float(np.max(np.abs(projected-stored_route)))
    result["replay_observation_encoding"] = "compute_queue[priority,destination]; transfer_queue[priority,source,destination]"
    result["queue_groups_predictions_only"] = queue_group_preferences(checkpoint, agent, indices)
    return result


def frame_step(mdp, agent, policy, first_service=False, hasher=None):
    env = mdp.env
    if env.global_micro_step % env.micro_steps_per_macro == 0:
        if env.global_micro_step:
            mdp.advance_macro_clock()
        mdp.start_macro(np.zeros(mdp.macro_action_dim))
    paused = env.macro_service_pause_steps_remaining > 0
    if paused:
        scores = np.zeros(mdp.micro_action_dim)
    else:
        state = mdp.prepare_service()
        scores = agent.sac_actor.choose_action(state, deterministic=True)
        if policy.startswith("sustained_") or (policy.startswith("pulse_") and first_service):
            route = mdp.route(scores)
            if policy.endswith("low_uniform"):
                route[1] = 1. / mdp.N
            elif policy.endswith("low_avoid_uav1"):
                route[1] = 0.5
                route[1, :, 1] = 0.0
            else:
                raise ValueError(policy)
            scores = route.ravel()
    if hasher is not None:
        for array in (env.user_coords, env.uav_coords, env.A_t_high, env.A_t_low):
            hasher.update(np.asarray(array, dtype=np.float64).tobytes())
    return mdp.step_frame(scores, mdp.config["training"]["gamma_per_resource_frame"]), not paused


def branch(snapshot, agent, mode, frames=60):
    mdp = copy.deepcopy(snapshot)
    before = mdp.metrics()
    hasher = hashlib.sha256()
    consumed = False
    cumulative_cost = discounted_reward = 0.0
    delay_cost = energy_cost = 0.0
    energy_components = {name: 0.0 for name in ("energy_comp", "energy_trans", "energy_uplink")}
    initial_completed = dict(mdp.env.completed_priority_count)
    gamma = mdp.config["training"]["gamma_per_resource_frame"]
    records = {}
    for frame in range(frames):
        result, service = frame_step(mdp, agent, mode, not consumed, hasher)
        consumed |= service
        cumulative_cost += result["cost"]
        discounted_reward += gamma**frame * result["reward"]
        for info in result["infos"]:
            reward_config = mdp.config["reward"]
            delay_cost += reward_config["delay_weight"] * info["task_holding_cost_weighted"] / reward_config["holding_norm_task_s"]
            energy_cost += reward_config["energy_weight"] * (info["total_energy"]+mdp.propulsion_delta_share) / reward_config["energy_norm_j"]
            for key in energy_components:
                energy_components[key] += info.get(key, 0.)
        if frame+1 in (1, 20, 60):
            after = mdp.metrics()
            np.testing.assert_allclose(delay_cost+energy_cost, cumulative_cost, atol=1e-7, rtol=1e-10)
            records[str(frame+1)] = {
                "elapsed_s": after["time_s"]-before["time_s"],
                "cost_integral": cumulative_cost,
                "discounted_reward": discounted_reward,
                "delay_cost_component": delay_cost,
                "energy_cost_component": energy_cost,
                "energy_components_j": dict(energy_components),
                "completed_since_fork": {key: mdp.env.completed_priority_count[key]-initial_completed[key]
                                         for key in initial_completed},
                "partially_completed_alive_low_tasks": sum(
                    p.priority == "low" and 0 < p.completed_chunks < p.total_chunks
                    for p in mdp.env.parent_tasks.values()),
                "queue_total_mbit": after["queue_total_mbit"],
                "queue_compute_mbit": after["queue_compute_mbit"],
                "queue_transfer_mbit": after["queue_transfer_mbit"],
                "low_compute_queue_mbit": mdp.env.Q_loc_low.tolist(),
            }
    return {"mode": mode, "exogenous_sha256": hasher.hexdigest(), "horizons_frames": records}


def encoding_probe(mdp):
    state = mdp.micro_observation()
    raw = mdp.env.get_raw_state()
    start, end = mdp.micro_feature_slices["compute_queue"]
    expected = np.stack([raw["Q_loc_high"], raw["Q_loc_low"]])
    decoded = 10*np.expm1(state[start:end].astype(np.float64)).reshape(2, 3)
    relative = float(np.max(np.abs(decoded-expected)/(1+np.abs(expected))))
    # Temporarily perturb the encoder only; no physical step is executed.
    old = mdp.env.Q_loc_low.copy()
    try:
        mdp.env.Q_loc_low[1] += 10.
        changed = mdp.micro_observation()
    finally:
        mdp.env.Q_loc_low[:] = old
    changes = np.flatnonzero(np.abs(changed-state) > 1e-7).tolist()
    assert changes == [start+4], changes
    assert changed[start+4] > state[start+4]
    np.testing.assert_array_equal(state, mdp.micro_observation())
    return {"compute_queue_decode_relative_error": relative,
            "encoder_only_added_10_mbit_low_uav1_changed_indices": changes,
            "expected_index": start+4, "monotone": True}


def forward_values(agent, mdp):
    state = tensor(mdp.prepare_service())[None]
    with torch.no_grad():
        mean, _ = agent.sac_actor.action_net(state)
        scores = mean.tanh()
        routes = project_rows_to_simplex_tensor(scores.reshape(-1, 2, 3, 3))
        changed = low_uniform_route(routes).reshape_as(scores)
        avoided = low_avoid_uav1_route(routes).reshape_as(scores)
        q1, q2 = agent.sac_critic.get_v(state, scores)
        r1, r2 = agent.sac_critic.get_v(state, changed)
        a1, a2 = agent.sac_critic.get_v(state, avoided)
    return {"actor_q1": float(q1), "actor_q2": float(q2),
            "low_uniform_q1": float(r1), "low_uniform_q2": float(r2),
            "low_uniform_minus_actor_q": float(torch.minimum(r1,r2)-torch.minimum(q1,q2)),
            "low_avoid_uav1_minus_actor_q": float(torch.minimum(a1,a2)-torch.minimum(q1,q2)),
            "actor_low_route": routes[0, 1].cpu().tolist()}


def fork_audit(config, agent):
    records = []
    for seed in (33500, 33501):
        mdp = QueueMDP(config)
        mdp.reset(seed, 90)
        for _ in range(1200):
            env = mdp.env
            if (mdp.macro_index in (3, 29, 59)
                    and env.global_micro_step % env.micro_steps_per_macro == env.deployment_steps):
                mdp.prepare_service()
                row = {"seed": seed, "time_s": env._current_time(),
                       "low_compute_queue_mbit": env.Q_loc_low.tolist(),
                       "encoding": encoding_probe(mdp), "critic": forward_values(agent, mdp)}
                row["branches"] = [branch(mdp, agent, mode) for mode in
                                    ("actor", "pulse_low_uniform", "sustained_low_uniform",
                                     "pulse_low_avoid_uav1", "sustained_low_avoid_uav1")]
                hashes = {b["exogenous_sha256"] for b in row["branches"]}
                assert len(hashes) == 1, "CRN mismatch"
                row["crn_equal"] = True
                records.append(row)
                print(f"fork seed={seed} t={row['time_s']:.1f}s done", flush=True)
                if mdp.macro_index == 59:
                    break
            frame_step(mdp, agent, "actor")
    assert len(records) == 6
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"results/split_mdp_mainline_seed1/micro_pretrained.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"results/micro_feedback_audit")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(94400)
    source_paths = [ROOT/name for name in (
        "hrl_uav/mdp.py", "hrl_uav/learning.py", "hrl_uav/agents/sac_agent.py",
        "hrl_uav/routing.py", "hrl_uav/env.py", "configs/main.json")]
    before = {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths}
    checkpoint_hash = file_hash(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mdp = QueueMDP(config)
    mdp.reset(94400, 90)
    agent = DirectHRL.from_mdp(mdp, config)
    agent.sac_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    agent.sac_critic.critic_v.load_state_dict(checkpoint["sac_critic"])
    agent.sac_critic.target_critic_v.load_state_dict(checkpoint["sac_target"])
    for network in (agent.sac_actor.action_net, agent.sac_critic.critic_v, agent.sac_critic.target_critic_v):
        network.eval()
        for param in network.parameters():
            param.requires_grad_(False)
    result = {
        "scope": "Exploratory read-only diagnostic of one final checkpoint. Not retraining or formal performance evidence.",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "audit_script_sha256": file_hash(Path(__file__)),
        "source_sha256": before, "config": config,
        "protocol": {"replay_states": 512, "stochastic_draws": 4,
                     "fork_seeds": [33500, 33501], "fork_times_s": [31, 291, 591],
                     "horizons_s": [.5, 10, 30], "macro": "centroid, frozen",
                     "pulse": "Only first service frame low routing uniform, then same frozen Actor",
                     "sustained": "Low routing uniform for 30 s; high routing remains frozen Actor",
                     "diagnostic_extension": "Exploratory addition after initial uniform probe: bypass UAV 1 entirely for one frame / 30 s. Prior uniform-only results retained separately. No gate or policy optimization.",
                     "limits": "Finite physical returns under deterministic continuation are not the learned soft Q under stochastic SAC continuation with future entropy. Tail also missing. States within seed correlated; no significance or optimality claim."},
        "replay": replay_audit(checkpoint, agent),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"replay_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["replay"]["deterministic"], indent=2), flush=True)
    if not args.replay_only:
        result["forks"] = fork_audit(config, agent)
    result["source_unchanged"] = before == {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths}
    result["checkpoint_unchanged"] = checkpoint_hash == file_hash(args.checkpoint)
    assert result["source_unchanged"] and result["checkpoint_unchanged"]
    (args.output/"report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Read-only feedback audit complete", flush=True)


if __name__ == "__main__":
    main()
