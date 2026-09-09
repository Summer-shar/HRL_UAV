"""Small same-state reward audit; frozen policy, no parameter or replay updates."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hrl_uav.experiment import isolated_rng
from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mainline_6mhz_routing_critic"


def start_next_frame(mdp):
    if mdp.env.global_micro_step and mdp.env.global_micro_step % mdp.env.micro_steps_per_macro == 0:
        mdp.advance_macro_clock()
        mdp.start_macro(np.zeros(mdp.macro_action_dim))
    if mdp.env.macro_service_pause_steps_remaining == 0:
        mdp.prepare_service()


def policy_action(mdp, agent):
    if mdp.env.macro_service_pause_steps_remaining:
        return np.zeros(mdp.micro_action_dim)
    return agent.sac_actor.choose_action(mdp.observation(), deterministic=True)


def probe(mdp, agent):
    state = mdp.observation()
    ready = np.zeros((2, mdp.N))
    for source, ids in enumerate(mdp.env.completed_upload_tasks_by_uav):
        for task_id in ids:
            task = mdp.env.parent_tasks[task_id]
            ready[int(task.priority == "low"), source] += task.size_mbits
    actor_action = policy_action(mdp, agent)
    cases = {"actor": actor_action, "uniform": np.zeros(mdp.micro_action_dim),
             "local": np.stack([np.eye(mdp.N)] * 2).ravel()}
    results = {}
    checkpoints = (1, 10, 20, 60)
    gamma = mdp.config["training"]["gamma_per_resource_frame"]
    w = mdp.config["reward"]
    for label, action in cases.items():
        with isolated_rng():
            fork = copy.deepcopy(mdp)
            discount_return, holding_component, energy_component = 0., 0., 0.
            snapshots = []
            with torch.no_grad():
                qa, qb = agent.sac_critic.get_v(tensor(state[None]), tensor(action[None]))
            for k in range(max(checkpoints)):
                if k:
                    start_next_frame(fork)
                executed = action if k == 0 else policy_action(fork, agent)
                frame = fork.step_frame(executed, gamma)
                discount_return += gamma ** k * frame["reward"]
                holding_component += sum(info["task_holding_cost_weighted"] for info in frame["infos"]) * w["delay_weight"] / w["holding_norm_task_s"]
                energy_component += sum(info["total_energy"] + fork.propulsion_delta_share for info in frame["infos"]) * w["energy_weight"] / w["energy_norm_j"]
                if k + 1 in checkpoints:
                    metrics = fork.metrics()
                    snapshots.append({"horizon_s": (k+1)*.5, "truncated_discounted_return": float(discount_return),
                                      "holding_cost_component": float(holding_component),
                                      "energy_cost_component": float(energy_component),
                                      "queue_mbit": metrics["queue_total_mbit"],
                                      "unfinished": metrics["unfinished"]})
            results[label] = {"route": mdp.route(action).tolist(),
                              "critic_Q1_Q2": [qa.item(), qb.item()], "horizons": snapshots,
                              "arrived_tasks": fork.env.arrived_task_count,
                              "user_coordinates": fork.env.user_coords.tolist()}
    for case in ("uniform", "local"):
        assert results[case]["arrived_tasks"] == results["actor"]["arrived_tasks"]
        assert np.array_equal(results[case]["user_coordinates"], results["actor"]["user_coordinates"])
    for value in results.values():
        del value["user_coordinates"]
    return {"start_s": mdp.env._current_time(), "load_mbps": mdp.env.zone_expected_offered_load_mbps.tolist(),
            "newborn_arrivals_high_low_mbit": np.stack([mdp.env.A_t_high, mdp.env.A_t_low]).tolist(),
            "uploaded_pending_route_high_low_mbit": ready.tolist(),
            "compute_queues_high_low": np.stack([mdp.env.Q_loc_high, mdp.env.Q_loc_low]).tolist(),
            "observed_assigned_cpu_seconds_encoded": state[slice(*mdp.feature_slices["assigned_cpu_seconds"])].tolist(),
            "cases": results, "crn_users_and_arrived_counts_match": True}


def main():
    torch.set_num_threads(1)
    path = OUT / "micro_pretrained.pt"
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    mdp = QueueMDP(config)
    state = mdp.reset(33000, 20)
    agent = DirectHRL(state.size, mdp.micro_action_dim, mdp.macro_action_dim, config)
    agent.sac_actor.action_net.load_state_dict(saved["sac_actor"])
    agent.sac_critic.critic_v.load_state_dict(saved["sac_critic"])
    mdp.start_macro(np.zeros(mdp.macro_action_dim))
    readings = []
    for _ in range(183):
        start_next_frame(mdp)
        if mdp.env.global_micro_step in (10, 310, 910):
            readings.append(probe(mdp, agent))
        mdp.step_frame(policy_action(mdp, agent))
    assert len(readings) == 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    result = {"scope": "Three descriptive state probes in validation scene 33000; no significance claim or infinite-horizon Q truth.",
              "continuation": "First service action differs; same frozen SAC thereafter, centroid macro, cloned state and paired randomness.",
              "holding_semantics": "Weighted alive parent-task count times dt; unfinished parents incur holding until all chunks finish.",
              "checkpoint_sha256_unchanged": original_hash, "readings": readings}
    (OUT / "state_reward_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in readings:
        print("STATE", row["start_s"], "queues", np.round(row["compute_queues_high_low"], 1).tolist(),
              "newborn", np.round(row["newborn_arrivals_high_low_mbit"], 2).tolist(),
              "ready", np.round(row["uploaded_pending_route_high_low_mbit"], 2).tolist())
        for label, case in row["cases"].items():
            print(label, "Q", np.round(case["critic_Q1_Q2"], 3).tolist(), "horizons", case["horizons"])


if __name__ == "__main__":
    main()
