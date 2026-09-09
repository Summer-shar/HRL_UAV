"""Frozen-objective routing comparisons; evaluation only, never training."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import append_csv, snapshot, write_json
from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import means, metered_evaluate, verify_crn
from scripts.audit_micro_feedback import file_hash


SEEDS = (33500, 33501, 33502)
FORK_MACROS = (3, 29, 59)
HORIZONS = (1, 20, 60)
MODES = ("actor", "pulse_uniform", "pulse_one_way", "sustained_uniform", "sustained_one_way")


def one_way_route(loads):
    """A diagnostic flow balance, restricted to one above-mean source."""
    loads = np.asarray(loads, dtype=np.float64)
    if loads.shape != (3,) or not np.isfinite(loads).all() or (loads < 0).any():
        raise ValueError("Expected three finite nonnegative observed loads")
    row = np.eye(3)
    target = float(loads.mean())
    if target > 0:
        heavy = int(loads.argmax())
        others = np.arange(3) != heavy
        if np.any(loads[others] > target + 1e-6):
            raise ValueError("Diagnostic supports only one above-mean source")
        row[heavy] = np.maximum(target - loads, 0) / loads[heavy]
        row[heavy, heavy] = 1.0 - row[heavy, others].sum()
    return np.stack([row, row])


class OneWayRouting(UniformRouting):
    """Use only the current-load slice already supplied to Direct SAC."""

    def __init__(self, load_slice):
        super().__init__()
        self.load_slice = slice(*load_slice)

    def choose_action(self, state, deterministic=False):
        loads = 10 * np.expm1(np.asarray(state, dtype=np.float64)[self.load_slice])
        return one_way_route(loads).ravel().astype(np.float32)


def candidate_scores(agent, state, rule, candidate):
    if candidate == "actor":
        return agent.sac_actor.choose_action(state, deterministic=True)
    if candidate == "uniform":
        return np.zeros(18, dtype=np.float32)
    if candidate == "one_way":
        return rule.choose_action(state, deterministic=True)
    raise ValueError(candidate)


def step(mdp, agent, rule, mode="actor", first_service=False, hasher=None):
    env = mdp.env
    if env.global_micro_step % env.micro_steps_per_macro == 0:
        if env.global_micro_step:
            mdp.advance_macro_clock()
        mdp.start_macro(np.zeros(mdp.macro_action_dim))
    service = env.macro_service_pause_steps_remaining == 0
    scores = np.zeros(mdp.micro_action_dim)
    if service:
        state = mdp.prepare_service()
        candidate = "actor"
        if mode.startswith("sustained_") or (mode.startswith("pulse_") and first_service):
            candidate = mode.split("_", 1)[1]
        scores = candidate_scores(agent, state, rule, candidate)
    if hasher is not None:
        for array in (env.user_coords, env.uav_coords, env.A_t_high, env.A_t_low,
                      env.zone_expected_offered_load_mbps):
            hasher.update(np.asarray(array, dtype=np.float64).tobytes())
    return mdp.step_frame(scores, mdp.config["training"]["gamma_per_resource_frame"]), service


@torch.no_grad()
def branch(snapshot_mdp, agent, rule, mode, horizons=HORIZONS):
    import hashlib

    mdp = copy.deepcopy(snapshot_mdp)
    initial = mdp.metrics()
    completed = dict(mdp.env.completed_priority_count)
    hasher = hashlib.sha256()
    consumed = False
    cost = discounted_reward = delay_cost = energy_cost = 0.0
    holding_high = holding_low = 0.0
    records = {}
    gamma = mdp.config["training"]["gamma_per_resource_frame"]
    for frame in range(max(horizons)):
        result, service = step(mdp, agent, rule, mode, not consumed, hasher)
        consumed |= service
        cost += result["cost"]
        discounted_reward += gamma ** frame * result["reward"]
        for info in result["infos"]:
            cfg = mdp.reward_config
            h = info["task_holding_cost_weighted"]
            delay_cost += cfg["delay_weight"] * h / cfg["holding_norm_task_s"]
            energy_cost += cfg["energy_weight"] * (info["total_energy"] + mdp.propulsion_delta_share) / cfg["energy_norm_j"]
            holding_high += info["task_alive_high_count"] * mdp.env.micro_dt
            holding_low += info["task_alive_low_count"] * mdp.env.micro_dt
        if frame + 1 in horizons:
            after = mdp.metrics()
            np.testing.assert_allclose(delay_cost + energy_cost, cost, atol=1e-7, rtol=1e-10)
            expected_delay = (mdp.env.w_D_high * holding_high + mdp.env.w_D_low * holding_low)
            expected_delay *= cfg["delay_weight"] / cfg["holding_norm_task_s"]
            np.testing.assert_allclose(expected_delay, delay_cost, atol=1e-7, rtol=1e-10)
            records[str(frame + 1)] = {
                "elapsed_s": after["time_s"] - initial["time_s"],
                "cost_integral": cost, "discounted_physical_return": discounted_reward,
                "delay_cost_component": delay_cost, "energy_cost_component": energy_cost,
                "high_unfinished_task_seconds": holding_high,
                "low_unfinished_task_seconds": holding_low,
                "queue_total_mbit": after["queue_total_mbit"],
                "queue_compute_mbit": after["queue_compute_mbit"],
                "queue_transfer_mbit": after["queue_transfer_mbit"],
                "low_compute_queue_mbit": mdp.env.Q_loc_low.tolist(),
                "unfinished": after["unfinished"],
                "oldest_age_s": after["oldest_age_s"],
                "completed_since_fork": {p: mdp.env.completed_priority_count[p] - completed[p] for p in completed},
                "exogenous_sha256": hasher.hexdigest(),
            }
    return records


@torch.no_grad()
def critic_values(mdp, agent, rule):
    state = mdp.prepare_service()
    values = {}
    for name in ("actor", "uniform", "one_way"):
        scores = candidate_scores(agent, state, rule, name)
        q1, q2 = agent.sac_critic.get_v(tensor(state)[None], tensor(scores)[None])
        route = mdp.route(scores)
        values[name] = {"q1": float(q1.item()), "q2": float(q2.item()),
                        "q_min": float(torch.minimum(q1, q2).item()), "route": route.tolist()}
    for value in values.values():
        value["q_minus_actor"] = value["q_min"] - values["actor"]["q_min"]
    return values


def summarize_forks(forks):
    summary = {}
    for mode in MODES[1:]:
        candidate = mode.split("_", 1)[1]
        summary[mode] = {}
        for horizon in HORIZONS:
            rows = []
            for fork in forks:
                base = fork["branches"]["actor"][str(horizon)]
                changed = fork["branches"][mode][str(horizon)]
                rows.append({
                    "seed": fork["seed"], "time_s": fork["time_s"],
                    "cost_delta": changed["cost_integral"] - base["cost_integral"],
                    "physical_return_delta": changed["discounted_physical_return"] - base["discounted_physical_return"],
                    "delay_cost_delta": changed["delay_cost_component"] - base["delay_cost_component"],
                    "energy_cost_delta": changed["energy_cost_component"] - base["energy_cost_component"],
                    "q_delta": fork["critic"][candidate]["q_minus_actor"],
                })
            summary[mode][str(horizon)] = {
                "states": len(rows), "lower_cost_states": sum(row["cost_delta"] < -1e-8 for row in rows),
                "mean_cost_delta": float(np.mean([r["cost_delta"] for r in rows])),
                "mean_physical_return_delta": float(np.mean([r["physical_return_delta"] for r in rows])),
                "mean_delay_cost_delta": float(np.mean([r["delay_cost_delta"] for r in rows])),
                "mean_energy_cost_delta": float(np.mean([r["energy_cost_delta"] for r in rows])),
                "mean_q_delta": float(np.mean([r["q_delta"] for r in rows])),
                "q_prefers_candidate_states": sum(r["q_delta"] > 1e-8 for r in rows),
                "per_state": rows,
            }
    return summary


def run(checkpoint_path, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(95008)
    checkpoint_path = checkpoint_path.resolve()
    source_hash = file_hash(checkpoint_path)
    paths = [p for folder in ("hrl_uav", "configs") for p in (ROOT/folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".json")]
    paths += [ROOT/"scripts/audit_micro_feedback.py", ROOT/"docs/current_objective_contract.md", Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mdp = QueueMDP(config)
    mdp.reset(SEEDS[0], 90)
    assert mdp.env.w_D_high == 10 and mdp.env.w_D_low == 1
    assert config["reward"] == {"delay_weight": .9, "energy_weight": .1,
                                 "holding_norm_task_s": 10, "energy_norm_j": 14}
    agent = DirectHRL.from_mdp(mdp, config)
    networks = {"sac_actor": agent.sac_actor.action_net, "sac_critic": agent.sac_critic.critic_v,
                "sac_target": agent.sac_critic.target_critic_v, "macro_actor": agent.macro_actor,
                "macro_critic": agent.macro_critic}
    for key, network in networks.items():
        network.load_state_dict(checkpoint[key])
        network.eval().requires_grad_(False)
    before_networks = {key: snapshot(network) for key, network in networks.items()}
    rule = OneWayRouting(mdp.observation_spec()["micro"]["slices"]["current_load"])
    protocol = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": source_hash,
        "source_sha256": hashes, "config": config,
        "objective_contract": "docs/current_objective_contract.md",
        "training": False, "macro": "Same centroid deployment for all arms",
        "development_seeds": list(SEEDS), "fork_times_s": [31, 291, 591],
        "horizons_s": [.5, 10, 30], "modes": list(MODES), "full_replay_seconds": 900,
        "one_way": "Current observed expected load only; one above-mean source exports to equalize mean destination work; other sources local; same matrix for both priorities",
        "limits": "Exploratory existing development scenes, one training seed. Finite deterministic physical returns are not the learned stochastic soft Q or its complete tail. No global optimum or training-root-cause claim.",
    }
    write_json(output/"protocol.json", protocol)
    write_json(output/"status.json", {"status": "RUNNING", "phase": "forks"})
    forks = []
    for seed in SEEDS:
        mdp = QueueMDP(config)
        mdp.reset(seed, 90)
        for _ in range(1200):
            env = mdp.env
            if mdp.macro_index in FORK_MACROS and env.global_micro_step % env.micro_steps_per_macro == env.deployment_steps:
                mdp.prepare_service()
                row = {"seed": seed, "time_s": env._current_time(), "critic": critic_values(mdp, agent, rule),
                       "low_compute_queue_mbit": env.Q_loc_low.tolist()}
                row["branches"] = {mode: branch(mdp, agent, rule, mode) for mode in MODES}
                for horizon in HORIZONS:
                    assert len({r[str(horizon)]["exogenous_sha256"] for r in row["branches"].values()}) == 1, "CRN mismatch"
                row["crn_equal"] = True
                forks.append(row)
                write_json(output/"forks.json", forks)
                print(f"Fork seed={seed} time={row['time_s']:.0f}s complete", flush=True)
                if mdp.macro_index == FORK_MACROS[-1]:
                    break
            step(mdp, agent, rule)
    assert len(forks) == len(SEEDS)*len(FORK_MACROS)
    full = {}
    for label, policy in (("actor", agent), ("uniform", UniformRouting()), ("one_way", rule)):
        full[label] = []
        for seed in SEEDS:
            write_json(output/"status.json", {"status": "RUNNING", "phase": "full_replay", "policy": label, "seed": seed})
            row = metered_evaluate(config, policy, [seed], 90, output, label)[0]
            full[label].append(row)
            append_csv(output/"full_metrics.csv", [row])
            print(f"Replay {label} seed={seed}: delay={row['task_delay_weighted_ms']:.3f}ms J={row['J']:.6f}", flush=True)
    crn = {label: verify_crn(output, SEEDS, "actor", label) for label in ("uniform", "one_way")}
    assert all(all(flags.values()) for rows in crn.values() for flags in rows.values())
    archive = json.loads((checkpoint_path.parent/"report.json").read_text(encoding="utf-8"))
    reproduction = {}
    for label, key in (("actor", "matched_centroid"), ("uniform", "uniform")):
        previous = {r["seed"]: r for r in archive[key]}
        reproduction[label] = max(abs(row[k]-previous[row["seed"]][k]) for row in full[label]
                                  for k in ("task_delay_weighted_ms", "J", "queue_total_mbit"))
        assert reproduction[label] < 1e-5, reproduction
    paired = {}
    for label in ("uniform", "one_way"):
        paired[label] = [{"seed": current["seed"], **{k: current[k]-base[k] for k in
                         ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "J", "power_w", "queue_total_mbit")}}
                         for current, base in zip(full[label], full["actor"])]
    network_equal = all(torch.equal(old[k], snapshot(networks[name])[k])
                        for name, old in before_networks.items() for k in old)
    source_equal = hashes == {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    assert network_equal and source_equal and source_hash == file_hash(checkpoint_path)
    assert agent.sac_updates == agent.sac_actor_updates == agent.ppo_updates == agent.replay.memory_counter == 0
    result = {"protocol": protocol, "forks": forks, "fork_summary": summarize_forks(forks),
              "full": full, "full_means": {label: means(rows) for label, rows in full.items()},
              "paired_minus_actor": paired, "full_crn": crn,
              "archive_reproduction_max_error": reproduction,
              "source_unchanged": source_equal, "networks_unchanged": network_equal,
              "checkpoint_unchanged": True, "updates": 0, "minutes": (time.perf_counter()-started)/60}
    write_json(output/"report.json", result)
    write_json(output/"status.json", {"status": "COMPLETE", "minutes": result["minutes"]})
    print(json.dumps(result["full_means"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"results/ppo_only_unlock_continue800_to1200_seed1/at1200/final.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"results/current_objective_audit1200")
    args = parser.parse_args()
    run(args.checkpoint, args.output)
