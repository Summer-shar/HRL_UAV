"""Two frozen-policy routing interventions, not training or an HRL method."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import write_json
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import means, metered_evaluate


BASE = ROOT / "results/ppo_only_unlock_seed1"
OUT = BASE / "priority_route_swap_audit"


class PrioritySwap:
    def __init__(self, actor, uniform_priority):
        self.sac_actor = self
        self.actor = actor
        self.priority = uniform_priority

    def macro_action(self, state, enabled, deterministic=True):
        assert not enabled
        return np.zeros(6), 0.0

    def choose_action(self, state, deterministic=True):
        assert deterministic
        action = self.actor.choose_action(state, deterministic=True).copy()
        assert action.shape == (18,)
        action.reshape(2, 3, 3)[self.priority] = 0.0
        return action


def check_crn(label, seed, priority=None):
    with np.load(BASE / f"centroid_same_sac_{seed}_trajectory.npz") as original:
        with np.load(OUT / f"{label}_{seed}_trajectory.npz") as new:
            for key in ("users", "uav", "centroid"):
                assert np.array_equal(original[key], new[key]), (label, seed, key)
            if priority is not None:
                assert np.allclose(new["executed_route"][:, priority], 1.0 / 3.0)
            else:
                assert np.array_equal(original["executed_route"], new["executed_route"])
    arrived = []
    for root, policy in ((BASE, "centroid_same_sac"), (OUT, label)):
        with (root / f"{policy}_{seed}_queues.csv").open(encoding="utf-8-sig") as handle:
            arrived.append([float(row["task_arrived_count"]) for row in csv.DictReader(handle)])
    assert arrived[0] == arrived[1], (label, seed, "arrivals")


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    path = BASE / "final.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    archived = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    mdp = QueueMDP(config)
    mdp.reset(1, config["training"]["window_macros"])
    agent = DirectHRL.from_mdp(mdp, config)
    agent.sac_actor.action_net.load_state_dict(saved["sac_actor"], strict=True)
    agent.macro_actor.load_state_dict(saved["macro_actor"], strict=True)
    actor_before = {k: v.detach().clone() for k, v in agent.sac_actor.action_net.state_dict().items()}
    seeds = config["test_seeds"]
    del saved
    report = {
        "scope": "Exploratory frozen-checkpoint interventions; one training seed, three reused development scenes, no training. Uniform overrides are diagnostic only, not learned behavior.",
        "protocol": "Centroid macro for every arm. Retained SAC output is recomputed on each intervened state. Replace all three source rows of exactly one priority by uniform routing. No policy, reward, or optimizer changes.",
        "checkpoint_sha256": digest,
        "seeds": seeds,
        "arms": {
            "sac_high_sac_low": archived["matched_centroid"],
            "uniform_high_uniform_low": archived["uniform"],
        },
    }
    write_json(OUT / "report.json", report)
    reproduced = metered_evaluate(config, agent, seeds[:1], config["test_macros"], OUT, "reproduce")
    reference = next(row for row in archived["matched_centroid"] if row["seed"] == seeds[0])
    error = max(abs(reproduced[0][key] - reference[key]) for key in means(reproduced))
    assert error < 1e-8, error
    check_crn("reproduce", seeds[0])
    report["reproduction_max_error"] = error
    print("Control reproduced:", error, flush=True)
    for label, priority in (("uniform_high_sac_low", 0), ("sac_high_uniform_low", 1)):
        rows = []
        policy = PrioritySwap(agent.sac_actor, priority)
        for seed in seeds:
            row = metered_evaluate(config, policy, [seed], config["test_macros"], OUT, label)[0]
            check_crn(label, seed, priority)
            rows.append(row)
            print(label, seed, {k: row[k] for k in ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "queue_total_mbit", "J")}, flush=True)
        report["arms"][label] = rows
        write_json(OUT / "report.json", report)
    report["means"] = {label: means(rows) for label, rows in report["arms"].items()}
    report["paired_vs_sac_ms"] = {
        label: [row["task_delay_weighted_ms"] - next(r["task_delay_weighted_ms"] for r in archived["matched_centroid"] if r["seed"] == row["seed"]) for row in rows]
        for label, rows in report["arms"].items()
    }
    assert all(torch.equal(v, actor_before[k]) for k, v in agent.sac_actor.action_net.state_dict().items())
    report["actor_parameters_unchanged"] = True
    report["checkpoint_unchanged"] = hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert report["checkpoint_unchanged"]
    report["crn_users_uav_centroids_arrivals_verified"] = True
    write_json(OUT / "report.json", report)
    print(json.dumps(report["means"], indent=2), flush=True)


if __name__ == "__main__":
    main()
