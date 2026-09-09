"""Read-only action-equivalence check on a completed native SAC checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from hrl_uav.experiment import write_json
from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mdp = QueueMDP(config)
    n = mdp.reset(1, 30).size
    a = mdp.micro_action_dim
    agent = DirectHRL(n, a, mdp.macro_action_dim, config)
    agent.sac_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    agent.sac_critic.critic_v.load_state_dict(checkpoint["sac_critic"])
    data = checkpoint["replay"]["transitions"]
    rows = np.random.default_rng(7351).choice(len(data), min(len(data), 256), replace=False)
    state = tensor(data[rows, :n])
    scores = torch.zeros((len(rows), a), device=state.device, requires_grad=True)
    qa, qb = agent.sac_critic.get_v(state, scores)
    q0 = torch.minimum(qa, qb)
    gradient, = torch.autograd.grad(q0.sum(), scores)
    gradient = gradient.reshape(-1, 2, mdp.N, mdp.N)
    shift_component = gradient.mean(dim=-1, keepdim=True).expand_as(gradient)
    ratios = shift_component.flatten(1).norm(dim=1)/gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
    variants = []
    with torch.no_grad():
        for shift in (-0.25, 0.25):
            q1, q2 = agent.sac_critic.get_v(state, scores.detach()+shift)
            difference = (torch.minimum(q1,q2)-q0).cpu().numpy().ravel()
            error = np.max(np.abs(mdp.route(np.full(a, shift))-mdp.route(np.zeros(a))))
            variants.append({"raw_shift": shift, "executed_route_max_difference": float(error),
                             "predicted_Q_difference_mean": float(difference.mean()),
                             "predicted_Q_difference_abs_mean": float(np.abs(difference).mean()),
                             "predicted_Q_difference_abs_max": float(np.abs(difference).max())})
        mean, _ = agent.sac_actor.action_net(state)
        action = mean.tanh()
        q1, q2 = agent.sac_critic.get_v(state, action)
        delta = (torch.minimum(q1,q2)-q0).cpu().numpy().ravel()
    result = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sample_count": len(rows), "sample_source": "Final training replay, private fixed RNG, no held-out adaptation",
        "equivalent_uniform_actions": variants,
        "critic_gradient_fraction_in_route_invariant_directions_mean": float(ratios.mean().item()),
        "actor_minus_uniform_predicted_Q_mean": float(delta.mean()),
        "actor_minus_uniform_predicted_Q_abs_mean": float(np.abs(delta).mean()),
        "critic_prefers_actor_fraction": float(np.mean(delta > 0)),
        "scope": "Same executed route implies identical transitions; Q should be invariant. This diagnoses approximation error, not its causal share of training failure.",
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
