"""Observational journal around the unchanged DirectHRL PPO update implementation."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import learning
from .experiment import append_csv, write_json
from .target_tracking_check import tracking_action


def b_actions(states, observation_spec, radius=60):
    """Decode causal float32 observations; not a new environment/controller call."""
    slices = observation_spec["macro"]["slices"]
    actions, projections = [], []
    for state in np.asarray(states):
        def field(name):
            lo, hi = slices[name]
            return np.asarray(state[lo:hi], dtype=float)
        current = field("uav_coordinates").reshape(3, 2) * 1000
        centroid = current + field("centroid_offset").reshape(3, 2) * 60
        encoded = field("current_load")
        loads = np.sign(encoded) * np.expm1(np.abs(encoded)) * 10
        action, detail = tracking_action(current, centroid, loads, [[0, 0]] * 3, radius, .5)
        actions.append(action)
        projections.append(detail["projected"].sum())
    return np.asarray(actions), np.asarray(projections)


def action_geometry(actions, reference):
    actions, reference = np.asarray(actions, float), np.asarray(reference, float)
    an, bn = np.linalg.norm(actions, axis=1), np.linalg.norm(reference, axis=1)
    dot = np.sum(actions * reference, axis=1)
    cosine = np.divide(dot, an * bn, out=np.full_like(dot, np.nan), where=(an * bn > 1e-12))
    return {
        "distance_to_B": np.linalg.norm(actions - reference, axis=1),
        "B_norm": bn, "action_norm": an,
        "angle_deg": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "projection_length": np.divide(dot, bn, out=np.full_like(dot, np.nan), where=bn > 1e-12),
        "projection_gain": np.divide(dot, bn**2, out=np.full_like(dot, np.nan), where=bn > 1e-12),
    }


def distribution(values, prefix):
    values = np.asarray(values, float)
    return {prefix + "_" + k: float(v) for k, v in {
        "mean": values.mean(), "std": values.std(), "variance": values.var(),
        "p10": np.percentile(values, 10), "p50": np.percentile(values, 50),
        "p90": np.percentile(values, 90),
    }.items()}


class PPOJournal:
    def __init__(self, directory, observation_spec, config):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.spec, self.radius = observation_spec, config["waypoint_radius_m"]
        self.pending = None
        write_json(self.directory / "schema.json", {
            "observations": observation_spec,
            "B": "world interpolation 0.5 towards current centroid, native local feasible projection",
            "B_precision": "decoded from stored float32 observations, not hidden exact environment state",
            "advantage": "raw GAE; advantage_normalized is the tensor used by the PPO objective",
            "return": "actual GAE return before PPO critic updates",
            "value_error": "pre-update value minus GAE return, not error against a known true value",
            "undefined_angle_projection": "NaN when reference/action norm is zero",
            "KL": "existing last_ppo_stats KL: last checked epoch, before that epoch's step",
            "loss": "existing last applied actor loss and last pre-step critic loss",
            "ppo_gradient_steps": "actor optimizer steps; critic steps separately reported",
            "scope": "observational only; no extra rollout, no imitation, no sample selection",
        })

    def before(self, agent, gae_arguments, gae_result):
        if self.pending is not None:
            raise AssertionError("Diagnostic batch already pending")
        states, actions, logprobs, rewards, next_states, continuations = map(np.asarray, zip(*agent.macro_rollout))
        actual_rewards, values, next_values, traces, discount, trace_lambda = gae_arguments
        np.testing.assert_array_equal(rewards, actual_rewards)
        np.testing.assert_array_equal(continuations, traces)
        advantages, returns = gae_result
        b, projected = b_actions(states, self.spec, self.radius)
        normalized = learning.tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        with torch.no_grad():
            mean, std = agent.macro_actor(learning.tensor(states))
        data = dict(state=states, sampled_raw_action=actions, B_action=b,
                    deterministic_mean=mean.detach().cpu().numpy(),
                    reward=rewards, value=values, next_value=next_values,
                    advantage=advantages, advantage_normalized=normalized.cpu().numpy(),
                    returns=returns, old_logprob=logprobs, exploration_std=std.detach().cpu().numpy(),
                    next_state=next_states, continuation=continuations,
                    B_projection_count=projected, **action_geometry(actions, b))
        update = agent.ppo_updates + 1
        path = self.directory / f"update{update:03d}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **data)
        self.pending = dict(update=update, transitions=len(states),
                            ppo_gradients_before=agent.ppo_gradient_steps,
                            exploration_std=float(std[0, 0]),
                            macro_discount=float(discount), gae_lambda=float(trace_lambda),
                            value_error_mse=float(np.mean((values - returns)**2)),
                            **distribution(data["distance_to_B"], "distance_to_B"),
                            **distribution(rewards, "reward"), **distribution(values, "value"),
                            **distribution(advantages, "advantage"),
                            **distribution(normalized.cpu().numpy(), "advantage_normalized"),
                            **distribution(returns, "return"))

    def after(self, agent):
        row = self.pending
        if row is None or agent.ppo_updates != row["update"]:
            raise AssertionError("Diagnostic batch/update mismatch")
        stats = agent.last_ppo_stats
        row.update(actual_actor_epochs=stats["epochs"], kl=stats["kl"], actor_loss=stats["actor_loss"],
                   critic_loss=stats["critic_loss"], pre_logprob_error=stats["pre_logprob_error"],
                   ppo_gradient_steps=agent.ppo_gradient_steps,
                   actor_steps_added=agent.ppo_gradient_steps - row["ppo_gradients_before"],
                   critic_steps_added=agent.cfg["ppo_epochs"],
                   kl_early_stop=stats["epochs"] < agent.cfg["ppo_epochs"],
                   std_after=float(agent.macro_actor.log_std.detach().exp()[0]))
        append_csv(self.directory / "updates.csv", [row])
        self.pending = None


class LoggedDirectHRL(learning.DirectHRL):
    def update_ppo(self, progress=0.0, force=False):
        journal = getattr(self, "ppo_journal", None)
        if journal is None:
            return super().update_ppo(progress, force)
        original_gae = learning.gae
        calls, before = 0, self.ppo_updates

        def observe(*args):
            nonlocal calls
            result = original_gae(*args)
            calls += 1
            # Observe the actual GAE arrays before the original method uses them.
            journal.before(self, args, result)
            return result

        # Single-threaded experiment scope. The original update function is unmodified.
        with patch.object(learning, "gae", observe):
            result = super().update_ppo(progress, force)
        if self.ppo_updates != before:
            if calls != 1:
                raise AssertionError("Expected exactly one observed GAE calculation")
            journal.after(self)
        elif calls:
            raise AssertionError("Observed GAE without an update")
        return result
