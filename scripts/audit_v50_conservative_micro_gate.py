"""Audit a conservative SAC-residual gate against residual SAC and pure SCA.

The macro PPO policy and all environment randomness are held fixed.  The gate
accepts the learned latent residual only when its pessimistic twin-Critic value
is greater than the optimistic value of the zero-residual (pure SCA) action.
This is a zero-training diagnostic; it does not modify any checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_v45_fair_screen as v45  # noqa: E402
from hrl_uav.agents.sac_agent import Actor as SACActor  # noqa: E402
from hrl_uav.agents.sac_agent import Critic as SACCritic  # noqa: E402
from train_centralized_macro import (  # noqa: E402
    compose_support_preserving_sca_residual,
    torch_load_checkpoint,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v49_support_residual_synchronous"
    / "formal"
    / "seed1"
    / "joint_ppo_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v50_conservative_micro_gate_audit"
ARMS = ("residual_sac", "pure_sca", "conservative_gate")
LABELS = {
    "residual_sac": "原始残差SAC",
    "pure_sca": "纯SCA",
    "conservative_gate": "保守优势门控",
}
COLORS = {
    "residual_sac": "#E07B39",
    "pure_sca": "#7A65A8",
    "conservative_gate": "#2878B5",
}


def conservative_twin_advantage(
    candidate_q1: float,
    candidate_q2: float,
    baseline_q1: float,
    baseline_q2: float,
) -> float:
    """Return candidate lower bound minus baseline upper bound."""
    return min(float(candidate_q1), float(candidate_q2)) - max(
        float(baseline_q1), float(baseline_q2)
    )


class ConservativeResidualPolicy:
    def __init__(self, actor, critic, args):
        self.actor = actor
        self.critic = critic
        self.args = args
        self.decisions = 0
        self.accepted = 0
        self.advantages: list[float] = []

    @property
    def device(self):
        return next(self.critic.critic_v.parameters()).device

    def snapshot(self) -> tuple[int, int, int]:
        return self.decisions, self.accepted, len(self.advantages)

    def delta_since(self, before: tuple[int, int, int]) -> dict[str, float]:
        decisions = self.decisions - before[0]
        accepted = self.accepted - before[1]
        values = np.asarray(self.advantages[before[2] :], dtype=np.float64)
        return {
            "gate_decisions": int(decisions),
            "gate_accepted": int(accepted),
            "gate_accept_ratio": float(accepted / decisions) if decisions else 0.0,
            "gate_advantage_mean": float(values.mean()) if values.size else 0.0,
            "gate_advantage_p95": float(np.percentile(values, 95)) if values.size else 0.0,
        }

    def __call__(self, env, _eval_mode):
        state_np = np.asarray(env.get_sac_state(), dtype=np.float32)
        candidate = self.actor.choose_action(state_np, deterministic=True)
        baseline = np.zeros_like(candidate, dtype=np.float32)
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        candidate_tensor = torch.as_tensor(
            candidate, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        baseline_tensor = torch.as_tensor(
            baseline, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            candidate_q = self.critic.critic_v(state, candidate_tensor)
            baseline_q = self.critic.critic_v(state, baseline_tensor)
        advantage = conservative_twin_advantage(
            candidate_q[0].item(),
            candidate_q[1].item(),
            baseline_q[0].item(),
            baseline_q[1].item(),
        )
        self.decisions += 1
        self.advantages.append(float(advantage))
        latent = baseline
        if advantage > 0.0:
            self.accepted += 1
            latent = candidate
        return compose_support_preserving_sca_residual(env, latent, self.args)


def zero_residual_policy(args):
    def _policy(env, _eval_mode):
        latent = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        return compose_support_preserving_sca_residual(env, latent, args)

    return _policy


def load_models(checkpoint_path: Path, requested_device: str):
    base = v45.load_method_policy(
        "joint_ppo_sac", checkpoint_path, requested_device
    )
    checkpoint = torch_load_checkpoint(str(checkpoint_path))
    args = base["args"]
    probe_env = v45.make_eval_env(args, record_trajectory=False)
    probe_env.reset(seed=int(args.seed))
    state_dim = int(probe_env.sac_obs_dim)
    action_dim = int(checkpoint["sac_actor"]["out.weight"].shape[0])
    actor = SACActor(state_dim, action_dim)
    critic = SACCritic(state_dim, action_dim)
    actor.action_net.load_state_dict(checkpoint["sac_actor"])
    critic.critic_v.load_state_dict(checkpoint["sac_critic"])
    actor.action_net.eval()
    critic.critic_v.eval()
    gate = ConservativeResidualPolicy(actor, critic, args)
    policies = {
        "residual_sac": {**base, "name": "residual_sac"},
        "pure_sca": {
            **base,
            "name": "pure_sca",
            "micro_policy": zero_residual_policy(args),
        },
        "conservative_gate": {
            **base,
            "name": "conservative_gate",
            "micro_policy": gate,
        },
    }
    return policies, gate


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(rows: list[dict], left: str, right: str, field: str) -> dict:
    left_values = {
        int(row["seed"]): float(row[field])
        for row in rows
        if row["method"] == left
    }
    right_values = {
        int(row["seed"]): float(row[field])
        for row in rows
        if row["method"] == right
    }
    seeds = sorted(set(left_values) & set(right_values))
    deltas = [left_values[seed] - right_values[seed] for seed in seeds]
    return {**v45.mean_ci95(deltas), "per_seed": dict(zip(seeds, deltas))}


def build_report(rows: list[dict], checkpoint_path: Path) -> dict:
    fields = (
        "task_delay_weighted_ms",
        "queue_total_mbits",
        "average_power_w",
        "soft_objective_j",
    )
    summaries = {
        arm: {
            field: v45.mean_ci95(
                [float(row[field]) for row in rows if row["method"] == arm]
            )
            for field in fields
        }
        for arm in ARMS
    }
    gate_rows = [row for row in rows if row["method"] == "conservative_gate"]
    summaries["conservative_gate"]["gate_accept_ratio"] = v45.mean_ci95(
        [float(row["gate_accept_ratio"]) for row in gate_rows]
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "statistics_unit": "paired_evaluation_seed (screening only; one training seed)",
        "gate_rule": "min(Q_candidate) > max(Q_zero_residual), margin=0",
        "summaries": summaries,
        "paired_delay_ms": {
            "gate_minus_residual": paired_summary(
                rows, "conservative_gate", "residual_sac", "task_delay_weighted_ms"
            ),
            "gate_minus_sca": paired_summary(
                rows, "conservative_gate", "pure_sca", "task_delay_weighted_ms"
            ),
            "residual_minus_sca": paired_summary(
                rows, "residual_sac", "pure_sca", "task_delay_weighted_ms"
            ),
        },
    }


def plot_report(report: dict, output: Path) -> None:
    v45.configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    panels = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 时延"),
        ("queue_total_mbits", "平均队列积压 / Mbit", "(b) 队列"),
        ("average_power_w", "平均功率 / W", "(c) 功率"),
    )
    x = np.arange(len(ARMS))
    for axis, (field, ylabel, title) in zip(axes, panels):
        stats = [report["summaries"][arm][field] for arm in ARMS]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=[COLORS[arm] for arm in ARMS],
            width=0.66,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3)
        axis.set_xticks(x, [LABELS[arm] for arm in ARMS])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("同一PPO上层下的保守残差门控审计（20个共同种子）")
    fig.tight_layout()
    fig.savefig(output / "01_保守残差门控对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies, gate = load_models(checkpoint_path, args.device)
    rows = []
    for seed in v45.parse_seed_spec(args.eval_seeds):
        for arm in ARMS:
            before = gate.snapshot() if arm == "conservative_gate" else None
            row = v45.evaluate_one(policies[arm], seed)
            if before is not None:
                row.update(gate.delta_since(before))
            else:
                row.update(
                    {
                        "gate_decisions": 0,
                        "gate_accepted": 0,
                        "gate_accept_ratio": 0.0,
                        "gate_advantage_mean": 0.0,
                        "gate_advantage_p95": 0.0,
                    }
                )
            rows.append(row)
            print(
                f"seed={seed} arm={arm} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms "
                f"accept={row['gate_accept_ratio']:.3f}",
                flush=True,
            )
    write_csv(output / "per_seed_raw.csv", rows)
    report = build_report(rows, checkpoint_path)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_report(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
