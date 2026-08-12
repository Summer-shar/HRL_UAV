import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import torch_load_checkpoint
from train_centralized_td3 import MacroActor
from train_centralized_td3 import TwinCritic


DEFAULT_CHECKPOINT = (
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_TRAIN_DATASET = (
    "result_archive/v28_td_vs_mc_counterfactual/train_transitions.npz"
)
DEFAULT_TEST_DATASET = (
    "result_archive/v28_td_vs_mc_counterfactual/test_transitions.npz"
)
DEFAULT_OUTPUT = "result_archive/v28_td_vs_mc_counterfactual"

HORIZON = 30
GAMMA = 0.99
HIDDEN_DIM = 256
INIT_SEEDS = (7101, 7102, 7103)
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
UPDATES = 20_000
TAU = 0.005
GRAD_CLIP = 10.0
LOG_INTERVAL = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen v28 matched TD/FQE and truncated-return Critic "
            "arms."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--test-dataset", default=DEFAULT_TEST_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path) as payload:
        return {key: payload[key].copy() for key in payload.files}


def branch_groups(payload):
    groups = {}
    for index, branch_id in enumerate(payload["branch_id"].astype(str)):
        groups.setdefault(branch_id, []).append(index)
    result = {}
    for branch_id, indices in groups.items():
        ordered = sorted(indices, key=lambda idx: int(payload["branch_step"][idx]))
        result[branch_id] = np.asarray(ordered, dtype=np.int64)
    return result


def validate_crn_pairs(payload, horizon=HORIZON):
    groups = branch_groups(payload)
    forks = {}
    for branch_id, indices in groups.items():
        indices = indices[: int(horizon)]
        fork_state_id = str(payload["fork_state_id"][indices[0]])
        branch_action = str(payload["branch_action"][indices[0]])
        forks.setdefault(fork_state_id, {})[branch_action] = payload[
            "external_trace_hash"
        ][indices].astype(str)
    for fork_state_id, branches in forks.items():
        if set(branches) != {"actor", "zero"}:
            return False
        if not np.array_equal(branches["actor"], branches["zero"]):
            return False
    return bool(forks)


def discounted_return_to_boundary(rewards, gamma=GAMMA):
    rewards = np.asarray(rewards, dtype=np.float64)
    powers = np.power(float(gamma), np.arange(rewards.size, dtype=np.float64))
    return float(np.sum(powers * rewards))


def prepare_finite_horizon_dataset(payload, horizon=HORIZON, gamma=GAMMA):
    groups = branch_groups(payload)
    selected = []
    returns = []
    remaining = []
    for branch_id, indices in sorted(groups.items()):
        if indices.size < int(horizon):
            raise RuntimeError(
                f"Branch {branch_id} has {indices.size} steps; expected >= {horizon}."
            )
        indices = indices[: int(horizon)]
        rewards = np.asarray(payload["reward"][indices], dtype=np.float64)
        for offset, index in enumerate(indices):
            selected.append(int(index))
            returns.append(discounted_return_to_boundary(rewards[offset:], gamma))
            remaining.append(int(horizon) - offset)

    selected = np.asarray(selected, dtype=np.int64)
    remaining = np.asarray(remaining, dtype=np.int16)
    horizon_feature = (remaining.astype(np.float32) / float(horizon)).reshape(-1, 1)
    next_horizon_feature = (
        np.maximum(remaining.astype(np.float32) - 1.0, 0.0) / float(horizon)
    ).reshape(-1, 1)
    return {
        "state": np.concatenate(
            (payload["state"][selected].astype(np.float32), horizon_feature),
            axis=1,
        ),
        "actor_state": payload["state"][selected].astype(np.float32),
        "action": payload["action"][selected].astype(np.float32),
        "reward": payload["reward"][selected].astype(np.float32).reshape(-1, 1),
        "next_state": np.concatenate(
            (
                payload["next_state"][selected].astype(np.float32),
                next_horizon_feature,
            ),
            axis=1,
        ),
        "next_actor_state": payload["next_state"][selected].astype(np.float32),
        "remaining": remaining.reshape(-1, 1),
        "truncated_return": np.asarray(returns, dtype=np.float32).reshape(-1, 1),
        "source_index": selected,
    }


def load_frozen_actor(checkpoint_path, device):
    payload = torch_load_checkpoint(str(checkpoint_path))
    args = payload["args"]
    state_dim = int(payload["macro_actor"]["net.0.weight"].shape[1])
    action_dim = int(payload["macro_actor"]["net.4.weight"].shape[0])
    hidden_dim = int(args["td3_hidden_dim"])
    actor = MacroActor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(payload["macro_actor"])
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, state_dim, action_dim


def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def soft_update(target, source, tau=TAU):
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - float(tau))
            target_parameter.add_(source_parameter, alpha=float(tau))


def train_paired_arms(train, actor, action_dim, init_seed, device):
    set_all_seeds(init_seed)
    state_dim = int(train["state"].shape[1])
    base = TwinCritic(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(device)
    initial_state = copy.deepcopy(base.state_dict())

    td_critic = TwinCritic(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(device)
    return_critic = TwinCritic(
        state_dim, action_dim, hidden_dim=HIDDEN_DIM
    ).to(device)
    td_critic.load_state_dict(initial_state)
    return_critic.load_state_dict(initial_state)
    td_target = copy.deepcopy(td_critic).to(device).eval()
    return_target = copy.deepcopy(return_critic).to(device).eval()

    td_optimizer = torch.optim.Adam(td_critic.parameters(), lr=LEARNING_RATE)
    return_optimizer = torch.optim.Adam(
        return_critic.parameters(), lr=LEARNING_RATE
    )
    rng = np.random.default_rng(int(init_seed) + 1_000_003)
    rows = []
    sample_count = int(train["state"].shape[0])

    for update in range(1, UPDATES + 1):
        indices = rng.integers(0, sample_count, size=BATCH_SIZE, endpoint=False)
        state_t = torch.as_tensor(train["state"][indices], device=device)
        actor_state_t = torch.as_tensor(
            train["actor_state"][indices], device=device
        )
        action_t = torch.as_tensor(train["action"][indices], device=device)
        reward_t = torch.as_tensor(train["reward"][indices], device=device)
        next_state_t = torch.as_tensor(train["next_state"][indices], device=device)
        next_actor_state_t = torch.as_tensor(
            train["next_actor_state"][indices], device=device
        )
        remaining_t = torch.as_tensor(
            train["remaining"][indices], device=device
        )
        return_t = torch.as_tensor(
            train["truncated_return"][indices], device=device
        )

        with torch.no_grad():
            next_action_t = actor(next_actor_state_t)
            next_q1, next_q2 = td_target(next_state_t, next_action_t)
            bootstrap = torch.minimum(next_q1, next_q2)
            td_label = reward_t + GAMMA * bootstrap
            td_label = torch.where(remaining_t <= 1, reward_t, td_label)

        td_q1, td_q2 = td_critic(state_t, action_t)
        td_loss = F.mse_loss(td_q1, td_label) + F.mse_loss(td_q2, td_label)
        td_optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(td_critic.parameters(), GRAD_CLIP)
        td_optimizer.step()

        return_q1, return_q2 = return_critic(state_t, action_t)
        return_loss = F.mse_loss(return_q1, return_t) + F.mse_loss(
            return_q2, return_t
        )
        return_optimizer.zero_grad(set_to_none=True)
        return_loss.backward()
        torch.nn.utils.clip_grad_norm_(return_critic.parameters(), GRAD_CLIP)
        return_optimizer.step()

        soft_update(td_target, td_critic)
        soft_update(return_target, return_critic)

        if update == 1 or update % LOG_INTERVAL == 0 or update == UPDATES:
            rows.append(
                {
                    "init_seed": int(init_seed),
                    "update": int(update),
                    "td_loss": float(td_loss.detach().cpu()),
                    "truncated_return_loss": float(return_loss.detach().cpu()),
                }
            )
    return td_critic.eval(), return_critic.eval(), rows


def initial_fork_queries(payload, horizon=HORIZON, gamma=GAMMA):
    groups = branch_groups(payload)
    by_fork = {}
    for branch_id, indices in groups.items():
        ordered = indices[: int(horizon)]
        fork_state_id = str(payload["fork_state_id"][ordered[0]])
        branch_action = str(payload["branch_action"][ordered[0]])
        entry = by_fork.setdefault(
            fork_state_id,
            {
                "episode_seed": int(payload["episode_seed"][ordered[0]]),
                "snapshot_policy": str(
                    payload["snapshot_policy"][ordered[0]]
                ),
                "fork_macro_index": int(
                    payload["fork_macro_index"][ordered[0]]
                ),
            },
        )
        entry[f"state_{branch_action}"] = payload["state"][ordered[0]].astype(
            np.float32
        )
        entry[f"action_{branch_action}"] = payload["action"][ordered[0]].astype(
            np.float32
        )
        entry[f"return_{branch_action}"] = discounted_return_to_boundary(
            payload["reward"][ordered], gamma
        )
    result = []
    for fork_state_id, entry in sorted(by_fork.items()):
        required = (
            "state_actor",
            "state_zero",
            "action_actor",
            "action_zero",
            "return_actor",
            "return_zero",
        )
        if any(key not in entry for key in required):
            raise RuntimeError(f"Incomplete paired fork {fork_state_id}.")
        if not np.array_equal(entry["state_actor"], entry["state_zero"]):
            raise RuntimeError(f"Fork state mismatch for {fork_state_id}.")
        result.append(
            {
                "fork_state_id": fork_state_id,
                **entry,
                "true_delta": float(
                    entry["return_actor"] - entry["return_zero"]
                ),
            }
        )
    return result


def predict_fork_deltas(critics, queries, device):
    states = np.stack([row["state_actor"] for row in queries]).astype(np.float32)
    horizon = np.ones((states.shape[0], 1), dtype=np.float32)
    states = np.concatenate((states, horizon), axis=1)
    actor_actions = np.stack([row["action_actor"] for row in queries]).astype(
        np.float32
    )
    zero_actions = np.stack([row["action_zero"] for row in queries]).astype(
        np.float32
    )
    state_t = torch.as_tensor(states, device=device)
    actor_action_t = torch.as_tensor(actor_actions, device=device)
    zero_action_t = torch.as_tensor(zero_actions, device=device)
    predictions = []
    with torch.no_grad():
        for critic in critics:
            actor_q1, actor_q2 = critic(state_t, actor_action_t)
            zero_q1, zero_q2 = critic(state_t, zero_action_t)
            actor_value = torch.minimum(actor_q1, actor_q2)
            zero_value = torch.minimum(zero_q1, zero_q2)
            predictions.append((actor_value - zero_value).squeeze(1).cpu().numpy())
    return np.mean(np.stack(predictions, axis=0), axis=0)


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return [mean, mean, mean]
    critical = float(stats.t.ppf(0.975, values.size - 1))
    sem = float(stats.sem(values))
    return [mean, mean - critical * sem, mean + critical * sem]


def build_prediction_rows(split, queries, td_delta, return_delta):
    rows = []
    for query, td_value, return_value in zip(queries, td_delta, return_delta):
        true_delta = float(query["true_delta"])
        rows.append(
            {
                "split": split,
                "episode_seed": int(query["episode_seed"]),
                "snapshot_policy": query["snapshot_policy"],
                "fork_macro_index": int(query["fork_macro_index"]),
                "fork_state_id": query["fork_state_id"],
                "true_delta": true_delta,
                "td_delta": float(td_value),
                "truncated_return_delta": float(return_value),
                "td_agreement": float(np.sign(td_value) == np.sign(true_delta)),
                "truncated_return_agreement": float(
                    np.sign(return_value) == np.sign(true_delta)
                ),
            }
        )
    return rows


def episode_rows(prediction_rows):
    grouped = {}
    for row in prediction_rows:
        key = (row["split"], int(row["episode_seed"]), row["snapshot_policy"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (split, seed, policy), rows in sorted(grouped.items()):
        if len(rows) != 6:
            raise RuntimeError(f"Expected six forks for {split}/{seed}/{policy}.")
        output.append(
            {
                "split": split,
                "episode_seed": seed,
                "snapshot_policy": policy,
                "fork_count": len(rows),
                "td_agreement": float(np.mean([row["td_agreement"] for row in rows])),
                "truncated_return_agreement": float(
                    np.mean([row["truncated_return_agreement"] for row in rows])
                ),
                "true_delta_mean": float(np.mean([row["true_delta"] for row in rows])),
                "td_delta_mean": float(np.mean([row["td_delta"] for row in rows])),
                "truncated_return_delta_mean": float(
                    np.mean([row["truncated_return_delta"] for row in rows])
                ),
            }
        )
    return output


def classify(agreement_ci, predicted_mean, true_mean):
    if agreement_ci[1] > 0.5 and np.sign(predicted_mean) == np.sign(true_mean):
        return "CORRECT"
    if agreement_ci[2] < 0.5:
        return "WRONG"
    return "INCONCLUSIVE"


def summarize_predictions(episodes):
    result = {}
    for split in ("train", "test"):
        result[split] = {}
        for policy in ("actor", "centroid"):
            selected = [
                row
                for row in episodes
                if row["split"] == split and row["snapshot_policy"] == policy
            ]
            policy_result = {"episodes": len(selected)}
            true_mean = float(np.mean([row["true_delta_mean"] for row in selected]))
            policy_result["true_delta_mean"] = true_mean
            for arm in ("td", "truncated_return"):
                agreement_ci = mean_ci95(
                    [row[f"{arm}_agreement"] for row in selected]
                )
                predicted_mean = float(
                    np.mean([row[f"{arm}_delta_mean"] for row in selected])
                )
                policy_result[arm] = {
                    "agreement_ci95": agreement_ci,
                    "predicted_delta_mean": predicted_mean,
                    "classification": classify(
                        agreement_ci, predicted_mean, true_mean
                    ),
                }
            result[split][policy] = policy_result
    return result


def apply_gates_and_verdict(
    summary,
    horizon_gate=True,
    crn_gate=True,
    finite_loss_gate=True,
):
    train_actor_return = summary["train"]["actor"]["truncated_return"]
    train_centroid_return = summary["train"]["centroid"][
        "truncated_return"
    ]
    test_actor_return = summary["test"]["actor"]["truncated_return"]
    gates = {
        "horizon_stability": bool(horizon_gate),
        "crn_and_counts": bool(crn_gate),
        "finite_training_losses": bool(finite_loss_gate),
        "truncated_return_train_actor_agreement_ge_90pct": bool(
            train_actor_return["agreement_ci95"][0] >= 0.90
        ),
        "truncated_return_train_centroid_agreement_ge_90pct": bool(
            train_centroid_return["agreement_ci95"][0] >= 0.90
        ),
        "truncated_return_test_actor_correct": bool(
            test_actor_return["classification"] == "CORRECT"
        ),
    }
    if not all(gates.values()):
        return gates, "INVALID_OR_INCONCLUSIVE"

    td = summary["test"]["centroid"]["td"]["classification"]
    direct = summary["test"]["centroid"]["truncated_return"][
        "classification"
    ]
    if direct == "CORRECT" and td == "WRONG":
        verdict = "SUPPORTS_BOOTSTRAP_BIAS_PROPAGATION"
    elif direct == "WRONG" and td == "WRONG":
        verdict = "BOTH_ARMS_WRONG_UNRESOLVED"
    elif direct == "WRONG":
        verdict = "JOINT_SUPPORT_OR_GENERALIZATION_NOT_RESOLVED"
    elif direct == "CORRECT" and td == "CORRECT":
        verdict = "OFFLINE_FIXED_POLICY_DID_NOT_REPRODUCE_FAILURE"
    else:
        verdict = "INCONCLUSIVE"
    return gates, verdict


def save_checkpoint(path, critic, init_seed, arm):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "critic": critic.state_dict(),
            "init_seed": int(init_seed),
            "arm": arm,
            "updates": UPDATES,
        },
        path,
    )


def write_report(path, result):
    test = result["ranking"]["test"]
    lines = [
        "# v28 TD-vs-truncated-return ranking report",
        "",
        f"- Verdict: `{result['verdict']}`",
        f"- Pipeline gates: `{json.dumps(result['gates'], ensure_ascii=False)}`",
        "",
        "| State region | Arm | Agreement mean [95% CI] | Mean predicted delta | Mean true delta | Classification |",
        "|---|---|---:|---:|---:|---|",
    ]
    for policy in ("actor", "centroid"):
        for arm in ("td", "truncated_return"):
            arm_result = test[policy][arm]
            mean, low, high = arm_result["agreement_ci95"]
            lines.append(
                f"| {policy} | {arm} | {mean:.4f} [{low:.4f}, {high:.4f}] "
                f"| {arm_result['predicted_delta_mean']:.8f} "
                f"| {test[policy]['true_delta_mean']:.8f} "
                f"| {arm_result['classification']} |"
            )
    lines.extend(
        [
            "",
            "Statistical unit is one episode. The six fork states are aggregated "
            "within each episode before the confidence interval is computed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    train_path = project_path(args.train_dataset)
    test_path = project_path(args.test_dataset)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actor, actor_state_dim, action_dim = load_frozen_actor(
        checkpoint_path, device
    )
    train_payload = load_npz(train_path)
    test_payload = load_npz(test_path)
    if train_payload["state"].shape[1] != actor_state_dim:
        raise RuntimeError("Train dataset and frozen Actor state dimensions differ.")
    if test_payload["state"].shape[1] != actor_state_dim:
        raise RuntimeError("Test dataset and frozen Actor state dimensions differ.")
    train = prepare_finite_horizon_dataset(train_payload)

    frozen_hyperparams = {
        "status": "FROZEN_V28_OFFLINE_ARMS",
        "checkpoint": str(checkpoint_path),
        "train_dataset": str(train_path),
        "test_dataset": str(test_path),
        "critic_input_state_dim": int(actor_state_dim + 1),
        "action_dim": int(action_dim),
        "hidden_dim": HIDDEN_DIM,
        "init_seeds": list(INIT_SEEDS),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_updates": UPDATES,
        "gamma": GAMMA,
        "tau": TAU,
        "gradient_clip": GRAD_CLIP,
        "target_smoothing": False,
        "horizon": HORIZON,
        "checkpoint_rule": "final_update_only",
        "shared_minibatch_indices_within_init_seed": True,
    }
    with (output_dir / "hyperparams_frozen.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(frozen_hyperparams, handle, ensure_ascii=False, indent=2)

    td_critics = []
    return_critics = []
    curve_rows = []
    for init_seed in INIT_SEEDS:
        print(f"[init {init_seed}] paired offline training start", flush=True)
        td_critic, return_critic, rows = train_paired_arms(
            train, actor, action_dim, init_seed, device
        )
        td_critics.append(td_critic)
        return_critics.append(return_critic)
        curve_rows.extend(rows)
        save_checkpoint(
            output_dir / "td_arm" / f"critic_seed_{init_seed}.pt",
            td_critic,
            init_seed,
            "td_fqe",
        )
        save_checkpoint(
            output_dir
            / "truncated_return_arm"
            / f"critic_seed_{init_seed}.pt",
            return_critic,
            init_seed,
            "truncated_return",
        )
        print(
            f"[init {init_seed}] done: td_loss={rows[-1]['td_loss']:.8f}, "
            f"return_loss={rows[-1]['truncated_return_loss']:.8f}",
            flush=True,
        )

    write_csv(output_dir / "offline_training_curve.csv", curve_rows)
    prediction_rows = []
    for split, payload in (("train", train_payload), ("test", test_payload)):
        queries = initial_fork_queries(payload)
        td_delta = predict_fork_deltas(td_critics, queries, device)
        return_delta = predict_fork_deltas(return_critics, queries, device)
        prediction_rows.extend(
            build_prediction_rows(split, queries, td_delta, return_delta)
        )
    episodes = episode_rows(prediction_rows)
    ranking = summarize_predictions(episodes)

    v28a_summary_path = output_dir / "v28a_target_horizon" / "summary.json"
    v28a_summary = json.loads(v28a_summary_path.read_text(encoding="utf-8"))
    horizon_gate = bool(v28a_summary["horizon_stability_gate_pass"])
    expected_train = 40 * 2 * 6 * 2 * HORIZON
    expected_test = 20 * 2 * 6 * 2 * 60
    crn_gate = bool(
        train_payload["reward"].shape[0] == expected_train
        and test_payload["reward"].shape[0] == expected_test
        and validate_crn_pairs(train_payload)
        and validate_crn_pairs(test_payload, horizon=60)
    )
    finite_loss_gate = bool(
        all(
            np.isfinite(float(row[field]))
            for row in curve_rows
            for field in ("td_loss", "truncated_return_loss")
        )
    )
    gates, verdict = apply_gates_and_verdict(
        ranking,
        horizon_gate=horizon_gate,
        crn_gate=crn_gate,
        finite_loss_gate=finite_loss_gate,
    )
    result = {
        "protocol": "v28 FROZEN",
        "ranking": ranking,
        "gates": gates,
        "verdict": verdict,
    }
    write_csv(output_dir / "per_state_critic_predictions.csv", prediction_rows)
    write_csv(output_dir / "episode_critic_predictions.csv", episodes)
    with (output_dir / "ranking_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    write_report(output_dir / "ranking_report.md", result)
    (output_dir / "verdict.md").write_text(
        "# v28 verdict\n\n"
        f"Final pre-registered verdict: **{verdict}**.\n\n"
        f"Pipeline gates: `{json.dumps(gates, ensure_ascii=False)}`\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
