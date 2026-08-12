import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from run_v28_offline_critic_arms import HIDDEN_DIM
from run_v28_offline_critic_arms import INIT_SEEDS
from run_v28_offline_critic_arms import TwinCritic
from run_v28_offline_critic_arms import initial_fork_queries
from run_v28_offline_critic_arms import load_npz
from run_v28_offline_critic_arms import predict_fork_deltas
from run_v28_offline_critic_arms import prepare_finite_horizon_dataset


DEFAULT_ROOT = "result_archive/v28_td_vs_mc_counterfactual"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose the frozen v28 offline absolute-return regression."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_critics(root, arm, state_dim, action_dim, device):
    critics = []
    for seed in INIT_SEEDS:
        payload = torch.load(
            root / arm / f"critic_seed_{seed}.pt",
            map_location=device,
            weights_only=False,
        )
        critic = TwinCritic(
            state_dim, action_dim, hidden_dim=HIDDEN_DIM
        ).to(device)
        critic.load_state_dict(payload["critic"])
        critic.eval()
        critics.append(critic)
    return critics


def ensemble_predictions(critics, states, actions, device, batch_size=1024):
    predictions = []
    with torch.no_grad():
        for critic in critics:
            chunks = []
            for start in range(0, states.shape[0], batch_size):
                state_t = torch.as_tensor(
                    states[start : start + batch_size], device=device
                )
                action_t = torch.as_tensor(
                    actions[start : start + batch_size], device=device
                )
                q1, q2 = critic(state_t, action_t)
                chunks.append(torch.minimum(q1, q2).squeeze(1).cpu().numpy())
            predictions.append(np.concatenate(chunks))
    return np.mean(np.stack(predictions), axis=0)


def regression_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    residual = prediction - target
    target_variance = float(np.var(target))
    return {
        "samples": int(target.size),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "prediction_mean": float(np.mean(prediction)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(
            1.0 - np.mean(np.square(residual)) / target_variance
        )
        if target_variance > 0.0
        else float("nan"),
    }


def fork_diagnostics(payload, critics, device):
    queries = initial_fork_queries(payload)
    predictions = predict_fork_deltas(critics, queries, device)
    rows = []
    for policy in ("actor", "centroid"):
        selected = [
            (query, prediction)
            for query, prediction in zip(queries, predictions)
            if query["snapshot_policy"] == policy
        ]
        truth = np.asarray([query["true_delta"] for query, _ in selected])
        predicted = np.asarray([value for _, value in selected])
        rows.append(
            {
                "snapshot_policy": policy,
                "forks": int(truth.size),
                "true_delta_mean": float(np.mean(truth)),
                "true_delta_std": float(np.std(truth)),
                "mean_abs_true_delta": float(np.mean(np.abs(truth))),
                "predicted_delta_mean": float(np.mean(predicted)),
                "prediction_error_rmse": float(
                    np.sqrt(np.mean(np.square(predicted - truth)))
                ),
                "sign_agreement": float(
                    np.mean(np.sign(predicted) == np.sign(truth))
                ),
            }
        )
    return rows


def draw_loss(root, output_dir):
    curve = pd.read_csv(root / "offline_training_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for seed, selected in curve.groupby("init_seed"):
        axes[0].plot(selected["update"], selected["td_loss"], label=str(seed))
        axes[1].plot(
            selected["update"],
            selected["truncated_return_loss"],
            label=str(seed),
        )
    for axis, title in zip(axes, ("TD/FQE", "Truncated-return")):
        axis.set_yscale("log")
        axis.set_xlabel("Gradient update")
        axis.set_ylabel("Twin-head MSE")
        axis.set_title(title)
        axis.legend(frameon=False, title="init")
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "offline_fit_loss.png", dpi=300)
    fig.savefig(output_dir / "offline_fit_loss.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    root = project_path(args.root)
    output_dir = root / "offline_fit_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_payload = load_npz(root / "train_transitions.npz")
    train = prepare_finite_horizon_dataset(train_payload)
    state_dim = int(train["state"].shape[1])
    action_dim = int(train["action"].shape[1])
    direct_critics = load_critics(
        root, "truncated_return_arm", state_dim, action_dim, device
    )
    prediction = ensemble_predictions(
        direct_critics, train["state"], train["action"], device
    )
    target = train["truncated_return"].reshape(-1)
    remaining = train["remaining"].reshape(-1)
    result = {
        "arm": "truncated_return",
        "overall": regression_metrics(target, prediction),
        "by_remaining_horizon": {},
        "initial_fork": fork_diagnostics(train_payload, direct_critics, device),
    }
    for horizon in (1, 5, 15, 30):
        selected = remaining == int(horizon)
        result["by_remaining_horizon"][str(horizon)] = regression_metrics(
            target[selected], prediction[selected]
        )
    for row in result["initial_fork"]:
        row["absolute_rmse_to_delta_std_ratio"] = float(
            result["by_remaining_horizon"]["30"]["rmse"]
            / max(row["true_delta_std"], 1e-12)
        )
    with (output_dir / "fit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    draw_loss(root, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
