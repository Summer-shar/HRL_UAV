import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import torch_load_checkpoint
from diagnose_td3_counterfactual import actor_action
from diagnose_td3_counterfactual import build_env
from diagnose_td3_counterfactual import load_models
import hrl_uav.agents.sac_agent as sac_agent_mod


DEFAULT_CHECKPOINT = Path(
    "results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/"
    "joint_td3_sac/checkpoints/final.pt"
)
DEFAULT_OUTPUT = Path("result_archive/v30_u2u_representation_audit")
DISTANCE_TABLE_M = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-training audit for the proposed U2U SE state block."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-base-seed", type=int, default=65000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def spectral_efficiency(env, coords):
    coords = np.asarray(coords, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    distance_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(distance_sq, np.inf)
    channel_gain = env.beta_0 / np.maximum(distance_sq, 1e-12)
    eta = np.log2(1.0 + (env.p_u2u * channel_gain) / env.noise_power)
    np.fill_diagonal(eta, 0.0)
    return eta


def eta_at_distance(env, distance_m):
    distance_sq = max(float(distance_m) ** 2, 1e-12)
    return float(
        np.log2(
            1.0
            + (env.p_u2u * env.beta_0)
            / (env.noise_power * distance_sq)
        )
    )


def anchor_eta_reference(wrapper):
    eta = spectral_efficiency(wrapper.env, wrapper.env.anchor_points)
    off_diagonal = ~np.eye(wrapper.N, dtype=bool)
    return float(np.median(eta[off_diagonal]))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def collect_episode(env, eval_args, actor, micro_policy, episode, seed, eta_ref):
    env.reset(seed=seed)
    rows = []
    off_diagonal = ~np.eye(env.N, dtype=bool)
    sample_index = 0
    done = False
    for macro_index in range(int(eval_args.macro_steps)):
        env.macro_step(actor_action(actor, env))
        for resource_frame_index in range(env.env.resource_frames_per_macro):
            paused = bool(
                env.env.pause_service_during_flight
                and env.env.macro_service_pause_steps_remaining > 0
            )
            if paused:
                action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
            else:
                env.prepare_resource_frame()
                action = np.asarray(micro_policy(env, True), dtype=np.float32)
            frame = env.step_resource_frame(action)
            for integration_index, info in enumerate(frame["infos"]):
                if "u2u_rate_bps" not in info:
                    continue
                bandwidth = np.asarray(
                    info["u2u_link_bandwidth_hz"], dtype=np.float64
                )
                executed_rate = np.asarray(info["u2u_rate_bps"], dtype=np.float64)
                eta = spectral_efficiency(env.env, env.env.uav_coords)
                reconstructed_rate = bandwidth * eta
                active = off_diagonal & (bandwidth > 0.0)
                coords = np.asarray(env.env.uav_coords, dtype=np.float64)
                distance = np.linalg.norm(
                    coords[:, None, :] - coords[None, :, :], axis=-1
                )
                old_feature = (
                    (env.env.B_u2u_total / (env.N * (env.N - 1)))
                    * eta
                    / 1e6
                    / env.max_u2u_rate_mbps
                )
                new_feature = eta / eta_ref
                if np.any(active):
                    absolute_error = np.abs(
                        reconstructed_rate[active] - executed_rate[active]
                    )
                    relative_error = absolute_error / np.maximum(
                        np.abs(executed_rate[active]), 1e-12
                    )
                else:
                    absolute_error = np.asarray([0.0])
                    relative_error = np.asarray([0.0])
                for i, j in zip(*np.where(off_diagonal)):
                    rows.append(
                        {
                            "episode": episode,
                            "seed": seed,
                            "macro_index": macro_index,
                            "resource_frame_index": resource_frame_index,
                            "integration_index": integration_index,
                            "sample_index": sample_index,
                            "source_uav": int(i),
                            "target_uav": int(j),
                            "distance_m": float(distance[i, j]),
                            "spectral_efficiency_bps_hz": float(eta[i, j]),
                            "old_nominal_rate_feature": float(old_feature[i, j]),
                            "new_anchor_normalized_eta": float(new_feature[i, j]),
                            "active_link_count": int(np.count_nonzero(active)),
                            "active": int(active[i, j]),
                            "actual_bandwidth_hz": float(bandwidth[i, j]),
                            "executed_rate_bps": float(executed_rate[i, j]),
                            "reconstructed_rate_bps": float(
                                reconstructed_rate[i, j]
                            ),
                            "sample_active_max_abs_error_bps": float(
                                np.max(absolute_error)
                            ),
                            "sample_active_max_rel_error": float(
                                np.max(relative_error)
                            ),
                        }
                    )
                sample_index += 1
            done = done or bool(frame["done"])
            if done:
                break
        env.get_macro_reward()
        if done:
            break
    return rows


def distance_table(env, eta_ref):
    rows = []
    old_scale = (
        env.B_u2u_total
        / (env.N * (env.N - 1))
        / 1e6
        / 60.0
    )
    for distance in DISTANCE_TABLE_M:
        eta = eta_at_distance(env, distance)
        eta_toward = eta_at_distance(env, max(distance - 20.0, 1e-6))
        rows.append(
            {
                "distance_m": distance,
                "eta_bps_hz": eta,
                "old_feature": old_scale * eta,
                "new_feature": eta / eta_ref,
                "old_feature_gain_toward_20m": old_scale * (eta_toward - eta),
                "new_feature_gain_toward_20m": (eta_toward - eta) / eta_ref,
            }
        )
    return rows


def summarize(rows, wrapper, eta_ref, table):
    distance = np.asarray([row["distance_m"] for row in rows], dtype=np.float64)
    eta = np.asarray(
        [row["spectral_efficiency_bps_hz"] for row in rows], dtype=np.float64
    )
    old_feature = np.asarray(
        [row["old_nominal_rate_feature"] for row in rows], dtype=np.float64
    )
    new_feature = np.asarray(
        [row["new_anchor_normalized_eta"] for row in rows], dtype=np.float64
    )
    active_rows = [row for row in rows if row["active"]]
    executed = np.asarray(
        [row["executed_rate_bps"] for row in active_rows], dtype=np.float64
    )
    reconstructed = np.asarray(
        [row["reconstructed_rate_bps"] for row in active_rows], dtype=np.float64
    )
    max_abs = float(
        np.max([row["sample_active_max_abs_error_bps"] for row in rows])
    )
    max_rel = float(
        np.max([row["sample_active_max_rel_error"] for row in rows])
    )
    correlation = float(np.corrcoef(reconstructed, executed)[0, 1])
    old_scale_denominator_eta = float(
        wrapper.max_u2u_rate_mbps
        / (wrapper.env.B_u2u_total / (wrapper.N * (wrapper.N - 1)) / 1e6)
    )
    return {
        "status": "ZERO_TRAINING_AUDIT",
        "episodes": len({row["episode"] for row in rows}),
        "link_samples": len(rows),
        "active_link_samples": len(active_rows),
        "channel_model": "beta_0 / d^2 with no additional fading term",
        "anchor_eta_reference_bps_hz": eta_ref,
        "normalization": "eta / median off-diagonal eta at initial anchors",
        "old_effective_eta_denominator": old_scale_denominator_eta,
        "feature_sensitivity_multiplier_new_over_old": (
            old_scale_denominator_eta / eta_ref
        ),
        "distance_m_p05_p50_p95": [
            float(value) for value in np.percentile(distance, [5, 50, 95])
        ],
        "eta_p05_p50_p95": [
            float(value) for value in np.percentile(eta, [5, 50, 95])
        ],
        "old_feature_p05_p50_p95": [
            float(value) for value in np.percentile(old_feature, [5, 50, 95])
        ],
        "new_feature_p05_p50_p95": [
            float(value) for value in np.percentile(new_feature, [5, 50, 95])
        ],
        "new_feature_max": float(np.max(new_feature)),
        "new_feature_fraction_above_3": float(np.mean(new_feature > 3.0)),
        "rate_reconstruction": {
            "pearson_r": correlation,
            "max_absolute_error_bps": max_abs,
            "max_relative_error": max_rel,
        },
        "distance_response_table": table,
        "validity_gates": {
            "formula_equivalent": bool(max_rel <= 1e-12),
            "correlation_at_least_0_999999": bool(correlation >= 0.999999),
            "no_feature_above_3": bool(np.max(new_feature) <= 3.0),
            "sensitivity_multiplier_descriptive": (
                old_scale_denominator_eta / eta_ref
            ),
        },
    }


def draw(rows, table, output_dir):
    distance = np.asarray([row["distance_m"] for row in rows], dtype=np.float64)
    eta = np.asarray(
        [row["spectral_efficiency_bps_hz"] for row in rows], dtype=np.float64
    )
    old = np.asarray(
        [row["old_nominal_rate_feature"] for row in rows], dtype=np.float64
    )
    new = np.asarray(
        [row["new_anchor_normalized_eta"] for row in rows], dtype=np.float64
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), constrained_layout=True)
    axes[0].scatter(distance, eta, s=8, alpha=0.15, color="#2F6690")
    axes[0].set_xlabel("UAV separation (m)")
    axes[0].set_ylabel("Spectral efficiency (bit/s/Hz)")
    axes[0].set_title("Action-independent channel quality")

    axes[1].hist(old, bins=30, alpha=0.75, label="old rate / 60 Mbps")
    axes[1].hist(new, bins=30, alpha=0.65, label="new eta / anchor eta")
    axes[1].set_xlabel("Encoded feature value")
    axes[1].set_ylabel("Link-sample count")
    axes[1].set_title("State dynamic range")
    axes[1].legend(frameon=False)

    d = [row["distance_m"] for row in table]
    old_delta = [row["old_feature_gain_toward_20m"] for row in table]
    new_delta = [row["new_feature_gain_toward_20m"] for row in table]
    axes[2].plot(d, old_delta, "o-", label="old")
    axes[2].plot(d, new_delta, "o-", label="new")
    axes[2].set_xlabel("Starting separation (m)")
    axes[2].set_ylabel("Feature increase after moving 20 m closer")
    axes[2].set_title("Twenty-meter response")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.savefig(output_dir / "v30_u2u_representation_audit.png", dpi=300)
    fig.savefig(output_dir / "v30_u2u_representation_audit.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    sac_agent_mod.device = device
    checkpoint = torch_load_checkpoint(os.path.abspath(args.checkpoint))
    eval_args = argparse.Namespace(**checkpoint["args"])
    eval_args.device = str(device)
    reference_env = build_env(eval_args)
    eta_ref = anchor_eta_reference(reference_env)
    actor, _, micro_policy = load_models(
        checkpoint, reference_env, eval_args, device
    )
    rows = []
    for episode in range(args.episodes):
        rows.extend(
            collect_episode(
                build_env(eval_args),
                eval_args,
                actor,
                micro_policy,
                episode,
                args.eval_base_seed + episode,
                eta_ref,
            )
        )
    table = distance_table(reference_env.env, eta_ref)
    summary = summarize(rows, reference_env, eta_ref, table)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "link_samples.csv", rows)
    write_csv(output_dir / "distance_response.csv", table)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    draw(rows, table, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
