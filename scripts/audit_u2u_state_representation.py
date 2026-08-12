import argparse
import csv
import json
import os

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare U2U rates exposed by the hierarchy state with rates "
            "actually used by dynamic active-link bandwidth allocation."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-base-seed", type=int, default=63000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_episode(env, eval_args, actor, micro_policy, episode_index, seed):
    env.reset(seed=seed)
    rows = []
    done = False
    sample_index = 0
    off_diagonal = ~np.eye(env.N, dtype=bool)
    for macro_index in range(int(eval_args.macro_steps)):
        action = actor_action(actor, env)
        env.macro_step(action)
        for resource_frame_index in range(env.env.resource_frames_per_macro):
            service_paused = bool(
                getattr(env.env, "pause_service_during_flight", False)
                and env.env.macro_service_pause_steps_remaining > 0
            )
            if service_paused:
                micro_action = np.zeros(
                    env.micro_action_space.shape[0],
                    dtype=np.float32,
                )
            else:
                env.prepare_resource_frame()
                micro_action = np.asarray(
                    micro_policy(env, True),
                    dtype=np.float32,
                )
            frame = env.step_resource_frame(micro_action)
            for integration_index, info in enumerate(frame["infos"]):
                if "u2u_rate_bps" not in info:
                    continue
                actual_rate = np.asarray(
                    info["u2u_rate_bps"],
                    dtype=np.float64,
                ) / 1e6
                actual_bandwidth = np.asarray(
                    info["u2u_link_bandwidth_hz"],
                    dtype=np.float64,
                )
                active = off_diagonal & (actual_bandwidth > 0.0)
                active_count = int(np.sum(active))
                nominal_rate = env.env.compute_u2u_rate_bps(
                    uav_coords=env.env.uav_coords,
                    active_link_mask=None,
                ) / 1e6
                nominal_offdiag = nominal_rate[off_diagonal]
                if active_count > 0:
                    actual_active = actual_rate[active]
                    nominal_active = nominal_rate[active]
                    ratio = np.divide(
                        actual_active,
                        nominal_active,
                        out=np.full_like(actual_active, np.nan),
                        where=nominal_active > 1e-12,
                    )
                    actual_mean = float(np.mean(actual_active))
                    nominal_active_mean = float(np.mean(nominal_active))
                    ratio_mean = float(np.nanmean(ratio))
                    bandwidth_active_mean_mhz = float(
                        np.mean(actual_bandwidth[active]) / 1e6
                    )
                else:
                    actual_mean = 0.0
                    nominal_active_mean = 0.0
                    ratio_mean = np.nan
                    bandwidth_active_mean_mhz = 0.0
                rows.append(
                    {
                        "episode_index": episode_index,
                        "seed": seed,
                        "macro_index": macro_index,
                        "resource_frame_index": resource_frame_index,
                        "integration_index": integration_index,
                        "sample_index": sample_index,
                        "active_link_count": active_count,
                        "nominal_bandwidth_per_link_mhz": float(
                            env.env.B_u2u_total / 1e6 / (env.N * (env.N - 1))
                        ),
                        "actual_bandwidth_active_mean_mhz": (
                            bandwidth_active_mean_mhz
                        ),
                        "nominal_rate_offdiag_mean_mbps": float(
                            np.mean(nominal_offdiag)
                        ),
                        "nominal_rate_active_mean_mbps": nominal_active_mean,
                        "actual_rate_active_mean_mbps": actual_mean,
                        "actual_to_nominal_rate_ratio": ratio_mean,
                        "nominal_state_rate_normalized_mean": float(
                            np.mean(nominal_offdiag)
                            / max(float(env.max_u2u_rate_mbps), 1e-8)
                        ),
                        "actual_active_rate_normalized_mean": float(
                            actual_mean
                            / max(float(env.max_u2u_rate_mbps), 1e-8)
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


def summarize(rows, env):
    active_rows = [row for row in rows if row["active_link_count"] > 0]
    active_counts = np.asarray(
        [row["active_link_count"] for row in rows],
        dtype=np.int64,
    )
    histogram = {
        str(count): int(np.sum(active_counts == count))
        for count in range(env.N * (env.N - 1) + 1)
    }
    return {
        "episodes": len({int(row["episode_index"]) for row in rows}),
        "integration_samples": len(rows),
        "active_samples": len(active_rows),
        "dynamic_u2u_bandwidth": bool(env.env.dynamic_u2u_bandwidth),
        "total_u2u_bandwidth_mhz": float(env.env.B_u2u_total / 1e6),
        "state_assumption": (
            "compute_u2u_rate_bps(active_link_mask=None), so all six "
            "directed links share the total bandwidth"
        ),
        "active_link_count_histogram": histogram,
        "active_link_count_mean_when_active": float(
            np.mean([row["active_link_count"] for row in active_rows])
        ),
        "nominal_state_rate_normalized_mean": float(
            np.mean(
                [row["nominal_state_rate_normalized_mean"] for row in rows]
            )
        ),
        "actual_active_rate_normalized_mean": float(
            np.mean(
                [row["actual_active_rate_normalized_mean"] for row in active_rows]
            )
        ),
        "actual_to_nominal_rate_ratio_mean": float(
            np.mean(
                [row["actual_to_nominal_rate_ratio"] for row in active_rows]
            )
        ),
        "actual_to_nominal_rate_ratio_p05_p50_p95": [
            float(value)
            for value in np.percentile(
                [row["actual_to_nominal_rate_ratio"] for row in active_rows],
                [5, 50, 95],
            )
        ],
    }


def draw_figure(rows, output_dir):
    active_rows = [row for row in rows if row["active_link_count"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    counts = np.asarray(
        [row["active_link_count"] for row in rows],
        dtype=np.int64,
    )
    bins = np.arange(-0.5, 7.5, 1.0)
    axes[0].hist(counts, bins=bins, color="#2878B5", edgecolor="white")
    axes[0].set_xticks(range(7))
    axes[0].set_xlabel("Active directed U2U links")
    axes[0].set_ylabel("Integration-step count")
    axes[0].set_title("Dynamic active-link distribution")

    nominal = [row["nominal_rate_active_mean_mbps"] for row in active_rows]
    actual = [row["actual_rate_active_mean_mbps"] for row in active_rows]
    colors = [row["active_link_count"] for row in active_rows]
    scatter = axes[1].scatter(
        nominal,
        actual,
        c=colors,
        cmap="viridis",
        alpha=0.55,
        s=14,
    )
    upper = max(max(nominal), max(actual))
    axes[1].plot([0.0, upper], [0.0, upper], "k--", linewidth=0.8)
    axes[1].set_xlabel("Rate encoded in state (Mbps)")
    axes[1].set_ylabel("Rate used by environment (Mbps)")
    axes[1].set_title("Nominal versus executed active-link rate")
    fig.colorbar(scatter, ax=axes[1], label="Active link count")
    fig.savefig(os.path.join(output_dir, "u2u_state_rate_audit.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "u2u_state_rate_audit.pdf"))
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
    probe_env = build_env(eval_args)
    probe_env.reset(seed=int(args.eval_base_seed))
    actor, _, micro_policy = load_models(
        checkpoint,
        probe_env,
        eval_args,
        device,
    )
    rows = []
    for episode_index in range(int(args.episodes)):
        seed = int(args.eval_base_seed) + episode_index
        rows.extend(
            collect_episode(
                build_env(eval_args),
                eval_args,
                actor,
                micro_policy,
                episode_index,
                seed,
            )
        )
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "u2u_rate_samples.csv"), rows)
    summary = summarize(rows, probe_env)
    with open(
        os.path.join(args.output_dir, "u2u_state_rate_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    draw_figure(rows, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
