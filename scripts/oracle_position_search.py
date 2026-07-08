import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = Path(bootstrap())

from compare_common import make_env_from_args


DEFAULT_SCENARIO = {
    "num_agents": 3,
    "num_users": 39,
    "traffic_profile": "normal",
    "traffic_scale": 1.0,
    "load_scale": 1.0,
    "arrival_dt": 0.1,
    "uplink_bandwidth_mhz": 20.0,
    "u2u_bandwidth_mhz": 3.0,
    "cpu_ghz": 8.0,
    "cpu_alloc_mode": "priority_kkt",
    "kkt_v": 1.0,
    "fixed_cpu_util": 0.5,
    "micro_diag_bias": 0.5,
    "micro_action_scale": 1.0,
    "micro_state_mode": "pressure",
    "macro_steps": 30,
    "micro_steps": 100,
    "eval_episodes": 2,
    "eval_base_seed": 1000,
    "seed": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search UAV positions that minimize a communication-only proxy: "
            "uplink batch time plus U2U forwarding batch time."
        ),
    )
    parser.add_argument(
        "--result_dir",
        default="",
        help="Result directory containing meta/scenario.json and eval/trajectory files.",
    )
    parser.add_argument(
        "--trajectory_npz",
        default="",
        help="Optional trajectory npz. Defaults to result_dir/trajectory_snapshots/trajectory_final.npz.",
    )
    parser.add_argument(
        "--detail_npz",
        default="",
        help="Optional eval detail npz. Defaults to result_dir/eval_sac_detail.npz.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory. Defaults to result_dir/oracle_positions or results/oracle_positions.",
    )
    parser.add_argument(
        "--flow_mode",
        choices=["learned_split", "local_only", "uniform_offload"],
        default="learned_split",
        help=(
            "learned_split uses the learned source-target split from eval detail, "
            "renormalized to expected zone load."
        ),
    )
    parser.add_argument(
        "--random_starts",
        type=int,
        default=256,
        help="Number of random multi-start positions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for oracle search starts.",
    )
    parser.add_argument(
        "--min_separation_m",
        type=float,
        default=20.0,
        help="Minimum UAV-UAV separation used by the oracle search.",
    )
    parser.add_argument(
        "--map_min",
        type=float,
        default=0.0,
        help="Minimum x/y coordinate.",
    )
    parser.add_argument(
        "--map_max",
        type=float,
        default=1000.0,
        help="Maximum x/y coordinate.",
    )
    parser.add_argument(
        "--uplink_weight",
        type=float,
        default=1.0,
        help="Weight on uplink batch-time proxy.",
    )
    parser.add_argument(
        "--u2u_weight",
        type=float,
        default=1.0,
        help="Weight on U2U forwarding batch-time proxy.",
    )
    return parser.parse_args()


def resolve_result_path(path_text):
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        direct = PROJECT_ROOT / path
        if direct.exists():
            return direct
        path = PROJECT_ROOT / "results" / path
    return path


def load_scenario(result_dir):
    scenario = dict(DEFAULT_SCENARIO)
    if result_dir is not None:
        scenario_path = result_dir / "meta" / "scenario.json"
        if scenario_path.exists():
            with open(scenario_path, "r", encoding="utf-8") as fp:
                scenario.update(json.load(fp))
    return scenario


def scenario_namespace(scenario):
    merged = dict(DEFAULT_SCENARIO)
    merged.update(scenario)
    return SimpleNamespace(**merged)


def make_env(scenario):
    args = scenario_namespace(scenario)
    env = make_env_from_args(args, record_trajectory=False)
    env.reset(seed=int(getattr(args, "seed", 1)))
    return env


def load_trajectory(result_dir, trajectory_npz):
    path = Path(trajectory_npz) if trajectory_npz else None
    if path is not None and not path.is_absolute():
        path = PROJECT_ROOT / path
    if path is None and result_dir is not None:
        path = result_dir / "trajectory_snapshots" / "trajectory_final.npz"
    if path is None or not path.exists():
        return None, None
    return path, np.load(path)


def load_detail(result_dir, detail_npz):
    path = Path(detail_npz) if detail_npz else None
    if path is not None and not path.is_absolute():
        path = PROJECT_ROOT / path
    if path is None and result_dir is not None:
        path = result_dir / "eval_sac_detail.npz"
    if path is None or not path.exists():
        return None, None
    return path, np.load(path)


def apply_trajectory_state(env, trajectory):
    if trajectory is None:
        return
    base_env = env.env
    if "user_allocations" in trajectory:
        base_env.user_allocations = np.asarray(
            trajectory["user_allocations"],
            dtype=np.int32,
        )
    if "anchor_points" in trajectory:
        base_env.anchor_points = np.asarray(trajectory["anchor_points"], dtype=np.float64)
    if "users_traj" in trajectory:
        base_env.user_coords = np.asarray(trajectory["users_traj"][-1], dtype=np.float64)
    if "centroid_traj" in trajectory:
        base_env.weighted_centroids = np.asarray(
            trajectory["centroid_traj"][-1],
            dtype=np.float64,
        )
    if "uav_traj" in trajectory:
        base_env.uav_coords = np.asarray(trajectory["uav_traj"][-1], dtype=np.float64)


def zone_load_mbps(env):
    base_env = env.env
    loads = np.asarray(base_env.zone_expected_offered_load_mbps, dtype=np.float64)
    if np.any(loads <= 0.0):
        loads = np.zeros(base_env.N, dtype=np.float64)
        for i in range(base_env.N):
            mask = base_env.user_allocations == i
            loads[i] = float(
                np.sum(
                    base_env.user_lambda_high[mask]
                    * base_env.user_task_size_high_mbits[mask]
                    + base_env.user_lambda_low[mask]
                    * base_env.user_task_size_low_mbits[mask]
                )
            )
    return loads


def load_flow_matrix(env, detail, scenario, flow_mode):
    loads = zone_load_mbps(env)
    n = env.N

    if flow_mode == "local_only":
        return np.diag(loads)

    if flow_mode == "uniform_offload":
        return np.repeat((loads / n)[:, None], n, axis=1)

    if detail is None or "coop_flow_mbits" not in detail:
        return np.diag(loads)

    raw_flow = np.asarray(detail["coop_flow_mbits"][-1], dtype=np.float64)
    row_sum = np.sum(raw_flow, axis=1, keepdims=True)
    split = np.divide(
        raw_flow,
        np.maximum(row_sum, 1e-12),
        out=np.eye(n, dtype=np.float64),
        where=row_sum > 1e-12,
    )
    return split * loads[:, None]


def uplink_rates_mbps(env, positions):
    base_env = env.env
    positions = np.asarray(positions, dtype=np.float64).reshape(base_env.N, 2)
    assigned = base_env.user_allocations
    offsets = base_env.user_coords - positions[assigned]
    dist_sq = np.sum(offsets * offsets, axis=1) + float(base_env.H) ** 2
    channel_gain = float(base_env.beta_0) / np.maximum(dist_sq, 1e-8)
    user_bandwidth_hz = float(base_env.B_gt_total) / max(base_env.M, 1)
    user_rates = user_bandwidth_hz * np.log2(
        1.0 + (float(base_env.p_gt) * channel_gain) / float(base_env.noise_power)
    )
    rates = np.zeros(base_env.N, dtype=np.float64)
    for i in range(base_env.N):
        rates[i] = np.sum(user_rates[assigned == i]) / 1e6
    return rates


def u2u_rates_mbps(env, positions, flow_mbps):
    base_env = env.env
    positions = np.asarray(positions, dtype=np.float64).reshape(base_env.N, 2)
    flow = np.asarray(flow_mbps, dtype=np.float64)
    active = flow > 1e-9
    np.fill_diagonal(active, False)
    active_count = int(np.count_nonzero(active))
    rates = np.zeros((base_env.N, base_env.N), dtype=np.float64)
    if active_count <= 0:
        return rates

    diff = positions[:, None, :] - positions[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    min_sep_sq = 1e-6
    channel_gain = float(base_env.beta_0) / np.maximum(dist_sq, min_sep_sq)
    np.fill_diagonal(channel_gain, 0.0)
    spectral_eff = np.log2(
        1.0 + (float(base_env.p_u2u) * channel_gain) / float(base_env.noise_power)
    )
    bandwidth_hz = float(base_env.B_u2u_total) / active_count
    rates[active] = bandwidth_hz * spectral_eff[active] / 1e6
    return rates


def separation_violation(positions, min_separation_m):
    positions = np.asarray(positions, dtype=np.float64)
    violation = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            violation += max(float(min_separation_m) - dist, 0.0)
    return violation


def communication_metrics(
    env,
    positions,
    flow_mbps,
    uplink_weight=1.0,
    u2u_weight=1.0,
    min_separation_m=20.0,
):
    loads = zone_load_mbps(env)
    uplink_rates = uplink_rates_mbps(env, positions)
    uplink_time = np.divide(
        loads,
        np.maximum(uplink_rates, 1e-12),
        out=np.full_like(loads, np.inf),
    )

    u2u_rates = u2u_rates_mbps(env, positions, flow_mbps)
    offdiag = np.ones_like(flow_mbps, dtype=bool)
    np.fill_diagonal(offdiag, False)
    u2u_time_matrix = np.zeros_like(flow_mbps, dtype=np.float64)
    active_offdiag = offdiag & (flow_mbps > 1e-9)
    u2u_time_matrix[active_offdiag] = flow_mbps[active_offdiag] / np.maximum(
        u2u_rates[active_offdiag],
        1e-12,
    )
    separation_penalty = separation_violation(positions, min_separation_m) * 1e3
    total = (
        float(uplink_weight) * float(np.sum(uplink_time))
        + float(u2u_weight) * float(np.sum(u2u_time_matrix))
        + separation_penalty
    )
    return {
        "objective": total,
        "uplink_time": float(np.sum(uplink_time)),
        "u2u_time": float(np.sum(u2u_time_matrix)),
        "separation_penalty": float(separation_penalty),
        "uplink_rates_mbps": uplink_rates,
        "u2u_rates_mbps": u2u_rates,
        "uplink_time_per_zone": uplink_time,
        "u2u_time_matrix": u2u_time_matrix,
    }


def local_refine(
    env,
    start_positions,
    flow_mbps,
    args,
    initial_step=120.0,
):
    positions = np.asarray(start_positions, dtype=np.float64).copy()
    positions = np.clip(positions, args.map_min, args.map_max)

    def score(pos):
        return communication_metrics(
            env,
            pos,
            flow_mbps,
            uplink_weight=args.uplink_weight,
            u2u_weight=args.u2u_weight,
            min_separation_m=args.min_separation_m,
        )["objective"]

    best_score = score(positions)
    step = float(initial_step)
    while step >= 1.0:
        improved = False
        for i in range(positions.shape[0]):
            for axis in range(2):
                for direction in (-1.0, 1.0):
                    candidate = positions.copy()
                    candidate[i, axis] += direction * step
                    candidate = np.clip(candidate, args.map_min, args.map_max)
                    candidate_score = score(candidate)
                    if candidate_score + 1e-9 < best_score:
                        positions = candidate
                        best_score = candidate_score
                        improved = True
        if not improved:
            step *= 0.5
    metrics = communication_metrics(
        env,
        positions,
        flow_mbps,
        uplink_weight=args.uplink_weight,
        u2u_weight=args.u2u_weight,
        min_separation_m=args.min_separation_m,
    )
    return positions, metrics


def search_oracle_positions(env, flow_mbps, args, rl_positions=None):
    base_env = env.env
    rng = np.random.default_rng(args.seed)
    starts = []
    starts.append(base_env.anchor_points.copy())
    starts.append(base_env.weighted_centroids.copy())
    starts.append(base_env.uav_initial_positions.copy())
    if rl_positions is not None:
        starts.append(np.asarray(rl_positions, dtype=np.float64).copy())

    lo = float(args.map_min)
    hi = float(args.map_max)
    for _ in range(max(int(args.random_starts), 0)):
        starts.append(rng.uniform(lo, hi, size=(base_env.N, 2)))

    best_positions = None
    best_metrics = None
    for start in starts:
        positions, metrics = local_refine(env, start, flow_mbps, args)
        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_positions = positions
            best_metrics = metrics
    return best_positions, best_metrics


def candidate_positions(env, rl_positions):
    base_env = env.env
    candidates = {
        "anchors": np.asarray(base_env.anchor_points, dtype=np.float64),
        "centroids": np.asarray(base_env.weighted_centroids, dtype=np.float64),
        "initial": np.asarray(base_env.uav_initial_positions, dtype=np.float64),
    }
    if rl_positions is not None:
        candidates["rl_final"] = np.asarray(rl_positions, dtype=np.float64)
    return candidates


def to_serializable_metrics(metrics):
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        else:
            payload[key] = float(value)
    return payload


def write_summary(output_dir, env, flow_mbps, evaluations, scenario, sources):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "oracle_position_summary.csv"
    fields = [
        "name",
        "objective",
        "uplink_time",
        "u2u_time",
        "separation_penalty",
        "positions",
        "uplink_rates_mbps",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        fp.write(",".join(fields) + "\n")
        for name, item in evaluations.items():
            positions = np.asarray(item["positions"], dtype=np.float64)
            metrics = item["metrics"]
            row = {
                "name": name,
                "objective": metrics["objective"],
                "uplink_time": metrics["uplink_time"],
                "u2u_time": metrics["u2u_time"],
                "separation_penalty": metrics["separation_penalty"],
                "positions": json.dumps(np.round(positions, 3).tolist()),
                "uplink_rates_mbps": json.dumps(
                    np.round(metrics["uplink_rates_mbps"], 6).tolist()
                ),
            }
            fp.write(",".join(str(row[field]) for field in fields) + "\n")

    json_path = output_dir / "oracle_position_summary.json"
    payload = {
        "scenario": scenario,
        "sources": sources,
        "zone_load_mbps": zone_load_mbps(env).tolist(),
        "flow_mbps": flow_mbps.tolist(),
        "evaluations": {
            name: {
                "positions": np.asarray(item["positions"], dtype=np.float64).tolist(),
                "metrics": to_serializable_metrics(item["metrics"]),
            }
            for name, item in evaluations.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return csv_path, json_path


def plot_positions(output_dir, env, flow_mbps, evaluations):
    base_env = env.env
    fig, axis = plt.subplots(figsize=(9.0, 8.0), dpi=150)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for zone_idx in range(base_env.N):
        mask = base_env.user_allocations == zone_idx
        axis.scatter(
            base_env.user_coords[mask, 0],
            base_env.user_coords[mask, 1],
            s=18,
            alpha=0.45,
            color=colors[zone_idx % len(colors)],
            label=f"Users Z{zone_idx + 1}",
        )
        axis.scatter(
            base_env.weighted_centroids[zone_idx, 0],
            base_env.weighted_centroids[zone_idx, 1],
            marker="x",
            s=90,
            color=colors[zone_idx % len(colors)],
            linewidth=2.0,
        )

    markers = {
        "anchors": "s",
        "centroids": "X",
        "initial": "^",
        "rl_final": "*",
        "oracle": "P",
    }
    for name, item in evaluations.items():
        positions = np.asarray(item["positions"], dtype=np.float64)
        marker = markers.get(name, "o")
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            marker=marker,
            s=120 if name != "oracle" else 150,
            linewidth=1.2,
            edgecolor="black",
            label=name,
        )
        for i, xy in enumerate(positions):
            axis.text(xy[0] + 5.0, xy[1] + 5.0, f"{name[:2]}{i + 1}", fontsize=7)

    if "oracle" in evaluations:
        positions = np.asarray(evaluations["oracle"]["positions"], dtype=np.float64)
        max_flow = np.max(flow_mbps)
        if max_flow > 0:
            for i in range(base_env.N):
                for j in range(base_env.N):
                    if i == j or flow_mbps[i, j] <= 1e-9:
                        continue
                    width = 0.4 + 2.2 * float(flow_mbps[i, j] / max_flow)
                    axis.annotate(
                        "",
                        xy=positions[j],
                        xytext=positions[i],
                        arrowprops={
                            "arrowstyle": "->",
                            "linewidth": width,
                            "alpha": 0.35,
                            "color": "0.25",
                        },
                    )

    axis.set_xlim(float(np.min(base_env.user_coords[:, 0])) - 120.0, float(np.max(base_env.user_coords[:, 0])) + 160.0)
    axis.set_ylim(float(np.min(base_env.user_coords[:, 1])) - 120.0, float(np.max(base_env.user_coords[:, 1])) + 160.0)
    axis.set_xlabel("X coordinate (m)")
    axis.set_ylabel("Y coordinate (m)")
    axis.set_title("Communication Oracle Positions")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=7, frameon=True)
    fig.tight_layout()
    path = output_dir / "oracle_positions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    result_dir = resolve_result_path(args.result_dir)
    scenario = load_scenario(result_dir)
    env = make_env(scenario)

    trajectory_path, trajectory = load_trajectory(result_dir, args.trajectory_npz)
    detail_path, detail = load_detail(result_dir, args.detail_npz)
    apply_trajectory_state(env, trajectory)

    rl_positions = None
    if trajectory is not None and "uav_traj" in trajectory:
        rl_positions = np.asarray(trajectory["uav_traj"][-1], dtype=np.float64)

    flow_mbps = load_flow_matrix(env, detail, scenario, args.flow_mode)
    oracle_positions, oracle_metrics = search_oracle_positions(
        env,
        flow_mbps,
        args,
        rl_positions=rl_positions,
    )

    evaluations = {}
    for name, positions in candidate_positions(env, rl_positions).items():
        evaluations[name] = {
            "positions": positions,
            "metrics": communication_metrics(
                env,
                positions,
                flow_mbps,
                uplink_weight=args.uplink_weight,
                u2u_weight=args.u2u_weight,
                min_separation_m=args.min_separation_m,
            ),
        }
    evaluations["oracle"] = {
        "positions": oracle_positions,
        "metrics": oracle_metrics,
    }

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir is None:
        output_dir = (
            result_dir / "oracle_positions"
            if result_dir is not None
            else PROJECT_ROOT / "results" / "oracle_positions"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "result_dir": str(result_dir) if result_dir is not None else "",
        "trajectory_npz": str(trajectory_path) if trajectory_path is not None else "",
        "detail_npz": str(detail_path) if detail_path is not None else "",
        "flow_mode": args.flow_mode,
        "min_separation_m": args.min_separation_m,
        "uplink_weight": args.uplink_weight,
        "u2u_weight": args.u2u_weight,
    }
    csv_path, json_path = write_summary(
        output_dir,
        env,
        flow_mbps,
        evaluations,
        scenario,
        sources,
    )
    fig_path = plot_positions(output_dir, env, flow_mbps, evaluations)

    print(f"Saved oracle CSV -> {csv_path}")
    print(f"Saved oracle JSON -> {json_path}")
    print(f"Saved oracle figure -> {fig_path}")
    print("\nCommunication objective summary:")
    for name, item in sorted(
        evaluations.items(),
        key=lambda pair: pair[1]["metrics"]["objective"],
    ):
        metrics = item["metrics"]
        positions = np.asarray(item["positions"], dtype=np.float64)
        print(
            f"  {name:10s} obj={metrics['objective']:.4f} "
            f"uplink={metrics['uplink_time']:.4f} "
            f"u2u={metrics['u2u_time']:.4f} "
            f"positions={np.round(positions, 2).tolist()}"
        )


if __name__ == "__main__":
    main()
