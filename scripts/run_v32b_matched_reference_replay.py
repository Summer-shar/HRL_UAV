import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import make_episode_load_switch_schedule
from run_load_switch_gate import RAW_FIELDNAMES
from run_load_switch_gate import paired_summary
from run_load_switch_gate import run_gate_episode
from run_load_switch_gate import save_macro_trace
from run_load_switch_gate import write_csv_atomic
from search_position_oracle import load_checkpoint_components
from train_centralized_td3 import td3_macro_policy


POLICIES = ("TD3", "P0", "P1_prime")
POWER_BUDGET_W = 840.0
PRACTICAL_GAIN_MS = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay v32b TD3, P0, and frozen P1-prime with one matched "
            "checkpoint, multiplier, micro policy, and CRN seed set."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--selection",
        default="result_archive/p1_prime_prescreen/selection.json",
    )
    parser.add_argument(
        "--output-dir",
        default="result_archive/v32b_matched_reference/seed1",
    )
    parser.add_argument("--eval-base-seed", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(values[0])
    return float(
        np.mean(values)
        + stats.t.ppf(0.95, values.size - 1) * stats.sem(values)
    )


def save_trace_npz(path, trace):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        uav_trajectory_m=trace["uav_trajectory_m"],
        centroid_trajectory_m=trace["centroid_trajectory_m"],
        trajectory_phases=trace["trajectory_phases"],
        phase_flow_mbits=trace["phase_flow_mbits"],
        total_flow_mbits=trace["total_flow_mbits"],
    )


def rows_by_policy(raw_rows, policy):
    return [row for row in raw_rows if row["policy"] == policy]


def paired_delta_rows(raw_rows, eval_seeds):
    by_policy_seed = {
        (str(row["policy"]), int(row["seed"])): row for row in raw_rows
    }
    paired = []
    for seed in eval_seeds:
        td3 = by_policy_seed[("TD3", seed)]
        p0 = by_policy_seed[("P0", seed)]
        prime = by_policy_seed[("P1_prime", seed)]
        hashes = {
            td3["external_trace_hash"],
            p0["external_trace_hash"],
            prime["external_trace_hash"],
        }
        if len(hashes) != 1:
            raise RuntimeError(f"CRN trace mismatch for seed {seed}")
        paired.append(
            {
                "seed": seed,
                "delay_td3_ms": td3["task_delay_weighted_ms"],
                "delay_p0_ms": p0["task_delay_weighted_ms"],
                "delay_p1_prime_ms": prime["task_delay_weighted_ms"],
                "delay_td3_minus_p0_ms": (
                    td3["task_delay_weighted_ms"]
                    - p0["task_delay_weighted_ms"]
                ),
                "delay_p1_prime_minus_p0_ms": (
                    prime["task_delay_weighted_ms"]
                    - p0["task_delay_weighted_ms"]
                ),
                "power_td3_w": td3["average_power_w"],
                "power_p0_w": p0["average_power_w"],
                "power_p1_prime_w": prime["average_power_w"],
                "power_td3_minus_p0_w": (
                    td3["average_power_w"] - p0["average_power_w"]
                ),
                "power_p1_prime_minus_p0_w": (
                    prime["average_power_w"] - p0["average_power_w"]
                ),
                "external_trace_hash": td3["external_trace_hash"],
            }
        )
    return paired


def summary_without_values(candidate, reference, field):
    summary = paired_summary(candidate, reference, field)
    return {key: value for key, value in summary.items() if key != "values"}


def draw_summary(paired, policy_rows, summary, output_dir):
    x = np.arange(len(paired))
    td3_delta = np.asarray(
        [row["delay_td3_minus_p0_ms"] for row in paired],
        dtype=np.float64,
    )
    prime_delta = np.asarray(
        [row["delay_p1_prime_minus_p0_ms"] for row in paired],
        dtype=np.float64,
    )
    colors = {"TD3": "#2878B5", "P0": "#777777", "P1_prime": "#E17C05"}

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    axes[0].plot(x, td3_delta, "o-", color=colors["TD3"], label="TD3 - P0")
    axes[0].plot(
        x,
        prime_delta,
        "s-",
        color=colors["P1_prime"],
        label="P1-prime - P0",
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("CRN evaluation seed index")
    axes[0].set_ylabel("Paired weighted delay difference (ms)")
    axes[0].set_title("Positioning benefit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.35)

    for policy in POLICIES:
        values = np.asarray(
            [row["average_power_w"] for row in policy_rows[policy]],
            dtype=np.float64,
        )
        axes[1].scatter(x, values, s=24, color=colors[policy], label=policy)
        axes[1].axhline(
            summary["policies"][policy]["power_ucb95_w"],
            color=colors[policy],
            linestyle=":",
            linewidth=1.0,
        )
    axes[1].axhline(POWER_BUDGET_W, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("CRN evaluation seed index")
    axes[1].set_ylabel("System-average power (W)")
    axes[1].set_title("Matched power feasibility")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.35)

    positions = np.arange(len(POLICIES))
    delay_means = [summary["policies"][name]["delay_mean_ms"] for name in POLICIES]
    axes[2].bar(
        positions,
        delay_means,
        color=[colors[name] for name in POLICIES],
        width=0.62,
    )
    for policy_index, policy in enumerate(POLICIES):
        values = np.asarray(
            [row["task_delay_weighted_ms"] for row in policy_rows[policy]],
            dtype=np.float64,
        )
        axes[2].scatter(
            np.full(values.size, policy_index),
            values,
            facecolors="white",
            edgecolors=colors[policy],
            linewidths=0.8,
            s=20,
            zorder=3,
        )
    axes[2].set_xticks(positions, ("TD3", "P0", "P1-prime"))
    axes[2].set_ylabel("Weighted end-to-end delay (ms)")
    axes[2].set_title("Absolute matched performance")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.savefig(output_dir / "matched_reference_summary.png", dpi=300)
    fig.savefig(output_dir / "matched_reference_summary.pdf")
    plt.close(fig)


def draw_trajectories(traces, seed, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    colors = ("#2878B5", "#E17C05", "#54A24B")
    titles = {
        "TD3": "TD3-SAC",
        "P0": "P0 centroid tracking",
        "P1_prime": "P1-prime causal support",
    }
    for axis, policy in zip(axes, POLICIES):
        trace = traces[(policy, seed)]
        uav = np.asarray(trace["uav_trajectory_m"], dtype=np.float64)
        centroid = np.asarray(trace["centroid_trajectory_m"], dtype=np.float64)
        for index, color in enumerate(colors):
            axis.plot(
                centroid[:, index, 0],
                centroid[:, index, 1],
                color=color,
                linestyle=":",
                linewidth=1.2,
            )
            axis.plot(
                uav[:, index, 0],
                uav[:, index, 1],
                color=color,
                linewidth=2.0,
                label=f"UAV {index + 1}",
            )
            axis.scatter(
                uav[0, index, 0],
                uav[0, index, 1],
                marker="s",
                color=color,
                s=30,
            )
            axis.scatter(
                uav[-1, index, 0],
                uav[-1, index, 1],
                marker="*",
                color=color,
                s=85,
            )
        axis.set_title(titles[policy])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle=":", alpha=0.35)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Matched load-switch trajectories (evaluation seed {seed})")
    fig.savefig(output_dir / "matched_reference_trajectories.png", dpi=300)
    fig.savefig(output_dir / "matched_reference_trajectories.pdf")
    plt.close(fig)


def render_verdict(summary):
    td3_delta = summary["paired_effects"]["td3_minus_p0_delay"]
    prime_delta = summary["paired_effects"]["p1_prime_minus_p0_delay"]
    return "\n".join(
        (
            "# v32b matched-reference replay: training seed "
            f"{summary['training_seed']}",
            "",
            f"- Selected checkpoint: `{summary['checkpoint']}`",
            f"- Frozen system multiplier: `{summary['system_lambda']:.9f}`",
            f"- TD3 power UCB95: `{summary['policies']['TD3']['power_ucb95_w']:.3f} W` "
            f"(feasible: `{summary['policies']['TD3']['power_feasible']}`)",
            f"- P1-prime power UCB95: `{summary['policies']['P1_prime']['power_ucb95_w']:.3f} W` "
            f"(feasible: `{summary['policies']['P1_prime']['power_feasible']}`)",
            "- TD3 minus P0 delay: "
            f"`{td3_delta['mean']:.3f} ms`, 95% CI "
            f"`[{td3_delta['ci95_low']:.3f}, {td3_delta['ci95_high']:.3f}] ms`",
            "- P1-prime minus P0 delay: "
            f"`{prime_delta['mean']:.3f} ms`, 95% CI "
            f"`[{prime_delta['ci95_low']:.3f}, {prime_delta['ci95_high']:.3f}] ms`",
            f"- Descriptive capture fraction: `{summary['capture_fraction']:.6f}`",
            f"- Seed-level TD3 direction beneficial: `{summary['td3_direction_beneficial']}`",
            "",
            "P1-prime participates in feasible-policy comparison only if its matched "
            "power UCB95 is at most 840 W.",
            "",
        )
    )


def main():
    args = parse_args()
    checkpoint = project_path(args.checkpoint).resolve()
    selection_path = project_path(args.selection).resolve()
    output_dir = project_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "FROZEN_FOR_SINGLE_REPLAY":
        raise RuntimeError("P1-prime selection is not frozen for replay.")
    offset_m = float(selection["selected"]["offset_m"])
    residual_cap_m = float(
        selection["selected"]["residual_cap_per_macro_m"]
    )

    (
        checkpoint_path,
        checkpoint_args,
        env,
        actor,
        micro_policy,
        _device,
    ) = load_checkpoint_components(checkpoint, args.device)
    if str(getattr(checkpoint_args, "load_switch_mode", "off")) != "cyclic_random":
        raise ValueError("Matched v32b replay requires cyclic_random load switching.")
    eval_seeds = tuple(
        range(
            int(args.eval_base_seed),
            int(args.eval_base_seed) + int(args.eval_episodes),
        )
    )
    learned_policy = td3_macro_policy(actor)
    raw_rows = []
    traces = {}
    schedule_manifest = {}

    for seed in eval_seeds:
        schedule = make_episode_load_switch_schedule(
            checkpoint_args,
            seed,
            int(env.N),
        )
        if schedule is None:
            raise RuntimeError("v32b checkpoint did not produce a load-switch schedule.")
        schedule_manifest[str(seed)] = {
            "switch_steps": list(schedule.switch_steps),
            "dwell_steps": list(schedule.dwell_steps),
            "phase_index_by_macro": schedule.phase_index_by_macro.tolist(),
        }
        hashes = {}
        for policy in POLICIES:
            if policy == "TD3":
                policy_name = "TD3"
                macro_policy = learned_policy
                slew_cap = None
            elif policy == "P0":
                policy_name = "P0"
                macro_policy = None
                slew_cap = None
            else:
                policy_name = "P1"
                macro_policy = None
                slew_cap = residual_cap_m
            row, trace = run_gate_episode(
                env,
                checkpoint_args,
                micro_policy,
                policy_name,
                seed,
                macro_steps=int(checkpoint_args.macro_steps),
                offset_m=offset_m,
                residual_slew_cap_m=slew_cap,
                load_switch_schedule=schedule,
                macro_policy=macro_policy,
            )
            row["policy"] = policy
            for record in trace["macro_records"]:
                record["policy"] = policy
            raw_rows.append(row)
            traces[(policy, seed)] = trace
            hashes[policy] = row["external_trace_hash"]
            save_macro_trace(
                trace_dir / f"{policy}_seed{seed}.csv",
                trace["macro_records"],
                int(env.env.N),
            )
            save_trace_npz(trace_dir / f"{policy}_seed{seed}.npz", trace)
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"CRN trace mismatch for seed {seed}: {hashes}")

    write_csv_atomic(output_dir / "per_seed_raw.csv", raw_rows, RAW_FIELDNAMES)
    paired = paired_delta_rows(raw_rows, eval_seeds)
    write_csv_atomic(
        output_dir / "paired_deltas.csv",
        paired,
        tuple(paired[0].keys()),
    )
    write_json(output_dir / "schedule_manifest.json", schedule_manifest)

    policy_rows = {
        policy: rows_by_policy(raw_rows, policy) for policy in POLICIES
    }
    policy_summary = {}
    for policy in POLICIES:
        delays = np.asarray(
            [row["task_delay_weighted_ms"] for row in policy_rows[policy]],
            dtype=np.float64,
        )
        powers = np.asarray(
            [row["average_power_w"] for row in policy_rows[policy]],
            dtype=np.float64,
        )
        power_ucb95 = one_sided_ucb95(powers)
        policy_summary[policy] = {
            "delay_mean_ms": float(np.mean(delays)),
            "delay_std_ms": float(np.std(delays, ddof=1)),
            "power_mean_w": float(np.mean(powers)),
            "power_std_w": float(np.std(powers, ddof=1)),
            "power_ucb95_w": power_ucb95,
            "power_feasible": bool(power_ucb95 <= POWER_BUDGET_W),
        }

    td3_delay = summary_without_values(
        policy_rows["TD3"],
        policy_rows["P0"],
        "task_delay_weighted_ms",
    )
    prime_delay = summary_without_values(
        policy_rows["P1_prime"],
        policy_rows["P0"],
        "task_delay_weighted_ms",
    )
    td3_power = summary_without_values(
        policy_rows["TD3"],
        policy_rows["P0"],
        "average_power_w",
    )
    prime_power = summary_without_values(
        policy_rows["P1_prime"],
        policy_rows["P0"],
        "average_power_w",
    )
    p1_gain_ms = -float(prime_delay["mean"])
    td3_gain_ms = -float(td3_delay["mean"])
    capture_fraction = (
        float(td3_gain_ms / p1_gain_ms)
        if p1_gain_ms > 0.0
        else float("nan")
    )
    system_lambda = float(
        getattr(checkpoint_args, "energy_lagrange_multiplier_current", 0.0)
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection": selection["selected"],
        "selection_sha256": sha256_file(selection_path),
        "training_seed": int(getattr(checkpoint_args, "seed", -1)),
        "eval_seeds": list(eval_seeds),
        "system_lambda": system_lambda,
        "load_switch_mode": str(checkpoint_args.load_switch_mode),
        "load_switch_dwell_min": int(checkpoint_args.load_switch_dwell_min),
        "load_switch_dwell_max": int(checkpoint_args.load_switch_dwell_max),
        "crn_hash_match": True,
        "policies": policy_summary,
        "paired_effects": {
            "td3_minus_p0_delay": td3_delay,
            "p1_prime_minus_p0_delay": prime_delay,
            "td3_minus_p0_power": td3_power,
            "p1_prime_minus_p0_power": prime_power,
        },
        "td3_gain_ms": td3_gain_ms,
        "p1_prime_gain_ms": p1_gain_ms,
        "capture_fraction": capture_fraction,
        "td3_direction_beneficial": bool(td3_delay["mean"] < 0.0),
        "p1_prime_delay_effective": bool(
            prime_delay["mean"] < -PRACTICAL_GAIN_MS
            and prime_delay["ci95_high"] < 0.0
        ),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "verdict.md").write_text(
        render_verdict(summary),
        encoding="utf-8",
    )
    draw_summary(paired, policy_rows, summary, output_dir)
    representative_seed = min(
        eval_seeds,
        key=lambda seed: abs(
            next(
                row["delay_td3_minus_p0_ms"]
                for row in paired
                if row["seed"] == seed
            )
            - td3_delay["mean"]
        ),
    )
    draw_trajectories(traces, representative_seed, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
