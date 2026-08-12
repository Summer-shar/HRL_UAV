import argparse
import csv
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

from run_load_switch_gate import RAW_FIELDNAMES
from run_load_switch_gate import paired_summary
from run_load_switch_gate import run_gate_episode
from run_load_switch_gate import save_macro_trace
from run_load_switch_gate import write_csv_atomic
from search_position_oracle import load_checkpoint_components


EVAL_SEEDS = tuple(range(65000, 65020))
POWER_BUDGET_W = 840.0
PRACTICAL_GAIN_MS = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the single frozen P1-prime CRN validation."
    )
    parser.add_argument(
        "--selection",
        default="result_archive/p1_prime_prescreen/selection.json",
    )
    parser.add_argument(
        "--output-dir",
        default="result_archive/p1_prime_validation",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
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


def paired_rows(p0_rows, prime_rows):
    p0_by_seed = {int(row["seed"]): row for row in p0_rows}
    prime_by_seed = {int(row["seed"]): row for row in prime_rows}
    rows = []
    for seed in EVAL_SEEDS:
        p0 = p0_by_seed[seed]
        prime = prime_by_seed[seed]
        if p0["external_trace_hash"] != prime["external_trace_hash"]:
            raise RuntimeError(f"CRN trace mismatch for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "delay_p0_ms": p0["task_delay_weighted_ms"],
                "delay_p1_prime_ms": prime["task_delay_weighted_ms"],
                "delay_p1_prime_minus_p0_ms": (
                    prime["task_delay_weighted_ms"]
                    - p0["task_delay_weighted_ms"]
                ),
                "power_p0_w": p0["average_power_w"],
                "power_p1_prime_w": prime["average_power_w"],
                "power_p1_prime_minus_p0_w": (
                    prime["average_power_w"] - p0["average_power_w"]
                ),
                "propulsion_delta_w": (
                    prime["propulsion_power_w"] - p0["propulsion_power_w"]
                ),
                "transient_propulsion_delta_w": (
                    prime["transient_propulsion_power_w"]
                    - p0["transient_propulsion_power_w"]
                ),
                "cpu_delta_w": prime["cpu_power_w"] - p0["cpu_power_w"],
                "communication_delta_w": (
                    prime["communication_power_w"]
                    - p0["communication_power_w"]
                ),
            }
        )
    return rows


def draw_summary(paired, power_ucb95, output_dir):
    delay = np.asarray(
        [row["delay_p1_prime_minus_p0_ms"] for row in paired],
        dtype=np.float64,
    )
    power = np.asarray(
        [row["power_p1_prime_w"] for row in paired], dtype=np.float64
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    axes[0].scatter(np.arange(delay.size), delay, color="#2878B5", zorder=3)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].axhline(-PRACTICAL_GAIN_MS, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("CRN evaluation seed index")
    axes[0].set_ylabel("P1-prime - P0 delay (ms)")
    axes[0].set_title("Paired delay effect")
    axes[0].grid(True, linestyle=":", alpha=0.4)

    axes[1].scatter(np.arange(power.size), power, color="#54A24B", zorder=3)
    axes[1].axhline(POWER_BUDGET_W, color="black", linestyle="--", linewidth=0.9, label="840 W")
    axes[1].axhline(power_ucb95, color="#C82423", linestyle=":", linewidth=1.1, label="P1-prime UCB95")
    axes[1].set_xlabel("CRN evaluation seed index")
    axes[1].set_ylabel("P1-prime average power (W)")
    axes[1].set_title("Power feasibility")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.4)
    fig.savefig(output_dir / "p1_prime_validation.png", dpi=300)
    fig.savefig(output_dir / "p1_prime_validation.pdf")
    plt.close(fig)


def draw_trajectory(p0_trace, prime_trace, seed, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)
    colors = ("#2878B5", "#E17C05", "#54A24B")
    for axis, title, trace in (
        (axes[0], "P0 centroid tracking", p0_trace),
        (axes[1], "P1-prime power-aware support", prime_trace),
    ):
        uav = np.asarray(trace["uav_trajectory_m"])
        centroid = np.asarray(trace["centroid_trajectory_m"])
        for index, color in enumerate(colors):
            axis.plot(uav[:, index, 0], uav[:, index, 1], color=color, linewidth=2.0, label=f"UAV {index + 1}")
            axis.plot(centroid[:, index, 0], centroid[:, index, 1], color=color, linestyle=":", linewidth=1.2)
            axis.scatter(uav[0, index, 0], uav[0, index, 1], marker="s", color=color, s=35)
            axis.scatter(uav[-1, index, 0], uav[-1, index, 1], marker="*", color=color, s=90)
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle=":", alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Representative load-switch trajectories (seed {seed})")
    fig.savefig(output_dir / "p1_prime_trajectory.png", dpi=300)
    fig.savefig(output_dir / "p1_prime_trajectory.pdf")
    plt.close(fig)


def render_verdict(summary):
    return "\n".join(
        (
            "# P1-prime single-validation verdict",
            "",
            f"- Frozen verdict: **{summary['verdict']}**",
            f"- P1-prime power UCB95: `{summary['p1_prime_power_ucb95_w']:.3f} W`",
            "- Paired delay P1-prime minus P0: "
            f"`{summary['delay_delta']['mean']:.3f} ms`, 95% CI "
            f"`[{summary['delay_delta']['ci95_low']:.3f}, {summary['delay_delta']['ci95_high']:.3f}] ms`",
            f"- Power feasible: `{summary['power_feasible']}`",
            f"- Delay gate passed: `{summary['delay_effective']}`",
            "",
            "No candidate parameter may be changed after this result.",
            "",
        )
    )


def main():
    args = parse_args()
    selection_path = project_path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "FROZEN_FOR_SINGLE_REPLAY":
        raise RuntimeError("P1-prime selection is not frozen for replay.")
    selected = selection["selected"]
    offset_m = float(selected["offset_m"])
    residual_cap_m = float(selected["residual_cap_per_macro_m"])
    checkpoint = selection["provenance"]["checkpoint"]
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"

    checkpoint_path, checkpoint_args, env, _actor, micro_policy, _device = (
        load_checkpoint_components(checkpoint, args.device)
    )
    raw_rows = []
    traces = {}
    for seed in EVAL_SEEDS:
        hashes = {}
        for label, policy_name, slew_cap in (
            ("P0", "P0", None),
            ("P1_prime", "P1", residual_cap_m),
        ):
            row, trace = run_gate_episode(
                env,
                checkpoint_args,
                micro_policy,
                policy_name,
                seed,
                offset_m=offset_m,
                residual_slew_cap_m=slew_cap,
            )
            row["policy"] = label
            for record in trace["macro_records"]:
                record["policy"] = label
            raw_rows.append(row)
            traces[(label, seed)] = trace
            hashes[label] = row["external_trace_hash"]
            save_macro_trace(
                trace_dir / f"{label}_seed{seed}.csv",
                trace["macro_records"],
                int(env.env.N),
            )
            save_trace_npz(trace_dir / f"{label}_seed{seed}.npz", trace)
        if hashes["P0"] != hashes["P1_prime"]:
            raise RuntimeError(f"CRN mismatch for seed {seed}")

    write_csv_atomic(output_dir / "per_seed_raw.csv", raw_rows, RAW_FIELDNAMES)
    p0_rows = [row for row in raw_rows if row["policy"] == "P0"]
    prime_rows = [row for row in raw_rows if row["policy"] == "P1_prime"]
    paired = paired_rows(p0_rows, prime_rows)
    write_csv_atomic(
        output_dir / "paired_deltas.csv", paired, tuple(paired[0].keys())
    )

    delay_summary = paired_summary(
        prime_rows, p0_rows, "task_delay_weighted_ms"
    )
    power_summary = paired_summary(prime_rows, p0_rows, "average_power_w")
    power_ucb95 = one_sided_ucb95(
        [row["average_power_w"] for row in prime_rows]
    )
    power_feasible = bool(power_ucb95 <= POWER_BUDGET_W)
    delay_effective = bool(
        delay_summary["mean"] < -PRACTICAL_GAIN_MS
        and delay_summary["ci95_high"] < 0.0
    )
    if power_feasible and delay_effective:
        verdict = "FEASIBLE_AND_EFFECTIVE"
    elif power_feasible:
        verdict = "FEASIBLE_BUT_WEAK"
    else:
        verdict = "POWER_INFEASIBLE"
    summary = {
        "verdict": verdict,
        "selected": selected,
        "checkpoint": str(checkpoint_path),
        "seeds": list(EVAL_SEEDS),
        "crn_hash_match": True,
        "p1_prime_power_ucb95_w": power_ucb95,
        "power_feasible": power_feasible,
        "delay_effective": delay_effective,
        "delay_delta": {
            key: value for key, value in delay_summary.items() if key != "values"
        },
        "power_delta": {
            key: value for key, value in power_summary.items() if key != "values"
        },
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "verdict.md").write_text(
        render_verdict(summary), encoding="utf-8"
    )
    draw_summary(paired, power_ucb95, output_dir)
    representative_seed = min(
        EVAL_SEEDS,
        key=lambda seed: abs(
            next(
                row["delay_p1_prime_minus_p0_ms"]
                for row in paired if row["seed"] == seed
            )
            - delay_summary["mean"]
        ),
    )
    draw_trajectory(
        traces[("P0", representative_seed)],
        traces[("P1_prime", representative_seed)],
        representative_seed,
        output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

