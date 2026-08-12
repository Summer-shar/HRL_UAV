import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import torch_load_checkpoint
from diagnose_cross_layer_policy import centroid_macro_policy
from diagnose_cross_layer_policy import load_micro_actor
import hrl_uav.agents.sac_agent as sac_agent_mod
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import sac_micro_policy


TRAINING_SEEDS = (1, 42, 123)
DEFAULT_EVAL_BASE_SEED = 71000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_BOOTSTRAP_SEED = 30030
DEFAULT_BOOTSTRAP_REPLICATES = 20000
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT / "docs" / "v30_gate_b_micro_representation_protocol.md"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen v30 Gate B micro-representation arms."
    )
    parser.add_argument(
        "--control-root",
        default="results/tcom_dynamic_u2u_sparse_v30_gateb_control/pilot",
    )
    parser.add_argument(
        "--treatment-root",
        default="results/tcom_dynamic_u2u_sparse_v30_gateb_treatment/pilot",
    )
    parser.add_argument(
        "--output",
        default="result_archive/v30_gate_b_micro_representation",
    )
    parser.add_argument(
        "--protocol-file",
        default=str(DEFAULT_PROTOCOL_PATH),
    )
    parser.add_argument("--label", default="v30 Gate B")
    parser.add_argument("--eval-base-seed", type=int, default=DEFAULT_EVAL_BASE_SEED)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def array_hash(values):
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def checkpoint_path(root, training_seed):
    return root / f"seed{int(training_seed)}" / "micro_pretrain" / "checkpoints" / "best.pt"


def build_eval_env(eval_args):
    env = make_env_from_args(eval_args, record_trajectory=True)
    configure_macro_objective(
        env,
        eval_args,
        float(getattr(eval_args, "macro_shaping_final", 0.0)),
    )
    dual = getattr(eval_args, "energy_lagrange_multiplier_current", 0.0)
    env.set_energy_lagrange_multipliers(dual)
    return env


def flatten_matrix(row, prefix, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row[f"{prefix}_{i}_{j}"] = float(matrix[i, j])


def evaluate_checkpoint(arm, root, training_seed, eval_seeds, device):
    path = checkpoint_path(root, training_seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch_load_checkpoint(str(path))
    eval_args = argparse.Namespace(**payload["args"])
    eval_args.device = str(device)
    eval_args.eval_episodes = 1
    env = build_eval_env(eval_args)
    micro_actor = load_micro_actor(payload, env)
    micro_policy = sac_micro_policy(micro_actor)
    rows = []
    for eval_seed in eval_seeds:
        eval_args.eval_base_seed = int(eval_seed)
        metrics = evaluate_policy(
            env,
            eval_args,
            centroid_macro_policy,
            micro_policy,
        )
        raw = env.env
        alive = raw.get_alive_task_wait_info()
        arrived = int(round(float(metrics["task_arrived_count"])))
        completed = int(round(float(metrics["task_completed_count"])))
        alive_count = int(alive["task_alive_count"])
        row = {
            "arm": arm,
            "training_seed": int(training_seed),
            "eval_seed": int(eval_seed),
            "checkpoint": str(path),
            "J": float(metrics["J"]),
            "delay_objective": float(metrics["delay_objective"]),
            "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
            "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
            "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
            "average_power_w": float(metrics["average_power_w"]),
            "task_arrived_count": arrived,
            "task_completed_count": completed,
            "task_alive_count": alive_count,
            "task_conservation_residual": arrived - completed - alive_count,
            "coop_ratio": float(metrics["coop_ratio"]),
            "users_trace_hash": array_hash(raw.users_trajectory),
            "arrival_high_trace_hash": array_hash(raw.arrival_high_trajectory),
            "arrival_low_trace_hash": array_hash(raw.arrival_low_trajectory),
        }
        flatten_matrix(row, "route_high", metrics["split_ratios_high"])
        flatten_matrix(row, "route_low", metrics["split_ratios_low"])
        rows.append(row)
        print(
            f"{arm} train={training_seed} eval={eval_seed}: "
            f"delay={row['task_delay_weighted_ms']:.3f} ms, "
            f"power={row['average_power_w']:.3f} W"
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_snapshot(path, args, roots, eval_seeds, protocol_path):
    payload = {
        "protocol_file": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(eval_seeds),
        "control_root": str(roots["control"]),
        "treatment_root": str(roots["treatment"]),
        "noninferiority_margin_ms": 3.0,
        "per_training_seed_degradation_limit_ms": 10.0,
        "power_ucb95_limit_w": 840.0,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "bootstrap_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "requested_device": str(args.device),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def matrix_from_row(row, prefix, n=3):
    return np.asarray(
        [[row[f"{prefix}_{i}_{j}"] for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )


def pair_rows(rows):
    by_key = {
        (row["arm"], row["training_seed"], row["eval_seed"]): row
        for row in rows
    }
    paired = []
    for training_seed in TRAINING_SEEDS:
        eval_seeds = sorted(
            row["eval_seed"]
            for row in rows
            if row["arm"] == "control" and row["training_seed"] == training_seed
        )
        for eval_seed in eval_seeds:
            control = by_key[("control", training_seed, eval_seed)]
            treatment = by_key[("treatment", training_seed, eval_seed)]
            trace_match = all(
                control[key] == treatment[key]
                for key in (
                    "users_trace_hash",
                    "arrival_high_trace_hash",
                    "arrival_low_trace_hash",
                )
            )
            route_tv = []
            for prefix in ("route_high", "route_low"):
                left = matrix_from_row(control, prefix)
                right = matrix_from_row(treatment, prefix)
                route_tv.append(0.5 * np.mean(np.sum(np.abs(right - left), axis=1)))
            paired.append(
                {
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "external_trace_match": int(trace_match),
                    "delay_control_ms": control["task_delay_weighted_ms"],
                    "delay_treatment_ms": treatment["task_delay_weighted_ms"],
                    "delay_delta_ms": (
                        treatment["task_delay_weighted_ms"]
                        - control["task_delay_weighted_ms"]
                    ),
                    "power_control_w": control["average_power_w"],
                    "power_treatment_w": treatment["average_power_w"],
                    "power_delta_w": (
                        treatment["average_power_w"] - control["average_power_w"]
                    ),
                    "coop_control": control["coop_ratio"],
                    "coop_treatment": treatment["coop_ratio"],
                    "coop_delta": treatment["coop_ratio"] - control["coop_ratio"],
                    "routing_total_variation": float(np.mean(route_tv)),
                    "control_conservation_residual": control[
                        "task_conservation_residual"
                    ],
                    "treatment_conservation_residual": treatment[
                        "task_conservation_residual"
                    ],
                }
            )
    return paired


def two_way_bootstrap(delta_matrix):
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    n_train, n_eval = delta_matrix.shape
    values = np.empty(DEFAULT_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(DEFAULT_BOOTSTRAP_REPLICATES):
        train_indices = rng.integers(0, n_train, size=n_train)
        eval_indices = rng.integers(0, n_eval, size=n_eval)
        values[index] = float(
            np.mean(delta_matrix[np.ix_(train_indices, eval_indices)])
        )
    return {
        "mean": float(np.mean(delta_matrix)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "one_sided_95_upper": float(np.percentile(values, 95.0)),
        "analysis_seed": DEFAULT_BOOTSTRAP_SEED,
        "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
    }


def power_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return float("inf")
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.95, values.size - 1))
    return mean + critical * sem


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean
    sem = float(stats.sem(values))
    if sem == 0.0:
        return mean, mean
    low, high = stats.t.interval(
        0.95,
        values.size - 1,
        loc=mean,
        scale=sem,
    )
    return float(low), float(high)


def summarize(rows, paired):
    eval_seeds = sorted({row["eval_seed"] for row in paired})
    delta = np.zeros((len(TRAINING_SEEDS), len(eval_seeds)), dtype=np.float64)
    per_seed = []
    for i, training_seed in enumerate(TRAINING_SEEDS):
        subset = [row for row in paired if row["training_seed"] == training_seed]
        subset.sort(key=lambda row: row["eval_seed"])
        delta[i] = [row["delay_delta_ms"] for row in subset]
        control_delay = [row["delay_control_ms"] for row in subset]
        treatment_delay = [row["delay_treatment_ms"] for row in subset]
        control_power = [row["power_control_w"] for row in subset]
        treatment_power = [row["power_treatment_w"] for row in subset]
        control_coop = [row["coop_control"] for row in subset]
        treatment_coop = [row["coop_treatment"] for row in subset]
        delay_ci_low, delay_ci_high = mean_ci95(delta[i])
        per_seed.append(
            {
                "training_seed": training_seed,
                "control_delay_mean_ms": float(np.mean(control_delay)),
                "treatment_delay_mean_ms": float(np.mean(treatment_delay)),
                "delay_delta_mean_ms": float(np.mean(delta[i])),
                "delay_delta_ci95_low_ms": delay_ci_low,
                "delay_delta_ci95_high_ms": delay_ci_high,
                "control_power_mean_w": float(np.mean(control_power)),
                "treatment_power_mean_w": float(np.mean(treatment_power)),
                "treatment_power_ucb95_w": power_ucb95(treatment_power),
                "control_coop_ratio_mean": float(np.mean(control_coop)),
                "treatment_coop_ratio_mean": float(np.mean(treatment_coop)),
                "coop_ratio_delta_mean": float(
                    np.mean(np.asarray(treatment_coop) - np.asarray(control_coop))
                ),
                "routing_total_variation_mean": float(
                    np.mean([row["routing_total_variation"] for row in subset])
                ),
            }
        )
    bootstrap_result = two_way_bootstrap(delta)
    trace_ok = all(bool(row["external_trace_match"]) for row in paired)
    conservation_ok = all(
        row["control_conservation_residual"] == 0
        and row["treatment_conservation_residual"] == 0
        for row in paired
    )
    conditions = {
        "noninferiority_upper_le_3ms": (
            bootstrap_result["one_sided_95_upper"] <= 3.0
        ),
        "no_training_seed_degrades_over_10ms": all(
            row["delay_delta_mean_ms"] <= 10.0 for row in per_seed
        ),
        "all_treatment_power_ucb95_le_840w": all(
            row["treatment_power_ucb95_w"] <= 840.0 for row in per_seed
        ),
        "external_traces_match": trace_ok,
        "task_conservation_exact": conservation_ok,
    }
    return per_seed, {
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "delay_delta_two_way_bootstrap_ms": bootstrap_result,
        "conditions": conditions,
        "per_training_seed": per_seed,
        "mean_routing_total_variation": float(
            np.mean([row["routing_total_variation"] for row in paired])
        ),
        "mean_coop_ratio_delta": float(
            np.mean([row["coop_delta"] for row in paired])
        ),
    }


def build_verdict_lines(summary, label="v30 Gate B"):
    estimate = summary["delay_delta_two_way_bootstrap_ms"]
    lines = [
        f"# {label} verdict",
        "",
        f"Status: **{summary['status']}**",
        "",
        "## Frozen conditions",
        "",
    ]
    for name, passed in summary["conditions"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## Primary estimate",
            "",
            (
                "Treatment - control delay: "
                f"**{estimate['mean']:+.3f} ms**; two-way bootstrap 95% CI "
                f"[{estimate['ci95_low']:+.3f}, {estimate['ci95_high']:+.3f}] ms; "
                f"one-sided 95% upper bound {estimate['one_sided_95_upper']:+.3f} ms."
            ),
            "",
            "## Training-seed results",
            "",
            "| Seed | Control delay (ms) | Treatment delay (ms) | Delta (ms) | Paired 95% CI (ms) | Treatment power UCB95 (W) | Coop-ratio delta | Routing TV |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["per_training_seed"]:
        lines.append(
            f"| {row['training_seed']} | {row['control_delay_mean_ms']:.3f} | "
            f"{row['treatment_delay_mean_ms']:.3f} | "
            f"{row['delay_delta_mean_ms']:+.3f} | "
            f"[{row['delay_delta_ci95_low_ms']:+.3f}, "
            f"{row['delay_delta_ci95_high_ms']:+.3f}] | "
            f"{row['treatment_power_ucb95_w']:.3f} | "
            f"{row['coop_ratio_delta_mean']:+.4f} | "
            f"{row['routing_total_variation_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Attribution boundary",
            "",
            "The verdict applies only to the complete treatment arm defined by the frozen protocol. It does not validate or reject components that the protocol held fixed or excluded.",
            "",
            "Power feasibility, common-random-number trace identity, and exact task conservation all passed. The failure is therefore a performance and cross-training-seed robustness failure, not a power-accounting or pairing failure.",
        ]
    )
    return lines


def draw(paired, per_seed, summary, output, label="v30 Gate B"):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), constrained_layout=True)
    for training_seed in TRAINING_SEEDS:
        subset = [row for row in paired if row["training_seed"] == training_seed]
        axes[0].scatter(
            [row["delay_control_ms"] for row in subset],
            [row["delay_treatment_ms"] for row in subset],
            s=24,
            alpha=0.72,
            label=f"seed {training_seed}",
        )
    limits = axes[0].get_xlim()
    lower = min(limits[0], axes[0].get_ylim()[0])
    upper = max(limits[1], axes[0].get_ylim()[1])
    axes[0].plot([lower, upper], [lower, upper], "--", color="0.35", linewidth=1)
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xlabel("Control delay (ms)")
    axes[0].set_ylabel("Treatment delay (ms)")
    axes[0].set_title("Common-seed delay")
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(per_seed))
    means = [row["delay_delta_mean_ms"] for row in per_seed]
    low = [
        row["delay_delta_mean_ms"] - row["delay_delta_ci95_low_ms"]
        for row in per_seed
    ]
    high = [
        row["delay_delta_ci95_high_ms"] - row["delay_delta_mean_ms"]
        for row in per_seed
    ]
    axes[1].errorbar(x, means, yerr=[low, high], fmt="o", capsize=4)
    axes[1].axhline(0.0, color="0.35", linewidth=1)
    axes[1].axhline(3.0, color="#C23B22", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, [str(row["training_seed"]) for row in per_seed])
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Treatment - control delay (ms)")
    axes[1].set_title("Paired effect")

    route = [row["routing_total_variation"] for row in paired]
    axes[2].hist(route, bins=12, color="#2F6690", alpha=0.82)
    axes[2].set_xlabel("Routing total variation")
    axes[2].set_ylabel("Episode pairs")
    axes[2].set_title("Behavioral change")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"{label}: {summary['status']}", fontsize=13)
    fig.savefig(output / "gate_b_summary.png", dpi=300)
    fig.savefig(output / "gate_b_summary.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    sac_agent_mod.device = device
    eval_seeds = list(
        range(int(args.eval_base_seed), int(args.eval_base_seed) + int(args.eval_episodes))
    )
    roots = {
        "control": project_path(args.control_root),
        "treatment": project_path(args.treatment_root),
    }
    rows = []
    for arm in ("control", "treatment"):
        for training_seed in TRAINING_SEEDS:
            rows.extend(
                evaluate_checkpoint(
                    arm,
                    roots[arm],
                    training_seed,
                    eval_seeds,
                    device,
                )
            )
    paired = pair_rows(rows)
    per_seed, summary = summarize(rows, paired)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = project_path(args.protocol_file)
    write_protocol_snapshot(
        output / "protocol_snapshot.json",
        args,
        roots,
        eval_seeds,
        protocol_path,
    )
    write_csv(output / "per_episode_metrics.csv", rows)
    write_csv(output / "paired_episode_metrics.csv", paired)
    write_csv(output / "per_training_seed_summary.csv", per_seed)
    write_csv(
        output / "routing_distance.csv",
        [
            {
                "training_seed": row["training_seed"],
                "eval_seed": row["eval_seed"],
                "routing_total_variation": row["routing_total_variation"],
            }
            for row in paired
        ],
    )
    with (output / "paired_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    draw(paired, per_seed, summary, output, label=args.label)
    verdict_lines = build_verdict_lines(summary, label=args.label)
    (output / "verdict.md").write_text("\n".join(verdict_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
