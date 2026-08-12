import argparse
import csv
import hashlib
import json
import math
import time
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
from run_load_switch_gate import run_gate_episode
from run_load_switch_gate import write_csv_atomic
from search_position_oracle import load_checkpoint_components
from search_position_oracle import target_centroid_waypoint_action


SEARCH_ARMS = ("unconstrained", "feasibility_first")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search a frozen noncausal load-stage residual policy family."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/tcom_dynamic_load_switch_v32b_formal/formal/seed1/"
            "joint_td3_sac/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="result_archive/v32c_stage_residual_benchmark",
    )
    parser.add_argument("--search-base-seed", type=int, default=2000)
    parser.add_argument("--search-episodes", type=int, default=5)
    parser.add_argument("--validation-base-seed", type=int, default=3000)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--initial-std-m", type=float, default=40.0)
    parser.add_argument("--minimum-std-m", type=float, default=5.0)
    parser.add_argument("--max-offset-m", type=float, default=80.0)
    parser.add_argument("--anticipation-steps", type=int, default=5)
    parser.add_argument("--search-power-gate-w", type=float, default=838.0)
    parser.add_argument("--power-budget-w", type=float, default=840.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_residuals(offsets, max_offset_m):
    offsets = np.asarray(offsets, dtype=np.float64).reshape(3, 3, 2).copy()
    norms = np.linalg.norm(offsets, axis=2, keepdims=True)
    scale = np.minimum(1.0, float(max_offset_m) / np.maximum(norms, 1e-12))
    return offsets * scale


def next_switch_after(schedule, macro_index):
    for switch in schedule.switch_steps:
        if int(switch) > int(macro_index):
            return int(switch)
    return None


class StageResidualPolicy:
    def __init__(self, offsets_m, schedule, anticipation_steps=5):
        self.offsets_m = np.asarray(offsets_m, dtype=np.float64).reshape(3, 3, 2)
        self.schedule = schedule
        self.anticipation_steps = int(anticipation_steps)
        self.macro_index = 0

    def desired_residual(self, macro_index):
        phase = int(self.schedule.phase_index(macro_index)) % 3
        residual = self.offsets_m[phase].copy()
        switch = next_switch_after(self.schedule, macro_index)
        if switch is None or self.anticipation_steps <= 0:
            return residual
        remaining = int(switch) - int(macro_index)
        if remaining > self.anticipation_steps:
            return residual
        next_phase = int(self.schedule.phase_index(switch)) % 3
        alpha = (
            self.anticipation_steps - remaining + 1
        ) / float(self.anticipation_steps)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (1.0 - alpha) * residual + alpha * self.offsets_m[next_phase]

    def __call__(self, env, _local_obs_batch, _deterministic):
        raw_state = env.env.get_raw_state()
        centroids = np.asarray(raw_state["weighted_centroids"], dtype=np.float64)
        targets = np.clip(
            centroids + self.desired_residual(self.macro_index),
            0.0,
            1000.0,
        )
        self.macro_index += 1
        return target_centroid_waypoint_action(
            targets,
            centroids,
            env.macro_waypoint_radius_m,
        ).astype(np.float32)


def one_sided_ucb95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(values[0])
    return float(
        np.mean(values)
        + stats.t.ppf(0.95, values.size - 1) * stats.sem(values)
    )


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def offsets_digest(offsets):
    array = np.asarray(offsets, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def evaluate_offsets(
    env,
    checkpoint_args,
    micro_policy,
    offsets,
    seeds,
    schedules,
    anticipation_steps,
    baseline_hashes,
    policy_name,
):
    rows = []
    for seed in seeds:
        policy = StageResidualPolicy(
            offsets,
            schedules[int(seed)],
            anticipation_steps=anticipation_steps,
        )
        row, trace = run_gate_episode(
            env,
            checkpoint_args,
            micro_policy,
            policy_name,
            int(seed),
            macro_steps=int(checkpoint_args.macro_steps),
            load_switch_schedule=schedules[int(seed)],
            macro_policy=policy,
        )
        expected_hash = baseline_hashes.get(int(seed))
        if expected_hash is not None and row["external_trace_hash"] != expected_hash:
            raise RuntimeError(
                f"CRN mismatch for seed {seed}: "
                f"{row['external_trace_hash']} != {expected_hash}"
            )
        rows.append(row)
    delay = np.asarray(
        [row["task_delay_weighted_ms"] for row in rows], dtype=np.float64
    )
    power = np.asarray([row["average_power_w"] for row in rows], dtype=np.float64)
    return rows, {
        "delay_mean_ms": float(np.mean(delay)),
        "delay_std_ms": float(np.std(delay, ddof=1)) if delay.size > 1 else 0.0,
        "power_mean_w": float(np.mean(power)),
        "power_std_w": float(np.std(power, ddof=1)) if power.size > 1 else 0.0,
        "power_ucb95_w": one_sided_ucb95(power),
    }


def candidate_rank(metrics, arm, power_gate_w):
    delay = float(metrics["delay_mean_ms"])
    power_ucb = float(metrics["power_ucb95_w"])
    if arm == "unconstrained":
        return (delay,)
    violation = max(power_ucb - float(power_gate_w), 0.0)
    return (int(violation > 0.0), violation, delay)


def run_cem(
    arm,
    rng_seed,
    args,
    evaluate,
    output_dir,
):
    rng = np.random.default_rng(int(rng_seed))
    mean = np.zeros((3, 3, 2), dtype=np.float64)
    std = np.full_like(mean, float(args.initial_std_m))
    elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
    best = None
    history = []
    candidate_rows = []
    for iteration in range(1, int(args.iterations) + 1):
        samples = rng.normal(mean, std, size=(int(args.population), 3, 3, 2))
        samples[0] = mean
        records = []
        for candidate_index, raw_offsets in enumerate(samples, start=1):
            offsets = project_residuals(raw_offsets, args.max_offset_m)
            metrics = evaluate(offsets, f"{arm}_i{iteration:02d}_c{candidate_index:02d}")
            rank = candidate_rank(metrics, arm, args.search_power_gate_w)
            record = {
                "iteration": iteration,
                "candidate": candidate_index,
                "offsets_m": offsets,
                "offsets_sha256": offsets_digest(offsets),
                "metrics": metrics,
                "rank": rank,
            }
            records.append(record)
            candidate_rows.append(
                {
                    "arm": arm,
                    "iteration": iteration,
                    "candidate": candidate_index,
                    "offsets_sha256": record["offsets_sha256"],
                    "delay_mean_ms": metrics["delay_mean_ms"],
                    "power_mean_w": metrics["power_mean_w"],
                    "power_ucb95_w": metrics["power_ucb95_w"],
                    "search_gate_feasible": int(
                        metrics["power_ucb95_w"] <= args.search_power_gate_w
                    ),
                    "offsets_json": json.dumps(offsets.tolist(), separators=(",", ":")),
                }
            )
            if best is None or rank < best["rank"]:
                best = record
        records.sort(key=lambda item: item["rank"])
        elites = np.stack([record["offsets_m"] for record in records[:elite_count]])
        mean = np.mean(elites, axis=0)
        std = np.maximum(np.std(elites, axis=0), float(args.minimum_std_m))
        snapshot = {
            "iteration": iteration,
            "iteration_best": records[0],
            "global_best": best,
            "mean_m": mean,
            "std_m": std,
        }
        history.append(snapshot)
        print(
            f"[{arm} {iteration}/{args.iterations}] "
            f"delay={best['metrics']['delay_mean_ms']:.3f} ms "
            f"power_ucb={best['metrics']['power_ucb95_w']:.3f} W"
        )
        (output_dir / "history.partial.json").write_text(
            json.dumps(json_ready(history), indent=2), encoding="utf-8"
        )
    (output_dir / "history.json").write_text(
        json.dumps(json_ready(history), indent=2), encoding="utf-8"
    )
    return best, candidate_rows


def collect_p0_baselines(
    env,
    checkpoint_args,
    micro_policy,
    seeds,
    schedules,
    policy_name,
):
    rows = []
    hashes = {}
    for seed in seeds:
        row, _trace = run_gate_episode(
            env,
            checkpoint_args,
            micro_policy,
            "P0",
            int(seed),
            macro_steps=int(checkpoint_args.macro_steps),
            load_switch_schedule=schedules[int(seed)],
            macro_policy=None,
        )
        row["policy"] = policy_name
        rows.append(row)
        hashes[int(seed)] = row["external_trace_hash"]
    return rows, hashes


def paired_validation(candidate_rows, p0_rows):
    candidate = {int(row["seed"]): row for row in candidate_rows}
    p0 = {int(row["seed"]): row for row in p0_rows}
    seeds = sorted(set(candidate) & set(p0))
    delay_delta = np.asarray(
        [
            float(candidate[seed]["task_delay_weighted_ms"])
            - float(p0[seed]["task_delay_weighted_ms"])
            for seed in seeds
        ],
        dtype=np.float64,
    )
    power = np.asarray(
        [float(candidate[seed]["average_power_w"]) for seed in seeds],
        dtype=np.float64,
    )
    mean, low, high = mean_ci95(delay_delta)
    return {
        "seeds": seeds,
        "delay_candidate_minus_p0_ms": {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "values": delay_delta.tolist(),
        },
        "gain_ms": -mean,
        "power_mean_w": float(np.mean(power)),
        "power_ucb95_w": one_sided_ucb95(power),
    }


def select_best_feasible(validation_summaries, power_budget_w):
    feasible = [
        (arm, summary)
        for arm, summary in validation_summaries.items()
        if float(summary["power_ucb95_w"]) <= float(power_budget_w)
    ]
    if not feasible:
        return None, None
    return max(feasible, key=lambda item: float(item[1]["gain_ms"]))


def classify_benchmark(validation_summaries, power_budget_w):
    unconstrained = validation_summaries["unconstrained"]
    feasible_arm, feasible = select_best_feasible(
        validation_summaries,
        power_budget_w,
    )
    feasible_gain = float(feasible["gain_ms"]) if feasible is not None else float("-inf")
    feasible_ci = (
        feasible["delay_candidate_minus_p0_ms"] if feasible is not None else None
    )
    unconstrained_gain = float(unconstrained["gain_ms"])
    if (
        feasible is not None
        and feasible_gain > 15.0
        and feasible_ci["ci95_high"] < 0.0
    ):
        verdict = "MEANINGFUL_FEASIBLE_POSITION_VALUE_FOUND"
    elif feasible is not None and feasible_gain >= 5.0:
        verdict = "SECONDARY_FEASIBLE_POSITION_INCREMENT_FOUND"
    elif unconstrained_gain > 15.0 and feasible_gain < 5.0:
        verdict = "POWER_FEASIBILITY_DOMINATES_BEST_FOUND_FAMILY"
    else:
        verdict = "NO_MATERIAL_FEASIBLE_GAIN_FOUND_IN_FROZEN_FAMILY"
    return verdict, feasible_arm


def draw_summary(p0_rows, candidate_rows, summaries, output_dir):
    policy_order = ("P0", "unconstrained", "feasibility_first")
    power_means = []
    power_ucbs = []
    grouped = {name: [] for name in policy_order}
    for row in p0_rows + candidate_rows:
        grouped[row["policy"]].append(row)
    for name in policy_order:
        powers = [float(row["average_power_w"]) for row in grouped[name]]
        power_means.append(float(np.mean(powers)))
        power_ucbs.append(one_sided_ucb95(powers))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    candidate_order = ("unconstrained", "feasibility_first")
    candidate_labels = ("Unconstrained", "Feasible-first")
    colors = ("#D55E00", "#2878B5")
    x_delay = np.arange(len(candidate_order))
    for x_value, name, color in zip(x_delay, candidate_order, colors):
        delta = summaries["validation"][name]["delay_candidate_minus_p0_ms"]
        values = np.asarray(delta["values"], dtype=np.float64)
        jitter = np.linspace(-0.08, 0.08, values.size)
        axes[0].scatter(
            np.full(values.size, x_value) + jitter,
            values,
            s=18,
            alpha=0.45,
            color=color,
            edgecolors="none",
        )
        mean = float(delta["mean"])
        yerr = np.asarray(
            [[mean - float(delta["ci95_low"])],
             [float(delta["ci95_high"]) - mean]],
            dtype=np.float64,
        )
        axes[0].errorbar(
            x_value,
            mean,
            yerr=yerr,
            fmt="o",
            markersize=6,
            capsize=4,
            linewidth=1.6,
            color=color,
        )
    axes[0].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xticks(x_delay, candidate_labels)
    axes[0].set_ylabel("Delay change relative to P0 (ms)")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(3)
    axes[1].bar(x, power_means, color=("#6C757D", "#D55E00", "#2878B5"))
    for x_value, name in zip(x, policy_order):
        powers = np.asarray(
            [float(row["average_power_w"]) for row in grouped[name]],
            dtype=np.float64,
        )
        jitter = np.linspace(-0.08, 0.08, powers.size)
        axes[1].scatter(
            np.full(powers.size, x_value) + jitter,
            powers,
            s=14,
            alpha=0.35,
            color="#222222",
            edgecolors="none",
        )
    axes[1].scatter(x, power_ucbs, marker="_", s=150, color="black", label="UCB95")
    axes[1].axhline(840.0, color="#C44E52", linestyle="--", label="840 W")
    axes[1].set_xticks(x, ("P0", "Unconstrained", "Feasible-first"))
    axes[1].set_ylabel("System-average power (W)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(summaries["verdict"].replace("_", " ").title(), fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "stage_residual_benchmark.png", dpi=240)
    fig.savefig(output_dir / "stage_residual_benchmark.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = project_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = project_path(args.checkpoint).resolve()
    (
        checkpoint_path,
        checkpoint_args,
        env,
        _actor,
        micro_policy,
        _device,
    ) = load_checkpoint_components(checkpoint, args.device)
    if str(getattr(checkpoint_args, "load_switch_mode", "off")) != "cyclic_random":
        raise ValueError("The frozen benchmark requires cyclic_random switching.")

    search_seeds = tuple(
        range(int(args.search_base_seed), int(args.search_base_seed) + int(args.search_episodes))
    )
    validation_seeds = tuple(
        range(
            int(args.validation_base_seed),
            int(args.validation_base_seed) + int(args.validation_episodes),
        )
    )
    if set(search_seeds) & set(validation_seeds):
        raise ValueError("Search and validation episode seeds must be disjoint.")
    all_seeds = search_seeds + validation_seeds
    schedules = {
        int(seed): make_episode_load_switch_schedule(
            checkpoint_args, int(seed), int(env.N)
        )
        for seed in all_seeds
    }
    if any(schedule is None for schedule in schedules.values()):
        raise RuntimeError("Failed to construct a frozen load-switch schedule.")

    search_p0, search_hashes = collect_p0_baselines(
        env,
        checkpoint_args,
        micro_policy,
        search_seeds,
        schedules,
        "P0_search",
    )
    candidate_cache = {}

    def evaluate_search(offsets, policy_name):
        digest = offsets_digest(offsets)
        if digest not in candidate_cache:
            rows, metrics = evaluate_offsets(
                env,
                checkpoint_args,
                micro_policy,
                offsets,
                search_seeds,
                schedules,
                args.anticipation_steps,
                search_hashes,
                policy_name,
            )
            candidate_cache[digest] = (rows, metrics)
        return dict(candidate_cache[digest][1])

    best_by_arm = {}
    candidate_metric_rows = []
    for arm, rng_seed in zip(SEARCH_ARMS, (3201, 3202)):
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        best, records = run_cem(
            arm,
            rng_seed,
            args,
            evaluate_search,
            arm_dir,
        )
        best_by_arm[arm] = best
        candidate_metric_rows.extend(records)

    write_csv(
        output_dir / "candidate_search_metrics.csv",
        candidate_metric_rows,
    )
    write_csv_atomic(
        output_dir / "search_baselines.csv",
        search_p0,
        RAW_FIELDNAMES,
    )

    validation_p0, validation_hashes = collect_p0_baselines(
        env,
        checkpoint_args,
        micro_policy,
        validation_seeds,
        schedules,
        "P0",
    )
    validation_candidates = []
    validation_summaries = {}
    for arm in SEARCH_ARMS:
        offsets = np.asarray(best_by_arm[arm]["offsets_m"], dtype=np.float64)
        rows, _metrics = evaluate_offsets(
            env,
            checkpoint_args,
            micro_policy,
            offsets,
            validation_seeds,
            schedules,
            args.anticipation_steps,
            validation_hashes,
            arm,
        )
        for row in rows:
            row["policy"] = arm
        validation_candidates.extend(rows)
        validation_summaries[arm] = paired_validation(rows, validation_p0)

    all_validation = validation_p0 + validation_candidates
    write_csv_atomic(
        output_dir / "validation_raw.csv",
        all_validation,
        RAW_FIELDNAMES,
    )
    paired_rows = []
    p0_by_seed = {int(row["seed"]): row for row in validation_p0}
    for row in validation_candidates:
        seed = int(row["seed"])
        paired_rows.append(
            {
                "seed": seed,
                "policy": row["policy"],
                "delay_candidate_minus_p0_ms": float(
                    row["task_delay_weighted_ms"]
                )
                - float(p0_by_seed[seed]["task_delay_weighted_ms"]),
                "power_candidate_minus_p0_w": float(row["average_power_w"])
                - float(p0_by_seed[seed]["average_power_w"]),
            }
        )
    write_csv(output_dir / "validation_paired.csv", paired_rows)

    verdict, best_feasible_arm = classify_benchmark(
        validation_summaries,
        args.power_budget_w,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "search_seeds": list(search_seeds),
        "validation_seeds": list(validation_seeds),
        "search_power_gate_w": float(args.search_power_gate_w),
        "validation_power_budget_w": float(args.power_budget_w),
        "best_offsets_m": {
            arm: json_ready(best_by_arm[arm]["offsets_m"]) for arm in SEARCH_ARMS
        },
        "search_best_metrics": {
            arm: json_ready(best_by_arm[arm]["metrics"]) for arm in SEARCH_ARMS
        },
        "validation": validation_summaries,
        "best_feasible_validation_arm": best_feasible_arm,
        "verdict": verdict,
        "elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "noncausal best-found benchmark in the frozen 18-dimensional "
            "stage-residual family; not a theoretical upper bound"
        ),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    protocol_snapshot = vars(args).copy()
    protocol_snapshot["checkpoint"] = str(checkpoint_path)
    (output_dir / "protocol_snapshot.json").write_text(
        json.dumps(protocol_snapshot, indent=2), encoding="utf-8"
    )
    (output_dir / "verdict.md").write_text(
        "\n".join(
            [
                "# v32c Noncausal Stage-Residual Benchmark",
                "",
                f"- Verdict: **{verdict}**",
                f"- Unconstrained best-found gain: "
                f"`{validation_summaries['unconstrained']['gain_ms']:.3f} ms`",
                f"- Unconstrained power UCB95: "
                f"`{validation_summaries['unconstrained']['power_ucb95_w']:.3f} W`",
                f"- Feasibility-first best-found gain: "
                f"`{validation_summaries['feasibility_first']['gain_ms']:.3f} ms`",
                f"- Feasibility-first power UCB95: "
                f"`{validation_summaries['feasibility_first']['power_ucb95_w']:.3f} W`",
                f"- Best feasible validated arm: `{best_feasible_arm}`",
                "- Scope: best found in the frozen policy family and search budget; "
                "not a theoretical upper bound.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    draw_summary(validation_p0, validation_candidates, summary, output_dir)
    print(json.dumps(json_ready(summary), indent=2))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
