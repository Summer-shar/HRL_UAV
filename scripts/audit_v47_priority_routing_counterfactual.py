"""Read-only v47 audit separating macro motion from priority routing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v47_independent_synchronous as v47  # noqa: E402
from hrl_uav.routing import project_rows_to_simplex  # noqa: E402
from train_centralized_macro import fixed_macro_policy, zero_macro_policy  # noqa: E402


DEFAULT_OUTPUT = ROOT / "result_archive" / "v47_priority_routing_counterfactual"


def parse_seeds(spec: str) -> list[int]:
    return v45.parse_seed_spec(spec)


def priority_limited_policy(base_policy, remote_cap: float):
    """Keep the learned low-priority routing and cap high-priority offload.

    The returned action is already on the simplex, so the environment's
    Euclidean projection is idempotent. This is an evaluation intervention,
    not a training rule.
    """
    cap = float(remote_cap)
    if not 0.0 <= cap <= 1.0:
        raise ValueError("remote_cap must lie in [0, 1]")

    def policy(env, deterministic):
        raw = np.asarray(base_policy(env, deterministic), dtype=np.float64)
        n = int(env.N)
        matrices = raw.reshape(2, n, n)
        projected = project_rows_to_simplex(matrices)
        high = projected[0].copy()
        for source in range(n):
            remote = high[source].copy()
            remote[source] = 0.0
            remote_total = float(np.sum(remote))
            if remote_total > cap and remote_total > 0.0:
                remote *= cap / remote_total
            high[source] = remote
            high[source, source] = 1.0 - float(np.sum(remote))
        return np.stack((high, projected[1]), axis=0).reshape(-1).astype(np.float32)

    return policy


def evaluate_variant(name: str, policy: dict, seed: int) -> dict:
    args = policy["args"]
    env = v45.make_eval_env(args, record_trajectory=False)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = v45.evaluate_policy(
        env,
        eval_args,
        policy["macro_policy"],
        policy["micro_policy"],
    )
    split_high = np.asarray(metrics["split_ratios_high"], dtype=np.float64)
    split_low = np.asarray(metrics["split_ratios_low"], dtype=np.float64)
    n = split_high.shape[0]
    return {
        "seed": int(seed),
        "variant": name,
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
        "queue_total_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "queue_high_mbits": float(metrics["q_high"]),
        "queue_low_mbits": float(metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "soft_objective_j": float(metrics["J"]),
        "coop_ratio": float(metrics["coop_ratio"]),
        "high_remote_ratio": float(1.0 - np.trace(split_high) / n),
        "low_remote_ratio": float(1.0 - np.trace(split_low) / n),
        "u2g_distance_m": float(metrics["u2g"]),
    }


def mean_ci95(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    half = (
        float(student_t.ppf(0.975, array.size - 1))
        * std
        / np.sqrt(array.size)
        if array.size > 1
        else 0.0
    )
    return {
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "std": std,
        "n": int(array.size),
    }


def summarize(rows: list[dict], variants: tuple[str, ...]) -> dict:
    fields = tuple(
        key for key in rows[0] if key not in {"seed", "variant"}
    )
    by_variant = {
        variant: {
            field: mean_ci95(
                [float(row[field]) for row in rows if row["variant"] == variant]
            )
            for field in fields
        }
        for variant in variants
    }
    baseline = {
        int(row["seed"]): row
        for row in rows
        if row["variant"] == "td3_macro_td3_sac"
    }
    paired = {}
    for variant in variants:
        if variant == "td3_macro_td3_sac":
            continue
        current = {
            int(row["seed"]): row
            for row in rows
            if row["variant"] == variant
        }
        common = sorted(set(baseline) & set(current))
        paired[variant] = {
            field: mean_ci95(
                [
                    float(current[seed][field]) - float(baseline[seed][field])
                    for seed in common
                ]
            )
            for field in (
                "task_delay_weighted_ms",
                "task_delay_high_ms",
                "task_delay_low_ms",
                "queue_total_mbits",
                "soft_objective_j",
            )
        }
    return {
        "summaries": by_variant,
        "paired_variant_minus_td3_macro_td3_sac": paired,
        "interpretation": "Negative paired delay means the intervention is faster.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    policies = v47.load_policies("formal", args.train_seed, args.device)
    td3 = policies["joint_td3_sac"]
    sca = policies["cppo_sca"]
    variants = {
        "td3_macro_td3_sac": td3,
        "td3_macro_high_local_sac": {
            **td3,
            "micro_policy": priority_limited_policy(td3["micro_policy"], 0.0),
        },
        "td3_macro_high_cap10_sac": {
            **td3,
            "micro_policy": priority_limited_policy(td3["micro_policy"], 0.10),
        },
        "td3_macro_sca": {**td3, "micro_policy": sca["micro_policy"]},
        "centroid_macro_td3_sac": {**td3, "macro_policy": zero_macro_policy},
        "fixed_macro_td3_sac": {**td3, "macro_policy": fixed_macro_policy},
        "sca_macro_td3_sac": {**td3, "macro_policy": sca["macro_policy"]},
        "centroid_macro_high_local_sac": {
            **td3,
            "macro_policy": zero_macro_policy,
            "micro_policy": priority_limited_policy(td3["micro_policy"], 0.0),
        },
        "fixed_macro_high_local_sac": {
            **td3,
            "macro_policy": fixed_macro_policy,
            "micro_policy": priority_limited_policy(td3["micro_policy"], 0.0),
        },
        "sca_macro_high_local_sac": {
            **td3,
            "macro_policy": sca["macro_policy"],
            "micro_policy": priority_limited_policy(td3["micro_policy"], 0.0),
        },
    }
    order = tuple(variants)
    rows: list[dict] = []
    for seed in parse_seeds(args.eval_seeds):
        for name, policy in variants.items():
            row = evaluate_variant(name, policy, seed)
            rows.append(row)
            print(
                f"seed={seed} {name}: delay={row['task_delay_weighted_ms']:.2f} "
                f"high_remote={row['high_remote_ratio']:.3f}",
                flush=True,
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_raw.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows, order)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
