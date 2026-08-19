"""Compare v47 policies with task-stage and link-level accounting.

This is a read-only evaluation audit. It loads the independently trained final
checkpoints, replays common environment seeds, and records where their delay
differences arise without changing either policy or environment dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_v44_physical_causal_chain import COMPONENTS, PhysicalAudit  # noqa: E402
from compare_common import evaluate_policy  # noqa: E402
from evaluate_v45_fair_screen import make_eval_env, mean_ci95  # noqa: E402
from evaluate_v47_independent_synchronous import load_policies  # noqa: E402


DEFAULT_OUTPUT = ROOT / "result_archive" / "v47_seed1_causal_chain"
METHODS = (
    "joint_td3_sac",
    "joint_ppo_sac",
    "cppo_sca",
    "centroid_sac",
    "fixed_sac",
)


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    return seeds


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_method(policy: dict, seed: int) -> tuple[dict, list[dict]]:
    args = policy["args"]
    env = make_eval_env(args, record_trajectory=False)
    audit = PhysicalAudit(env)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(
        env,
        eval_args,
        policy["macro_policy"],
        policy["micro_policy"],
    )
    row = {
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
        "queue_total_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "queue_high_mbits": float(metrics["q_high"]),
        "queue_low_mbits": float(metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "soft_objective_j": float(metrics["J"]),
        "coop_ratio": float(metrics["coop_ratio"]),
        "pressure_imbalance": float(metrics["pressure_imbalance"]),
        "u2g_distance_m": float(metrics["u2g"]),
        **audit.summary(),
    }
    return row, audit.completed_records


def summarize(rows: list[dict], seeds: list[int]) -> dict:
    fields = [key for key in rows[0] if key not in {"seed", "method"}]
    by_method = {
        method: {
            int(row["seed"]): row
            for row in rows
            if row["method"] == method
        }
        for method in METHODS
    }
    means = {
        method: {
            field: float(
                np.mean([float(by_method[method][seed][field]) for seed in seeds])
            )
            for field in fields
        }
        for method in METHODS
    }
    paired_vs_td3 = {
        method: {
            field: mean_ci95(
                [
                    float(by_method[method][seed][field])
                    - float(by_method["joint_td3_sac"][seed][field])
                    for seed in seeds
                ]
            )
            for field in fields
        }
        for method in METHODS
        if method != "joint_td3_sac"
    }
    closure = {
        method: float(
            sum(means[method][f"stage_weighted_{component[:-2]}_ms"] for component in COMPONENTS)
            - means[method]["task_delay_weighted_ms"]
        )
        for method in METHODS
    }
    return {
        "methods": list(METHODS),
        "seeds": seeds,
        "means": means,
        "paired_method_minus_td3": paired_vs_td3,
        "decomposition_closure_error_ms": closure,
        "difference_definition": (
            "method minus joint_td3_sac; negative means the comparator is lower"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    seeds = parse_seed_spec(args.eval_seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = load_policies(args.stage, args.train_seed, args.device)

    rows: list[dict] = []
    task_rows: list[dict] = []
    for seed in seeds:
        for method in METHODS:
            result, completed = evaluate_method(policies[method], seed)
            rows.append({"seed": seed, "method": method, **result})
            task_rows.extend(
                {"seed": seed, "method": method, **task} for task in completed
            )
            print(
                f"seed={seed} method={method} "
                f"delay={result['task_delay_weighted_ms']:.2f}ms "
                f"u2g_active={result['u2g_active_link_time_s']:.3f}s "
                f"u2u_active={result['u2u_active_link_time_s']:.3f}s",
                flush=True,
            )

    write_csv(output / "per_seed_raw.csv", rows)
    write_csv(output / "per_task_stage_components.csv", task_rows)
    report = summarize(rows, seeds)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
