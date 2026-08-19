"""Read-only v48 counterfactual audit for macro motion and routing structure."""

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

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v48_priority_gated as v48  # noqa: E402
from compare_sca import solve_sca_micro_action  # noqa: E402
from hrl_uav.routing import priority_gated_to_simplex  # noqa: E402
from hrl_uav.routing import project_rows_to_simplex  # noqa: E402
from train_centralized_macro import configure_sca_wrapper  # noqa: E402
from train_centralized_macro import fixed_macro_policy  # noqa: E402
from train_centralized_macro import zero_macro_policy  # noqa: E402


DEFAULT_OUTPUT = ROOT / "result_archive" / "v48_macro_micro_counterfactual"


def decode_policy_action(base_policy, env, deterministic: bool) -> np.ndarray:
    """Decode either latent priority-gated or full-simplex policy output."""
    raw = np.asarray(base_policy(env, deterministic), dtype=np.float64).reshape(-1)
    n = int(env.N)
    latent_dim = 2 * n * (n - 1)
    full_dim = 2 * n * n
    if raw.size == latent_dim:
        matrices = priority_gated_to_simplex(
            np.clip(raw.reshape(2, n, n - 1), -1.0, 1.0),
            priority_local_biases=np.asarray(
                [np.log(float(env.micro_priority_weight_ratio)), 0.0],
                dtype=np.float64,
            ),
            logit_scale=float(env.micro_stick_breaking_scale),
        )
    elif raw.size == full_dim:
        matrices = project_rows_to_simplex(raw.reshape(2, n, n))
    else:
        raise ValueError(
            f"Unexpected micro action size {raw.size}; expected {latent_dim} or {full_dim}."
        )
    return np.asarray(matrices, dtype=np.float64)


def full_simplex_policy(matrix_fn):
    def policy(env, deterministic):
        matrices = np.asarray(matrix_fn(env, deterministic), dtype=np.float64)
        return project_rows_to_simplex(matrices).reshape(-1).astype(np.float32)

    return policy


def capped_policy(base_policy, high_remote_cap=None, low_remote_cap=None):
    caps = (high_remote_cap, low_remote_cap)

    def matrices(env, deterministic):
        routed = decode_policy_action(base_policy, env, deterministic).copy()
        n = int(env.N)
        for priority, cap in enumerate(caps):
            if cap is None:
                continue
            cap = float(cap)
            if not 0.0 <= cap <= 1.0:
                raise ValueError("Remote caps must lie in [0, 1].")
            for source in range(n):
                remote = routed[priority, source].copy()
                remote[source] = 0.0
                remote_total = float(np.sum(remote))
                if remote_total > cap and remote_total > 0.0:
                    remote *= cap / remote_total
                routed[priority, source] = remote
                routed[priority, source, source] = 1.0 - float(np.sum(remote))
        return routed

    return full_simplex_policy(matrices)


def make_sca_policy(args):
    sca_defaults = {
        "sca_update_interval": 1,
        "sca_max_iter": 3,
        "sca_tol": 1e-3,
        "sca_solver": "CLARABEL",
        "sca_delay_weight": 0.9,
        "sca_energy_weight": 0.1,
        "sca_proximal_weight": 1e-3,
    }
    for name, value in sca_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)

    def policy(env, deterministic):
        configure_sca_wrapper(env, args)
        return solve_sca_micro_action(env, deterministic=bool(deterministic))

    return policy


def hybrid_policy(sac_policy, sca_policy, *, local_source: str):
    """Combine local gates from one policy with remote destinations from the other."""
    if local_source not in {"sac", "sca"}:
        raise ValueError("local_source must be 'sac' or 'sca'.")

    def matrices(env, deterministic):
        sac = decode_policy_action(sac_policy, env, deterministic)
        sca = decode_policy_action(sca_policy, env, deterministic)
        local = sac if local_source == "sac" else sca
        remote = sca if local_source == "sac" else sac
        n = int(env.N)
        output = np.zeros_like(local)
        for priority in range(2):
            for source in range(n):
                local_fraction = float(local[priority, source, source])
                remote_weights = remote[priority, source].copy()
                remote_weights[source] = 0.0
                remote_total = float(np.sum(remote_weights))
                if remote_total <= 1e-12:
                    peers = [index for index in range(n) if index != source]
                    remote_weights[peers] = 1.0 / len(peers)
                else:
                    remote_weights /= remote_total
                output[priority, source] = (1.0 - local_fraction) * remote_weights
                output[priority, source, source] = local_fraction
        return output

    return full_simplex_policy(matrices)


def blended_policy(sac_policy, sca_policy, learned_weight: float):
    """Blend a learned routing matrix with the per-frame SCA baseline."""
    learned_weight = float(learned_weight)
    if not 0.0 <= learned_weight <= 1.0:
        raise ValueError("learned_weight must lie in [0, 1].")

    def matrices(env, deterministic):
        learned = decode_policy_action(sac_policy, env, deterministic)
        baseline = decode_policy_action(sca_policy, env, deterministic)
        return (1.0 - learned_weight) * baseline + learned_weight * learned

    return full_simplex_policy(matrices)


def log_residual_policy(sac_policy, sca_policy, residual_scale: float):
    """Apply SAC as a multiplicative log-ratio residual around SCA."""
    residual_scale = float(residual_scale)
    if residual_scale < 0.0:
        raise ValueError("residual_scale must be non-negative.")

    def matrices(env, deterministic):
        learned = decode_policy_action(sac_policy, env, deterministic)
        baseline = decode_policy_action(sca_policy, env, deterministic)
        zero_latent = np.zeros((2, env.N, env.N - 1), dtype=np.float64)
        prior = priority_gated_to_simplex(
            zero_latent,
            priority_local_biases=np.asarray(
                [np.log(float(env.micro_priority_weight_ratio)), 0.0],
                dtype=np.float64,
            ),
            logit_scale=float(env.micro_stick_breaking_scale),
        )
        eps = 1e-6
        logits = np.log(np.maximum(baseline, eps)) + residual_scale * (
            np.log(np.maximum(learned, eps)) - np.log(np.maximum(prior, eps))
        )
        logits -= np.max(logits, axis=-1, keepdims=True)
        output = np.exp(logits)
        return output / np.sum(output, axis=-1, keepdims=True)

    return full_simplex_policy(matrices)


def support_preserving_log_residual_policy(
    sac_policy,
    sca_policy,
    residual_scale: float,
    support_threshold: float = 1e-6,
):
    """Reweight SCA-selected links without activating inactive MAC links."""
    residual_scale = float(residual_scale)
    support_threshold = float(support_threshold)

    def matrices(env, deterministic):
        learned = decode_policy_action(sac_policy, env, deterministic)
        baseline = decode_policy_action(sca_policy, env, deterministic)
        zero_latent = np.zeros((2, env.N, env.N - 1), dtype=np.float64)
        prior = priority_gated_to_simplex(
            zero_latent,
            priority_local_biases=np.asarray(
                [np.log(float(env.micro_priority_weight_ratio)), 0.0],
                dtype=np.float64,
            ),
            logit_scale=float(env.micro_stick_breaking_scale),
        )
        support = baseline > support_threshold
        logits = np.full_like(baseline, -np.inf, dtype=np.float64)
        logits[support] = np.log(baseline[support]) + residual_scale * (
            np.log(np.maximum(learned[support], 1e-12))
            - np.log(np.maximum(prior[support], 1e-12))
        )
        row_max = np.max(logits, axis=-1, keepdims=True)
        output = np.where(support, np.exp(logits - row_max), 0.0)
        return output / np.maximum(np.sum(output, axis=-1, keepdims=True), 1e-12)

    return full_simplex_policy(matrices)


def evaluate_variant(policy: dict, seed: int) -> dict:
    env = v45.make_eval_env(policy["args"], record_trajectory=False)
    eval_args = argparse.Namespace(**vars(policy["args"]))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = v45.evaluate_policy(
        env,
        eval_args,
        policy["macro_policy"],
        policy["micro_policy"],
    )
    high_raw = np.asarray(metrics["split_ratios_high"], dtype=np.float64)
    low_raw = np.asarray(metrics["split_ratios_low"], dtype=np.float64)
    high = high_raw / np.maximum(np.sum(high_raw, axis=1, keepdims=True), 1e-12)
    low = low_raw / np.maximum(np.sum(low_raw, axis=1, keepdims=True), 1e-12)
    n = int(env.N)
    return {
        "seed": int(seed),
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
        "queue_total_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "queue_high_mbits": float(metrics["q_high"]),
        "queue_low_mbits": float(metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "soft_objective_j": float(metrics["J"]),
        "coop_ratio": float(metrics["coop_ratio"]),
        "u2g_distance_m": float(metrics["u2g"]),
        "high_local_ratio": float(np.trace(high) / n),
        "low_local_ratio": float(np.trace(low) / n),
    }


def summarize(rows: list[dict], order: tuple[str, ...]) -> dict:
    fields = tuple(key for key in rows[0] if key not in {"seed", "variant"})
    summaries = {
        variant: {
            field: v45.mean_ci95(
                [float(row[field]) for row in rows if row["variant"] == variant]
            )
            for field in fields
        }
        for variant in order
    }
    baseline = {
        int(row["seed"]): row for row in rows if row["variant"] == "learned"
    }
    paired = {}
    for variant in order:
        if variant == "learned":
            continue
        current = {
            int(row["seed"]): row for row in rows if row["variant"] == variant
        }
        common = sorted(set(baseline) & set(current))
        paired[variant] = {
            field: v45.mean_ci95(
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
        "summaries": summaries,
        "paired_variant_minus_learned": paired,
        "difference_definition": "variant minus learned; negative means improvement",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated subset of variant names.",
    )
    args = parser.parse_args()

    checkpoint = v48.checkpoint(args.stage, args.train_seed)
    learned = v45.load_method_policy("joint_td3_sac", checkpoint, args.device)
    sac_policy = learned["micro_policy"]
    sca_policy = make_sca_policy(learned["args"])
    variants = {
        "learned": learned,
        "centroid_macro": {**learned, "macro_policy": zero_macro_policy},
        "fixed_macro": {**learned, "macro_policy": fixed_macro_policy},
        "high_cap10": {
            **learned,
            "micro_policy": capped_policy(sac_policy, high_remote_cap=0.10),
        },
        "low_cap25": {
            **learned,
            "micro_policy": capped_policy(sac_policy, low_remote_cap=0.25),
        },
        "both_caps": {
            **learned,
            "micro_policy": capped_policy(
                sac_policy,
                high_remote_cap=0.10,
                low_remote_cap=0.25,
            ),
        },
        "sca_micro": {**learned, "micro_policy": sca_policy},
        "sac_gates_sca_destinations": {
            **learned,
            "micro_policy": hybrid_policy(
                sac_policy,
                sca_policy,
                local_source="sac",
            ),
        },
        "sca_gates_sac_destinations": {
            **learned,
            "micro_policy": hybrid_policy(
                sac_policy,
                sca_policy,
                local_source="sca",
            ),
        },
        "blend10": {
            **learned,
            "micro_policy": blended_policy(sac_policy, sca_policy, 0.10),
        },
        "blend20": {
            **learned,
            "micro_policy": blended_policy(sac_policy, sca_policy, 0.20),
        },
        "blend30": {
            **learned,
            "micro_policy": blended_policy(sac_policy, sca_policy, 0.30),
        },
        "blend50": {
            **learned,
            "micro_policy": blended_policy(sac_policy, sca_policy, 0.50),
        },
        "logres10": {
            **learned,
            "micro_policy": log_residual_policy(sac_policy, sca_policy, 0.10),
        },
        "logres02": {
            **learned,
            "micro_policy": log_residual_policy(sac_policy, sca_policy, 0.02),
        },
        "logres05": {
            **learned,
            "micro_policy": log_residual_policy(sac_policy, sca_policy, 0.05),
        },
        "logres25": {
            **learned,
            "micro_policy": log_residual_policy(sac_policy, sca_policy, 0.25),
        },
        "logres50": {
            **learned,
            "micro_policy": log_residual_policy(sac_policy, sca_policy, 0.50),
        },
        "logres100": {
            **learned,
            "micro_policy": log_residual_policy(sac_policy, sca_policy, 1.00),
        },
        "slogres05": {
            **learned,
            "micro_policy": support_preserving_log_residual_policy(
                sac_policy, sca_policy, 0.05
            ),
        },
        "slogres10": {
            **learned,
            "micro_policy": support_preserving_log_residual_policy(
                sac_policy, sca_policy, 0.10
            ),
        },
        "slogres25": {
            **learned,
            "micro_policy": support_preserving_log_residual_policy(
                sac_policy, sca_policy, 0.25
            ),
        },
    }
    if args.variants.strip():
        selected = tuple(
            token.strip() for token in args.variants.split(",") if token.strip()
        )
        unknown = sorted(set(selected) - set(variants))
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}")
        variants = {name: variants[name] for name in selected}
    order = tuple(variants)
    rows: list[dict] = []
    for seed in v45.parse_seed_spec(args.eval_seeds):
        for name, policy in variants.items():
            result = evaluate_variant(policy, seed)
            rows.append({"variant": name, **result})
            print(
                f"seed={seed} {name}: delay={result['task_delay_weighted_ms']:.2f}ms "
                f"local(H/L)={result['high_local_ratio']:.3f}/"
                f"{result['low_local_ratio']:.3f}",
                flush=True,
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_seed_raw.csv", rows)
    report = summarize(rows, order)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
