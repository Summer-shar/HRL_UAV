"""Calibrate delay/energy weights from directly logged v37 proxy costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v37_load_imbalance_gate import (
    BASE_CONFIG,
    TREATMENT_CONFIG,
    paired_delta,
    read_rows,
    run_gate,
)


DEFAULT_OUTPUT = ROOT / "result_archive" / "v38_objective_calibration"
DEFAULT_RHO_MS_PER_W = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="5000-5019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Recompute the report from existing per-seed CSV files.",
    )
    parser.add_argument(
        "--rho-ms-per-w",
        type=float,
        default=DEFAULT_RHO_MS_PER_W,
    )
    return parser.parse_args()


def paired_values(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray(paired_delta(rows, field)["values"], dtype=np.float64)


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise ValueError("Cannot fit a through-origin slope with zero input norm.")
    return float(np.dot(x, y) / denominator)


def weights_from_slopes(
    delay_proxy_per_ms: float,
    energy_proxy_per_w: float,
    rho_ms_per_w: float,
) -> dict:
    if delay_proxy_per_ms <= 0.0 or energy_proxy_per_w <= 0.0:
        raise ValueError("Proxy sensitivities must be positive.")
    ratio = (
        float(rho_ms_per_w)
        * float(delay_proxy_per_ms)
        / float(energy_proxy_per_w)
    )
    delay_weight = 1.0 / (1.0 + ratio)
    energy_weight = ratio / (1.0 + ratio)
    return {
        "energy_to_delay_weight_ratio": ratio,
        "delay_weight": delay_weight,
        "energy_weight": energy_weight,
    }


def summarize_scenario(rows: list[dict], rho_ms_per_w: float) -> dict:
    delay_ms = paired_values(rows, "task_delay_weighted_ms")
    power_w = paired_values(rows, "average_power_w")
    delay_proxy = paired_values(rows, "cost_delay_mean")
    delay_proxy_raw = paired_values(rows, "cost_delay_raw_mean")
    energy_proxy = paired_values(rows, "cost_energy_mean")
    energy_proxy_raw = paired_values(rows, "cost_energy_raw_mean")
    overflow = paired_values(rows, "overflow_cost_mean")
    drift = paired_values(rows, "drift_cost_mean")
    objective = paired_values(rows, "objective_cost_mean")
    soft_alignment = np.asarray(
        [
            float(row["soft_objective_j"])
            - float(row["objective_cost_mean"])
            for row in rows
        ],
        dtype=np.float64,
    )

    k_delay = slope_through_origin(delay_ms, delay_proxy)
    k_energy = slope_through_origin(power_w, energy_proxy)
    weights = weights_from_slopes(abs(k_delay), abs(k_energy), rho_ms_per_w)
    reconstructed = 0.9 * delay_proxy + 0.1 * energy_proxy + overflow + drift
    mean_delay_gain_ms = -float(np.mean(delay_ms))
    mean_power_increase_w = float(np.mean(power_w))
    actual_exchange_rate = (
        mean_delay_gain_ms / mean_power_increase_w
        if mean_power_increase_w > 0.0
        else float("inf")
    )
    return {
        "delay_delta_ms_mean": float(np.mean(delay_ms)),
        "power_delta_w_mean": float(np.mean(power_w)),
        "cost_delay_delta_mean": float(np.mean(delay_proxy)),
        "cost_energy_delta_mean": float(np.mean(energy_proxy)),
        "overflow_cost_delta_mean": float(np.mean(overflow)),
        "drift_cost_delta_mean": float(np.mean(drift)),
        "objective_cost_delta_mean": float(np.mean(objective)),
        "objective_reconstruction_max_abs_error": float(
            np.max(np.abs(objective - reconstructed))
        ),
        "soft_objective_alignment_max_abs_error": float(
            np.max(np.abs(soft_alignment))
        ),
        "delay_raw_clip_delta_max_abs": float(
            np.max(np.abs(delay_proxy_raw - delay_proxy))
        ),
        "energy_raw_clip_delta_max_abs": float(
            np.max(np.abs(energy_proxy_raw - energy_proxy))
        ),
        "delay_clip_fraction_max": float(
            max(float(row["delay_clip_fraction"]) for row in rows)
        ),
        "energy_clip_fraction_max": float(
            max(float(row["energy_clip_fraction"]) for row in rows)
        ),
        "k_delay_proxy_per_ms": abs(k_delay),
        "k_energy_proxy_per_w": abs(k_energy),
        "actual_exchange_rate_ms_per_w": actual_exchange_rate,
        "meets_declared_exchange_rate": bool(
            actual_exchange_rate >= float(rho_ms_per_w)
        ),
        "candidate_weights": weights,
        "paired": {
            "delay_delta_ms": delay_ms.tolist(),
            "power_delta_w": power_w.tolist(),
            "cost_delay_delta": delay_proxy.tolist(),
            "cost_energy_delta": energy_proxy.tolist(),
        },
    }


def pooled_weights(
    base_rows: list[dict],
    treatment_rows: list[dict],
    rho_ms_per_w: float,
) -> dict:
    delay_ms = np.concatenate(
        [
            paired_values(base_rows, "task_delay_weighted_ms"),
            paired_values(treatment_rows, "task_delay_weighted_ms"),
        ]
    )
    power_w = np.concatenate(
        [
            paired_values(base_rows, "average_power_w"),
            paired_values(treatment_rows, "average_power_w"),
        ]
    )
    delay_proxy = np.concatenate(
        [
            paired_values(base_rows, "cost_delay_mean"),
            paired_values(treatment_rows, "cost_delay_mean"),
        ]
    )
    energy_proxy = np.concatenate(
        [
            paired_values(base_rows, "cost_energy_mean"),
            paired_values(treatment_rows, "cost_energy_mean"),
        ]
    )
    k_delay = abs(slope_through_origin(delay_ms, delay_proxy))
    k_energy = abs(slope_through_origin(power_w, energy_proxy))
    return {
        "k_delay_proxy_per_ms": k_delay,
        "k_energy_proxy_per_w": k_energy,
        **weights_from_slopes(k_delay, k_energy, rho_ms_per_w),
    }


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    base_output = output / "base_load"
    treatment_output = output / "imbalanced_load"

    if not args.reuse_existing:
        run_gate(BASE_CONFIG, base_output, args.seeds, args.device)
        run_gate(TREATMENT_CONFIG, treatment_output, args.seeds, args.device)
    base_rows = read_rows(base_output / "per_seed_raw.csv")
    treatment_rows = read_rows(treatment_output / "per_seed_raw.csv")

    base = summarize_scenario(base_rows, args.rho_ms_per_w)
    treatment = summarize_scenario(treatment_rows, args.rho_ms_per_w)
    pooled = pooled_weights(base_rows, treatment_rows, args.rho_ms_per_w)
    scenario_energy_weights = [
        base["candidate_weights"]["energy_weight"],
        treatment["candidate_weights"]["energy_weight"],
    ]
    freeze_checks = {
        "zero_delay_clipping": max(
            base["delay_clip_fraction_max"],
            treatment["delay_clip_fraction_max"],
        ) == 0.0,
        "zero_energy_clipping": max(
            base["energy_clip_fraction_max"],
            treatment["energy_clip_fraction_max"],
        ) == 0.0,
        "zero_overflow_and_drift_delta": max(
            abs(base["overflow_cost_delta_mean"]),
            abs(treatment["overflow_cost_delta_mean"]),
            abs(base["drift_cost_delta_mean"]),
            abs(treatment["drift_cost_delta_mean"]),
        ) <= 1e-12,
        "objective_reconstruction_exact": max(
            base["objective_reconstruction_max_abs_error"],
            treatment["objective_reconstruction_max_abs_error"],
            base["soft_objective_alignment_max_abs_error"],
            treatment["soft_objective_alignment_max_abs_error"],
        ) <= 1e-12,
    }
    weights_frozen = bool(all(freeze_checks.values()))
    report = {
        "status": (
            "WEIGHTS_FROZEN_CALIBRATION_ONLY_NOT_FORMAL_EVIDENCE"
            if weights_frozen
            else "WEIGHTS_NOT_FROZEN"
        ),
        "seeds": args.seeds,
        "exchange_rate_ms_per_w": float(args.rho_ms_per_w),
        "base_load": base,
        "imbalanced_load": treatment,
        "pooled": pooled,
        "cross_operating_point": {
            "energy_weight_min": float(min(scenario_energy_weights)),
            "energy_weight_max": float(max(scenario_energy_weights)),
            "energy_weight_span": float(np.ptp(scenario_energy_weights)),
        },
        "freeze_checks": freeze_checks,
        "weights_frozen": weights_frozen,
        "freeze_rule": (
            "Use the mechanically computed pooled weights after confirming zero "
            "clipping, zero overflow/drift deltas, and exact objective reconstruction."
        ),
    }
    (output / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if weights_frozen:
        frozen = {
            "status": "FROZEN_FOR_NEXT_SAC_RETRAIN",
            "exchange_rate_ms_per_w": float(args.rho_ms_per_w),
            "delay_weight": float(pooled["delay_weight"]),
            "energy_weight": float(pooled["energy_weight"]),
            "paper_display": {"delay_weight": 0.96, "energy_weight": 0.04},
            "calibration_seeds": args.seeds,
            "formal_gate_seed_rule": (
                "Use 20 new CRN seeds disjoint from calibration seeds 5000-5019."
            ),
        }
        (output / "frozen_weights.json").write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
