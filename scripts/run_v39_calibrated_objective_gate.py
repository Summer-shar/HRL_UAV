"""Run the frozen v39 two-load gate with the retrained micro SAC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v37_load_imbalance_gate import (
    mean_ci,
    paired_delta,
    plot_report,
    policy_mean,
    read_rows,
)


GATE_SCRIPT = ROOT / "scripts" / "run_v34_cooperative_mobility_gate.py"
BASE_CONFIG = ROOT / "configs" / "tcom_v39_calibrated_soft_objective_base.json"
TREATMENT_CONFIG = ROOT / "configs" / "tcom_v39_calibrated_soft_objective.json"
CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v39_calibrated_soft_objective"
    / "formal"
    / "seed1"
    / "micro_pretrain"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v39_calibrated_objective_gate"
DEFAULT_SEEDS = "6000-6019"
DECLARED_EXCHANGE_RATE_MS_PER_W = 0.25
DELAY_WEIGHT = 0.960408028711155
ENERGY_WEIGHT = 0.03959197128884489


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Rebuild summaries and plots from the archived per-seed CSV files.",
    )
    return parser.parse_args()


def run_gate(
    config: Path,
    checkpoint: Path,
    output: Path,
    seeds: str,
    device: str,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--seeds",
            seeds,
            "--offset_m",
            "60",
            "--device",
            device,
        ],
        cwd=ROOT,
        check=True,
    )


def exchange_rate_summary(delay_gain_ms: float, power_delta_w: float) -> dict:
    delay_gain_ms = float(delay_gain_ms)
    power_delta_w = float(power_delta_w)
    if delay_gain_ms <= 0.0:
        status = "NO_DELAY_IMPROVEMENT"
        value = 0.0
    elif power_delta_w <= 0.0:
        status = "PARETO_DOMINANT"
        value = float("inf")
    else:
        status = "FINITE_EXCHANGE_RATE"
        value = delay_gain_ms / power_delta_w
    return {
        "status": status,
        "ms_per_w": value,
        "declared_threshold_ms_per_w": DECLARED_EXCHANGE_RATE_MS_PER_W,
        "meets_declared_threshold": bool(
            delay_gain_ms > 0.0
            and (
                power_delta_w <= 0.0
                or value >= DECLARED_EXCHANGE_RATE_MS_PER_W
            )
        ),
    }


def raw_soft_objective_value(row: dict) -> float:
    return (
        DELAY_WEIGHT * float(row["cost_delay_raw_mean"])
        + ENERGY_WEIGHT * float(row["cost_energy_raw_mean"])
    )


def scenario_summary(rows: list[dict]) -> dict:
    for row in rows:
        row["raw_soft_objective_j"] = raw_soft_objective_value(row)
    delay = paired_delta(rows, "task_delay_weighted_ms")
    queue = paired_delta(rows, "mean_queue_mbits")
    power = paired_delta(rows, "average_power_w")
    soft_j = paired_delta(rows, "soft_objective_j")
    raw_soft_j = paired_delta(rows, "raw_soft_objective_j")
    delay_gain_ms = -float(delay["mean"])
    return {
        "delay_delta_ms": delay,
        "delay_gain_ms": delay_gain_ms,
        "queue_delta_mbits": queue,
        "power_delta_w": power,
        "soft_j_delta": soft_j,
        "raw_soft_j_delta": raw_soft_j,
        "proxy_clipping": {
            "delay_clip_fraction_mean": {
                policy: policy_mean(rows, policy, "delay_clip_fraction")
                for policy in ("fixed", "centroid", "cooperative")
            },
            "delay_clip_fraction_max": {
                policy: max(
                    float(row["delay_clip_fraction"])
                    for row in rows
                    if row["policy"] == policy
                )
                for policy in ("fixed", "centroid", "cooperative")
            },
            "energy_clip_fraction_max": max(
                float(row["energy_clip_fraction"]) for row in rows
            ),
        },
        "actual_exchange_rate": exchange_rate_summary(
            delay_gain_ms,
            float(power["mean"]),
        ),
        "cooperative_motion": {
            field: policy_mean(rows, "cooperative", field)
            for field in (
                "target_occupancy_fraction",
                "target_reach_segment_fraction",
                "target_arrival_time_reached_s",
                "target_arrival_time_capped_s",
                "target_error_mean_m",
            )
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    checkpoint = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else ROOT / args.checkpoint
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing retrained micro checkpoint: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    base_output = output / "base_load"
    treatment_output = output / "imbalanced_load"
    if not args.reuse_existing:
        run_gate(BASE_CONFIG, checkpoint, base_output, args.seeds, args.device)
        run_gate(TREATMENT_CONFIG, checkpoint, treatment_output, args.seeds, args.device)
    else:
        for path in (
            base_output / "per_seed_raw.csv",
            treatment_output / "per_seed_raw.csv",
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing archived gate rows: {path}")

    base_rows = read_rows(base_output / "per_seed_raw.csv")
    treatment_rows = read_rows(treatment_output / "per_seed_raw.csv")
    base = scenario_summary(base_rows)
    treatment = scenario_summary(treatment_rows)
    base_delta = np.asarray(base["delay_delta_ms"]["values"], dtype=np.float64)
    treatment_delta = np.asarray(
        treatment["delay_delta_ms"]["values"], dtype=np.float64
    )
    gain_difference = mean_ci(base_delta - treatment_delta)
    centroid_queue = policy_mean(treatment_rows, "centroid", "mean_queue_mbits")
    queue_margin = 0.05 * centroid_queue
    checks = {
        "delay_gain_gt_5ms": treatment["delay_gain_ms"] > 5.0,
        "delay_ci_excludes_zero": treatment["delay_delta_ms"]["ci_high"] < 0.0,
        "soft_j_mean_improves": treatment["soft_j_delta"]["mean"] < 0.0,
        "soft_j_ci_excludes_zero": treatment["soft_j_delta"]["ci_high"] < 0.0,
        "queue_noninferior": (
            treatment["queue_delta_mbits"]["ci_high"] < queue_margin
        ),
    }
    passed = bool(all(checks.values()))
    report = {
        "status": "PASS" if passed else "FAIL",
        "evidence_role": "FORMAL_ZERO_TRAINING_GATE_WITH_RETRAINED_MICRO_SAC",
        "seeds": args.seeds,
        "micro_checkpoint": str(checkpoint),
        "delay_weight": DELAY_WEIGHT,
        "energy_weight": ENERGY_WEIGHT,
        "base_load_mbps": [12.5, 5.5, 3.5],
        "imbalanced_load_mbps": [14.5, 4.5, 2.5],
        "cooperative_offset_m": 60.0,
        "base": base,
        "treatment": treatment,
        "cooperative_gain_difference_ms": gain_difference,
        "queue_noninferiority_margin_mbits": queue_margin,
        "checks": checks,
        "known_invalid_legacy_fields": {
            "coop_ratio": (
                "The archived v39 gate CSV computed this field from the "
                "off-diagonal transmission-service matrix only. It is not "
                "used by any gate decision and must not be interpreted."
            )
        },
    }
    (output / "gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_report(base, treatment, gain_difference, output)

    delay = treatment["delay_delta_ms"]
    soft_j = treatment["soft_j_delta"]
    raw_soft_j = treatment["raw_soft_j_delta"]
    queue = treatment["queue_delta_mbits"]
    power = treatment["power_delta_w"]
    exchange = treatment["actual_exchange_rate"]
    lines = [
        "# v39 校准软目标正式门禁结果",
        "",
        f"**判决：{report['status']}**",
        "",
        "## 增强不均衡负载主判据",
        "",
        (
            f"- 时延收益：{treatment['delay_gain_ms']:.3f} ms；配对差 95% CI "
            f"[{delay['ci_low']:.3f}, {delay['ci_high']:.3f}] ms；"
        ),
        (
            f"- 软目标差：{soft_j['mean']:.6f}；95% CI "
            f"[{soft_j['ci_low']:.6f}, {soft_j['ci_high']:.6f}]；"
        ),
        (
            f"- 未裁剪原始代价重算差：{raw_soft_j['mean']:.6f}；95% CI "
            f"[{raw_soft_j['ci_low']:.6f}, {raw_soft_j['ci_high']:.6f}]；"
        ),
        (
            f"- 队列差：{queue['mean']:.3f} Mbit；95% CI "
            f"[{queue['ci_low']:.3f}, {queue['ci_high']:.3f}] Mbit；"
        ),
        (
            f"- 功率差：{power['mean']:.3f} W；实际交换率："
            f"{exchange['ms_per_w']:.3f} ms/W；"
        ),
        "",
        "## 判据核对",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "基准负载仅作描述性对照，不改变正式判决。功率无硬阈值。",
            "",
            "注：归档 CSV 中的 `coop_ratio` 为旧统计口径错误，未进入任何判据，不得用于解释。",
        ]
    )
    (output / "verdict.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
