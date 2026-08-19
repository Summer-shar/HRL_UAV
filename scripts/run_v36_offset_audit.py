"""Pre-registered search/validation audit for cooperative offset amplitude."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = ROOT / "scripts" / "run_v34_cooperative_mobility_gate.py"
DEFAULT_CONFIG = ROOT / "configs" / "tcom_v35_reference_like_mobility.json"
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v33_soft_energy_random_mobility"
    / "formal"
    / "seed1"
    / "micro_pretrain"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v36_cooperative_offset_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offsets", default="60,90,120")
    parser.add_argument("--search_seeds", default="3000-3004")
    parser.add_argument("--validation_seeds", default="4000-4019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_gate(
    config: Path,
    checkpoint: Path,
    output: Path,
    seeds: str,
    offset: float,
    device: str,
) -> dict:
    command = [
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
        str(offset),
        "--device",
        device,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return read_report(output / "gate_report.json")


def summarize(report: dict, offset: float, split: str) -> dict:
    means = report["policy_means"]
    paired = report["paired"]["cooperative_vs_centroid_delay"]
    coop = means["cooperative"]
    centroid = means["centroid"]
    return {
        "split": split,
        "offset_m": float(offset),
        "delay_gain_ms": float(paired["mean_gain"]),
        "delay_delta_ci_low_ms": float(paired["ci_low"]),
        "delay_delta_ci_high_ms": float(paired["ci_high"]),
        "power_delta_w": float(
            coop["average_power_w"]["mean"]
            - centroid["average_power_w"]["mean"]
        ),
        "soft_j_delta": float(
            coop["soft_objective_j"]["mean"]
            - centroid["soft_objective_j"]["mean"]
        ),
        "target_occupancy_fraction": float(
            coop["target_occupancy_fraction"]["mean"]
        ),
        "target_reach_segment_fraction": float(
            coop["target_reach_segment_fraction"]["mean"]
        ),
        "arrival_time_reached_s": float(
            coop["target_arrival_time_reached_s"]["mean"]
        ),
        "arrival_time_capped_s": float(
            coop["target_arrival_time_capped_s"]["mean"]
        ),
        "target_error_mean_m": float(coop["target_error_mean_m"]["mean"]),
        "gate_pass": bool(report["pass"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_search(rows: list[dict], selected_offset: float, output: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["offset_m"])
    offsets = np.asarray([row["offset_m"] for row in ordered], dtype=float)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    axes[0].plot(offsets, [row["delay_gain_ms"] for row in ordered], "o-", lw=2)
    axes[0].axhline(5.0, color="#B23A48", ls="--", lw=1.2, label="5 ms 门槛")
    axes[0].set(ylabel="相对质心的时延收益 / ms", xlabel="协作偏移 / m")
    axes[0].legend(frameon=False)

    axes[1].plot(
        offsets,
        [100.0 * row["target_occupancy_fraction"] for row in ordered],
        "o-",
        color="#2E7D6E",
        lw=2,
        label="目标位置驻留占比",
    )
    axes[1].plot(
        offsets,
        [100.0 * row["target_reach_segment_fraction"] for row in ordered],
        "s--",
        color="#F28E2B",
        lw=1.8,
        label="阶段到位比例",
    )
    axes[1].set(ylabel="比例 / %", xlabel="协作偏移 / m")
    axes[1].legend(frameon=False)

    axes[2].plot(
        offsets,
        [row["power_delta_w"] for row in ordered],
        "o-",
        color="#7A5195",
        lw=2,
    )
    axes[2].set(ylabel="相对质心的功率增量 / W", xlabel="协作偏移 / m")
    for axis in axes:
        axis.axvline(selected_offset, color="#333333", ls=":", lw=1.1)
        axis.grid(alpha=0.25)
    fig.suptitle("协作偏移幅度搜索：收益、到位能力与软能耗代价", fontsize=13)
    fig.savefig(output / "01_偏移幅度搜索.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = resolve(args.config)
    checkpoint = resolve(args.checkpoint)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    offsets = [float(item.strip()) for item in args.offsets.split(",") if item.strip()]
    if offsets != [60.0, 90.0, 120.0]:
        raise ValueError("The frozen v36 offsets must be exactly 60,90,120 m.")

    search_rows = []
    for offset in offsets:
        arm_output = output / "search" / f"offset_{int(offset):03d}m"
        report = run_gate(
            config,
            checkpoint,
            arm_output,
            args.search_seeds,
            offset,
            args.device,
        )
        search_rows.append(summarize(report, offset, "search"))

    selected = max(
        search_rows,
        key=lambda row: (row["delay_gain_ms"], -row["offset_m"]),
    )
    selected_offset = float(selected["offset_m"])
    validation_output = output / "validation" / f"offset_{int(selected_offset):03d}m"
    validation_report = run_gate(
        config,
        checkpoint,
        validation_output,
        args.validation_seeds,
        selected_offset,
        args.device,
    )
    validation_row = summarize(validation_report, selected_offset, "validation")

    write_csv(output / "search_summary.csv", search_rows)
    write_csv(output / "validation_summary.csv", [validation_row])
    plot_search(search_rows, selected_offset, output)

    delay_pass = (
        validation_row["delay_gain_ms"] > 5.0
        and validation_row["delay_delta_ci_high_ms"] < 0.0
    )
    queue_pass = bool(
        validation_report["checks"]["queue_noninferior_to_centroid"]
    )
    final_pass = bool(delay_pass and queue_pass)
    payload = {
        "status": "PASS" if final_pass else "FAIL",
        "selection_rule": (
            "maximum search-set mean paired delay gain; exact ties choose the "
            "smaller offset"
        ),
        "offset_candidates_m": offsets,
        "search_seeds": args.search_seeds,
        "validation_seeds": args.validation_seeds,
        "selected_offset_m": selected_offset,
        "search": search_rows,
        "validation": validation_row,
        "checks": {
            "validation_delay_gain_gt_5ms": validation_row["delay_gain_ms"] > 5.0,
            "validation_delay_ci_excludes_zero": (
                validation_row["delay_delta_ci_high_ms"] < 0.0
            ),
            "validation_queue_noninferior": queue_pass,
        },
        "power_role": "soft_objective_component_and_descriptive_ledger_no_hard_gate",
        "config": str(config),
        "micro_checkpoint": str(checkpoint),
    }
    (output / "audit_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# v36 协作偏移幅度审计",
        "",
        f"**判决：{payload['status']}**",
        "",
        f"搜索集选中偏移：**{selected_offset:g} m**。",
        (
            "独立验证集相对质心的配对时延收益："
            f"**{validation_row['delay_gain_ms']:.3f} ms**，差值 95% CI "
            f"[{validation_row['delay_delta_ci_low_ms']:.3f}, "
            f"{validation_row['delay_delta_ci_high_ms']:.3f}] ms。"
        ),
        (
            "功率增量："
            f"{validation_row['power_delta_w']:.3f} W；软目标 J 差值："
            f"{validation_row['soft_j_delta']:.6f}（负值更优）。"
        ),
        (
            "目标位置驻留占比："
            f"{100.0 * validation_row['target_occupancy_fraction']:.1f}%；"
            "阶段到位比例："
            f"{100.0 * validation_row['target_reach_segment_fraction']:.1f}%。"
        ),
        "",
        "功率不设硬阈值；本审计仅以时延实用收益、统计方向和队列非劣性判决。",
    ]
    (output / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
