"""Matched zero-training gate for the v37 load-imbalance treatment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = ROOT / "scripts" / "run_v34_cooperative_mobility_gate.py"
BASE_CONFIG = ROOT / "configs" / "tcom_v35_reference_like_mobility.json"
TREATMENT_CONFIG = ROOT / "configs" / "tcom_v37_imbalanced_load_gate.json"
CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v33_soft_energy_random_mobility"
    / "formal"
    / "seed1"
    / "micro_pretrain"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v37_load_imbalance_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="5000-5019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def run_gate(config: Path, output: Path, seeds: str, device: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--config",
            str(config),
            "--checkpoint",
            str(CHECKPOINT),
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


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in tuple(row):
            if key not in {"seed", "policy", "external_trace_hash"}:
                row[key] = float(row[key])
    return rows


def mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    half = (
        float(student_t.ppf(0.975, values.size - 1) * std / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return {
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "std": std,
        "n": int(values.size),
        "values": values.tolist(),
    }


def paired_delta(rows: list[dict], field: str) -> dict:
    by = {
        (str(row["policy"]), int(row["seed"])): float(row[field])
        for row in rows
    }
    seeds = sorted(seed for policy, seed in by if policy == "centroid")
    values = np.asarray(
        [by[("cooperative", seed)] - by[("centroid", seed)] for seed in seeds],
        dtype=np.float64,
    )
    result = mean_ci(values)
    result["seeds"] = seeds
    result["cooperative_better_count"] = int(np.sum(values < 0.0))
    return result


def policy_mean(rows: list[dict], policy: str, field: str) -> float:
    return float(np.mean([
        float(row[field]) for row in rows if str(row["policy"]) == policy
    ]))


def scenario_summary(rows: list[dict]) -> dict:
    delay = paired_delta(rows, "task_delay_weighted_ms")
    queue = paired_delta(rows, "mean_queue_mbits")
    power = paired_delta(rows, "average_power_w")
    soft_j = paired_delta(rows, "soft_objective_j")
    return {
        "delay_delta_ms": delay,
        "delay_gain_ms": -float(delay["mean"]),
        "queue_delta_mbits": queue,
        "power_delta_w": power,
        "soft_j_delta": soft_j,
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


def plot_report(base: dict, treatment: dict, did: dict, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    labels = ["基准负载", "增强不均衡"]
    gains = [base["delay_gain_ms"], treatment["delay_gain_ms"]]
    axes[0].bar(labels, gains, color=["#7F8C8D", "#2E7D6E"], width=0.58)
    axes[0].axhline(5.0, color="#B23A48", ls="--", lw=1.2, label="5 ms 门槛")
    axes[0].set_ylabel("协作相对质心的时延收益 / ms")
    axes[0].legend(frameon=False)

    j_values = [base["soft_j_delta"]["mean"], treatment["soft_j_delta"]["mean"]]
    axes[1].bar(labels, j_values, color=["#7F8C8D", "#2E7D6E"], width=0.58)
    axes[1].axhline(0.0, color="#222222", lw=1.0)
    axes[1].set_ylabel("协作相对质心的软目标 J 差值\n负值表示协作更优")

    did_values = np.asarray(did["values"], dtype=np.float64)
    axes[2].scatter(
        np.ones_like(did_values), did_values, color="#2878B5", alpha=0.78, s=25
    )
    axes[2].errorbar(
        [1.0],
        [did["mean"]],
        yerr=[[did["mean"] - did["ci_low"]], [did["ci_high"] - did["mean"]]],
        fmt="D",
        color="#B23A48",
        capsize=5,
        markersize=7,
    )
    axes[2].axhline(0.0, color="#222222", lw=1.0)
    axes[2].set_xticks([1.0], ["负载不均衡带来的\n协作收益增量"])
    axes[2].set_ylabel("配对差分 / ms")
    axes[2].set_xlim(0.65, 1.35)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("负载空间不均衡零训练门禁", fontsize=13)
    fig.savefig(output / "01_负载不均衡门禁.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    base_output = output / "base_load"
    treatment_output = output / "imbalanced_load"
    run_gate(BASE_CONFIG, base_output, args.seeds, args.device)
    run_gate(TREATMENT_CONFIG, treatment_output, args.seeds, args.device)

    base_rows = read_rows(base_output / "per_seed_raw.csv")
    treatment_rows = read_rows(treatment_output / "per_seed_raw.csv")
    base = scenario_summary(base_rows)
    treatment = scenario_summary(treatment_rows)
    base_delta = np.asarray(base["delay_delta_ms"]["values"], dtype=np.float64)
    treatment_delta = np.asarray(
        treatment["delay_delta_ms"]["values"], dtype=np.float64
    )
    # Positive values mean load imbalance increased the cooperative delay gain.
    gain_difference = mean_ci(base_delta - treatment_delta)

    centroid_queue = policy_mean(treatment_rows, "centroid", "mean_queue_mbits")
    queue_margin = 0.05 * centroid_queue
    checks = {
        "delay_gain_gt_5ms": treatment["delay_gain_ms"] > 5.0,
        "delay_ci_excludes_zero": (
            treatment["delay_delta_ms"]["ci_high"] < 0.0
        ),
        "soft_j_mean_improves": treatment["soft_j_delta"]["mean"] < 0.0,
        "soft_j_ci_excludes_zero": treatment["soft_j_delta"]["ci_high"] < 0.0,
        "queue_noninferior": treatment["queue_delta_mbits"]["ci_high"] < queue_margin,
    }
    passed = bool(all(checks.values()))
    report = {
        "status": "PASS" if passed else "FAIL",
        "seeds": args.seeds,
        "base_load_mbps": [12.5, 5.5, 3.5],
        "imbalanced_load_mbps": [14.5, 4.5, 2.5],
        "total_load_mbps": 21.5,
        "cooperative_offset_m": 60.0,
        "base": base,
        "treatment": treatment,
        "cooperative_gain_difference_ms": gain_difference,
        "queue_noninferiority_margin_mbits": queue_margin,
        "checks": checks,
        "objective_alignment": {
            "macro_reward_mode": "strict_global",
            "macro_shaping_weight": 0.0,
            "macro_move_penalty_weight": 0.0,
            "delay_weight": 0.9,
            "energy_weight": 0.1,
        },
    }
    (output / "gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_report(base, treatment, gain_difference, output)

    delay = treatment["delay_delta_ms"]
    soft_j = treatment["soft_j_delta"]
    lines = [
        "# v37 负载不均衡零训练门禁结果",
        "",
        f"**判决：{report['status']}**",
        "",
        (
            "增强不均衡负载下，60 m 协作相对质心的时延收益为 "
            f"{treatment['delay_gain_ms']:.3f} ms，差值 95% CI "
            f"[{delay['ci_low']:.3f}, {delay['ci_high']:.3f}] ms。"
        ),
        (
            "软目标 J 差值为 "
            f"{soft_j['mean']:.6f}，95% CI "
            f"[{soft_j['ci_low']:.6f}, {soft_j['ci_high']:.6f}]；负值表示协作更优。"
        ),
        (
            "相对匹配基准负载，协作时延收益增加 "
            f"{gain_difference['mean']:.3f} ms，95% CI "
            f"[{gain_difference['ci_low']:.3f}, {gain_difference['ci_high']:.3f}] ms。"
        ),
        "",
        "## 判据",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] `{name}`")
    (output / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# v37 负载不均衡门禁归档\n\n"
        "最终裁决以本目录 `verdict.md` 和 `gate_report.json` 为准。"
        "`base_load/` 与 `imbalanced_load/` 内的 `gate_verdict.md` 是通用 v34 "
        "重放器按旧时延门槛生成的组件报告，未包含 v37 新增的软目标 J 主判据，"
        "不得单独作为 v37 PASS/FAIL 使用。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
