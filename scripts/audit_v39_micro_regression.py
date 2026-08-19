"""Audit the v39 micro-SAC regression against the known-good v33 policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hrl_uav.load_switching import build_load_switch_schedule
from run_load_switch_gate import run_gate_episode
from run_v34_cooperative_mobility_gate import (
    V34MacroPolicy,
    load_frozen_micro_policy,
    make_gate_env,
    mean_ci,
    parse_seed_list,
    queue_metrics,
)


CONFIG = ROOT / "configs" / "tcom_v39_calibrated_soft_objective.json"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v39_micro_regression_audit"
CHECKPOINTS = {
    "v33_final": (
        ROOT
        / "results"
        / "tcom_v33_soft_energy_random_mobility"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "checkpoints"
        / "final.pt"
    ),
    "v39_best": (
        ROOT
        / "results"
        / "tcom_v39_calibrated_soft_objective"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "checkpoints"
        / "best.pt"
    ),
    "v39_final": (
        ROOT
        / "results"
        / "tcom_v39_calibrated_soft_objective"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "checkpoints"
        / "final.pt"
    ),
}
TRAIN_LOGS = {
    "v33": (
        ROOT
        / "results"
        / "tcom_v33_soft_energy_random_mobility"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "train_centralized.csv"
    ),
    "v39": (
        ROOT
        / "results"
        / "tcom_v39_calibrated_soft_objective"
        / "formal"
        / "seed1"
        / "micro_pretrain"
        / "train_centralized.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="6000-6019")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def read_training_log(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    fields = (
        "episode",
        "sac_alpha",
        "micro_reward",
        "sac_critic_loss",
        "micro_q_total_mean",
        "micro_cost_delay_mean",
    )
    return {
        field: np.asarray(
            [float(row[field]) if row[field] else np.nan for row in rows],
            dtype=np.float64,
        )
        for field in fields
    }


def rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        if np.any(np.isfinite(segment)):
            result[index] = np.nanmean(segment)
    return result


def configure_chinese_font() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "STSong"):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_diagnostics(logs: dict[str, dict], output: Path) -> None:
    configure_chinese_font()
    colors = {"v33": "#2878B5", "v39": "#D9534F"}
    labels = {"v33": "v33（已知良好）", "v39": "v39（当前重训）"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    for name, data in logs.items():
        episode = data["episode"]
        axes[0].plot(episode, data["sac_alpha"], color=colors[name], label=labels[name])
        axes[1].plot(
            episode,
            rolling_mean(data["micro_reward"]),
            color=colors[name],
            label=labels[name],
        )
        axes[2].plot(
            episode,
            rolling_mean(data["sac_critic_loss"]),
            color=colors[name],
            label=labels[name],
        )
    axes[0].set_title("SAC 熵温度")
    axes[0].set_ylabel("alpha")
    axes[1].set_title("微观训练奖励（20轮滑动均值）")
    axes[1].set_ylabel("累计奖励")
    axes[2].set_title("SAC Critic 损失（20轮滑动均值）")
    axes[2].set_ylabel("Critic loss")
    for axis in axes:
        axis.set_xlabel("训练轮次")
        axis.grid(alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "01_alpha_reward_critic对照.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def training_summary(data: dict[str, np.ndarray]) -> dict:
    mask = (data["episode"] >= 241) & (data["episode"] <= 300)
    return {
        "alpha_start": float(data["sac_alpha"][0]),
        "alpha_final": float(data["sac_alpha"][-1]),
        "late_reward_mean": float(np.mean(data["micro_reward"][mask])),
        "late_reward_std": float(np.std(data["micro_reward"][mask], ddof=1)),
        "late_critic_loss_mean": float(
            np.nanmean(data["sac_critic_loss"][mask])
        ),
        "late_queue_mean": float(np.mean(data["micro_q_total_mean"][mask])),
        "late_delay_cost_std": float(
            np.std(data["micro_cost_delay_mean"][mask], ddof=1)
        ),
    }


def evaluate_checkpoints(
    config: dict,
    seeds: list[int],
    device: str,
) -> list[dict]:
    phases = np.asarray(config["load_switch_phases_mbps"], dtype=np.float64)
    rows: list[dict] = []
    trace_hashes: dict[int, dict[str, str]] = {seed: {} for seed in seeds}
    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        checkpoint_args, micro_policy, resolved_device = load_frozen_micro_policy(
            checkpoint_path,
            config,
            device,
        )
        print(
            f"checkpoint={checkpoint_name}; path={checkpoint_path}; "
            f"device={resolved_device}"
        )
        for seed in seeds:
            schedule = build_load_switch_schedule(
                mode=config["load_switch_mode"],
                phases_mbps=phases,
                macro_steps=int(config["macro_steps"]),
                episode_seed=seed,
                dwell_min=int(config["load_switch_dwell_min"]),
                dwell_max=int(config["load_switch_dwell_max"]),
            )
            env = make_gate_env(checkpoint_args)
            macro_policy = V34MacroPolicy("centroid", 60.0, 10.0)
            result, trace = run_gate_episode(
                env=env,
                checkpoint_args=checkpoint_args,
                micro_policy=micro_policy,
                policy_name="centroid",
                seed=seed,
                macro_steps=int(config["macro_steps"]),
                load_switch_schedule=schedule,
                macro_policy=macro_policy,
                macro_metadata_provider=macro_policy.metadata,
            )
            result.update(queue_metrics(trace, float(config["macro_period_s"])))
            row = {
                "checkpoint": checkpoint_name,
                "seed": seed,
                "task_delay_weighted_ms": float(result["task_delay_weighted_ms"]),
                "mean_queue_mbits": float(result["mean_queue_mbits"]),
                "terminal_queue_mbits": float(result["terminal_queue_mbits"]),
                "average_power_w": float(result["average_power_w"]),
                "external_trace_hash": str(result["external_trace_hash"]),
            }
            rows.append(row)
            trace_hashes[seed][checkpoint_name] = row["external_trace_hash"]
            print(
                f"seed={seed} checkpoint={checkpoint_name}: "
                f"delay={row['task_delay_weighted_ms']:.2f}ms, "
                f"queue={row['mean_queue_mbits']:.2f}Mbit"
            )
    for seed, hashes in trace_hashes.items():
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"CRN mismatch for seed {seed}: {hashes}")
    return rows


def summarize_evaluation(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    by_checkpoint = {
        name: [row for row in rows if row["checkpoint"] == name]
        for name in CHECKPOINTS
    }
    old_by_seed = {row["seed"]: row for row in by_checkpoint["v33_final"]}
    for name, group in by_checkpoint.items():
        delays = np.asarray(
            [row["task_delay_weighted_ms"] for row in group], dtype=np.float64
        )
        queues = np.asarray(
            [row["mean_queue_mbits"] for row in group], dtype=np.float64
        )
        powers = np.asarray(
            [row["average_power_w"] for row in group], dtype=np.float64
        )
        paired = np.asarray(
            [
                row["task_delay_weighted_ms"]
                - old_by_seed[row["seed"]]["task_delay_weighted_ms"]
                for row in group
            ],
            dtype=np.float64,
        )
        summary[name] = {
            "delay_ms": mean_ci(delays),
            "mean_queue_mbits": mean_ci(queues),
            "average_power_w": mean_ci(powers),
            "delay_delta_vs_v33_ms": mean_ci(paired),
        }
    return summary


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = parse_seed_list(args.seeds)
    for path in (*CHECKPOINTS.values(), *TRAIN_LOGS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    logs = {name: read_training_log(path) for name, path in TRAIN_LOGS.items()}
    plot_training_diagnostics(logs, output)
    rows = evaluate_checkpoints(config, seeds, args.device)
    write_rows(output / "per_seed_checkpoint_comparison.csv", rows)
    report = {
        "config": str(CONFIG),
        "seeds": seeds,
        "checkpoints": {name: str(path) for name, path in CHECKPOINTS.items()},
        "training_diagnostics": {
            name: training_summary(data) for name, data in logs.items()
        },
        "checkpoint_evaluation": summarize_evaluation(rows),
        "external_trace_hash_manifest_sha256": hashlib.sha256(
            "\n".join(
                f"{row['seed']}:{row['checkpoint']}:{row['external_trace_hash']}"
                for row in rows
            ).encode("utf-8")
        ).hexdigest(),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
