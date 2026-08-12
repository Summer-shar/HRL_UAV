import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from search_position_oracle import load_checkpoint_components


OFFSETS_M = (10.0, 20.0, 30.0, 40.0)
RESIDUAL_CAPS_M = (5.0, 10.0, 15.0, 20.0)
ACTIVE_BLOCK_LENGTHS = (10, 9, 9)
POWER_BUDGET_W = 840.0
CROSS_LAYER_RESERVE_W = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analytically pre-screen the frozen P1-prime grid."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/tcom_dynamic_u2u_sparse_v23_target_noise/pilot/"
            "seed1/joint_td3_sac/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--power-audit-dir",
        default="result_archive/load_switch_gate_power_audit",
    )
    parser.add_argument(
        "--output-dir",
        default="result_archive/p1_prime_prescreen",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty P1-prime grid.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archived_metric(rows, policy, metric):
    matches = [
        row for row in rows
        if row["policy"] == policy and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one archived row for {policy}/{metric}, got {len(matches)}."
        )
    return matches[0]


def archived_delta_metric(rows, metric):
    matches = [row for row in rows if row["metric"] == metric]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one archived delta row for {metric}, got {len(matches)}."
        )
    return matches[0]


def one_sided_ucb95(mean, std, sample_count):
    if sample_count <= 1:
        return float(mean)
    return float(
        mean
        + stats.t.ppf(0.95, sample_count - 1)
        * std
        / math.sqrt(sample_count)
    )


def split_traversal(offset_m, residual_cap_m):
    offset_m = float(offset_m)
    residual_cap_m = float(residual_cap_m)
    if offset_m <= 0.0 or residual_cap_m <= 0.0:
        raise ValueError("Offset and residual cap must be positive.")
    full_legs = int(math.floor(offset_m / residual_cap_m + 1e-12))
    remainder = offset_m - full_legs * residual_cap_m
    legs = [residual_cap_m] * full_legs
    if remainder > 1e-9:
        legs.append(remainder)
    return legs


def integrated_offset_exposure(offset_m, residual_cap_m):
    return float(
        sum(
            sum(
                min(step * residual_cap_m, offset_m)
                for step in range(1, block_length + 1)
            )
            for block_length in ACTIVE_BLOCK_LENGTHS
        )
    )


def maneuver_reserve(base_env, offset_m, residual_cap_m):
    legs = split_traversal(offset_m, residual_cap_m)
    deployment_dt = float(base_env.deployment_dt)
    episode_duration_s = 30.0 * float(base_env.macro_dt)
    hover_power = float(base_env._compute_tcom_flying_power([0.0])[0])
    steady_energy_j = 0.0
    transient_energy_j = 0.0
    for distance_m in legs:
        deployment_speed = distance_m / deployment_dt
        steady_energy_j += (
            float(base_env._compute_tcom_flying_power([deployment_speed])[0])
            - hover_power
        ) * deployment_dt
        transient_energy_j += float(
            base_env._compute_deployment_transient_energy(
                [deployment_speed]
            )[0]
        )
    traversal_count = 5
    steady_power_w = traversal_count * steady_energy_j / episode_duration_s
    transient_power_w = (
        traversal_count * transient_energy_j / episode_duration_s
    )
    return {
        "legs_m": legs,
        "steady_reserve_w": float(steady_power_w),
        "transient_reserve_w": float(transient_power_w),
        "maneuver_reserve_w": float(steady_power_w + transient_power_w),
    }


def build_grid(base_env, baseline, paired_delta_std_w):
    rows = []
    p0_ucb95_w = one_sided_ucb95(
        baseline["mean"], baseline["std"], baseline["sample_count"]
    )
    for offset_m in OFFSETS_M:
        for residual_cap_m in RESIDUAL_CAPS_M:
            reserve = maneuver_reserve(base_env, offset_m, residual_cap_m)
            legs = reserve["legs_m"]
            timing_feasible = len(legs) <= min(ACTIVE_BLOCK_LENGTHS)
            predicted_ucb95_w = (
                p0_ucb95_w
                + reserve["maneuver_reserve_w"]
                + CROSS_LAYER_RESERVE_W
            )
            margin_w = POWER_BUDGET_W - predicted_ucb95_w
            analytically_safe = bool(
                timing_feasible and margin_w >= paired_delta_std_w
            )
            rows.append(
                {
                    "offset_m": offset_m,
                    "residual_cap_per_macro_m": residual_cap_m,
                    "macro_average_approach_mps": (
                        residual_cap_m / float(base_env.macro_dt)
                    ),
                    "deployment_speed_full_leg_mps": (
                        residual_cap_m / float(base_env.deployment_dt)
                    ),
                    "legs_per_traversal": len(legs),
                    "leg_pattern_m": ";".join(f"{value:g}" for value in legs),
                    "timing_feasible": timing_feasible,
                    "integrated_offset_exposure_m_periods": (
                        integrated_offset_exposure(offset_m, residual_cap_m)
                    ),
                    "steady_reserve_w": reserve["steady_reserve_w"],
                    "transient_reserve_w": reserve["transient_reserve_w"],
                    "maneuver_reserve_w": reserve["maneuver_reserve_w"],
                    "cross_layer_reserve_w": CROSS_LAYER_RESERVE_W,
                    "p0_ucb95_w": p0_ucb95_w,
                    "predicted_p1_prime_ucb95_w": predicted_ucb95_w,
                    "budget_margin_w": margin_w,
                    "required_margin_w": paired_delta_std_w,
                    "analytically_safe": analytically_safe,
                }
            )
    return rows


def select_candidate(rows):
    safe = [row for row in rows if row["analytically_safe"]]
    if not safe:
        raise RuntimeError("No P1-prime grid point satisfies the frozen rule.")
    return max(
        safe,
        key=lambda row: (
            float(row["offset_m"]),
            float(row["residual_cap_per_macro_m"]),
        ),
    )


def plot_grid(rows, selected, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), constrained_layout=True)
    colors = {10.0: "#2878B5", 20.0: "#54A24B", 30.0: "#E17C05", 40.0: "#C82423"}
    for offset in OFFSETS_M:
        group = sorted(
            (row for row in rows if row["offset_m"] == offset),
            key=lambda row: row["macro_average_approach_mps"],
        )
        x = [row["macro_average_approach_mps"] for row in group]
        y = [row["predicted_p1_prime_ucb95_w"] for row in group]
        axes[0].plot(x, y, marker="o", label=f"offset {offset:g} m", color=colors[offset])
    axes[0].axhline(POWER_BUDGET_W, color="black", linestyle="--", linewidth=1.0, label="840 W budget")
    axes[0].axhline(
        POWER_BUDGET_W - float(selected["required_margin_w"]),
        color="#666666",
        linestyle=":",
        linewidth=1.0,
        label="required safety line",
    )
    axes[0].scatter(
        [selected["macro_average_approach_mps"]],
        [selected["predicted_p1_prime_ucb95_w"]],
        marker="*",
        s=180,
        color="black",
        zorder=5,
        label="selected",
    )
    axes[0].set_xlabel("Residual approach rate averaged over 10 s (m/s)")
    axes[0].set_ylabel("Predicted one-sided power UCB95 (W)")
    axes[0].set_title("Analytic power screen")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.45)

    safe = [row for row in rows if row["analytically_safe"]]
    unsafe = [row for row in rows if not row["analytically_safe"]]
    axes[1].scatter(
        [row["integrated_offset_exposure_m_periods"] for row in safe],
        [row["predicted_p1_prime_ucb95_w"] for row in safe],
        color="#2878B5",
        label="safe",
    )
    axes[1].scatter(
        [row["integrated_offset_exposure_m_periods"] for row in unsafe],
        [row["predicted_p1_prime_ucb95_w"] for row in unsafe],
        facecolors="none",
        edgecolors="#C82423",
        label="rejected",
    )
    axes[1].scatter(
        [selected["integrated_offset_exposure_m_periods"]],
        [selected["predicted_p1_prime_ucb95_w"]],
        marker="*",
        s=180,
        color="black",
        zorder=5,
    )
    axes[1].axhline(POWER_BUDGET_W, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Integrated residual-offset exposure (m-periods)")
    axes[1].set_ylabel("Predicted one-sided power UCB95 (W)")
    axes[1].set_title("Geometry exposure versus power")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.45)
    fig.savefig(output_dir / "prescreen_grid.png", dpi=300)
    fig.savefig(output_dir / "prescreen_grid.pdf")
    plt.close(fig)


def render_report(baseline, delta_std, selected, safe_count):
    return "\n".join(
        (
            "# P1-prime analytic pre-screen",
            "",
            "## Frozen selection",
            "",
            f"- Offset: **{selected['offset_m']:.0f} m**",
            "- Residual movement cap: "
            f"**{selected['residual_cap_per_macro_m']:.0f} m per 10 s macro period**",
            "- Macro-average approach rate: "
            f"**{selected['macro_average_approach_mps']:.1f} m/s**",
            "- Actual 1 s deployment speed for a full leg: "
            f"**{selected['deployment_speed_full_leg_mps']:.1f} m/s**",
            "",
            "## Power screen",
            "",
            f"- Archived P0 mean: `{baseline['mean']:.3f} W`",
            f"- Archived P0 one-sided UCB95: `{selected['p0_ucb95_w']:.3f} W`",
            f"- Analytic maneuver reserve: `{selected['maneuver_reserve_w']:.3f} W`",
            f"- Fixed cross-layer reserve: `{selected['cross_layer_reserve_w']:.3f} W`",
            f"- Predicted P1-prime UCB95: `{selected['predicted_p1_prime_ucb95_w']:.3f} W`",
            f"- Remaining budget margin: `{selected['budget_margin_w']:.3f} W`",
            f"- Required margin: `{delta_std:.3f} W`",
            f"- Safe grid points: `{safe_count}/{len(OFFSETS_M) * len(RESIDUAL_CAPS_M)}`",
            "",
            "The selected parameters are frozen before P1-prime replay. No grid fallback is allowed.",
            "",
        )
    )


def main():
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    power_audit_dir = project_path(args.power_audit_dir)
    policy_metrics_path = power_audit_dir / "power_components_by_policy.csv"
    delta_metrics_path = power_audit_dir / "power_component_paired_deltas.csv"
    protocol_path = PROJECT_ROOT / "docs" / "p1_prime_prescreen_protocol.md"

    policy_metrics = read_csv(policy_metrics_path)
    delta_metrics = read_csv(delta_metrics_path)
    p0 = archived_metric(policy_metrics, "P0", "average_power_w")
    baseline = {
        "mean": float(p0["mean"]),
        "std": float(p0["std"]),
        "sample_count": int(p0["sample_count"]),
    }
    delta = archived_delta_metric(delta_metrics, "average_power_w")
    paired_delta_std_w = float(delta["std"])

    _path, _args, env, _actor, _micro, _device = load_checkpoint_components(
        args.checkpoint, args.device
    )
    rows = build_grid(env.env, baseline, paired_delta_std_w)
    selected = select_candidate(rows)
    write_csv(output_dir / "grid.csv", rows)

    payload = {
        "status": "FROZEN_FOR_SINGLE_REPLAY",
        "selected": selected,
        "selection_rule": "max_offset_then_max_residual_cap_among_safe",
        "grid": {
            "offsets_m": list(OFFSETS_M),
            "residual_caps_per_macro_m": list(RESIDUAL_CAPS_M),
            "active_block_lengths": list(ACTIVE_BLOCK_LENGTHS),
        },
        "baseline": baseline,
        "paired_original_p1_minus_p0_power_std_w": paired_delta_std_w,
        "power_budget_w": POWER_BUDGET_W,
        "cross_layer_reserve_w": CROSS_LAYER_RESERVE_W,
        "provenance": {
            "protocol_sha256": sha256_file(protocol_path),
            "policy_metrics_sha256": sha256_file(policy_metrics_path),
            "delta_metrics_sha256": sha256_file(delta_metrics_path),
            "checkpoint": str(project_path(args.checkpoint)),
        },
    }
    (output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    safe_count = sum(bool(row["analytically_safe"]) for row in rows)
    (output_dir / "prescreen_report.md").write_text(
        render_report(baseline, paired_delta_std_w, selected, safe_count),
        encoding="utf-8",
    )
    plot_grid(rows, selected, output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

