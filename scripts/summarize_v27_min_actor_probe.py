import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_V27 = (
    "result_archive/v27_min_actor_pilot_audit/fixed_probe/branch_raw.csv"
)
DEFAULT_V26 = (
    "result_archive/v26_joint_support_audit/fixed_probe_support/branch_raw.csv"
)
DEFAULT_AUDIT_SUMMARY = (
    "result_archive/v27_min_actor_pilot_audit/summary.json"
)
DEFAULT_OUTPUT = "result_archive/v27_min_actor_pilot_audit"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare v27 min-Actor and v26 Q1-Actor same-state probes using "
            "episode-level paired statistics."
        )
    )
    parser.add_argument("--v27-branch-csv", default=DEFAULT_V27)
    parser.add_argument("--v26-branch-csv", default=DEFAULT_V26)
    parser.add_argument("--audit-summary", default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return [mean, mean, mean]
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return [mean, mean - critical * sem, mean + critical * sem]


def same_sign(left, right):
    return float(np.sign(left) == np.sign(right))


def aggregate_episode(rows):
    grouped = {}
    for row in rows:
        key = (
            row["snapshot_policy"],
            int(row["episode_index"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)

    result = []
    for (policy, episode_index, seed), branch_rows in sorted(grouped.items()):
        if not all(row["trace_match"].lower() == "true" for row in branch_rows):
            raise RuntimeError(
                f"Counterfactual trace mismatch: {policy}, episode {episode_index}."
            )
        values = {
            "q1_diff": [],
            "q2_diff": [],
            "qmin_diff": [],
            "return_diff": [],
            "q1_agreement": [],
            "q2_agreement": [],
            "qmin_agreement": [],
        }
        for row in branch_rows:
            q1_diff = float(row["q1_actor"]) - float(row["q1_zero"])
            q2_diff = float(row["q2_actor"]) - float(row["q2_zero"])
            qmin_diff = float(row["q_actor_minus_zero"])
            return_diff = float(row["return_actor_minus_zero"])
            values["q1_diff"].append(q1_diff)
            values["q2_diff"].append(q2_diff)
            values["qmin_diff"].append(qmin_diff)
            values["return_diff"].append(return_diff)
            values["q1_agreement"].append(same_sign(q1_diff, return_diff))
            values["q2_agreement"].append(same_sign(q2_diff, return_diff))
            values["qmin_agreement"].append(same_sign(qmin_diff, return_diff))
        aggregate = {
            "snapshot_policy": policy,
            "episode_index": episode_index,
            "seed": seed,
            "branch_count": len(branch_rows),
        }
        for field, field_values in values.items():
            aggregate[f"{field}_mean"] = float(np.mean(field_values))
        result.append(aggregate)
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_version(rows):
    result = {}
    fields = (
        "q1_diff_mean",
        "q2_diff_mean",
        "qmin_diff_mean",
        "return_diff_mean",
        "q1_agreement_mean",
        "q2_agreement_mean",
        "qmin_agreement_mean",
    )
    for policy in ("actor", "centroid"):
        selected = [row for row in rows if row["snapshot_policy"] == policy]
        if not selected:
            raise RuntimeError(f"Probe is missing {policy} snapshots.")
        result[policy] = {
            "episodes": len(selected),
            **{
                f"{field}_ci95": mean_ci95([row[field] for row in selected])
                for field in fields
            },
        }
    return result


def paired_comparison(v27_rows, v26_rows):
    key_fields = ("snapshot_policy", "episode_index", "seed")
    v27 = {tuple(row[field] for field in key_fields): row for row in v27_rows}
    v26 = {tuple(row[field] for field in key_fields): row for row in v26_rows}
    if set(v27) != set(v26):
        raise RuntimeError("v27 and v26 probe episode keys do not match.")

    fields = (
        "q1_diff_mean",
        "q2_diff_mean",
        "qmin_diff_mean",
        "return_diff_mean",
        "q1_agreement_mean",
        "q2_agreement_mean",
        "qmin_agreement_mean",
    )
    paired_rows = []
    for key in sorted(v27):
        row = {
            "snapshot_policy": key[0],
            "episode_index": key[1],
            "seed": key[2],
        }
        for field in fields:
            row[f"{field}_v27"] = v27[key][field]
            row[f"{field}_v26"] = v26[key][field]
            row[f"{field}_v27_minus_v26"] = v27[key][field] - v26[key][field]
        paired_rows.append(row)

    summary = {}
    for policy in ("actor", "centroid"):
        selected = [
            row for row in paired_rows if row["snapshot_policy"] == policy
        ]
        summary[policy] = {
            f"{field}_v27_minus_v26_ci95": mean_ci95(
                [row[f"{field}_v27_minus_v26"] for row in selected]
            )
            for field in fields
        }
    return paired_rows, summary


def draw_figure(v27_rows, v26_rows, paired_rows, output_dir):
    versions = ("v26 Q1-Actor", "v27 min-Actor")
    version_rows = (v26_rows, v27_rows)
    policies = ("actor", "centroid")
    colors = {"v26 Q1-Actor": "#808080", "v27 min-Actor": "#2878B5"}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)

    x = np.arange(len(policies), dtype=np.float64)
    width = 0.34
    for index, (label, rows) in enumerate(zip(versions, version_rows)):
        means = []
        errors = []
        for policy in policies:
            values = [
                row["qmin_agreement_mean"]
                for row in rows
                if row["snapshot_policy"] == policy
            ]
            mean, low, high = mean_ci95(values)
            means.append(mean)
            errors.append([mean - low, high - mean])
        axes[0].bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=np.asarray(errors).T,
            capsize=4,
            color=colors[label],
            label=label,
            alpha=0.88,
        )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, policies)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Ranking agreement")
    axes[0].set_title("Conservative-Q ranking")
    axes[0].legend(frameon=False, fontsize=8)

    centroid = [row for row in v27_rows if row["snapshot_policy"] == "centroid"]
    series = (
        ("Q1", "q1_diff_mean", "#C82423"),
        ("Q2", "q2_diff_mean", "#2878B5"),
        ("min Q", "qmin_diff_mean", "#7A5195"),
        ("True return", "return_diff_mean", "#2E8B57"),
    )
    for index, (label, field, color) in enumerate(series):
        mean, low, high = mean_ci95([row[field] for row in centroid])
        axes[1].bar(
            index,
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            capsize=4,
            color=color,
            alpha=0.88,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(series)), [item[0] for item in series])
    axes[1].set_ylabel("Actor action - zero action")
    axes[1].set_title("v27 centroid-state ranking")

    paired_centroid = [
        row for row in paired_rows if row["snapshot_policy"] == "centroid"
    ]
    agreement_delta = np.asarray(
        [row["qmin_agreement_mean_v27_minus_v26"] for row in paired_centroid]
    )
    axes[2].scatter(
        np.linspace(-0.08, 0.08, agreement_delta.size),
        agreement_delta,
        s=30,
        alpha=0.75,
        color="#2878B5",
    )
    mean, low, high = mean_ci95(agreement_delta)
    axes[2].errorbar(
        [0.0], [mean], yerr=[[mean - low], [high - mean]],
        fmt="D", color="black", capsize=5
    )
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_xlim(-0.25, 0.25)
    axes[2].set_xticks([])
    axes[2].set_ylabel("v27 - v26 agreement")
    axes[2].set_title("Paired centroid-state improvement")

    fig.savefig(output_dir / "min_actor_fixed_probe.png", dpi=300)
    fig.savefig(output_dir / "min_actor_fixed_probe.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v27_rows = aggregate_episode(read_rows(args.v27_branch_csv))
    v26_rows = aggregate_episode(read_rows(args.v26_branch_csv))
    paired_rows, paired_summary = paired_comparison(v27_rows, v26_rows)
    v27_summary = summarize_version(v27_rows)
    v26_summary = summarize_version(v26_rows)

    centroid_delta = paired_summary["centroid"][
        "qmin_agreement_mean_v27_minus_v26_ci95"
    ]
    mechanism_pass = bool(centroid_delta[1] > 0.0)
    mechanism = {
        "criterion": (
            "Centroid-state conservative-Q ranking agreement v27-v26 has "
            "a paired 95% CI lower bound above zero."
        ),
        "verdict": "PASS" if mechanism_pass else "FAIL",
        "criterion_pass": mechanism_pass,
        "statistical_unit": "episode",
        "branch_points_are_aggregated_within_episode": True,
        "v27": v27_summary,
        "v26": v26_summary,
        "paired_v27_minus_v26": paired_summary,
    }
    write_csv(output_dir / "v27_probe_episode_summary.csv", v27_rows)
    write_csv(output_dir / "v26_probe_episode_summary.csv", v26_rows)
    write_csv(output_dir / "probe_paired_v27_minus_v26.csv", paired_rows)
    with (output_dir / "mechanism_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(mechanism, handle, ensure_ascii=False, indent=2)

    audit_summary_path = Path(args.audit_summary)
    with audit_summary_path.open("r", encoding="utf-8") as handle:
        audit_summary = json.load(handle)
    audit_summary["mechanism_verdict"] = mechanism["verdict"]
    audit_summary["mechanism_criterion"] = mechanism
    audit_summary["overall_verdict"] = (
        "PASS"
        if audit_summary["behavior_verdict"] == "PASS" and mechanism_pass
        else "FAIL"
    )
    with audit_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_summary, handle, ensure_ascii=False, indent=2)
    draw_figure(v27_rows, v26_rows, paired_rows, output_dir)

    print(json.dumps(mechanism, ensure_ascii=False, indent=2))
    print(f"Overall verdict: {audit_summary['overall_verdict']}")


if __name__ == "__main__":
    main()
