import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


DEFAULT_ROOT = Path("result_archive/v29_online_nstep_pilot")
DEFAULT_CONTROL_DIAGNOSTICS = Path(
    "results/tcom_dynamic_u2u_sparse_v29_control_n1/pilot/seed1/"
    "joint_td3_sac/nstep_replay_diagnostics.csv"
)
DEFAULT_TREATMENT_DIAGNOSTICS = Path(
    "results/tcom_dynamic_u2u_sparse_v29_treatment_n5/pilot/seed1/"
    "joint_td3_sac/nstep_replay_diagnostics.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the frozen v29 decision table to completed audits."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--control-diagnostics", default=str(DEFAULT_CONTROL_DIAGNOSTICS)
    )
    parser.add_argument(
        "--treatment-diagnostics", default=str(DEFAULT_TREATMENT_DIAGNOSTICS)
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size <= 1:
        return [mean, mean, mean]
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return [mean, mean - critical * sem, mean + critical * sem]


def age_effect(path):
    rows = read_csv(path)
    keyed = {
        (int(row["episode"]), row["age_bucket"]): row
        for row in rows
        if int(row["episode"]) >= 60
    }
    episodes = sorted(
        episode
        for episode, bucket in keyed
        if bucket == "0-4" and (episode, ">=30") in keyed
    )
    fields = (
        "abs_y5_minus_y1_mean",
        "q_y5_gap_abs_mean",
        "policy_mismatch_mean",
    )
    result = {"episodes": episodes, "n_episodes": len(episodes)}
    for field in fields:
        deltas = []
        for episode in episodes:
            young = float(keyed[(episode, "0-4")][field])
            old = float(keyed[(episode, ">=30")][field])
            if np.isfinite(young) and np.isfinite(old):
                deltas.append(old - young)
        result[f"old_minus_young_{field}_ci95"] = mean_ci95(deltas)
    target_ci = result[
        "old_minus_young_abs_y5_minus_y1_mean_ci95"
    ]
    mismatch_ci = result[
        "old_minus_young_policy_mismatch_mean_ci95"
    ]
    result["clear_age_effect"] = bool(
        target_ci[1] > 0.0 or mismatch_ci[1] > 0.0
    )
    return result


def render_verdict_markdown(payload):
    mechanism = payload["mechanism"]
    final = payload["final_evaluation"]
    reproduction_ci = final["control_reproduction"][
        "task_delay_weighted_ms"
    ]["control_n1_minus_historical_v24_ci95"]
    delay_ci = final["n5_minus_n1"]["task_delay_weighted_ms"][
        "treatment_n5_minus_control_n1_ci95"
    ]
    mechanism_ci = mechanism["centroid"][
        "q1_agreement_n5_minus_n1_ci95"
    ]

    def format_mean_ci(values, unit=""):
        mean, low, high = (float(value) for value in values)
        suffix = f" {unit}" if unit else ""
        return f"{mean:.6f}{suffix}, 95% CI [{low:.6f}, {high:.6f}]{suffix}"

    return "\n".join(
        (
            "# v29 online n-step pilot verdict",
            "",
            f"- Frozen verdict: **{payload['verdict']}**",
            "- Control reproduction delay difference: "
            f"`{format_mean_ci(reproduction_ci, 'ms')}`",
            "- Centroid Q1 agreement difference (n5-n1): "
            f"`{format_mean_ci(mechanism_ci)}`",
            "- Final delay difference (n5-n1): "
            f"`{format_mean_ci(delay_ci, 'ms')}`",
            f"- n=1 power UCB95: `{final['control_power_ucb95_w']:.3f} W`",
            f"- n=5 power UCB95: `{final['treatment_power_ucb95_w']:.3f} W`",
            "",
            "The statistical unit for the mechanism test is the evaluation "
            "episode seed. Fork points and checkpoints were averaged within "
            "each seed before the paired confidence interval was computed.",
            "",
        )
    )


def main():
    args = parse_args()
    root = Path(args.root)
    with (root / "fixed_probe_progression" / "summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        mechanism = json.load(handle)
    with (root / "final_crn_evaluation" / "summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        final = json.load(handle)

    ages = {
        "control_n1": age_effect(Path(args.control_diagnostics)),
        "treatment_n5": age_effect(Path(args.treatment_diagnostics)),
    }
    if not final["control_reproduced"]:
        verdict = "CONTROL_NOT_REPRODUCED"
    elif not final["constraint_pass"]:
        verdict = "CONSTRAINT_FAIL"
    elif mechanism["mechanism_pass"] and final["delay_improvement_pass"]:
        verdict = "FULL_PASS"
    elif mechanism["mechanism_pass"]:
        verdict = "MECHANISM_ONLY"
    elif ages["treatment_n5"]["clear_age_effect"]:
        verdict = "OFF_POLICY_STALENESS_CANDIDATE"
    else:
        verdict = "BOOTSTRAP_HYPOTHESIS_WEAKENED"
    payload = {
        "verdict": verdict,
        "mechanism": mechanism,
        "final_evaluation": final,
        "replay_age_diagnostics": ages,
    }
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with (root / "verdict.md").open("w", encoding="utf-8") as handle:
        handle.write(render_verdict_markdown(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
