import argparse
import csv
import json
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def read_rows(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training log: {path}")
    return rows


def number(row, key):
    value = float(row[key])
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite {key}: {row[key]}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "diagnostic", "pilot", "formal"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results-version", default="joint_hrl_v1")
    args = parser.parse_args()

    root = os.path.join(
        PROJECT_ROOT,
        "results",
        args.results_version,
        args.stage,
        f"seed{args.seed}",
    )
    micro_rows = read_rows(
        os.path.join(root, "micro_pretrain", "train_centralized.csv")
    )
    joint_rows = read_rows(
        os.path.join(root, "joint_td3_sac", "train_td3.csv")
    )

    if int(float(micro_rows[-1]["sac_replay_size"])) <= 0:
        raise RuntimeError("Micro pretraining replay did not grow")
    if int(float(joint_rows[-1]["sac_replay_size"])) <= 0:
        raise RuntimeError("Joint-training SAC replay did not grow")
    if int(float(joint_rows[-1]["sac_trainable_micro_steps"])) <= 0:
        raise RuntimeError("Joint-training SAC remained frozen")
    if int(float(joint_rows[-1]["service_resource_decisions"])) <= 0:
        raise RuntimeError("No micro resource decisions were executed")

    for row in joint_rows:
        number(row, "macro_actor_reward")
        number(row, "micro_reward")
        number(row, "delay_objective_train")
        number(row, "average_power_w")

    summary = {
        "stage": args.stage,
        "seed": args.seed,
        "micro_pretrain_episodes": len(micro_rows),
        "joint_episodes": len(joint_rows),
        "joint_sac_trainable_micro_steps": int(
            float(joint_rows[-1]["sac_trainable_micro_steps"])
        ),
        "joint_sac_replay_size": int(
            float(joint_rows[-1]["sac_replay_size"])
        ),
        "joint_macro_replay_size": int(
            float(joint_rows[-1]["td3_buffer_size"])
        ),
        "status": "passed",
    }
    output_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        args.results_version,
        args.stage,
        "summaries",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"verification_seed{args.seed}.json",
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
