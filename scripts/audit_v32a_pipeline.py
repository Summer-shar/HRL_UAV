import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap


PROJECT_ROOT = Path(bootstrap())

from train_centralized_td3 import MacroReplayBuffer
from train_centralized_td3 import discounted_sequence_target


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen v32a pipeline gate.")
    parser.add_argument(
        "--output",
        default="result_archive/v32a_clean_nstep/pipeline_gate.json",
    )
    parser.add_argument(
        "--snapshot-output",
        default="result_archive/v32a_clean_nstep/protocol_snapshot.json",
    )
    return parser.parse_args()


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fill_replay(buffer):
    data_rng = np.random.default_rng(32001)
    for episode in range(1, 7):
        for macro in range(30):
            buffer.add(
                data_rng.normal(size=8).astype(np.float32),
                data_rng.uniform(-0.5, 0.5, size=6).astype(np.float32),
                reward=float(data_rng.normal()),
                next_state=data_rng.normal(size=8).astype(np.float32),
                done=False,
                energy_ratio=0.9,
                episode_index=episode,
                macro_index=macro,
            )


def main():
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_output = Path(args.snapshot_output)
    if not snapshot_output.is_absolute():
        snapshot_output = PROJECT_ROOT / snapshot_output

    frozen_files = {
        "protocol": PROJECT_ROOT / "docs" / "v32a_clean_nstep_protocol.md",
        "control_config": (
            PROJECT_ROOT
            / "configs"
            / "tcom_dynamic_u2u_sparse_v32a_control_n1.json"
        ),
        "treatment_config": (
            PROJECT_ROOT
            / "configs"
            / "tcom_dynamic_u2u_sparse_v32a_treatment_n5.json"
        ),
    }
    snapshot = {
        "status": "FROZEN",
        "training_seeds": [1, 42, 123],
        "evaluation_seeds": list(range(1000, 1020)),
        "files": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in frozen_files.items()
        },
    }
    snapshot_output.write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )

    buffers = [
        MacroReplayBuffer(
            8,
            6,
            512,
            rng=np.random.default_rng(32002),
        )
        for _ in range(2)
    ]
    for buffer in buffers:
        fill_replay(buffer)

    np.random.seed(32003)
    expected_global = np.random.random(32)
    np.random.seed(32003)
    control = buffers[0].sample_sequence_batch(
        64,
        torch.device("cpu"),
        delay_n_step=1,
        sequence_span=5,
        episode_horizon=30,
        current_episode_index=6,
    )
    observed_global = np.random.random(32)
    np.random.normal(size=1000)
    treatment = buffers[1].sample_sequence_batch(
        64,
        torch.device("cpu"),
        delay_n_step=5,
        sequence_span=5,
        episode_horizon=30,
        current_episode_index=6,
    )

    bootstrap_value = torch.linspace(-0.4, 0.5, 64).reshape(-1, 1)
    n1_target = discounted_sequence_target(
        control["reward_sequence"],
        control["target_length"],
        bootstrap_value,
        control["done_n"],
        gamma=0.99,
    )
    historical_target = control["reward_sequence"][:, 0] + (
        (1.0 - control["done_one"]) * 0.99 * bootstrap_value
    )

    checks = {
        "n1_target_exact": bool(torch.equal(n1_target, historical_target)),
        "global_numpy_rng_unchanged": bool(
            np.array_equal(expected_global, observed_global)
        ),
        "start_indices_equal": bool(
            np.array_equal(
                control["start_indices"],
                treatment["start_indices"],
            )
        ),
        "sampled_states_equal": bool(
            torch.equal(control["state"], treatment["state"])
        ),
        "reward_sequences_equal": bool(
            torch.equal(
                control["reward_sequence"],
                treatment["reward_sequence"],
            )
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "control_resamples": int(control["sequence_resamples"]),
        "treatment_resamples": int(treatment["sequence_resamples"]),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
