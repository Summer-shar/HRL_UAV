"""Predeclared one-variable PPO pilot; no arbitrary hyperparameter overrides."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.ppo_frozen import run_frozen_ppo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_frozen_ppo(args.output, ppo_windows=1 if args.smoke else 120,
                            smoke=args.smoke, initial_std=1/6, sample_diagnostics=True)
    print(result["status"], "windows=", result["ppo_windows"], "PPO updates=", result["ppo_updates"], flush=True)
