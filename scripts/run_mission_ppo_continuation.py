"""Explicit opt-in for the fixed PPO-only window120->600 continuation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.mission_ppo_continuation import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-authorized-continuation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run_authorized_continuation:
        parser.error("No training started; explicit --run-authorized-continuation is required")
    print(run(args.output, resume=args.resume))
