"""Run one per-micro PPO qualification (cvx|random|local); starts the full budget."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.mission_ppo_permicro import MICROS, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro", required=True, choices=MICROS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-require-committed", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / f"results/mission_ppo_micro_{args.micro}_qualification"
    run(output, args.micro, resume=args.resume, require_committed=not args.no_require_committed)
