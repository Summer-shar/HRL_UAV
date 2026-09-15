"""Explicit opt-in for the six fixed-CPU Xu-CVX-R development comparisons."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.mission_xu_cvx_r import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-fixed-comparison", action="store_true")
    args = parser.parse_args()
    if not args.run_fixed_comparison:
        parser.error("Explicit --run-fixed-comparison is required; this is not a training entry")
    print(run(args.output))
