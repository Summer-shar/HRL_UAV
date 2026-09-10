"""Run only the authorized fresh seed102/103 replicas, then summarize seed101-103."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.wireless_replicates import aggregate, run_fresh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(102, 103))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--resume-anchor", type=Path)
    args = parser.parse_args()
    if args.aggregate and args.seed is None:
        aggregate()
    elif args.seed is not None and not args.aggregate:
        run_fresh(args.seed, args.resume_anchor)
    else:
        parser.error("Select exactly one of --seed or --aggregate")
