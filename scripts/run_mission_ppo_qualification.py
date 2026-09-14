"""Explicit opt-in only; --help and imports do not simulate or train."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.mission_ppo_qualification import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-authorized-qualification", action="store_true",
                        help="Starts3x120 windows; requires separate user authorization after code audit")
    parser.add_argument("--resume", action="store_true", help="Restore only complete900s stream boundaries")
    parser.add_argument("--skip-commit-check", action="store_true",
                        help="Skip commit verification (for development with intentional code changes)")
    args = parser.parse_args()
    if not args.run_authorized_qualification:
        parser.error("No training started. Explicit --run-authorized-qualification is required.")
    print(run(args.output, resume=args.resume, require_committed=not args.skip_commit_check))
