"""Only the explicitly authorized one-window integration smoke; no pilot mode."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.ppo_residual_b import run_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(args.output)
    print(report["status"])
    print(report["training"])


if __name__ == "__main__":
    main()
