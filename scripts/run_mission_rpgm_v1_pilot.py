"""Run only the registered ten-case no-learning mission pilot."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrl_uav.mission_pilot import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/mission_rpgm_v1_pilot")
    args = parser.parse_args()
    print(run(args.output))
