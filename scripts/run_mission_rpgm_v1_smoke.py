"""Run only the fixed mission_rpgm_v1 60 s smoke, with no training modes."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrl_uav.mission_smoke import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mission_rpgm_v1_smoke"))
    args = parser.parse_args()
    print(run(args.output))
