"""Execute only the authorized fixed 120-window residual PPO development pilot."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrl_uav.ppo_residual_pilot import run_pilot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot(args.output)
    print(result["status"], flush=True)
