"""Offline recovery only; forbids environment creation and evaluation."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.env import UAV_MEC_Env
from hrl_uav.mission_xu_cvx_r import finalize_completed_results
from hrl_uav.trajectory_check import forbidden


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    with patch.object(UAV_MEC_Env, "__init__", forbidden):
        print(finalize_completed_results(args.output))
