"""Only the authorized four-run target x tracking diagnostic."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from hrl_uav.target_tracking_check import prepare,smoke,run


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","smoke","run"))
    parser.add_argument("--output",type=Path,default=ROOT/"results/target_tracking_sac101")
    args=parser.parse_args()
    torch.set_num_threads(1)
    {"prepare":prepare,"smoke":smoke,"run":run}[args.stage](args.output.resolve())
