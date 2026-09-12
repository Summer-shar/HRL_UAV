"""Run the single fixed same-state3/6MHz diagnosis; no experimental knobs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hrl_uav.trajectory_bandwidth_stress import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/trajectory_bandwidth_stress_sac101')
    print(run(parser.parse_args().output))
