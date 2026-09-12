"""Run only the preregistered B-centered 40-state/13-action/six-macro diagnosis."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hrl_uav.local_counterfactual import run

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/b_centered_local_counterfactual_sac101"))
    args=parser.parse_args()
    print(run(args.output))
