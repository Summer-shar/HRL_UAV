"""Run only the authorized fresh SAC 120-window candidate-wireless pilot."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.wireless_pilot import run_pilot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-report", type=Path,
                        default=ROOT/"results/sac_only_wireless6_pilot_smoke/manifest.json")
    args = parser.parse_args()
    output = args.output or ROOT/("results/sac_only_wireless6_pilot_smoke" if args.smoke
                                 else "results/sac_only_wireless6_pilot_seed101")
    run_pilot(output, args.smoke, args.smoke_report)
