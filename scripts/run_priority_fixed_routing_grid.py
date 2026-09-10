"""Frozen 31-condition priority routing grid. No training or model changes."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.priority_grid import run_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-report", type=Path,
                        default=ROOT/"results/priority_fixed_routing_grid_smoke/manifest.json")
    args = parser.parse_args()
    output = args.output or ROOT/("results/priority_fixed_routing_grid" + ("_smoke" if args.smoke else ""))
    if output.exists():
        output = output.with_name(output.name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    print(f"Output: {output}", flush=True)
    run_grid(output, args.smoke, args.smoke_report)
