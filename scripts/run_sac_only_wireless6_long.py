"""Continue the authorized native SAC pilot to a fixed total of 2000 windows."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.wireless_long import run_long


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/sac_only_wireless6_long_seed101")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume-anchor", type=Path,
                        help="True stream anchor only; recovery uses a new output directory")
    args = parser.parse_args()
    run_long(args.output, args.audit_only, args.resume_anchor)
