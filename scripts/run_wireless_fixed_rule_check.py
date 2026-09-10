"""One bounded smoke or six-arm fixed-rule wireless comparison. Never trains."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.wireless_check import run_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-report", type=Path,
                        default=ROOT / "results/wireless_model_fixed_rule_check_smoke/report.json")
    args = parser.parse_args()
    output = args.output or ROOT / ("results/wireless_model_fixed_rule_check" + ("_smoke" if args.smoke else ""))
    if output.exists():
        output = output.with_name(output.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    cfg = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text(encoding="utf-8"))
    print(f"Output: {output}", flush=True)
    run_check(cfg, output, args.smoke, args.smoke_report)
