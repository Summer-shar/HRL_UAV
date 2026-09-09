"""Run smoke first, then the fixed six-arm deployment/communication screen."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.scene_prescreen import run_prescreen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-report", type=Path,
                        default=ROOT / "results/scene_prescreen_deployment_compute_smoke/report.json")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text(encoding="utf-8"))
    output = args.output or ROOT / ("results/scene_prescreen_deployment_compute"
                                   + ("_smoke" if args.smoke else ""))
    run_prescreen(config, output, args.smoke, args.smoke_report)
