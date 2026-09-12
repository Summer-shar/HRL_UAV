"""Run only the fixed second-stage mission micro comparison."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.mission_micro_pilot import run

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("This runner accepts no scene, controller or parameter overrides")
    print(json.dumps(run(), ensure_ascii=False))
