"""Run only Mission Stage3B's frozen320-branch validation."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hrl_uav.mission_future_check import run

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("No action/future/scene overrides are authorized")
    print(run())
