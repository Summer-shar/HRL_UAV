"""Only Stage4A functional smoke is authorized; no training-budget overrides."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrl_uav.mission_ppo_stage4a import run_smoke

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("This entry exposes only fixed Stage4A smoke, not formal training/resume")
    print(run_smoke())
