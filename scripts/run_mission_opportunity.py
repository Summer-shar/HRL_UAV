"""Execute only the predeclared Mission Stage3A matrix."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.mission_opportunity import run

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("No scene, epsilon, horizon or parameter overrides are allowed")
    print(run())
