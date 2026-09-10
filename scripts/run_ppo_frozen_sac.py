"""Run macro PPO training on the frozen seed101 SAC2000 reference."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.ppo_frozen import DEFAULT_SOURCE, run_frozen_ppo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--ppo-windows", type=int, required=True)
    parser.add_argument("--training-seed-base", type=int, default=72000)
    parser.add_argument("--validation-every", type=int, default=30)
    parser.add_argument("--smoke", action="store_true",
                        help="One native training window and short paired checks; no hyperparameter changes. No resume support.")
    args = parser.parse_args()
    run_frozen_ppo(
        args.output,
        source=args.source,
        ppo_windows=args.ppo_windows,
        training_seed_base=args.training_seed_base,
        validation_every=args.validation_every,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
