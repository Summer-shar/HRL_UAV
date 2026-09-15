"""
Explicit opt-in for the joint HRL training experiments.

Usage:
    # Joint training (recommended 1:10 ratio)
    python scripts/run_mission_joint_training.py --output results/joint_training --experiment joint_1_10 --run-authorized

    # PPO-only baseline (validates PPO trajectory tracking improvement)
    python scripts/run_mission_joint_training.py --output results/ppo_only_baseline --experiment ppo_only --run-authorized

    # SAC-only baseline (validates SAC routing improvement)
    python scripts/run_mission_joint_training.py --output results/sac_only_baseline --experiment sac_only --run-authorized

    # Ratio ablation studies
    python scripts/run_mission_joint_training.py --output results/joint_1_5 --experiment joint_1_5 --run-authorized
    python scripts/run_mission_joint_training.py --output results/joint_1_20 --experiment joint_1_20 --run-authorized

    # Resume interrupted run
    python scripts/run_mission_joint_training.py --output results/joint_training --experiment joint_1_10 --run-authorized --resume
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrl_uav.mission_joint_training import (
    JointTrainingConfig, run, TrainingMode, protocol as experiment_protocol
)


EXPERIMENT_CONFIGS = {
    'joint_1_5': JointTrainingConfig(
        mode=TrainingMode.JOINT,
        ppo_update_interval=5,
        total_micro_updates=600_000,
    ),
    'joint_1_10': JointTrainingConfig(
        mode=TrainingMode.JOINT,
        ppo_update_interval=10,
        total_micro_updates=600_000,
    ),
    'joint_1_20': JointTrainingConfig(
        mode=TrainingMode.JOINT,
        ppo_update_interval=20,
        total_micro_updates=600_000,
    ),
    'ppo_only': JointTrainingConfig(
        mode=TrainingMode.PPO_ONLY,
        ppo_update_interval=10,
        total_micro_updates=600_000,
    ),
    'sac_only': JointTrainingConfig(
        mode=TrainingMode.SAC_ONLY,
        ppo_update_interval=10,
        total_micro_updates=600_000,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment',
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help='Experiment configuration to run'
    )
    parser.add_argument(
        '--run-authorized', action='store_true',
        help='Explicit opt-in to begin training'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from interruption (uses same output directory)'
    )
    parser.add_argument(
        '--seeds',
        type=lambda s: tuple(int(x) for x in s.split(',')),
        default='201,202,203',
        help='PPO seeds to use (comma-separated, default: 201,202,203)'
    )
    parser.add_argument(
        '--budget',
        type=int,
        default=600_000,
        help='Total micro-updates budget (default: 600000)'
    )

    args = parser.parse_args()

    # Explicit opt-in check
    if not args.run_authorized:
        parser.error(
            "Training not started. Explicit --run-authorized flag is required.\n"
            "This is a safety check to prevent accidental training launches."
        )

    # Get configuration
    config = EXPERIMENT_CONFIGS[args.experiment]
    config.ppo_seeds = args.seeds
    config.total_micro_updates = args.budget

    print(f"\n{'='*60}")
    print(f"Joint HRL Training Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Mode: {config.mode.value}")
    print(f"PPO update interval: {config.ppo_update_interval} macros")
    print(f"Total micro-update budget: {config.total_micro_updates:,}")
    print(f"PPO seeds: {config.ppo_seeds}")
    print(f"Output: {args.output}")
    print(f"Resume: {args.resume}")
    print(f"{'='*60}\n")

    # Verify output directory status
    if args.output.exists() and not args.resume:
        existing = list(args.output.glob("seed*"))
        if existing:
            print(f"WARNING: Output directory has existing results:")
            for d in existing:
                print(f"  - {d.name}")
            print(f"\nUse --resume to continue from existing results,")
            print(f"or specify a different --output directory.")
            sys.exit(1)

    # Print protocol summary
    prot = experiment_protocol()
    print("Protocol summary:")
    print(f"  Training scenes: {len(prot['training_scenes'])}")
    print(f"  Validation scenes: {prot['validation_scenes']}")
    print(f"  Metrics: {len(prot['metrics'])} tracked")
    print()

    # Run experiment
    try:
        results = run(args.output, config, resume=args.resume)
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {args.output}")
        print(f"\nKey results:")
        for key in ['J', 'J_holding', 'J_energy', 'task_delay_weighted_ms', 'centroid_distance_mean']:
            if key in results:
                val = results[key]
                if isinstance(val, dict) and 'mean' in val:
                    print(f"  {key}: {val['mean']:.4f} +/- {val['std']:.4f}")
                elif isinstance(val, (int, float)):
                    print(f"  {key}: {val:.4f}")
        return results

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Resume with: python scripts/run_mission_joint_training.py \\")
        print(f"  --output {args.output} --experiment {args.experiment} \\")
        print(f"  --run-authorized --resume")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nTraining failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print(f"\nCheck logs in: {args.output}")
        raise


if __name__ == "__main__":
    main()
