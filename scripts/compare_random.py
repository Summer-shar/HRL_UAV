import argparse

import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import add_common_args
from compare_common import alpha_to_micro_action
from compare_common import run_fixed_policy_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random macro/micro baseline on the shared HRL environment",
    )
    add_common_args(parser, default_results_subdir="random")
    return parser.parse_args()


def random_macro_policy(env, local_obs_batch, deterministic=False):
    return env.macro_action_space.sample()


def random_micro_policy(env, deterministic=False):
    rng = getattr(getattr(env, "env", None), "np_random", np.random)
    split_ratios = np.zeros((2, env.N, env.N), dtype=np.float32)

    for channel in range(2):
        for source_uav in range(env.N):
            split_ratios[channel, source_uav] = rng.dirichlet(np.ones(env.N))

    action_high = alpha_to_micro_action(
        split_ratios[0],
        diagonal_bias=float(getattr(env, "micro_diag_bias", 0.5)),
        action_scale=float(getattr(env, "micro_action_scale", 1.0)),
        channels=1,
    ).reshape(1, env.N, env.N)
    action_low = alpha_to_micro_action(
        split_ratios[1],
        diagonal_bias=float(getattr(env, "micro_diag_bias", 0.5)),
        action_scale=float(getattr(env, "micro_action_scale", 1.0)),
        channels=1,
    ).reshape(1, env.N, env.N)
    return np.concatenate([action_high, action_low], axis=0).reshape(-1).astype(np.float32)


def seed_random_spaces(env, seed):
    env.macro_action_space.seed(seed)
    env.micro_action_space.seed(seed + 1009)


def main():
    args = parse_args()
    run_fixed_policy_benchmark(
        experiment_label="Random",
        args=args,
        csv_filename="eval_random.csv",
        macro_policy=random_macro_policy,
        micro_policy=random_micro_policy,
        on_train_episode_start=seed_random_spaces,
        on_eval_episode_start=seed_random_spaces,
    )


if __name__ == "__main__":
    main()
