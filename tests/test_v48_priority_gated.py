import unittest

import numpy as np

from hrl_uav.hrl_wrappers import HRL_Wrapper
from scripts.run_experiment import common_args


class V48PriorityGatedTest(unittest.TestCase):
    def test_wrapper_exposes_twelve_semantic_actions_for_three_uavs(self):
        env = HRL_Wrapper(
            num_uavs=3,
            num_users=6,
            micro_action_encoding="priority_gated",
            micro_stick_breaking_scale=4.0,
            micro_priority_weight_ratio=10.0,
            users_static=True,
            pause_service_during_flight=False,
        )
        self.assertEqual(env.micro_action_space.shape, (12,))
        env.reset(seed=1)
        env.macro_step(np.zeros((3, 2), dtype=np.float32))
        _, _, _, _, info = env.micro_step(np.zeros(12, dtype=np.float32))
        high = np.asarray(info["split_ratios_high"])
        low = np.asarray(info["split_ratios_low"])
        np.testing.assert_allclose(np.diagonal(high), 10.0 / 11.0)
        np.testing.assert_allclose(np.diagonal(low), 0.5)

    def test_protocol_forwards_priority_ratio_to_training_scripts(self):
        args = common_args(
            {
                "episodes": 1,
                "macro_steps": 2,
                "eval_interval": 1,
                "eval_episodes": 1,
                "trajectory_interval": 0,
                "sac_batch_size": 32,
                "sac_memory_warmup": 0,
            },
            "cpu",
        )
        index = args.index("--micro_priority_weight_ratio")
        self.assertEqual(float(args[index + 1]), 10.0)


if __name__ == "__main__":
    unittest.main()
