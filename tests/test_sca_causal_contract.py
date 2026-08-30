import unittest
from types import SimpleNamespace

import numpy as np

from scripts.compare_sca import _build_zone_priority_task_bits
from scripts.compare_xu2024_sca import solve_xu2024_micro_action
from hrl_uav.hrl_wrappers import HRL_Wrapper


class _ForbiddenExpectedLoad:
    def __array__(self, *_args, **_kwargs):
        raise AssertionError("Fair causal mode must not read model arrival rates")


class SCACausalContractTest(unittest.TestCase):
    def test_current_arrival_mode_uses_only_observed_frame_load(self):
        base = SimpleNamespace(
            N=3,
            user_lambda_high=_ForbiddenExpectedLoad(),
            user_lambda_low=_ForbiddenExpectedLoad(),
        )
        env = SimpleNamespace(env=base, sca_load_source="current_arrivals")
        raw = {
            "A_t_high": np.asarray([1.0, 2.0, 3.0]),
            "A_t_low": np.asarray([4.0, 5.0, 6.0]),
        }
        bits = _build_zone_priority_task_bits(env, raw_state=raw)
        np.testing.assert_allclose(
            bits,
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 1e6,
        )

    def test_xu_adapter_does_not_invent_expected_tasks_on_empty_frame(self):
        env = HRL_Wrapper(
            num_uavs=3,
            num_users=9,
            micro_action_encoding="priority_gated",
            micro_diag_bias=0.0,
        )
        env.reset(seed=11)
        env.env.current_arrival_tasks_by_uav = [[], [], []]
        action = solve_xu2024_micro_action(env)
        self.assertEqual(action.shape, (2 * env.N * env.N,))
        self.assertFalse(env._xu2024_used_expected_fallback)
        self.assertEqual(env._xu2024_empty_arrival_frames, 1)


if __name__ == "__main__":
    unittest.main()
