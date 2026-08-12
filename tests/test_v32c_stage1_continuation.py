import unittest

import numpy as np

from scripts.audit_v32c_stage1_continuation import BranchPolicy
from scripts.audit_v32c_stage1_continuation import discounted_suffix
from scripts.audit_v32c_stage1_continuation import select_fork_index


class _FakeEnv:
    def get_macro_state(self):
        return [], np.asarray([1.0, 2.0], dtype=np.float32)


def _actor_policy(_env, _obs, _deterministic):
    return np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)


class Stage1ContinuationTests(unittest.TestCase):
    def test_branch_policy_zero_once_and_return_have_distinct_continuations(self):
        env = _FakeEnv()
        once = BranchPolicy(_actor_policy, fork_index=1, arm="ZERO_ONCE")
        sustained = BranchPolicy(_actor_policy, fork_index=1, arm="RETURN")
        once_actions = [once(env, None, True) for _ in range(3)]
        sustained_actions = [sustained(env, None, True) for _ in range(3)]
        np.testing.assert_allclose(once_actions[0], _actor_policy(None, None, True))
        np.testing.assert_allclose(once_actions[1], 0.0)
        np.testing.assert_allclose(once_actions[2], _actor_policy(None, None, True))
        np.testing.assert_allclose(
            sustained_actions[0], _actor_policy(None, None, True)
        )
        np.testing.assert_allclose(sustained_actions[1], 0.0)
        np.testing.assert_allclose(sustained_actions[2], 0.0)

    def test_select_fork_uses_state_before_macro_action(self):
        trace = {
            "uav_trajectory_m": np.asarray(
                [
                    [[0.0, 0.0]],
                    [[20.0, 0.0]],
                    [[55.0, 0.0]],
                    [[80.0, 0.0]],
                ]
            ),
            "centroid_trajectory_m": np.zeros((4, 1, 2)),
        }
        self.assertEqual(select_fork_index(trace, 50.0, 1, 2), 2)

    def test_discounted_suffix_uses_exact_replay_reward_field(self):
        records = [
            {"td3_replay_reward": 99.0},
            {"td3_replay_reward": 1.0},
            {"td3_replay_reward": 2.0},
        ]
        self.assertEqual(discounted_suffix(records, 1, 0.5), 2.0)
