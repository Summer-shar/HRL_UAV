import unittest

import numpy as np

from hrl_uav.load_switching import apply_load_switch_phase
from hrl_uav.load_switching import build_load_switch_schedule
from hrl_uav.load_switching import parse_load_switch_phases


PHASES = np.asarray(
    [
        [12.5, 5.5, 3.5],
        [3.5, 12.5, 5.5],
        [5.5, 3.5, 12.5],
    ],
    dtype=np.float64,
)


class DummyBaseEnv:
    def __init__(self):
        self.loads = None

    def set_zone_target_loads(self, loads):
        self.loads = np.asarray(loads, dtype=np.float64).copy()


class DummyWrapper:
    def __init__(self):
        self.env = DummyBaseEnv()


class LoadSwitchScheduleTests(unittest.TestCase):
    def test_parser_preserves_equal_total_phases(self):
        parsed = parse_load_switch_phases(
            "12.5,5.5,3.5;3.5,12.5,5.5;5.5,3.5,12.5",
            3,
        )
        np.testing.assert_allclose(parsed, PHASES)

    def test_schedule_is_deterministic_and_cyclic(self):
        first = build_load_switch_schedule(
            "cyclic_random", PHASES, 30, 1001, 8, 12
        )
        second = build_load_switch_schedule(
            "cyclic_random", PHASES, 30, 1001, 8, 12
        )
        np.testing.assert_array_equal(
            first.phase_index_by_macro,
            second.phase_index_by_macro,
        )
        self.assertEqual(first.switch_steps, second.switch_steps)
        self.assertEqual(first.phase_index(0), 0)
        for switch_index, step in enumerate(first.switch_steps, start=1):
            self.assertTrue(first.switched(step))
            self.assertEqual(first.phase_index(step), switch_index % 3)
        self.assertTrue(all(8 <= item <= 12 for item in first.dwell_steps[:-1]))
        self.assertEqual(sum(first.dwell_steps), 30)

    def test_schedule_does_not_consume_numpy_global_rng(self):
        np.random.seed(77)
        expected = np.random.random(5)
        np.random.seed(77)
        build_load_switch_schedule(
            "cyclic_random", PHASES, 30, 42, 8, 12
        )
        observed = np.random.random(5)
        np.testing.assert_allclose(observed, expected)

    def test_phase_is_applied_only_at_episode_start_and_switch(self):
        schedule = build_load_switch_schedule(
            "cyclic_random", PHASES, 30, 123, 8, 12
        )
        wrapper = DummyWrapper()
        apply_load_switch_phase(wrapper, schedule, 0, force=True)
        np.testing.assert_allclose(wrapper.env.loads, PHASES[0])
        first_switch = schedule.switch_steps[0]
        changed = apply_load_switch_phase(
            wrapper,
            schedule,
            first_switch,
            force=False,
        )
        self.assertTrue(changed)
        np.testing.assert_allclose(wrapper.env.loads, PHASES[1])

    def test_unequal_total_load_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_load_switch_phases("12,5,3;4,5,3", 3)


if __name__ == "__main__":
    unittest.main()

