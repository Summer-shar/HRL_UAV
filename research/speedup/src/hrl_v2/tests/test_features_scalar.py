"""Exact scalar validation equivalence; one physical step for real reports."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np

from hrl_v2 import features


def original_L(x, scale=1.):
    """Pre-optimization implementation, kept independent as the oracle."""
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Features contain negative or nonfinite physical values")
    return np.log1p(a / scale)


class ScalarFeatureTests(unittest.TestCase):
    def assert_exact_result(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        actual_array, expected_array = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual_array.dtype, expected_array.dtype)
        self.assertEqual(actual_array.shape, expected_array.shape)
        self.assertEqual(actual_array.tobytes(), expected_array.tobytes())

    def test_scalar_and_zero_dimensional_results_are_bitwise_identical(self):
        values = [0., -0., np.nextafter(0., 1.), 1e-320, 1e-300,
                  np.finfo(np.float64).tiny, np.finfo(np.float64).eps,
                  1., 10., 1e6, 1e15, np.finfo(np.float64).max,
                  0, 1, 2**53 + 1, np.int64(2**62), np.uint64(2**64 - 1),
                  np.float32(.1), np.float64(.1), True, False]
        scales = [1., 10., 1e6, 1e9, np.float32(1e3), np.asarray(10.),
                  np.asarray([1., 1e6, 1e9]), np.asarray([1., 1e6], dtype=np.float32)]
        for value in values:
            for form in (value, np.asarray(value)):
                for scale in scales:
                    with self.subTest(value=repr(value), form=type(form).__name__, scale=repr(scale)):
                        with np.errstate(all="ignore"):
                            self.assert_exact_result(features.L(form, scale), original_L(form, scale))

    def test_array_path_is_unchanged(self):
        arrays = [[], [0., 1., 1e6], np.asarray([0., -0., np.nextafter(0., 1.)]),
                  np.asarray([[1., 2.], [3., 4.]], dtype=np.float32),
                  np.asarray([1, 2, 3], dtype=np.int64)]
        for value in arrays:
            for scale in (1., 10., 1e9):
                with self.subTest(value=repr(value), scale=scale):
                    self.assert_exact_result(features.L(value, scale), original_L(value, scale))

    def test_invalid_values_raise_the_same_error(self):
        invalid = [-1., -np.nextafter(0., 1.), np.nan, np.inf, -np.inf]
        for value in invalid:
            for form in (value, np.asarray(value), [0., value], np.asarray([[value, 1.]])):
                with self.subTest(value=repr(form)):
                    with self.assertRaises(ValueError) as expected:
                        original_L(form)
                    with self.assertRaises(ValueError) as actual:
                        features.L(form)
                    self.assertEqual(str(actual.exception), str(expected.exception))

    def test_scale_semantics_and_signed_zero_are_unchanged(self):
        # L historically validates x, not scale. Preserve this boundary,
        # including IEEE infinities/NaNs and the sign of zero in the result.
        for value in (0., -0., .5, 1., 2.):
            for scale in (0., -0., -1., np.inf, np.nan):
                with self.subTest(value=value, scale=scale), np.errstate(all="ignore"):
                    self.assert_exact_result(features.L(value, scale), original_L(value, scale))


class CompleteReportFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hrl_v2.physics_adapter import PhysicsAdapter
        from hrl_v2.tests.test_physical_contract import quiet_scene, admit
        adapter = PhysicsAdapter(quiet_scene())
        adapter.reset(arrival_seed=71, layout_seed=72, load_seed=73)
        adapter.env._scene.frame = lambda frame: ()
        adapter.start_macro(np.zeros(6))
        adapter.prepare()
        # Two parents per class/source give ordered CPU and transfer queues;
        # two unfinished uploads per source exercise terminal FIFO edges.
        for source in range(3):
            for priority in range(2):
                for _ in range(2):
                    admit(adapter.env, source=source, priority=priority,
                          bits=1e10, cycles=1e12, ready=True)
            for _ in range(2):
                admit(adapter.env, source=source, priority=1, bits=1e12, cycles=1e12)
        before = adapter.report()
        route = np.full((2, 3, 3), 1 / 3, dtype=np.float64)
        tick = adapter.env.step_index
        adapter.step(route)
        cls.physical_steps = adapter.env.step_index - tick
        cls.reports = (before, adapter.report())

    def assert_exact_tree(self, actual, expected, path="graph"):
        self.assertIs(type(actual), type(expected), path)
        if isinstance(expected, dict):
            self.assertEqual(list(actual), list(expected), path)
            for key in expected:
                self.assert_exact_tree(actual[key], expected[key], path + "." + str(key))
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), path)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_exact_tree(left, right, path + "[" + str(index) + "]")
        elif isinstance(expected, (float, np.floating)):
            self.assertEqual(np.asarray(actual).tobytes(), np.asarray(expected).tobytes(), path)
        else:
            self.assertEqual(actual, expected, path)

    def test_all_graph_features_relations_and_fifo_are_exact(self):
        self.assertEqual(self.physical_steps, 1)
        populated = set()
        for report in self.reports:
            with patch.object(features, "L", original_L):
                expected = features.build(report)
            actual = features.build(report)
            self.assert_exact_tree(asdict(actual), asdict(expected))
            self.assertEqual(tuple(actual.edges), features.RELATIONS)
            for kind, queues in actual.queues.items():
                if any(queues):
                    populated.add(kind)
                for queue in queues:
                    for first, second in zip(queue, queue[1:]):
                        self.assertIn((first, second, ()), actual.edges["prev"])
                        self.assertIn((second, first, ()), actual.edges["next"])
        self.assertEqual(populated, {"upload", "ready", "transfer", "cpu"})


if __name__ == "__main__":
    unittest.main()
