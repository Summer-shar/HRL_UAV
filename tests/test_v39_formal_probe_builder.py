import unittest

import numpy as np

from scripts.build_v39_formal_probes import cooperative_probe_action


class V39FormalProbeBuilderTests(unittest.TestCase):
    def test_cooperative_action_targets_lightest_support_toward_heavy_uav(self):
        loads = np.asarray([14.5, 4.5, 2.5])
        uavs = np.asarray([[0.0, 0.0], [0.0, 100.0], [100.0, 0.0]])
        centroids = uavs.copy()
        action, heavy, support = cooperative_probe_action(
            loads, uavs, centroids, waypoint_radius_m=120.0
        )
        self.assertEqual(heavy, 0)
        self.assertEqual(support, 2)
        np.testing.assert_allclose(action[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(action[1], 0.0, atol=1e-7)
        self.assertLess(action[2, 0], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(action[2])), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
