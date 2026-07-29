import unittest

import numpy as np

from scripts.diagnose_cross_layer_policy import pin_uav_to_centroid


class CrossLayerDiagnosticsTest(unittest.TestCase):
    def test_pin_uav_to_centroid_preserves_other_actions(self):
        expected = np.asarray(
            [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]],
            dtype=np.float32,
        )

        def base_policy(_env, _obs, _eval_mode):
            return expected

        action = pin_uav_to_centroid(base_policy, 1)(None, None, True)

        np.testing.assert_allclose(action[0], expected[0])
        np.testing.assert_allclose(action[1], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(action[2], expected[2])
        np.testing.assert_allclose(expected[1], [0.4, 0.3])


if __name__ == "__main__":
    unittest.main()
