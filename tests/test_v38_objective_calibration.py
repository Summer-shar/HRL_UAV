import unittest

import numpy as np

from scripts.run_v38_objective_calibration import (
    slope_through_origin,
    weights_from_slopes,
)


class V38ObjectiveCalibrationTests(unittest.TestCase):
    def test_slope_through_origin(self):
        x = np.asarray([1.0, 2.0, 3.0])
        y = 0.25 * x
        self.assertAlmostEqual(slope_through_origin(x, y), 0.25)

    def test_weights_encode_declared_exchange_rate(self):
        result = weights_from_slopes(
            delay_proxy_per_ms=0.001190,
            energy_proxy_per_w=1.0 / 140.0,
            rho_ms_per_w=0.25,
        )
        delay = result["delay_weight"] * 0.001190 * 5.0
        energy = result["energy_weight"] * (1.0 / 140.0) * 20.0
        self.assertAlmostEqual(delay, energy)
        self.assertAlmostEqual(
            result["delay_weight"] + result["energy_weight"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
