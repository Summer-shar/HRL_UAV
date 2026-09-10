import copy
import json
import unittest

import numpy as np

from hrl_uav.mdp import QueueMDP
from hrl_uav.wireless_check import ROOT, PSD, MEASURES, arm_configs, assert_local_equal, fixed_rollout
from hrl_uav.scene_prescreen import LocalRouting


class WirelessCheckTests(unittest.TestCase):
    def test_only_authorized_fields_change_and_local_route(self):
        cfg = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text())
        before = copy.deepcopy(cfg)
        arms = arm_configs(cfg)
        self.assertEqual(cfg, before)
        self.assertEqual(list(arms), ["w0_local_b6", "w1_local_b6", "w0_uniform_b3",
                                      "w1_uniform_b3", "w0_uniform_b6", "w1_uniform_b6"])
        for name, changed in arms.items():
            self.assertTrue(changed["environment"].pop("compute_during_deployment"))
            self.assertEqual(changed["environment"].pop("wireless_model"),
                             "legacy_fixed_noise" if name.startswith("w0") else "psd_per_uav_budget")
            if name.startswith("w1"):
                for key, value in PSD.items():
                    self.assertEqual(changed["environment"].pop(key), value)
            changed["environment"]["u2u_bandwidth_mhz"] = 6
            self.assertEqual(changed, cfg)
        mdp = QueueMDP(cfg)
        np.testing.assert_array_equal(mdp.route(LocalRouting().choose_action(None)), np.tile(np.eye(3), (2, 1, 1)))

    def test_local_calibration_stops_on_difference(self):
        row = {k: 1.0 for k in MEASURES}
        row.update(energy_u2u_j=0, tx_served_mbit=0, queue_transfer_mbit=0)
        self.assertTrue(assert_local_equal(row, row, np.ones((2, 39)), np.ones((2, 39)),
                                          np.zeros((2, 3)), np.zeros((2, 3)))["passed"])
        with self.assertRaises(AssertionError):
            assert_local_equal(row, row, np.ones((2, 39)), np.ones((2, 39))*1.01,
                               np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaises(AssertionError):
            assert_local_equal(row, dict(row, task_completed_count=2), np.ones((2, 39)), np.ones((2, 39)),
                               np.zeros((2, 3)), np.zeros((2, 3)))

    def test_learning_and_tail_are_forbidden(self):
        for kwargs in ({"train": True}, {"complete_tail": True}, {"macro_enabled": True}):
            with self.assertRaises(ValueError):
                fixed_rollout(**kwargs)


if __name__ == "__main__":
    unittest.main()
