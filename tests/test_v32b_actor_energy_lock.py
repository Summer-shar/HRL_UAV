import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_v32b_actor_energy_lock import actor_loss_components
from audit_v32b_actor_energy_lock import hover_actions_from_states
from audit_v32b_actor_energy_lock import patch_system_dual_state
from audit_v32b_actor_energy_lock import predicted_movement_m


class V32BActorEnergyLockTests(unittest.TestCase):
    def test_hover_action_places_waypoint_at_current_uav(self):
        states = np.zeros((1, 2, 4), dtype=np.float32)
        states[0, 0, :2] = [0.10, 0.0]
        states[0, 1, :2] = [0.0, -0.04]
        states = states.reshape(1, -1)

        hover, raw_norm, exact = hover_actions_from_states(states, 2, 200.0)
        np.testing.assert_allclose(
            hover.reshape(1, 2, 2),
            [[[-0.5, 0.0], [0.0, 0.2]]],
            atol=1e-7,
        )
        np.testing.assert_allclose(raw_norm, [[0.5, 0.2]], atol=1e-7)
        self.assertTrue(exact[0])
        movement = predicted_movement_m(states, hover, 2, 200.0, 20.0)
        np.testing.assert_allclose(movement, 0.0, atol=1e-5)

        return_movement = predicted_movement_m(
            states,
            np.zeros((1, 4), dtype=np.float32),
            2,
            200.0,
            20.0,
        )
        np.testing.assert_allclose(return_movement, [[20.0, 20.0]])

    def test_system_dual_state_is_patched_for_each_agent(self):
        args = {
            "macro_flow_state": True,
            "macro_load_state": True,
            "macro_capacity_state": True,
            "power_constraint_scope": "system",
            "system_dual_state": True,
            "energy_dual_state_norm": 10.0,
        }
        states = np.zeros((2, 56), dtype=np.float32)
        patched = patch_system_dual_state(states, args, 4.0, 2)
        local = patched.reshape(2, 2, 28)
        np.testing.assert_allclose(local[:, :, 11], 0.4)
        self.assertEqual(np.count_nonzero(patched), 4)

    def test_actor_loss_components_keep_delay_and_energy_signs(self):
        delay_q = np.asarray([-2.0, -3.0])
        energy_q = np.asarray([0.25, -0.10])
        delay_loss, energy_loss, total = actor_loss_components(
            delay_q,
            energy_q,
            4.0,
        )
        np.testing.assert_allclose(delay_loss, [2.0, 3.0])
        np.testing.assert_allclose(energy_loss, [1.0, -0.4])
        np.testing.assert_allclose(total, [3.0, 2.6])


if __name__ == "__main__":
    unittest.main()
