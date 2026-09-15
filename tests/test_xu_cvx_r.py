import unittest

import numpy as np
from scipy.optimize import linprog

from hrl_uav.routing import project_rows_to_simplex
from hrl_uav.xu_cvx_r import (StaticModel, XuCVXRouting, SLICES, SUPPORTS,
    decode_observation, hypothetical_full_link_resources, predictor_coefficients)


def static_model():
    return StaticModel(6e6, 1.95e-19, 1., 20e6/39, 1., 1000., 1e-28,
                       10., 1., .9, .1, 10., 14.)


def encoded_state():
    state = np.zeros(68, dtype=np.float32)
    state[slice(*SLICES["cpu_capacity"])] = 1
    eta = np.full((3, 3), 8.)
    np.fill_diagonal(eta, 0)
    state[slice(*SLICES["u2u_spectral_efficiency"])] = np.log1p(eta).ravel()
    state[slice(*SLICES["current_arrivals"])] = np.log1p([1, 2, 3, 4, 5, 6])
    return state


class XuCVXTests(unittest.TestCase):
    def test_fixed_cpu_convex_special_case_against_closed_form(self):
        actor = XuCVXRouting(static_model())
        costs = np.tile([1., 2., 4.], (6, 1))
        action = actor.solve_coefficients(np.zeros((6, 3)), costs, np.zeros((6, 3)))
        expected = np.tile([4/7, 2/7, 1/7], (6, 1))
        np.testing.assert_allclose(action.reshape(6, 3), expected, atol=3e-6, rtol=0)
        np.testing.assert_allclose(project_rows_to_simplex(action.reshape(2, 3, 3)),
                                   action.reshape(2, 3, 3), atol=1e-14)
        self.assertTrue(actor.problem.is_dcp())
        self.assertTrue(actor.problem.is_dpp())

    def test_unused_destination_has_no_waiting_time_charge(self):
        actor = XuCVXRouting(static_model())
        waits = np.tile([0., 0., 100.], (6, 1))
        action = actor.solve_coefficients(waits, np.ones((6, 3)), np.zeros((6, 3))).reshape(6, 3)
        np.testing.assert_allclose(action, np.tile([.5, .5, 0.], (6, 1)), atol=3e-6)

    def test_independent_highs_solver_for_all_supports(self):
        actor = XuCVXRouting(static_model())
        rng = np.random.default_rng(934)
        wait, duration, energy = (rng.uniform(.1, 3., (6, 3)) for _ in range(3))
        actor.solve_coefficients(wait, duration, energy)
        expected = np.zeros((6, 7))
        for row in range(6):
            for j, support in enumerate(SUPPORTS):
                matrix = np.zeros((int(support.sum()), 4))
                active = np.flatnonzero(support)
                for k, dest in enumerate(active):
                    matrix[k, dest] = duration[row, dest]
                    matrix[k, 3] = -1
                result = linprog(np.r_[actor.energy_weight*energy[row], actor.delay_weights[row]],
                    A_ub=matrix, b_ub=-wait[row, active], A_eq=[[1, 1, 1, 0]], b_eq=[1],
                    bounds=[(0, 1 if s else 0) for s in support]+[(0, None)], method="highs")
                self.assertTrue(result.success)
                expected[row, j] = result.fun
        np.testing.assert_allclose(actor.candidate_costs[-1], expected, atol=3e-6, rtol=1e-6)

    def test_same_information_no_environment_and_empty_row_carry(self):
        actor = XuCVXRouting(static_model())
        state = encoded_state()
        action = actor.choose_action(state, deterministic=True)
        snapshot = state.copy()
        state[slice(*SLICES["current_arrivals"])] = 0
        carried = actor.choose_action(state, deterministic=True)
        np.testing.assert_array_equal(action, carried)
        self.assertNotIn("env", vars(actor))
        self.assertNotIn("meter", vars(actor))
        self.assertEqual(actor.records[-1]["empty_rows"], 6)
        self.assertTrue(np.isfinite(snapshot).all())
        with self.assertRaises(ValueError):
            actor.choose_action(np.zeros(67), deterministic=True)
        with self.assertRaises(ValueError):
            actor.choose_action(snapshot, deterministic=False)

    def test_wireless_inverse_and_scalar_rate_budget(self):
        model = static_model()
        decoded = decode_observation(encoded_state())
        b, p, n, rate = hypothetical_full_link_resources(decoded["eta"], model)
        gain = (2**8-1)*model.noise_psd*model.reference_bandwidth_hz
        expected = 1e6*np.log2(1+.5*gain/(model.noise_psd*1e6))/1e6
        active = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(rate[active], expected, rtol=2e-6)
        np.testing.assert_allclose(b.sum(), 6e6)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_allclose(n, b*model.noise_psd)
        np.testing.assert_array_equal(rate.diagonal(), 0)

    def test_priority_fifo_predictor_and_energy_units(self):
        model = static_model()
        decoded = decode_observation(encoded_state())
        decoded["compute"][0] = [10, 20, 30]
        decoded["compute"][1] = [40, 50, 60]
        decoded["transfer"].fill(0)
        wait, duration, energy = predictor_coefficients(decoded, model)
        self.assertAlmostEqual(wait[0, 0], 1)
        self.assertAlmostEqual(wait[3, 0], 5)
        self.assertAlmostEqual(duration[0, 0], .1, places=6)
        self.assertAlmostEqual(energy[0, 0], 10., places=5)
        self.assertGreater(energy[0, 1], energy[0, 0])

    def test_invalid_constants_rejected(self):
        from dataclasses import replace
        for field, value in (("noise_psd", 0), ("bandwidth_hz", -1), ("kappa", float("nan"))):
            with self.assertRaises(ValueError):
                replace(static_model(), **{field: value})

    def test_real_entry_imports_and_requires_opt_in(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root/"scripts/run_mission_xu_cvx_r.py")]
        help_result = subprocess.run(command+["--help"], cwd=root, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        rejected = subprocess.run(command+["--output", "must_not_be_created"], cwd=root, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2, rejected.stderr)
        self.assertIn("Explicit --run-fixed-comparison", rejected.stderr)
        self.assertFalse((root/"must_not_be_created").exists())


if __name__ == "__main__":
    unittest.main()
