import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_xu2024_sca import configure_xu2024_wrapper
from compare_xu2024_sca import _aggregate_source_priority_tasks
from compare_xu2024_sca import _queue_wait_costs_s
from compare_xu2024_sca import solve_xu2024_micro_action
from hrl_uav.hrl_wrappers import HRL_Wrapper


class Xu2024SCAAdapterTest(unittest.TestCase):
    def test_source_priority_aggregation_preserves_upload_time(self):
        priorities = np.asarray(["high", "high", "low"], dtype=object)
        sources = np.asarray([0, 0, 1], dtype=np.int64)
        bits = np.asarray([2.0, 3.0, 4.0], dtype=np.float64)
        rates = np.asarray([10.0, 30.0, 20.0], dtype=np.float64)
        out_priority, out_source, out_bits, out_rates = (
            _aggregate_source_priority_tasks(
                priorities,
                sources,
                bits,
                rates,
                num_uavs=3,
            )
        )
        high = (out_priority == "high") & (out_source == 0)
        self.assertEqual(int(np.count_nonzero(high)), 1)
        np.testing.assert_allclose(out_bits[high], [5.0])
        np.testing.assert_allclose(
            out_bits[high] / out_rates[high],
            [2.0 / 10.0 + 3.0 / 30.0],
        )

    def test_external_cpu_override_is_finite_duration(self):
        env = HRL_Wrapper(
            num_uavs=3,
            num_users=9,
            cpu_alloc_mode="priority_demand",
        )
        env.reset(seed=1)
        base = env.env
        high = np.asarray([1.0, 2.0, 3.0]) * 1e9
        low = np.asarray([0.5, 0.5, 0.5]) * 1e9
        base.set_external_cpu_frequencies(high, low, duration_steps=1)

        got_high, got_low, got_total, model = base._allocate_cpu_frequencies()
        np.testing.assert_allclose(got_high, high)
        np.testing.assert_allclose(got_low, low)
        np.testing.assert_allclose(got_total, high + low)
        self.assertEqual(model, "total_cubic")
        self.assertIsNone(base._external_cpu_frequency_override)

    def test_queue_wait_cost_respects_priority_and_link_backlog(self):
        base = SimpleNamespace(
            N=3,
            C=1000.0,
            F_max_array=np.full(3, 10e9, dtype=np.float64),
            Q_loc_high=np.asarray([1.0, 2.0, 3.0]),
            Q_loc_low=np.asarray([10.0, 20.0, 30.0]),
            Q_tx_high=np.asarray(
                [[0.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            Q_tx_low=np.asarray(
                [[0.0, 8.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        )
        rates = np.full((3, 3), 2.0, dtype=np.float64)
        waits = _queue_wait_costs_s(
            base,
            np.asarray(["high", "low"], dtype=object),
            np.asarray([0, 0], dtype=np.int64),
            rates,
        )

        self.assertGreater(waits[1, 1], waits[0, 1])
        self.assertGreater(waits[0, 1], waits[0, 0])

    def test_adapter_executes_solver_route_and_cpu_allocation(self):
        env = HRL_Wrapper(
            num_uavs=3,
            num_users=9,
            cpu_alloc_mode="priority_demand",
            micro_diag_bias=1.0,
            micro_action_scale=1.0,
            strict_hierarchy_validation=True,
            pause_service_during_flight=True,
        )
        env.env.F_max_array[:] = 22e9
        args = SimpleNamespace(
            xu_sca_max_iter=2,
            xu_sca_tol=0.01,
            xu_sca_proximal_rho=1e-3,
            xu_sca_proximal_cpu=1e-3,
            xu_sca_enforce_frame_deadline=0,
            xu_sca_use_priority_delay_weights=1,
            sca_solver="CLARABEL",
            sca_delay_weight=0.3,
            sca_energy_weight=0.7,
        )
        configure_xu2024_wrapper(env, args)
        env.reset(seed=17)
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))

        zero_action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(env.env.deployment_steps):
            _, _, _, _, info = env.micro_step(zero_action)
            self.assertTrue(info["service_paused"])

        env.prepare_resource_frame()
        action = solve_xu2024_micro_action(env)
        expected_route = np.asarray(env._xu2024_last_route)
        frame = env.step_resource_frame(action)
        info = frame["infos"][0]

        np.testing.assert_allclose(
            info["split_ratios_high"],
            expected_route[0],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            info["split_ratios_low"],
            expected_route[1],
            atol=2e-6,
        )
        self.assertEqual(getattr(env, "_xu2024_solve_count", 0), 1)
        self.assertGreater(
            getattr(env, "_xu2024_total_solve_seconds", 0.0),
            0.0,
        )
        self.assertTrue(np.all(np.isfinite(info["f_total_per_uav"])))
        self.assertTrue(
            np.all(info["f_total_per_uav"] <= env.env.F_max_array + 1e-5)
        )


if __name__ == "__main__":
    unittest.main()
