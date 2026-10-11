"""Fixed-task parent identity, completion and service-order regression tests."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.tests.test_physical_contract import LOCAL, SCENE, admit, step


def fixed_environment():
    env = HotspotIndustrialEnv(SCENE, ledger_enabled=True)
    # An explicit empty external arrival tape; no stochastic frame generation.
    env._scene.frame = lambda frame: ()
    return env


class ReferenceParentTests(unittest.TestCase):
    def test_service_energy_uses_actual_work_below_each_budget(self):
        env = fixed_environment()
        env.physics.compute_u2g_rate_bps = lambda: np.full(39, 10.)
        allocate = env.physics.allocate_u2u_resources

        def known_transfer_rate(active):
            resources = allocate(active)
            resources["u2u_rate_bps"] = np.asarray(active, float) * 10.
            return resources

        env.physics.allocate_u2u_resources = known_transfer_rate
        cpu_frequency = env.config["resources"]["cpu_hz"][1] / 4
        admit(env, bits=.25, cycles=1e12)
        admit(env, bits=1., cycles=cpu_frequency * env.dt, source=1, ready=True)
        admit(env, bits=.125, cycles=1e12, source=2, ready=True)
        route = LOCAL.copy()
        route[0, 2] = [1., 0., 0.]
        reading = step(env, route)
        expected = dict(upload=.25 / 10 * env.physics.p_gt,
            u2u=.125 / 10 * env.physics.u2u_tx_budget_w,
            cpu=env.config["resources"]["cpu_kappa"] * cpu_frequency**3 * env.dt)
        for name, energy in expected.items():
            self.assertAlmostEqual(reading["energy_j"][name], energy)
            self.assertAlmostEqual(sum(row["energy_j"] for row in env.ledger
                if row["kind"] == name + "_service"), energy)
        self.assertEqual(env.cpu_served_cycles, cpu_frequency * env.dt)
        self.assertAlmostEqual(reading["holding_task_s"], 3.)
        self.assertAlmostEqual(reading["cost"], .09 * 3 +
            (sum(expected.values()) + .005*reading["energy_j"]["propulsion"]) / 140)

    def test_training_requires_confirmed_flags_and_strict_discounts(self):
        from hrl_v2.run import require_research_protocol, validate_discounts
        from hrl_v2.timing import ClockConfig
        flags = dict(training_enabled=True, discounts_confirmed=True, schedule_confirmed=True)
        for config in ({}, {"research_protocol_confirmed": True},
                       {"research": dict(flags, schedule_confirmed=False)}):
            with self.assertRaisesRegex(RuntimeError, "Research training is disabled"):
                require_research_protocol(config)
        with self.assertRaisesRegex(RuntimeError, "0 < gamma < 1"):
            require_research_protocol(dict(research=flags, objective=dict(gamma_micro=1., gamma_macro=1.)))
        require_research_protocol(dict(research=flags, objective=dict(gamma_micro=.99, gamma_macro=.99**10)))
        clocks = ClockConfig(.1, .5, 5., 900.)
        good = dict(objective=dict(gamma_micro=.99, gamma_macro=.99**10), ppo=dict(gamma=.99**10, gae_lambda=.95))
        validate_discounts(good, clocks)
        for bad in (dict(good, ppo=dict(gamma=.99, gae_lambda=.95)),
                    dict(good, objective=dict(gamma_micro=.99, gamma_macro=.99), ppo=dict(gamma=.99, gae_lambda=.95))):
            with self.assertRaises(ValueError):
                validate_discounts(bad, clocks)

    def test_forward_keeps_identity_birth_and_waits_for_destination_cpu(self):
        env = fixed_environment()
        for _ in range(5):
            step(env)
        parent = admit(env, bits=1., cycles=1e9, source=0, ready=True)
        arrival = parent.arrival
        route = LOCAL.copy()
        route[0, 0] = [0., 0., 1.]
        step(env, route)
        fid, = parent.fragment_ids
        fragment = env.fragments[fid]
        self.assertIs(env.parents[arrival.task_id], parent)
        self.assertIs(parent.arrival, arrival)
        self.assertEqual((fragment.parent_id, fragment.source, fragment.destination), (arrival.task_id, 0, 2))
        self.assertEqual(arrival.birth_s, .5)
        self.assertEqual(fragment.stage, "cpu")
        self.assertIn(fid, env.compute[0][2])
        self.assertNotIn(fid, env.compute[0][0])
        self.assertFalse(env.transfer[0][0][2])
        self.assertIsNone(parent.finish_s)
        self.assertEqual(fragment.compute_remaining, arrival.compute_cycles)
        step(env, route)
        self.assertEqual(fragment.stage, "complete")
        self.assertAlmostEqual(parent.finish_s, .7)
        self.assertEqual(parent.arrival.birth_s, .5)
        cpu = [row for row in env.ledger if row["kind"] == "cpu_service"]
        self.assertEqual([row["fragment_id"] for row in cpu], [fid])

    def test_deliberately_late_fragment_blocks_parent_without_double_completion(self):
        env = fixed_environment()
        parent = admit(env, bits=100., cycles=1e9, ready=True)
        route = LOCAL.copy()
        route[0, 0] = [.3, .5, .2]
        original_allocate = env.physics.allocate_u2u_resources

        def delayed_link(active):
            resources = original_allocate(active)
            if env.step_index < 2:
                resources["u2u_rate_bps"][0, 2] = 0.
            return resources

        with patch.object(env.physics, "allocate_u2u_resources", side_effect=delayed_link):
            readings = [step(env, route), step(env, route)]
            children = [env.fragments[fid] for fid in parent.fragment_ids]
            self.assertEqual([f.finish_s is not None for f in children], [True, True, False])
            self.assertIsNone(parent.finish_s)
            self.assertIn(parent.arrival.task_id, env._active_parents)
            self.assertFalse(any(row["kind"] == "parent_complete" for row in env.ledger))
            allocations = [(f.fragment_id, f.destination, f.input_bits, f.compute_cycles) for f in children]
            routed_at = parent.route_time_s
            with self.assertRaisesRegex(RuntimeError, "only be routed once"):
                env._route((parent.arrival.task_id,), LOCAL)
            # Change the control while a child is still blocked. The existing
            # remote work must remain remote, and no second split is permitted.
            for _ in range(2):
                readings.append(step(env, LOCAL))
            self.assertEqual([(f.fragment_id, f.destination, f.input_bits, f.compute_cycles)
                              for f in env.fragments.values()], allocations)
            self.assertEqual(parent.route_time_s, routed_at)
        self.assertEqual(parent.finish_s, .4)
        self.assertEqual(parent.finish_s, max(f.finish_s for f in children))
        self.assertAlmostEqual(sum(f.input_bits for f in children), parent.arrival.input_bits)
        self.assertAlmostEqual(sum(f.compute_cycles for f in children), parent.arrival.compute_cycles)
        self.assertEqual(len(children), 3)
        self.assertEqual(sum(r["holding_task_s"] for r in readings), 4.)
        # The route can change after creation; existing fragments are immutable allocations.
        frozen = copy.deepcopy(env.fragments)
        for _ in range(3):
            self.assertEqual(step(env)["holding_task_s"], 0.)
        self.assertEqual(env.fragments, frozen)
        self.assertEqual(len([r for r in env.ledger if r["kind"] == "parent_complete"]), 1)
        self.assertEqual(len([r for r in env.ledger if r["kind"] == "fragment_created"]), 3)
        with self.assertRaisesRegex(RuntimeError, "only be routed once"):
            env._route((parent.arrival.task_id,), LOCAL)
        self.assertEqual(len(parent.fragment_ids), 3)

    def test_u2u_preempts_low_priority_but_preserves_fifo_within_priority(self):
        env = fixed_environment()
        actual = env.physics.allocate_u2u_resources

        def capped_rate(active):
            resources = actual(active)
            resources["u2u_rate_bps"] = np.asarray(active, float) * 100.
            return resources

        route = LOCAL.copy()
        route[:, 0] = [0., 1., 0.]
        low = admit(env, bits=40., cycles=1e12, priority=1, ready=True)
        with patch.object(env.physics, "allocate_u2u_resources", side_effect=capped_rate):
            step(env, route)
            high1 = admit(env, bits=15., cycles=1e12, ready=True)
            high2 = admit(env, bits=15., cycles=1e12, ready=True)
            step(env, route)
            get = lambda p: env.fragments[p.fragment_ids[0]].transfer_remaining
            self.assertEqual((get(low), get(high1), get(high2)), (30., 5., 15.))
            step(env, route)
            self.assertEqual((get(low), get(high1), get(high2)), (30., 0., 10.))
            step(env, route)
            self.assertEqual((get(low), get(high1), get(high2)), (30., 0., 0.))
            step(env, route)
            self.assertEqual(get(low), 20.)
        self.assertEqual(env.tx_served_bits, 50.)


if __name__ == "__main__":
    unittest.main()
