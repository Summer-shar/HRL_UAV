"""Causal controller inputs and continuing-window evaluation, without training."""
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.industrial_scene import Arrival
from hrl_v2.features import arrival_array, build
from hrl_v2.metrics import finish, pool, snapshot
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.reports import ReportV2
from hrl_v2.timing import ClockConfig


SCENE = json.loads((Path(__file__).parents[1]/"configs/physical_scene.json").read_text())
LOCAL = np.tile(np.eye(3), (2, 1, 1)).astype(np.float64)


def adapter(seconds=5.):
    a = PhysicsAdapter(SCENE, timing=ClockConfig(.1, .5, 5., seconds))
    a.reset(arrival_seed=1, layout_seed=2, load_seed=3)
    return a


def advance(a, ticks):
    for _ in range(ticks):
        if a.env.step_index % a.timing.physical_per_upper == 0:
            a.start_macro(np.full(6, .2))
        if a.env.step_index % a.timing.physical_per_lower == 0:
            a.prepare()
        a.step(LOCAL)


class ObservationTests(unittest.TestCase):
    def test_rollout_length_cannot_change_report_or_policy_features(self):
        short, long = adapter(5.), adapter(60.)
        for a in (short, long):
            advance(a, 5)
        self.assertEqual(short.report(), long.report())
        self.assertEqual(asdict(build(short.report())), asdict(build(long.report())))
        names = {f.name for f in fields(ReportV2)}
        self.assertNotIn("remaining_ticks", names)
        self.assertFalse(any("rollout" in name for name in names))

    def test_absolute_clock_translation_preserves_policy_input(self):
        a = adapter()
        advance(a, 5)
        r = a.report()
        offset = 10000
        seconds = offset*r.physical_dt
        shifted = replace(r, tick=r.tick+offset,
            parents=tuple(replace(p, birth=p.birth+seconds,
                routed_at=None if p.routed_at is None else p.routed_at+seconds) for p in r.parents),
            fragments=tuple(replace(f, entered=f.entered+seconds) for f in r.fragments))
        original, later = build(r), build(shifted)
        for kind in original.features:
            np.testing.assert_allclose(original.features[kind], later.features[kind], atol=1e-12)
        self.assertEqual(original.clock, later.clock)
        self.assertEqual(original.edges, later.edges)

    def test_task_age_motion_phase_and_remaining_motion_are_retained(self):
        a = adapter()
        a.start_macro(np.full(6, .2))
        r = a.prepare()
        self.assertTrue(r.parents)
        graph = build(r)
        a.step(LOCAL)
        later = a.report()
        self.assertEqual(r.phase, "adjustment")
        self.assertEqual(later.phase, "adjustment")
        self.assertAlmostEqual(later.remaining_adjustment, r.remaining_adjustment-a.env.dt)
        self.assertGreater(build(later).features["parent"][0][5], graph.features["parent"][0][5])
        self.assertNotEqual(build(later).clock, graph.clock)
        for _ in range(9):
            if a.env.step_index % a.env.frame_steps == 0:
                a.prepare()
            a.step(LOCAL)
        self.assertEqual(a.report().phase, "hold")
        self.assertEqual(a.report().remaining_adjustment, 0.)

    def test_report_never_reads_hidden_generator_or_future_streams(self):
        a = adapter()
        advance(a, 5)
        expected = a.report()

        class HiddenGenerator:
            def __getattribute__(self, name):
                raise AssertionError("Policy report read hidden generator: "+name)

        with patch.object(a.env, "_scene", HiddenGenerator()):
            a.env._invalidate_view()
            actual = a.report()
            self.assertEqual(expected, actual)
            self.assertEqual(asdict(build(expected)), asdict(build(actual)))

    def test_arrival_features_read_observed_prefix_only(self):
        history = np.ones((4, 39, 6))
        changed_future = history.copy()
        changed_future[2:] = np.nan
        np.testing.assert_array_equal(arrival_array(history, 2, "cpu"),
                                      arrival_array(changed_future, 2, "cpu"))
        for cursor in (-1, 5):
            with self.assertRaises(ValueError):
                arrival_array(history, cursor, "cpu")


class WindowMetricsTests(unittest.TestCase):
    def test_cutoff_censors_live_work_without_reset_or_failed_labels(self):
        a = adapter()
        advance(a, 50)
        before = a.report()
        result = finish(a)
        self.assertEqual(before, a.report())
        self.assertFalse(result["terminated"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["seconds"], 5.)
        self.assertEqual(result["unfinished_parents"], len(before.parents))
        self.assertTrue(any(p["censored"] for p in result["parents"]))
        for p in result["parents"]:
            if p["censored"]:
                self.assertIsNone(p["delay_s"])
                self.assertIsNone(p["finish_s"])
                self.assertGreaterEqual(p["cutoff_age_s"], 0.)
        for forbidden in ("failed", "dropped", "deadline_missed"):
            self.assertNotIn(forbidden, result)
        self.assertEqual(set(result["energy_j"]), {"upload", "u2u", "cpu", "propulsion"})
        self.assertTrue({"upload_bits", "ready_cycles", "transfer_bits", "cpu_cycles"}
                        <= set(result["cutoff_backlog"]))
        self.assertAlmostEqual(result["holding_identity_error"], 0., places=8)

    def test_adjacent_windows_account_for_carry_in_and_lifetime_delay(self):
        a = adapter()
        events = {
            0: (Arrival(0, 0, 0, "sensor", 0, 0., 1., 6e10),
                Arrival(1, 13, 1, "sensor", 1, 0., 1e12, 1e14)),
            10: (Arrival(2, 26, 2, "sensor", 1, 5., 1., 1.),)}
        with patch.object(a.env._scene, "frame", side_effect=lambda frame: events.get(frame, ())):
            begin = snapshot(a)
            advance(a, 50)
            first = finish(a, begin)
            begin = snapshot(a)
            advance(a, 50)
            second = finish(a, begin)
        self.assertEqual((first["arrived_parents"], first["unfinished_parents"]), (2, 2))
        self.assertEqual((second["carry_in_parents"], second["arrived_parents"]), (2, 1))
        self.assertEqual((second["completed_parents"], second["unfinished_parents"]), (2, 1))
        self.assertAlmostEqual(second["unfinished_rate"], 1/3)
        carried = next(p for p in second["parents"] if p["parent"] == 0)
        self.assertTrue(carried["carry_in"])
        self.assertAlmostEqual(carried["delay_s"], 6.1)
        self.assertAlmostEqual(carried["observed_residence_s"], 1.1)
        self.assertAlmostEqual(first["cost_sum"]+second["cost_sum"], a.env.cost_sum)
        self.assertAlmostEqual(first["holding_task_s"]+second["holding_task_s"], a.env.holding)
        for name, value in a.env.energy.items():
            self.assertAlmostEqual(first["energy_j"][name]+second["energy_j"][name], value)
        for name, value in a.queue_integrals.items():
            self.assertAlmostEqual(first["queue_integrals_physical_grid"][name]
                +second["queue_integrals_physical_grid"][name], value)
        for row in second["priority"]:
            self.assertEqual(row["carry_in"]+row["arrived"], row["completed"]+row["unfinished"])
        pooled = pool((first, second))
        self.assertAlmostEqual(pooled["cost_per_s"], a.env.cost_sum/10)
        self.assertEqual(pooled["completed_parents"], 2)
        self.assertEqual(pooled["energy_j"], a.env.energy)
        self.assertEqual(pooled["oldest_unfinished_age_s"], 10.)
        self.assertEqual(pooled["mean_cutoff_backlog"]["upload_parents"], 1.)
        self.assertIsNotNone(pooled["priority"][0]["mean_delay_ms"])
        self.assertIsNotNone(pooled["priority"][1]["mean_delay_ms"])


if __name__ == "__main__":
    unittest.main()
