"""Reproducible, zero-physics unit checks for the external budget tool."""
import importlib.util
from pathlib import Path
import unittest


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("current_seed_budget_tool",ROOT/"tools/estimate_seed_budget.py")
budget=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


class PerformanceToolTests(unittest.TestCase):
    def test_simulated_seconds_are_collection_scaling_denominator(self):
        row=dict(seconds=60,physical_steps=600,physical_seconds=.25248,wall_s=.9149183)
        clocks=dict(physical_dt=.1,rollout_seconds=900)
        actual=budget.full_window_collection_seconds(row,clocks)
        self.assertAlmostEqual(actual,13.7237745,places=10)
        # physical_seconds is CPU time spent in micro_step, not the duration
        # of the simulated window. Changing it must not change extrapolation.
        row["physical_seconds"]=999.
        self.assertEqual(actual,budget.full_window_collection_seconds(row,clocks))

    def test_collection_duration_must_match_physical_tick_count(self):
        with self.assertRaisesRegex(ValueError,"duration and physical tick"):
            budget.full_window_collection_seconds(dict(seconds=30,physical_steps=600,wall_s=1.),
                                                  dict(physical_dt=.1,rollout_seconds=900))

    def test_empirical_range_uses_actual_samples_not_interpolated_p95(self):
        result=budget.coefficient({"sample":dict(median_s=2.,p95_s=999.,
            samples=[dict(seconds=1.),dict(seconds=2.),dict(seconds=3.)])},["sample"])
        self.assertEqual((result["low_s"],result["high_s"]),(1.,3.))

    def test_complete_moving_replay_replaces_its_inference_and_window_json(self):
        # Three two-decision windows: two moving, one hover, one panel.
        # Independent expected price: physics 3*5 + moving 2*10 + hover
        # lower(1+2) + JSON(hover+panel)*3 = 44. Upper standalone costs
        # are deliberately huge to expose accidental double charging.
        def measured(seconds,**extra):
            return dict(median_s=seconds,samples=[dict(seconds=seconds)],**extra)
        times={"evaluation.replay900s":measured(10.,workload=dict(lower_calls=2,upper_calls=2)),
               "inference.lower.h1.rebuild":measured(1.),
               "inference.lower.h120.cached":measured(2.),
               "inference.upper.h0.rebuild":measured(100.),
               "inference.upper.h120.cached":measured(100.),
               "io.metrics_json":measured(3.)}
        rows,scenario=budget.evaluation_prices(dict(rollout_seconds=2.,lower_interval=1.,upper_interval=2.),times,
            windows=3,moving=2,panel_summaries=1,physical_price=dict(low_s=5.,high_s=5.,keys=["physical"]))
        self.assertEqual([sum(row["seconds"][i] for row in rows) for i in (0,1)],[44.,44.])
        by_name={row["name"]:row for row in rows}
        self.assertEqual(by_name["moving_complete_report_replay"]["count"],2)
        self.assertEqual(by_name["hover_lower_cold_inference"]["count"],1)
        self.assertEqual(by_name["hover_lower_cached_inference"]["count"],1)
        self.assertEqual(by_name["uncovered_window_and_panel_json"]["count"],2)
        self.assertFalse(any("upper" in row["name"] for row in rows))
        # Separate all-moving-price scenario: 3*5 + 3*10 + panel*3 = 48.
        self.assertEqual(scenario["conditional_seconds"],[48.,48.])
        self.assertFalse(scenario["strict_upper_bound"])


if __name__=="__main__": unittest.main()
