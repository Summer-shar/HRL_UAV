"""Bounded fixture checks; archived timings cannot describe current research."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch

from hrl_v2.benchmark_optimization import make_fixture,fixture_clock,episodes_for,fixed_samples,package,measure,upper_auxiliary_rows
from hrl_v2.budget import require_archived_measurement
from hrl_v2.timing import ClockConfig
from hrl_v2.replay import unpack


class BenchmarkSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clock=ClockConfig(.1,.5,5,5)
        with tempfile.TemporaryDirectory() as directory:
            cls.fixture=make_fixture(Path(directory)/"short.pkl",timing=cls.clock)

    def test_fixture_and_replay_samples_follow_configured_window(self):
        self.assertEqual(self.fixture["physics_steps"],50)
        self.assertEqual(fixture_clock(self.fixture["episode"]),self.clock)
        replicas=episodes_for(package("optimized"),self.fixture,5)
        self.assertEqual(len(replicas),5)
        for replica in replicas:
            self.assertEqual(replica.expected_lower_steps,10)
            self.assertEqual(replica.expected_upper_steps,1)
            self.assertEqual(replica.truncated,[False]*9+[True])
        samples=fixed_samples(package("optimized"),replicas)
        self.assertEqual(len(samples),128)
        self.assertTrue(all(0<=sample.index<10 for sample in samples))

    def test_current_measurement_rejects_frozen_baseline(self):
        with self.assertRaisesRegex(ValueError,"incompatible"):
            measure(SimpleNamespace(variant="baseline"))

    def test_current_measurement_rejects_terminal_fixture(self):
        episode=copy.deepcopy(self.fixture["episode"])
        episode.terminated[-1]=True
        with self.assertRaisesRegex(ValueError,"continuing-service"):
            fixture_clock(episode)

    def test_ppo_timing_auxiliary_uses_real_successor_value(self):
        episode=self.fixture["episode"]
        calls=[]
        class Policy:
            _episode=object()
            def upper(self,report,history): return None,dict(value=report.time)
        def value(report,history,**kwargs):
            calls.append(report)
            return torch.tensor(9+report.time)
        rows=upper_auxiliary_rows(episode,unpack(episode.history),Policy(),value)
        self.assertEqual(calls,[episode.report(episode.high[0]["successor"])])
        self.assertEqual(rows[0]["next_value"],14.)
        self.assertFalse(rows[0]["terminated"])
        self.assertTrue(rows[0]["truncated"])

    def test_budget_requires_archived_opt_in_and_matching_metadata(self):
        archived=dict(status="complete",baseline_commit="137d2f1",
            fixture=dict(physical_steps=9000,history_frames=1800),measurements={"ppo_kl900":{}})
        with self.assertRaisesRegex(ValueError,"archive-only"):
            require_archived_measurement(archived)
        require_archived_measurement(archived,archived_finite_window=True)
        with self.assertRaisesRegex(ValueError,"only archived"):
            require_archived_measurement(dict(archived,schema="continuing_service_benchmark_v1"),
                                         archived_finite_window=True)
        changed=copy.deepcopy(archived)
        changed["fixture"]["physical_steps"]=50
        with self.assertRaisesRegex(ValueError,"only archived"):
            require_archived_measurement(changed,archived_finite_window=True)


if __name__=="__main__": unittest.main()
