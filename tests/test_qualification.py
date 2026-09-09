import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import evaluate, smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import metered_evaluate, pretraining_verdict, verify_crn


ROOT = Path(__file__).resolve().parents[1]


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/main.json").read_text())
        self.rules = self.config["qualification"]
        self.baseline = [{"seed": seed, "task_delay_weighted_ms": 620.0, "J": 0.7,
                          "task_completion_rate": 0.999, "queue_total_mbit": 12.0,
                          "mean_queue_compute_mbit": 9.0, "mean_queue_transfer_mbit": 1.0,
                          "mean_queue_upload_mbit": 2.0, "late_queue_slope_mbit_s": 0.0}
                         for seed in (33000, 33001)]
        self.history = [{"window": w, "rows": copy.deepcopy(self.baseline)} for w in (48, 54, 60)]

    def test_finite_stable_quality_passes(self):
        self.assertTrue(pretraining_verdict(self.history, self.baseline, self.rules)["passed"])

    def test_good_final_does_not_hide_bad_tail(self):
        self.history[0]["rows"][0]["task_delay_weighted_ms"] = 1500
        result = pretraining_verdict(self.history, self.baseline, self.rules)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["delay_quality"])
        self.assertFalse(result["checks"]["validation_stability"])

    def test_missing_checkpoint_or_seed_never_passes(self):
        self.assertFalse(pretraining_verdict(self.history[1:], self.baseline, self.rules)["passed"])
        self.history[1]["rows"].pop()
        self.assertFalse(pretraining_verdict(self.history, self.baseline, self.rules)["passed"])

    def test_small_completed_delay_cannot_hide_unfinished_work(self):
        self.history[-1]["rows"][0].update(task_delay_weighted_ms=500, queue_total_mbit=200,
                                          task_completion_rate=0.98, late_queue_slope_mbit_s=0.5)
        result = pretraining_verdict(self.history, self.baseline, self.rules)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["terminal_queue"])
        self.assertFalse(result["checks"]["completion"])

    def test_nan_cannot_pass(self):
        self.history[-1]["rows"][0]["J"] = float("nan")
        self.assertFalse(pretraining_verdict(self.history, self.baseline, self.rules)["passed"])

    def test_mainline_freezes_only_new_bandwidth_not_cpu_or_reward(self):
        self.assertEqual(self.config["environment"]["u2u_bandwidth_mhz"], 6)
        self.assertEqual(self.config["cpu_ghz"], 10)
        self.assertAlmostEqual(sum(self.config["environment"]["zone_loads"])*self.config["environment"]["load_scale"], 24.725)
        self.assertEqual(self.config["reward"]["delay_weight"], 0.9)

    def test_read_only_diagnostics_preserve_actions_rng_and_replay(self):
        torch.set_num_threads(1)
        config = smoke_config(self.config)
        env = QueueMDP(config)
        state = env.reset(1, 6)
        agent = DirectHRL.from_mdp(env, config)
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            initial = metered_evaluate(config, agent, [33000], 6, path, "initial")
            uniform = metered_evaluate(config, UniformRouting(), [33000], 6, path, "uniform")
            ordinary = evaluate(config, agent, [33000], 6, path, "ordinary", macro_enabled=False)
            for key in ("J", "task_delay_weighted_ms", "queue_total_mbit"):
                self.assertEqual(initial[0][key], uniform[0][key])
                self.assertEqual(initial[0][key], ordinary[0][key])
            self.assertTrue(all(verify_crn(path, [33000], "initial", "uniform")["33000"].values()))
        self.assertEqual(agent.replay.memory_counter, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(agent.sac_actor_updates, 0)


if __name__ == "__main__":
    unittest.main()
