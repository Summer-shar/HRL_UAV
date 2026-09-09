import copy
import unittest

from scripts.compare_training_budget import budget_difference_only, prefix_check


class TrainingBudgetAnalysisTests(unittest.TestCase):
    def test_prefix_requires_all_old_rows_and_all_fields(self):
        old = [{"window": "1", "reward": "-1"}, {"window": "2", "reward": "-2"}]
        self.assertTrue(prefix_check(old, old + [{"window": "3"}])["equal"])
        self.assertFalse(prefix_check(old, old[:1])["equal"])
        changed = copy.deepcopy(old)
        changed[1]["reward"] = "-3"
        self.assertEqual(prefix_check(old, changed)["different_fields"], ["reward"])


    def test_budget_comparison_rejects_reward_or_lr_changes(self):
        old = {"training": {"pretrain_windows": 60, "actor_lr": 3e-5},
               "qualification": {"max_stage_seconds": 1200}, "reward": {"gamma": .99}}
        new = copy.deepcopy(old)
        new["training"]["pretrain_windows"] = 300
        new["qualification"]["max_stage_seconds"] = 7200
        self.assertTrue(budget_difference_only(old, new))
        new["reward"]["gamma"] = .995
        self.assertFalse(budget_difference_only(old, new))
        new["reward"]["gamma"] = .99
        new["training"]["actor_lr"] = 1e-4
        self.assertFalse(budget_difference_only(old, new))


if __name__ == "__main__":
    unittest.main()
