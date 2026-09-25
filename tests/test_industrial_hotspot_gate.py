"""Pure admission, cache and budget tests: no environment steps or torch imports."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from hrl_uav.industrial_hotspot_gate import (TechnicalEvaluationError, digest,
    evaluate_admission, initial_gate_state, advance_gate, pooled_metrics)
from hrl_uav.industrial_hotspot_evaluation import (CausalReferenceDeployment,
    EvaluationBudget, EvaluationBudgetExceeded, _load_cache, _cache_identity,
    atomic_json, validate_panel)


RULES = dict(version="test_v1", cost_ratio_to_local_max=.95,
    per_scene_cost_ratio_to_local_max=1.02, per_scene_required_count=4,
    pooled_high_completion_deficit_max_percentage_points=1.,
    pooled_low_unfinished_ratio_to_local_max=.9)


def rows(cost=90., low=9, high_completed=99):
    result = []
    for i, perm in enumerate(itertools.permutations(range(3))):
        result.append(dict(status="complete", technical_valid=True, frozen_verified=True,
            metrics_frozen_before_priming=True, scenario_id=str(i), scenario_sha=str(i),
            realization_sha=str(i), event_sha256=str(i), profile_permutation=list(perm), low_unfinished=low,
            metrics=dict(seconds=900., cost_sum=cost, holding_task_s=100.,
                energy_j=dict(cpu=1., propulsion=1., upload=1., u2u=1.),
                priority=[dict(arrived=100, completed=high_completed, delay_sum_s=10.),
                          dict(arrived=100, completed=100-low, delay_sum_s=20.)])))
    return result


def receipt(candidate, passed=True):
    measurement = evaluate_admission(rows(90 if passed else 110), rows(100, 10), rows(80, 8), RULES)
    return dict(status="complete", candidate_sha=candidate, evaluation_file_sha=digest(measurement),
        identity=dict(candidate_sha=candidate, scene_contract_sha="scene", dev_manifest_sha="dev",
            admission_rule_version=RULES["version"], evaluator_source_sha="source"),
        gate_result=measurement, quality_passed=measurement["quality_passed"])


class AdmissionTests(unittest.TestCase):
    def test_threshold_and_pooled_delay(self):
        result = evaluate_admission(rows(), rows(100, 10), rows(80, 8), RULES)
        self.assertTrue(result["quality_passed"])
        pooled = result["pooled"]["sac"]
        self.assertEqual(pooled["priority"][1]["unfinished"], 54)
        self.assertAlmostEqual(pooled["priority"][0]["mean_delay_ms"], 1000*60/594)
        self.assertTrue(result["cvx_reference_only"])

    def test_low_tail_cutoff_not_cpu_queue(self):
        result = evaluate_admission(rows(low=10), rows(100, 10), rows(80, 8), RULES)
        self.assertFalse(result["checks"]["low_unfinished"])
        wrong = rows()
        wrong[0]["low_unfinished"] += 1
        with self.assertRaises(TechnicalEvaluationError):
            evaluate_admission(wrong, rows(100, 10), rows(80, 8), RULES)

    def test_local_zero_requires_zero(self):
        result = evaluate_admission(rows(low=1), rows(100, 0), rows(80, 0), RULES)
        self.assertFalse(result["quality_passed"])
        self.assertTrue(evaluate_admission(rows(low=0), rows(100, 0), rows(80, 0), RULES)["quality_passed"])

    def test_nan_and_unpaired_are_technical_failures(self):
        bad = rows()
        bad[0]["metrics"]["cost_sum"] = float("nan")
        with self.assertRaises(TechnicalEvaluationError):
            evaluate_admission(bad, rows(100, 10), rows(80, 8), RULES)
        bad = rows()
        bad[0]["event_sha256"] = "wrong"
        with self.assertRaises(TechnicalEvaluationError):
            evaluate_admission(bad, rows(100, 10), rows(80, 8), RULES)

    def test_no_high_arrivals_is_not_an_automatic_pass(self):
        groups = [rows(), rows(100, 10), rows(80, 8)]
        for group in groups:
            for row in group:
                row["metrics"]["priority"][0] = dict(arrived=0, completed=0, delay_sum_s=0.)
        self.assertIsNone(pooled_metrics(groups[0])["priority"][0]["completion_rate"])
        with self.assertRaises(TechnicalEvaluationError):
            evaluate_admission(*groups, RULES)

    def test_two_distinct_candidates_and_idempotency(self):
        original = initial_gate_state([2, 3, 4], 4)
        once = advance_gate(original, 2, "one", receipt("one"))
        self.assertEqual(original["history"], [])
        self.assertEqual(once, advance_gate(once, 2, "one", receipt("one")))
        self.assertEqual(once["next_check_episode"], 3)
        final = advance_gate(once, 3, "two", receipt("two"))
        self.assertEqual(final["status"], "admitted")
        self.assertEqual(final["consecutive_passes"], 2)
        self.assertIsNone(final["next_check_episode"])

    def test_fail_resets_and_cap_first_pass_needs_review(self):
        state = initial_gate_state([2, 3, 4], 4)
        state = advance_gate(state, 2, "one", receipt("one"))
        state = advance_gate(state, 3, "two", receipt("two", False))
        self.assertEqual(state["consecutive"], 0)
        state = advance_gate(state, 4, "three", receipt("three"))
        self.assertEqual(state["status"], "needs_review")

    def test_changed_context_same_candidate_or_missing_node_rejected(self):
        state = advance_gate(initial_gate_state([2, 3, 4], 4), 2, "one", receipt("one"))
        with self.assertRaises(TechnicalEvaluationError):
            advance_gate(state, 3, "one", receipt("one"))
        with self.assertRaises(TechnicalEvaluationError):
            advance_gate(state, 4, "two", receipt("two"))
        changed = receipt("two")
        changed["identity"]["dev_manifest_sha"] = "other"
        with self.assertRaises(TechnicalEvaluationError):
            advance_gate(state, 3, "two", changed)
        changed = receipt("two")
        changed["status"] = "failed"
        with self.assertRaises(TechnicalEvaluationError):
            advance_gate(state, 3, "two", changed)

    def test_forged_quality_flag_rejected(self):
        bad = receipt("one", False)
        bad["quality_passed"] = bad["gate_result"]["quality_passed"] = True
        with self.assertRaises(TechnicalEvaluationError):
            advance_gate(initial_gate_state([1], 1), 1, "one", bad)


class EvaluationBookkeepingTests(unittest.TestCase):
    def test_crashed_reservation_cannot_be_forgotten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"budget.json"
            a = EvaluationBudget(path, 100, {"scope":"test"})
            token = a.begin("first", 60)
            a.progress(token, 20)
            b = EvaluationBudget(path, 100, {"scope":"test"})
            self.assertEqual(b.snapshot()["charged_steps"], 60)
            with self.assertRaises(EvaluationBudgetExceeded):
                b.begin("retry", 60)
            a.finish(token, 20, "failed", "controlled failure")
            retry = b.begin("retry", 60)
            b.finish(retry, 60, "complete")
            self.assertEqual(b.snapshot()["charged_steps"], 80)

    def test_budget_contract_and_double_commit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"budget.json"
            a = EvaluationBudget(path, 50, {"scope":"test"})
            token = a.begin("one", 50)
            a.finish(token, 50, "complete")
            with self.assertRaises(ValueError):
                a.finish(token, 50, "complete")
            with self.assertRaises(ValueError):
                EvaluationBudget(path, 100, {"scope":"test"})

    def test_cache_digest_and_load_assignment_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"cached.json"
            identity = _cache_identity({"scene":"s"}, {"load_seed":1}, "hover_sac", "checkpoint")
            changed = _cache_identity({"scene":"s"}, {"load_seed":2}, "hover_sac", "checkpoint")
            self.assertNotEqual(identity, changed)
            row = rows()[0]
            atomic_json(path, dict(identity=identity, row=row, row_sha256=digest(row)))
            self.assertEqual(_load_cache(path, identity), row)
            with self.assertRaises(TechnicalEvaluationError):
                _load_cache(path, changed)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["row"]["metrics"]["cost_sum"] += 1
            atomic_json(path, data)
            with self.assertRaises(TechnicalEvaluationError):
                _load_cache(path, identity)

    def test_six_explicit_permutations(self):
        panel = dict(seconds=5, scenarios=[dict(scenario_id=str(i), layout_seed=i,
            load_seed=10+i, arrival_seed=20+i, routing_seed=30+i, profile_permutation=list(p))
            for i,p in enumerate(itertools.permutations(range(3)))])
        self.assertEqual(validate_panel(panel, research=False), panel)
        panel["scenarios"][0]["profile_permutation"] = panel["scenarios"][1]["profile_permutation"]
        with self.assertRaises(ValueError):
            validate_panel(panel, research=False)

    def test_reference_reads_only_legal_report_and_keeps_once_chosen_target(self):
        rule = CausalReferenceDeployment()
        terminals = tuple(SimpleNamespace(source=i, position_m=(100.+10*i, 50.), recent_bits=(1.,2.)) for i in range(3))
        base = SimpleNamespace(time_s=5., terminals=terminals, positions_m=((0.,0.),)*3)
        self.assertTrue(np.array_equal(rule(SimpleNamespace(base=base)), np.zeros(6)))
        base.time_s = 10.
        action = rule(SimpleNamespace(base=base)).reshape(3,2)
        self.assertTrue(np.all(np.linalg.norm(action, axis=1) <= 1.+1e-12))
        target = rule.target.copy()
        for terminal in terminals:
            terminal.position_m = (900.,900.)
        base.time_s = 15.
        rule(SimpleNamespace(base=base))
        self.assertTrue(np.array_equal(rule.target, target))


if __name__ == "__main__":
    unittest.main()
