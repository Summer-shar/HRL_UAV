"""Run only fixed-task/scripted environment tests with an audited step cap.

Usage: python -m hrl_v2.tests.reference_validation --output-dir outputs/reference_validation
The persistent output-dir budget includes earlier runs, including failed tests.
No learner, optimizer, reference framework or random policy is executed.
"""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys
import unittest
from unittest.mock import patch

from hrl_uav.industrial_env import IndustrialEnv


LIMIT = 20000
MODULES = (
    "hrl_v2.tests.test_physical_contract",
    "hrl_v2.tests.test_reference_parents",
    "hrl_v2.tests.test_reference_physics",
    "hrl_v2.tests.test_reference_continuation",
    "hrl_v2.tests.test_observation_metrics.ObservationTests.test_report_never_reads_hidden_generator_or_future_streams",
    "hrl_v2.tests.test_observation_metrics.ObservationTests.test_arrival_features_read_observed_prefix_only",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test", action="append", help="Rerun a selected test within the fixed-task allowlist")
    args = parser.parse_args()
    selection = args.test or MODULES
    if any(not any(name == module or name.startswith(module + ".") for module in MODULES)
           for name in selection):
        parser.error("Selected test is outside the fixed-task environment allowlist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = args.output_dir / "budget.json"
    prior = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else {"runs": []}
    previous = sum(row["physical_steps"] for row in prior["runs"])
    if previous >= LIMIT:
        raise RuntimeError("This validation output directory has exhausted its physical step budget")
    counters = {"physical_steps": 0, "rejected_step_calls": 0}
    original = IndustrialEnv.micro_step

    def counted(env, route):
        if previous + counters["physical_steps"] >= LIMIT:
            raise RuntimeError("Reference validation physical step limit reached")
        before = env.step_index
        try:
            return original(env, route)
        finally:
            delta = env.step_index - before
            counters["physical_steps"] += delta
            if not delta:
                counters["rejected_step_calls"] += 1

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            self.start_steps = counters["physical_steps"]
            super().startTest(test)

        def stopTest(self, test):
            per_test.append(dict(test=test.id(), physical_steps=counters["physical_steps"]-self.start_steps))
            super().stopTest(test)

    per_test = []
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(selection)
    # Existing quiet_scene tests have zero arrival rate. Bypass even their RNG
    # draws: all work is explicitly injected through their admit() helper.
    from hrl_uav.industrial_hotspot_scene import HotspotIndustrialScene
    with patch.object(IndustrialEnv, "micro_step", counted), \
         patch.object(HotspotIndustrialScene, "frame", lambda self, frame: ()):
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    run_index = len(prior["runs"]) + 1
    record = dict(run=run_index, utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), tests=result.testsRun, successful=result.wasSuccessful(),
        failures=len(result.failures), errors=len(result.errors), **counters,
        fixture_or_class_setup_steps=counters["physical_steps"]-sum(r["physical_steps"] for r in per_test),
        tests_detail=per_test, modules=list(selection), budget_limit=LIMIT,
        cumulative_physical_steps=previous+counters["physical_steps"])
    prior["runs"].append(record)
    budget_path.write_text(json.dumps(prior, indent=2)+"\n", encoding="utf-8")
    (args.output_dir / f"run_{run_index:02d}.txt").write_text(log.getvalue(), encoding="utf-8")
    (args.output_dir / f"run_{run_index:02d}.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(log.getvalue())
    print(json.dumps({k: v for k, v in record.items() if k not in ("tests_detail", "modules")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
