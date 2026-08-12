import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_eval_reproduction import compare_rows


class EvalReproductionTests(unittest.TestCase):
    def test_runtime_fields_are_ignored_but_scientific_fields_are_not(self):
        reference = {
            20: {"episode": "20", "J": "0.5", "runtime": "1.0"}
        }
        candidate = {
            20: {"episode": "20", "J": "0.5", "runtime": "2.0"}
        }
        matched = compare_rows(
            reference,
            candidate,
            ignored_fields={"runtime"},
        )
        self.assertTrue(matched["all_scientific_fields_match"])

        candidate[20]["J"] = "0.6"
        mismatched = compare_rows(
            reference,
            candidate,
            ignored_fields={"runtime"},
        )
        self.assertFalse(mismatched["all_scientific_fields_match"])


if __name__ == "__main__":
    unittest.main()
