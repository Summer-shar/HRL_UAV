import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
V40 = ROOT / "configs" / "tcom_v40_micro_warmstart.json"
V41 = ROOT / "configs" / "tcom_v41_micro_warmstart_500ep.json"


class V41MicroWarmstart500EpisodeTests(unittest.TestCase):
    def test_v41_changes_only_identity_and_training_horizon(self):
        old = json.loads(V40.read_text(encoding="utf-8"))
        new = json.loads(V41.read_text(encoding="utf-8"))
        allowed = {
            "results_version",
            "micro_pretrain_source_results_version",
            "micro_pretrain_episodes",
        }
        changed = {
            key
            for key in set(old) | set(new)
            if old.get(key) != new.get(key)
        }
        self.assertEqual(changed, allowed)
        self.assertEqual(new["micro_pretrain_episodes"], 500)

    def test_v41_keeps_frozen_warmstart_schedule(self):
        config = json.loads(V41.read_text(encoding="utf-8"))
        self.assertEqual(config["micro_pretrain_prefill_episodes"], 20)
        self.assertEqual(config["micro_pretrain_actor_freeze_episodes"], 50)
        self.assertEqual(config["micro_pretrain_checkpoint"], "final")
        self.assertEqual(config["formal_seeds"], [1])


if __name__ == "__main__":
    unittest.main()
