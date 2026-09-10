"""Read-only restoration checks of the interrupted replicas' own anchors."""

from pathlib import Path
import tempfile
import unittest

import torch

from hrl_uav.experiment import isolated_rng
from hrl_uav.priority_grid import file_hash
from hrl_uav.wireless_replicates import ROOT, copy_completed_prefix, load_json, restore_replica
from hrl_uav.wireless_long import rows


class RecoveryTest(unittest.TestCase):
    def test_each_own_anchor_restores_full_learning_state_without_updates(self):
        with isolated_rng(), tempfile.TemporaryDirectory() as tmp:
            for seed in (102, 103):
                parent = ROOT / f"results/sac_only_wireless6_long_seed{seed}"
                manifest = load_json(parent / "manifest.json")
                anchor = manifest["latest_resume_anchor"]
                self.assertEqual(file_hash(anchor["path"]), anchor["sha256"])
                checkpoint = torch.load(anchor["path"], map_location="cpu", weights_only=False)
                agent, _, _, _, audit = restore_replica(manifest["config"], Path(tmp), checkpoint, Path(anchor["path"]))
                self.assertTrue(all(audit["exact_restore_checks"].values()))
                self.assertEqual(agent.replay.memory_counter, 369 * 540)
                self.assertEqual(agent.sac_updates, 9914)
                self.assertEqual(agent.sac_actor_updates, 4907)
                self.assertEqual(agent.ppo_updates, 0)
                self.assertNotEqual(audit["alpha"], .05)

    def test_prefix_excludes_interrupted_windows_and_preserves_source(self):
        parent = ROOT / "results/sac_only_wireless6_long_seed102"
        before = file_hash(parent / "training.csv")
        with tempfile.TemporaryDirectory() as tmp:
            provenance = copy_completed_prefix(parent, Path(tmp), 369)
            kept = rows(Path(tmp) / "training.csv")
            self.assertEqual([r["window"] for r in kept], list(range(1, 370)))
            self.assertEqual(provenance["training.csv"]["excluded_rows"], 2)
            self.assertEqual(file_hash(parent / "training.csv"), before)


if __name__ == "__main__":
    unittest.main()
