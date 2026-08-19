import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from hrl_uav.agents.sac_agent import Actor as SACActor
from hrl_uav.agents.sac_agent import Critic as SACCritic
from hrl_uav.agents.sac_agent import Entroy as SACEntropy
from scripts.train_centralized_macro import load_micro_actor_checkpoint
from scripts.train_centralized_macro import micro_warmstart_phase
from scripts.train_centralized_macro import state_dict_sha256
from scripts.train_centralized_macro import write_micro_actor_only_initialization_audit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tcom_v40_micro_warmstart.json"


class V40MicroWarmstartTests(unittest.TestCase):
    def test_frozen_protocol_values(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["formal_seeds"], [1])
        self.assertEqual(config["micro_pretrain_episodes"], 300)
        self.assertEqual(config["micro_pretrain_prefill_episodes"], 20)
        self.assertEqual(config["micro_pretrain_actor_freeze_episodes"], 50)
        self.assertEqual(config["micro_pretrain_checkpoint"], "final")
        self.assertIn("tcom_v33_soft_energy_random_mobility", config["micro_pretrain_actor_init_checkpoint"])

    def test_phase_boundaries(self):
        args = Namespace(
            sac_prefill_episodes=20,
            sac_actor_freeze_episodes=50,
        )
        self.assertEqual(micro_warmstart_phase(args, 0), "prefill")
        self.assertEqual(micro_warmstart_phase(args, 19), "prefill")
        self.assertEqual(micro_warmstart_phase(args, 20), "critic_calibration")
        self.assertEqual(micro_warmstart_phase(args, 49), "critic_calibration")
        self.assertEqual(micro_warmstart_phase(args, 50), "finetune")

    def test_actor_only_load_keeps_fresh_critic_and_optimizers(self):
        torch.manual_seed(11)
        source_actor = SACActor(8, 4)
        source_critic = SACCritic(8, 4)
        with torch.no_grad():
            for parameter in source_actor.action_net.parameters():
                parameter.add_(0.25)

        torch.manual_seed(29)
        actor = SACActor(8, 4)
        critic = SACCritic(8, 4)
        entropy = SACEntropy(4, initial_alpha=0.1)
        fresh_critic_hash = state_dict_sha256(critic.critic_v.state_dict())

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "source.pt"
            torch.save(
                {
                    "sac_actor": source_actor.action_net.state_dict(),
                    "sac_critic": source_critic.critic_v.state_dict(),
                },
                checkpoint_path,
            )
            resolved = load_micro_actor_checkpoint(str(checkpoint_path), actor)
            self.assertEqual(Path(resolved), checkpoint_path)
            self.assertEqual(
                state_dict_sha256(actor.action_net.state_dict()),
                state_dict_sha256(source_actor.action_net.state_dict()),
            )
            self.assertEqual(
                state_dict_sha256(critic.critic_v.state_dict()),
                fresh_critic_hash,
            )
            args = Namespace(
                sac_prefill_episodes=20,
                sac_actor_freeze_episodes=50,
                micro_replay="",
            )
            audit_path = write_micro_actor_only_initialization_audit(
                tmpdir,
                resolved,
                actor,
                critic,
                entropy,
                args,
            )
            audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
            self.assertTrue(audit["actor_matches_source"])
            self.assertFalse(audit["critic_matches_source"])
            self.assertEqual(audit["actor_optimizer_state_entries"], 0)
            self.assertEqual(audit["critic_optimizer_state_entries"], 0)
            self.assertEqual(audit["entropy_optimizer_state_entries"], 0)
            self.assertFalse(audit["replay_loaded"])


if __name__ == "__main__":
    unittest.main()
