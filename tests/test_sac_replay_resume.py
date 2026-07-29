import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hrl_uav.agents.sac_agent import Actor
from hrl_uav.agents.sac_agent import Entroy
from hrl_uav.agents.sac_agent import Memory


class SACReplayResumeTest(unittest.TestCase):
    def test_entropy_temperature_uses_configured_initial_value(self):
        entropy = Entroy(action_dim=12, initial_alpha=0.1)
        self.assertAlmostEqual(float(entropy.alpha.item()), 0.1, places=6)

    def test_temperature_loss_optimizes_log_alpha(self):
        entropy_a = Entroy(action_dim=2, initial_alpha=1.0)
        entropy_b = Entroy(action_dim=2, initial_alpha=0.1)
        log_prob = torch.full(
            (4, 1),
            -1.0,
            device=entropy_a.log_alpha.device,
        )

        grad_a = torch.autograd.grad(
            entropy_a.temperature_loss(log_prob),
            entropy_a.log_alpha,
        )[0]
        grad_b = torch.autograd.grad(
            entropy_b.temperature_loss(log_prob),
            entropy_b.log_alpha,
        )[0]
        torch.testing.assert_close(grad_a, grad_b)

    def test_actor_update_clips_large_gradients(self):
        actor = Actor(state_dim=3, action_dim=2)
        loss = sum(parameter.sum() for parameter in actor.action_net.parameters())
        loss = loss * 1e6

        preclip_norm = actor.learn(loss, max_grad_norm=0.1)
        postclip_norm = torch.sqrt(
            sum(
                torch.sum(parameter.grad.detach() ** 2)
                for parameter in actor.action_net.parameters()
            )
        ).item()

        self.assertGreater(preclip_norm, 0.1)
        self.assertLessEqual(postclip_norm, 0.1001)

    def test_round_trip_preserves_valid_transitions(self):
        source = Memory(capacity=5, dims=4, recent_window=5, recent_ratio=0.0)
        for value in range(7):
            source.mem[source.memory_counter % source.capacity, :] = value
            source.memory_counter += 1

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.pt"
            torch.save(source.state_dict(), path)
            restored = Memory(
                capacity=5,
                dims=4,
                recent_window=5,
                recent_ratio=0.0,
            )
            restored.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False)
            )

        self.assertEqual(restored.memory_counter, 5)
        np.testing.assert_array_equal(
            restored.mem[:5],
            np.repeat(
                np.arange(2, 7, dtype=np.float32)[:, None],
                4,
                axis=1,
            ),
        )

    def test_width_mismatch_is_rejected(self):
        memory = Memory(capacity=5, dims=4)
        with self.assertRaisesRegex(ValueError, "width"):
            memory.load_state_dict(
                {"transitions": np.zeros((3, 5), dtype=np.float32)}
            )


if __name__ == "__main__":
    unittest.main()
