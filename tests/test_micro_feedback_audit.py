"""Checks for the read-only feedback audit, not a claim of learning quality."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.experiment import smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.routing import project_rows_to_simplex_tensor
from scripts.audit_micro_feedback import branch, encoding_probe, low_uniform_route, queue_group_preferences
from scripts.check_soft_continuation import sample_with_noise, soft_branch


ROOT = Path(__file__).resolve().parents[1]


class MicroFeedbackAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = smoke_config(json.loads((ROOT/"configs/main.json").read_text()))
        self.mdp = QueueMDP(self.config)
        self.mdp.reset(744, 90)
        self.agent = DirectHRL.from_mdp(self.mdp, self.config)

    def test_strict_simplex_vertex_has_zero_local_score_derivative(self):
        scores = torch.tensor([[-0.8, 0.8, -0.8]], dtype=torch.float64, requires_grad=True)
        route = project_rows_to_simplex_tensor(scores)
        values = torch.tensor([[2., -3., 1.]], dtype=torch.float64)
        gradient, = torch.autograd.grad((route*values).sum(), scores)
        torch.testing.assert_close(route, torch.tensor([[0., 1., 0.]], dtype=torch.float64))
        torch.testing.assert_close(gradient, torch.zeros_like(scores))
        for index in range(3):
            shifted = scores.detach().clone()
            shifted[0, index] += 1e-4
            torch.testing.assert_close(project_rows_to_simplex_tensor(shifted), route)
        # A finite feasible move still improves this explicitly known value.
        better = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float64)
        self.assertGreater(float((better*values).sum()), float((route*values).sum()))

    def test_interior_route_retains_nonzero_value_derivative(self):
        scores = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        route = project_rows_to_simplex_tensor(scores)
        gradient, = torch.autograd.grad((route*torch.tensor([[2., -3., 1.]])).sum(), scores)
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertAlmostEqual(float(gradient.sum()), 0)

    def test_low_only_probe_keeps_high_route_and_input_unchanged(self):
        route = torch.tensor([[[[.8,.1,.1]]*3, [[0.,1.,0.]]*3]])
        original = route.clone()
        changed = low_uniform_route(route)
        torch.testing.assert_close(route, original)
        torch.testing.assert_close(changed[:,0], route[:,0])
        torch.testing.assert_close(changed[:,1], torch.full((1,3,3), 1/3))

    def test_encoder_probe_restores_state_and_only_correct_queue_index_changes(self):
        before = self.mdp.micro_observation().copy()
        result = encoding_probe(self.mdp)
        self.assertEqual(result["expected_index"], 4)
        self.assertTrue(result["monotone"])
        np.testing.assert_array_equal(before, self.mdp.micro_observation())

    def test_queue_groups_partition_predictions_without_claiming_error_labels(self):
        n = self.mdp.observation_spec()["micro"]["dim"]
        data = np.zeros((5, 2*n+38), dtype=np.float32)
        data[:, :n] = self.mdp.micro_observation()
        data[:, 4] = np.log1p(np.asarray([0.,5.,20.,200.,1000.])/10.)
        data[:, n+20:n+38] = 1/3
        checkpoint = {"replay": {"transitions": data}, "micro_state_dim": n,
                      "observations": self.mdp.observation_spec()}
        result = queue_group_preferences(checkpoint, self.agent, np.arange(5))
        self.assertEqual([g["states"] for g in result["groups"]], [2,1,2])
        self.assertEqual(len(result["per_state"]), 5)
        self.assertIn("No physical return labels", result["scope"])
        self.assertNotIn("error_rate", result)

    def test_branches_are_deterministic_copies_without_network_updates(self):
        self.mdp.start_macro(np.zeros(6))
        for _ in range(self.mdp.env.deployment_resource_frames):
            self.mdp.step_frame(np.zeros(18))
        self.mdp.prepare_service()
        state = self.mdp.micro_observation().copy()
        parameters = copy.deepcopy(self.agent.sac_actor.action_net.state_dict())
        first = branch(self.mdp, self.agent, "actor", frames=20)
        second = branch(self.mdp, self.agent, "actor", frames=20)
        self.assertEqual(first, second)
        # Initially the deterministic Actor is uniform; low correction is identical.
        corrected = branch(self.mdp, self.agent, "sustained_low_uniform", frames=20)
        self.assertEqual(first["exogenous_sha256"], corrected["exogenous_sha256"])
        self.assertEqual(first["horizons_frames"], corrected["horizons_frames"])
        np.testing.assert_array_equal(state, self.mdp.micro_observation())
        for key, value in parameters.items():
            torch.testing.assert_close(value, self.agent.sac_actor.action_net.state_dict()[key])
        self.assertEqual(self.agent.sac_updates, 0)

    def test_soft_sampling_matches_actor_distribution(self):
        state = self.mdp.micro_observation()
        from hrl_uav.learning import tensor
        torch.manual_seed(42)
        expected_action, expected_logprob = self.agent.sac_actor.evaluate(tensor(state)[None])
        torch.manual_seed(42)
        epsilon = torch.randn_like(expected_action).cpu().numpy()
        action, logprob = sample_with_noise(self.agent.sac_actor, state, epsilon)
        torch.testing.assert_close(action, expected_action)
        torch.testing.assert_close(logprob, expected_logprob.ravel())

    def test_lower_actor_rate_changes_no_initial_parameters_or_other_rates(self):
        torch.manual_seed(1)
        control = DirectHRL.from_mdp(self.mdp, self.config)
        treatment_config = copy.deepcopy(self.config)
        treatment_config["training"]["pretrain_lr"][0] = 3e-5
        torch.manual_seed(1)
        treatment = DirectHRL.from_mdp(self.mdp, treatment_config)
        for left, right in ((control.sac_actor.action_net, treatment.sac_actor.action_net),
                            (control.sac_critic.critic_v, treatment.sac_critic.critic_v)):
            for a, b in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(treatment.sac_actor.optimizer.param_groups[0]["lr"], 3e-5)
        self.assertEqual(treatment.sac_critic.optimizer.param_groups[0]["lr"],
                         control.sac_critic.optimizer.param_groups[0]["lr"])
        self.assertEqual(treatment.entropy.optimizer.param_groups[0]["lr"],
                         control.entropy.optimizer.param_groups[0]["lr"])
        self.assertEqual(treatment.replay.memory_counter, 0)

    def test_soft_branch_excludes_initial_entropy_and_repeats_exactly(self):
        self.mdp.start_macro(np.zeros(6))
        for _ in range(self.mdp.env.deployment_resource_frames):
            self.mdp.step_frame(np.zeros(18))
        self.mdp.prepare_service()
        options = dict(bypass=False, noise_seed=98, alpha=.05, horizons=(1, 2))
        first = soft_branch(self.mdp, self.agent, **options)
        self.assertEqual(first, soft_branch(self.mdp, self.agent, **options))
        self.assertEqual(first["1"]["entropy_prefix"], 0.)
        self.assertEqual(first["2"]["future_service_entropy_events"], 1)
        for row in first.values():
            self.assertAlmostEqual(row["soft_prefix"], row["physical_prefix"]+row["entropy_prefix"])
            self.assertAlmostEqual(row["soft_with_target_tail"], row["soft_prefix"]+row["target_tail"])
        self.assertEqual(self.agent.sac_updates, 0)


if __name__ == "__main__":
    unittest.main()
