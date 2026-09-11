"""Synthetic short functional checks; the real SAC101 smoke is a separate command."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.experiment import isolated_rng
from hrl_uav.geometry import cooperation_aligned_basis
from hrl_uav.learning import tensor
from hrl_uav.mdp import QueueMDP
from hrl_uav.ppo_frozen import RESERVED_SEEDS, SOURCE_DIGESTS, assert_sac_frozen, sac_learning_state
from hrl_uav.ppo_residual_b import (
    HYPERPARAMETERS, ROOT, SEMANTICS, ResidualAgent, ResidualFrozen, ResidualJournal, ResidualMeter,
    baseline_b, evaluate_pair, frozen_guards, prepare_residual_source, project_action, train_smoke_window,
)
from hrl_uav.resumption import capture_rng, restore_rng, state_equal


def config():
    cfg = json.loads((ROOT/"configs/sac_only_wireless6_pilot.json").read_text())
    # Short synthetic fixture only; the real smoke reads the unchanged PPO120 config.
    cfg["training"].update(replay_capacity=64, window_macros=2, joint_windows=120)
    return cfg


def make_agent(cfg):
    model = QueueMDP(cfg)
    model.reset(72000, 2)
    agent = ResidualAgent.from_mdp(model, cfg)
    agent.set_stage("joint")
    return agent


class ResidualGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()

    def test_b_world_target_and_inverse_basis_independent_formula(self):
        current = np.array([[180., 170.], [430., 220.], [610., 700.]])
        centroids = current + np.array([[16., -8.], [-12., 2.], [3., -6.]])
        loads = np.array([12.5, 5.5, 3.5])
        with patch("pathlib.Path.read_text", side_effect=AssertionError("Controller read a file")):
            action, detail = baseline_b(current, centroids, loads, 60)
        expected = current+.5*(centroids-current)
        np.testing.assert_array_equal(detail["interpolated_target"], expected)
        basis, _ = cooperation_aligned_basis(centroids, current, loads)
        world = centroids+60*np.einsum("nij,nj->ni", basis, action.reshape(3, 2))
        np.testing.assert_allclose(world, expected, atol=1e-10)
        self.assertFalse(detail["projected"].any())

    def test_b_unreachable_target_uses_existing_feasible_projection(self):
        current = np.array([[100., 100.], [500., 100.], [100., 500.]])
        centroids = current+200
        action, detail = baseline_b(current, centroids, np.array([12., 6., 3.]), 60)
        self.assertTrue(detail["projected"].all())
        np.testing.assert_allclose(np.linalg.norm(action.reshape(3, 2), axis=1), 1, atol=1e-12)
        self.assertGreater(np.linalg.norm(detail["interpolated_target"]-detail["feasible_target"]), 0)

    def test_projection_component_clip_then_unit_circle_without_mutating_residual(self):
        action = np.array([2., .5, -.3, .4, -2., -2.])
        original = action.copy()
        result, flags = project_action(action)
        expected = np.array([1/np.sqrt(1.25), .5/np.sqrt(1.25), -.3, .4,
                             -1/np.sqrt(2), -1/np.sqrt(2)])
        np.testing.assert_allclose(result, expected, atol=1e-14)
        np.testing.assert_array_equal(action, original)
        np.testing.assert_array_equal(flags, [True, False, True])
        for invalid in ([0]*5, [0]*7, [np.nan]*6, [np.inf]*6):
            with self.assertRaises(ValueError):
                project_action(invalid)

    def test_nonzero_residual_uses_native_start_and_real_movement_constraints(self):
        delta = np.array([2., .7, -2., -.4, .2, -.3])
        meter = ResidualMeter(self.cfg)
        meter.reset(33600, 2)
        before = meter.env.uav_coords.copy()
        native = QueueMDP.start_macro
        calls = []

        def observe(mdp, action):
            calls.append(np.array(action))
            return native(mdp, action)

        with patch.object(QueueMDP, "start_macro", observe):
            meter.start_macro(delta)
        np.testing.assert_array_equal(meter.env.uav_coords, before)
        self.assertEqual(len(calls), 1)
        record = meter.controls[0]
        np.testing.assert_array_equal(record["sampled_delta"], delta)
        np.testing.assert_allclose(calls[0], project_action(record["a_B"]+delta)[0])
        self.assertTrue(record["projected"].any())
        for _ in range(meter.env.deployment_resource_frames):
            frame = meter.step_frame(np.zeros(meter.micro_action_dim))
            self.assertEqual(frame["phase"], "deployment")
        distance = np.linalg.norm(record["actual_displacement"], axis=1)
        self.assertTrue(record["execution_complete"])
        self.assertTrue((distance <= 20+1e-5).all())
        self.assertTrue(((meter.env.uav_coords >= 0) & (meter.env.uav_coords <= 1000)).all())
        sep = np.linalg.norm(meter.env.uav_coords[:, None]-meter.env.uav_coords[None], axis=-1)
        self.assertGreaterEqual(float(sep[np.triu_indices(3, 1)].min()), self.cfg["environment"]["min_uav_separation_m"])
        np.testing.assert_allclose(meter.cost_sum, meter.objective_holding+meter.objective_energy, atol=1e-10)


class ResidualIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.cfg = config()
        self.agent = make_agent(self.cfg)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def test_window0_zero_head_reproduces_B_routes_metrics_and_CRN(self):
        report = evaluate_pair(self.cfg, self.agent, self.out, "zero", require_zero=True)
        self.assertTrue(report["window0_B_equal"])
        self.assertTrue(report["CRN_equal"])
        self.assertTrue(report["learning_replay_rng_unchanged"])
        self.assertLessEqual(max(abs(v) for v in report["residual_minus_B"].values()), 1e-9)
        for name in ("residual", "B"):
            metrics = report["metrics"][name]
            self.assertAlmostEqual(metrics["J"], metrics["J_holding"]+metrics["J_energy"], places=11)

    def test_nonzero_initial_head_fails_window0_gate(self):
        with torch.no_grad():
            self.agent.macro_actor.out.bias.add_(.2)
        with self.assertRaises(AssertionError):
            evaluate_pair(self.cfg, self.agent, self.out, "badzero", require_zero=True)

    def test_reserved_and_unknown_seeds_blocked_before_any_rollout(self):
        with patch("hrl_uav.ppo_residual_b.rollout") as run:
            for seed in (*RESERVED_SEEDS, 71000, 72000, True, 33600.0):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    evaluate_pair(self.cfg, self.agent, self.out, "forbidden", seed=seed)
            run.assert_not_called()

    def test_CRN_tampering_detected(self):
        from hrl_uav.ppo_residual_b import crn_records
        calls = 0

        def corrupt(meter):
            nonlocal calls
            calls += 1
            rows = crn_records(meter)
            if calls == 4:
                rows[0]["arrivals_sha256"] = "incorrect"
            return rows

        with patch("hrl_uav.ppo_residual_b.crn_records", side_effect=corrupt):
            with self.assertRaisesRegex(AssertionError, "CRN mismatch"):
                evaluate_pair(self.cfg, self.agent, self.out, "badcrn")

    def test_frozen_guards_block_optimizer_sampling_and_eval_store(self):
        before = sac_learning_state(self.agent, self.cfg)
        with frozen_guards(self.agent, evaluation=True):
            for call in (lambda: self.agent.replay.sample(1), lambda: self.agent.store(),
                         self.agent.sac_actor.optimizer.step, self.agent.sac_critic.optimizer.step,
                         self.agent.entropy.optimizer.step):
                with self.assertRaisesRegex(AssertionError, "forbidden"):
                    call()
        self.assertTrue(all(assert_sac_frozen(self.agent, self.cfg, before).values()))

    def test_native_300_threshold_and_final_flush_do_not_repeat_update(self):
        mdp = QueueMDP(self.cfg)
        mdp.reset(72000, 2)
        state = mdp.macro_observation()
        proxy = ResidualFrozen(self.agent)
        for index in range(300):
            delta, logprob = proxy.macro_action(state)
            self.agent.macro_rollout.append((state.copy(), delta.copy(), logprob,
                -1-.01*np.sin(index), state.copy(), float(index < 299)))
            proxy.update_ppo(10/120)
            self.assertEqual(self.agent.ppo_updates, int(index == 299))
        self.assertEqual(proxy.consumed_macro_transitions, 300)
        self.assertFalse(proxy.ppo_update_events[0]["forced"])
        before = copy.deepcopy(self.agent.checkpoint(self.cfg))
        proxy.update_ppo(10/120, force=True)
        self.assertTrue(state_equal(before, self.agent.checkpoint(self.cfg)))
        self.assertLess(self.agent.last_ppo_stats["pre_logprob_error"], 1e-4)

    def test_actual_residual_logprob_GAE_update_and_boundary_conservation(self):
        journal = ResidualJournal(self.out/"samples")
        report, _, _ = train_smoke_window(self.cfg, self.agent, self.out, journal=journal)
        self.assertEqual(report["ppo_updates"], 1)
        self.assertEqual(report["macro_transitions_consumed"], 2)
        self.assertEqual(report["replay_added"], 36)
        self.assertEqual(report["boundary_transitions"], 2)
        self.assertEqual(report["pending_dropped"], 0)
        self.assertEqual(report["sac_update_increment"], 0)
        self.assertEqual(report["sac_updates_at_entry"], report["sac_updates_at_exit"])
        self.assertLess(report["last_ppo_stats"]["pre_logprob_error"], 1e-4)
        self.assertTrue(all(report["sac_frozen_checks"].values()))
        self.assertGreater(report["actor_max_delta"], 0)
        self.assertGreater(report["critic_max_delta"], 0)
        self.assertEqual(journal.cursor, len(journal.records))
        with np.load(journal.directory/"update001.npz") as data:
            dist = torch.distributions.Normal(tensor(data["residual_mean"]), tensor(data["residual_std"]))
            raw_log = dist.log_prob(tensor(data["sampled_delta"])).sum(-1).cpu().numpy()
            wrong_log = dist.log_prob(tensor(data["a_exec"])).sum(-1).cpu().numpy()
            np.testing.assert_allclose(raw_log, data["old_logprob"], atol=1e-6)
            self.assertGreater(float(np.max(np.abs(wrong_log-data["old_logprob"]))), 1e-4)
            np.testing.assert_array_equal(data["pre_projection"], data["a_B"]+data["sampled_delta"])
            np.testing.assert_allclose(data["returns"], data["value"]+data["advantage"], atol=1e-7)
            self.assertEqual(len(data["scene_seed"]), 2)
            self.assertEqual(data["state"].shape[1], self.agent.macro_state_dim)
        saved = torch.load(self.out/"smoke_window1.pt", map_location="cpu", weights_only=False)
        self.assertEqual(saved["experiment"]["probability_semantics"], SEMANTICS)
        self.assertFalse(saved["experiment"]["resume_supported"])
        self.assertEqual(saved["config"], self.cfg)

    def test_logging_on_off_identical_rng_networks_replay_and_physics(self):
        rng = capture_rng()
        initial_checkpoint = copy.deepcopy(self.agent.checkpoint(self.cfg))
        off_agent = self.agent
        restore_rng(rng)
        a, traj_a, routes_a = train_smoke_window(self.cfg, off_agent, self.out)
        checkpoint_a, rng_a = copy.deepcopy(off_agent.checkpoint(self.cfg)), capture_rng()
        # Restore weights and the pre-rollout RNG, rather than copying nonleaf alpha tensors.
        cfg = self.cfg
        on_agent = make_agent(cfg)
        from hrl_uav.resumption import restore_agent
        restore_agent(on_agent, initial_checkpoint, dict(replay_counter=0, legacy_rng_seed=101))
        restore_rng(rng)
        directory = self.out/"on"
        directory.mkdir()
        journal = ResidualJournal(directory/"samples")
        b, traj_b, routes_b = train_smoke_window(cfg, on_agent, directory, journal=journal)
        self.assertTrue(state_equal(checkpoint_a, on_agent.checkpoint(cfg)))
        self.assertTrue(state_equal(rng_a, capture_rng()))
        for left, right in zip(traj_a, traj_b):
            for x, y in zip(left, right):
                np.testing.assert_array_equal(x, y)
        for left, right in zip(routes_a, routes_b):
            for x, y in zip(left, right):
                np.testing.assert_array_equal(x, y)
        self.assertEqual(a["last_ppo_stats"], b["last_ppo_stats"])

class ResidualProvenanceTests(unittest.TestCase):
    def test_fixed_source_rejects_hash_mismatch_before_deserialize(self):
        with patch("hrl_uav.ppo_frozen.file_hash", return_value="wrong"), patch("hrl_uav.ppo_frozen.torch.load") as load:
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                prepare_residual_source()
            load.assert_not_called()

    def test_original_config_required_and_no_lowstd_or_source_rewrite(self):
        prior = json.loads((ROOT/"results/ppo_frozen_sac_pilot120_seed101/manifest.json").read_text())
        cfg = copy.deepcopy(prior["config"])
        original = copy.deepcopy(cfg)
        return_value = (cfg, {}, dict(source_hashes={}), dict(status="LONG_COMPLETE"), prior["training_scene_seeds"])
        with patch("hrl_uav.ppo_residual_b.prepare_source", return_value=return_value) as prepare:
            checked, _, _ = prepare_residual_source()
            self.assertEqual(checked, original)
            prepare.assert_called_once_with(unittest.mock.ANY, 120, 72000)
            self.assertEqual({k: checked["training"][k] for k in HYPERPARAMETERS}, HYPERPARAMETERS)
            cfg["training"]["ppo_initial_std"] = 1/6
            with self.assertRaisesRegex(ValueError, "provenance/config"):
                prepare_residual_source()
        self.assertEqual(SOURCE_DIGESTS["checkpoint"], "be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932")


if __name__ == "__main__":
    unittest.main()
