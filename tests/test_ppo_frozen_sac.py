import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.experiment import rollout, smoke_config, write_json
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.ppo_frozen import (
    DEVELOPMENT_SEEDS, RESERVED_SEEDS, STAGE_TRANSITION, VALIDATION_SEEDS,
    FrozenSACUpdates, assert_sac_frozen, exogenous_records, paired_evaluation, prepare_source,
    run_frozen_ppo, sac_learning_state,
)
from hrl_uav.priority_grid import file_hash
from hrl_uav.resumption import capture_rng, state_equal


ROOT = Path(__file__).resolve().parents[1]


class FrozenSACPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        base = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text())
        self.config = smoke_config(base)
        self.config["training"].update(joint_windows=2, replay_capacity=64, batch_size=8, replay_warmup=8)
        env = QueueMDP(self.config)
        env.reset(1, 3)
        self.agent = DirectHRL.from_mdp(env, self.config)
        self.agent.set_stage("joint")
        self.proxy = FrozenSACUpdates(self.agent)
        self.env = env

    def test_blocked_update_does_not_change_sac_learning_state(self):
        state = self.env.micro_observation()
        action = self.agent.sac_actor.choose_action(state, deterministic=True)
        route = self.env.route(action)
        for _ in range(8):
            self.agent.store(state, action, route, -0.1, state, 0.99)
        before = sac_learning_state(self.agent, self.config)
        for _ in range(5):
            self.proxy.update_sac()
        checks = assert_sac_frozen(self.agent, self.config, before)
        self.assertTrue(all(checks.values()))
        self.assertEqual(self.proxy.blocked_update_calls, 5)

    def test_replay_collection_is_allowed_while_sac_parameters_remain_frozen(self):
        before = sac_learning_state(self.agent, self.config)
        counter = self.agent.replay.memory_counter
        state = self.env.micro_observation()
        action = np.zeros(self.env.micro_action_dim, dtype=np.float32)
        route = self.env.route(action)
        self.proxy.store(state, action, route, -0.2, state, 0.99)
        self.assertEqual(self.agent.replay.memory_counter, counter + 1)
        checks = assert_sac_frozen(self.agent, self.config, before)
        self.assertTrue(all(checks.values()))

    def test_ppo_update_can_change_macro_without_changing_sac(self):
        before_sac = sac_learning_state(self.agent, self.config)
        macro_before = copy.deepcopy(self.agent.macro_actor.state_dict())
        state = self.env.macro_observation()
        for _ in range(self.config["training"]["ppo_update_macros"]):
            action, logprob = self.proxy.macro_action(state, enabled=True, deterministic=False)
            self.agent.macro_rollout.append((state.copy(), action.copy(), logprob, -0.1, state.copy(), 1.0))
        self.proxy.update_ppo(0.5, force=True)
        checks = assert_sac_frozen(self.agent, self.config, before_sac)
        self.assertTrue(all(checks.values()))
        changed = any(not torch.equal(macro_before[k], v) for k, v in self.agent.macro_actor.state_dict().items()
                      if k != "log_std")
        self.assertTrue(changed)
        self.assertGreater(self.agent.ppo_updates, 0)

    def test_frozen_assertion_covers_runtime_state_outside_state_dict(self):
        before = sac_learning_state(self.agent, self.config)
        alpha = self.agent.entropy.alpha
        self.agent.entropy.alpha = alpha * 1.1
        with self.assertRaisesRegex(AssertionError, "Frozen SAC"):
            assert_sac_frozen(self.agent, self.config, before)
        self.agent.entropy.alpha = alpha
        self.agent.sac_actor.min_action = -0.8
        with self.assertRaisesRegex(AssertionError, "Frozen SAC"):
            assert_sac_frozen(self.agent, self.config, before)
        self.agent.sac_actor.min_action = -1.0
        self.agent.sac_actor.action_net.eval()
        with self.assertRaisesRegex(AssertionError, "Frozen SAC"):
            assert_sac_frozen(self.agent, self.config, before)

    def test_validation_rejects_reserved_or_undeclared_scenes_before_rollout(self):
        with patch("hrl_uav.ppo_frozen.metered_evaluate") as evaluate:
            for seeds in ([s] for s in (*RESERVED_SEEDS, 72000)):
                with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                    paired_evaluation(self.config, self.agent, seeds, ROOT, "forbidden", 0)
            evaluate.assert_not_called()

    def test_validation_independent_snapshot_detects_parameter_mutation(self):
        def mutate(*args, **kwargs):
            with torch.no_grad():
                next(self.agent.macro_actor.parameters()).add_(1)
            return []

        with patch("hrl_uav.ppo_frozen.metered_evaluate", side_effect=mutate):
            with self.assertRaisesRegex(AssertionError, "learning state or RNG"):
                paired_evaluation(self.config, self.agent, VALIDATION_SEEDS, ROOT, "mutate", 0)

    def test_validation_independent_snapshot_detects_optimizer_mutation(self):
        parameter = next(self.agent.macro_actor.parameters())
        self.agent.ppo_actor_optimizer.state[parameter]["audit_tensor"] = torch.zeros_like(parameter)

        def mutate(*args, **kwargs):
            self.agent.ppo_actor_optimizer.state[parameter]["audit_tensor"].add_(1)
            return []

        with patch("hrl_uav.ppo_frozen.metered_evaluate", side_effect=mutate):
            with self.assertRaisesRegex(AssertionError, "learning state or RNG"):
                paired_evaluation(self.config, self.agent, VALIDATION_SEEDS, ROOT, "mutate", 0)

    def test_validation_detects_rng_or_replay_mutation(self):
        state = self.env.micro_observation()
        action = np.zeros(self.env.micro_action_dim, dtype=np.float32)
        for target in ("rng", "replay"):
            def mutate(*args, **kwargs):
                if target == "rng":
                    np.random.random()
                else:
                    self.agent.store(state, action, self.env.route(action), 0, state, 0.99)
                return []

            with self.subTest(target=target), patch("hrl_uav.ppo_frozen.metered_evaluate", side_effect=mutate):
                with self.assertRaisesRegex(AssertionError, "learning state or RNG"):
                    paired_evaluation(self.config, self.agent, VALIDATION_SEEDS, ROOT, "mutate", 0)

    def test_pairing_detects_task_or_motion_or_schedule_changes_with_same_counts(self):
        for key in ("arrivals_sha256", "users_microsteps_sha256", "schedule_sha256"):
            calls = 0

            def changed(meter):
                nonlocal calls
                calls += 1
                records = exogenous_records(meter)
                if calls == 2:
                    records[0][key] = "changed-without-changing-arrival-count"
                return records

            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with patch("hrl_uav.ppo_frozen.exogenous_records", side_effect=changed):
                    with self.assertRaisesRegex(AssertionError, "CRN mismatch"):
                        paired_evaluation(self.config, self.agent, [33600], Path(directory), "crn", 0, macros=6)


class FrozenSourceTests(unittest.TestCase):
    """Small synthetic checkpoints; never substitutes for the real-source smoke."""

    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source" / "checkpoints" / "window2000.pt"
        self.source.parent.mkdir(parents=True)
        self.run = self.source.parent.parent
        base = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text())
        self.config = smoke_config(base)
        self.config["training"].update(seed=101, pretrain_windows=2000, joint_windows=0,
                                       replay_capacity=64, batch_size=8, replay_warmup=8,
                                       ppo_update_macros=4)
        self.config["validation_seeds"] = list(VALIDATION_SEEDS)
        self.config["test_seeds"] = list(DEVELOPMENT_SEEDS)
        self.config["pilot"]["reserved_final_seeds"] = list(RESERVED_SEEDS)
        env = QueueMDP(self.config)
        env.reset(71000, 2)
        agent = DirectHRL.from_mdp(env, self.config)
        state = env.micro_observation()
        action = np.zeros(env.micro_action_dim, dtype=np.float32)
        agent.store(state, action, env.route(action), -0.2, state, 0.99)
        self.saved = copy.deepcopy(agent.checkpoint(self.config))
        self.saved["run_state"] = {"completed_windows": 2000, "at_rollout_boundary": True,
                                   "next_training_scene_seed": 71667, "rng": capture_rng()}
        self.manifest = {
            "status": "LONG_COMPLETE", "training_initialization_seed": 101, "macro_enabled": False,
            "config": copy.deepcopy(self.config), "source_hashes": {"hrl_uav/env.py": file_hash(ROOT / "hrl_uav/env.py")},
        }
        self.report = {"status": "LONG_COMPLETE", "final_window": 2000}
        self.digests = self.write_fixture()
        patcher = patch("hrl_uav.ppo_frozen.SOURCE_DIGESTS", self.digests)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_fixture(self):
        torch.save(self.saved, self.source)
        self.manifest["checkpoints"] = [{"window": 2000, "sha256": file_hash(self.source)}]
        write_json(self.run / "manifest.json", self.manifest)
        write_json(self.run / "report.json", self.report)
        return {"checkpoint": file_hash(self.source), "manifest.json": file_hash(self.run / "manifest.json"),
                "report.json": file_hash(self.run / "report.json")}

    def test_prepare_source_additive_branch_and_new_stage_without_mutating_source(self):
        cfg, saved, _, _, scenes = prepare_source(self.source, 31, 72000)
        expected = copy.deepcopy(self.config)
        expected["training"]["joint_windows"] = 31
        self.assertEqual(cfg, expected)
        self.assertTrue(state_equal(saved, self.saved))
        self.assertEqual(scenes, list(range(72000, 72031)))
        self.assertEqual(saved["config"]["training"]["joint_windows"], 0)
        self.assertTrue((ROOT / "hrl_uav/ppo_frozen.py").is_file())
        self.assertNotIn("hrl_uav/ppo_frozen.py", self.manifest["source_hashes"])

    def test_source_hash_checked_before_deserializing(self):
        self.digests["checkpoint"] = "0" * 64
        with patch("hrl_uav.ppo_frozen.torch.load") as load:
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                prepare_source(self.source, 1, 72000)
            load.assert_not_called()

    def test_source_manifest_and_report_hashes_are_checked(self):
        for name in ("manifest.json", "report.json"):
            old = self.digests[name]
            self.digests[name] = "0" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                prepare_source(self.source, 1, 72000)
            self.digests[name] = old

    def test_prepare_rejects_wrong_source_metadata_or_nonfresh_ppo(self):
        original_saved, original_manifest = copy.deepcopy(self.saved), copy.deepcopy(self.manifest)
        for case in ("seed", "stage", "ppo_gradients", "ppo_optimizer", "config", "source_hash", "missing_source", "rng"):
            self.saved, self.manifest = copy.deepcopy(original_saved), copy.deepcopy(original_manifest)
            if case == "seed":
                self.manifest["training_initialization_seed"] = 102
            elif case == "stage":
                self.saved["stage"] = "joint"
            elif case == "ppo_gradients":
                self.saved["updates"]["ppo_gradients"] = 1
            elif case == "ppo_optimizer":
                self.saved["optimizers"][3]["state"] = {0: {"step": torch.tensor(1)}}
            elif case == "config":
                self.saved["config"]["cpu_ghz"] = 99
            elif case == "source_hash":
                self.manifest["source_hashes"]["hrl_uav/env.py"] = "0" * 64
            elif case == "missing_source":
                self.manifest["source_hashes"]["hrl_uav/missing_source.py"] = "0" * 64
            elif case == "rng":
                self.saved["run_state"].pop("rng")
            self.digests.update(self.write_fixture())
            with self.subTest(case=case), self.assertRaises(ValueError):
                prepare_source(self.source, 1, 72000)

    def test_training_scenes_cannot_overlap_validation_development_or_reserved(self):
        for seed in (*VALIDATION_SEEDS, *DEVELOPMENT_SEEDS, *RESERVED_SEEDS, 33599):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "overlap"):
                prepare_source(self.source, 4, seed)

    def test_no_lossy_budget_or_seed_coercion(self):
        for windows, seed in ((1.5, 72000), (True, 72000), (0, 72000), (1, -1), (1, 72000.5)):
            with self.subTest(windows=windows, seed=seed), self.assertRaises(ValueError):
                prepare_source(self.source, windows, seed)

    def test_final_validation_and_checkpoint_include_forced_update_exactly_once(self):
        report = run_frozen_ppo(self.root / "result", source=self.source, ppo_windows=1, smoke=True)
        self.check_run(report, forced=True)

    def test_exact_threshold_does_not_repeat_final_update(self):
        self.saved["config"]["training"]["ppo_update_macros"] = 2
        self.manifest["config"] = copy.deepcopy(self.saved["config"])
        self.digests.update(self.write_fixture())
        report = run_frozen_ppo(self.root / "result", source=self.source, ppo_windows=1, smoke=True)
        self.check_run(report, forced=False)

    def check_run(self, report, forced):
        output = self.root / "result"
        final = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
        monitored = torch.load(output / "checkpoints/window1.pt", map_location="cpu", weights_only=False)
        boundary = torch.load(output / "boundary_latest.pt", map_location="cpu", weights_only=False)
        self.assertTrue(state_equal(final, monitored))
        self.assertTrue(state_equal(final, boundary))
        self.assertEqual(final["experiment"]["stage_transition"], STAGE_TRANSITION)
        self.assertEqual(final["experiment"]["learning_scope"], "ppo_only_sac_frozen")
        self.assertEqual(final["experiment"]["source_checkpoint_sha256"], self.digests["checkpoint"])
        self.assertFalse(final["experiment"]["resume_supported"])
        self.assertEqual(report["status"], "SMOKE_ONLY")
        self.assertEqual(report["stage_transition"], STAGE_TRANSITION)
        self.assertFalse(report["resume_supported"])
        self.assertEqual(report["sac_updates_at_entry"], report["sac_updates_at_exit"])
        self.assertTrue(all(report["sac_frozen_checks"].values()))
        self.assertEqual(report["sac_actor_max_delta"], 0)
        self.assertGreater(report["macro_mean_max_delta"], 0)
        self.assertEqual(report["ppo_updates"], 1)
        self.assertEqual(report["ppo_transitions_consumed"], 2)
        self.assertEqual(report["ppo_transitions_pending"], 0)
        self.assertEqual(report["ppo_update_events"][0]["forced"], forced)
        training = self.saved["config"]["training"]
        progress = 1 / training["ppo_std_decay_windows"]
        expected_std = np.exp((1-progress) * np.log(training["ppo_initial_std"]) + progress * np.log(training["ppo_final_std"]))
        np.testing.assert_allclose(report["ppo_update_events"][0]["std_after"], expected_std, rtol=1e-6)
        self.assertEqual(report["replay_at_exit"] - report["replay_at_entry"], 36)
        self.assertEqual(report["reserved_scenes_executed"], [])
        for check in report["validation_history"] + [report["development_final"]]:
            self.assertTrue(check["learning_state_unchanged"] and check["rng_unchanged"])
            self.assertTrue(all(row["task_births"] and row["load_schedule"] and row["user_microsteps"]
                                for row in check["crn"].values()))
        initial = report["validation_history"][0]
        self.assertTrue(all(row["uav"] for row in initial["crn"].values()))
        self.assertEqual(initial["deltas"][0]["ppo_minus_centroid_J"], 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue(all(manifest["restore_audit"]["exact_restore_checks"].values()))
        self.assertEqual(manifest["restore_audit"]["completed_ppo_windows"], 0)
        self.assertEqual(file_hash(self.source), self.digests["checkpoint"])
        self.assertTrue(state_equal(torch.load(self.source, map_location="cpu", weights_only=False), self.saved))

    def test_missing_boundary_experience_is_detected(self):
        def corrupt(*args, **kwargs):
            result = rollout(*args, **kwargs)
            result[0]["final_pending_dropped"] = 1
            return result

        with patch("hrl_uav.ppo_frozen.rollout", side_effect=corrupt):
            with self.assertRaisesRegex(AssertionError, "conservation failed"):
                run_frozen_ppo(self.root / "bad", source=self.source, ppo_windows=1, smoke=True)


if __name__ == "__main__":
    unittest.main()
