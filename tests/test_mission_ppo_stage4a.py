"""Bounded Stage4A checks; clock doubles are not900s physical evaluations."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.agents.sac_agent import Memory
from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import isolated_rng, rollout
from hrl_uav.mission_smoke import ROOT, load_source, load_spec, mission_config
from hrl_uav.mission_pilot import crn_records
from hrl_uav.mission_ppo_stage4a import (
    PPO_SEEDS, TRAIN_SCENES, MissionMacroPPO, Stage4Meter, collect_stream,
    frozen_sac_guard, protocol, stage_config, fixed_short, evaluate_short,
)
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.resumption import capture_rng, state_equal
from hrl_uav.trajectory_check import ReadOnlySAC


class FixedMacro:
    def __init__(self, cfg):
        self.cfg = cfg["training"]
        self.macro_rollout = []
        self.collected_macros = self.consumed_macros = self.ppo_updates = 0
        self.batches = []
        self.flags = []
        self.micro_flags = []
        self.sac = SimpleNamespace(sac_actor=self)

    def choose_action(self, state, deterministic=False):
        self.micro_flags.append(deterministic)
        if not deterministic:
            raise AssertionError("Micro controller was asked to explore")
        return np.zeros(18, dtype=np.float32)

    def macro_action(self, state, enabled=True, deterministic=False):
        self.flags.append(deterministic)
        return np.full(6, .1, dtype=np.float32), 0.

    def update_ppo(self, progress=0., force=False):
        if len(self.macro_rollout) >= 300:
            if len(self.macro_rollout) != 300 or force:
                raise AssertionError("Unexpected batch or force update")
            self.batches.append(copy.deepcopy(self.macro_rollout))
            self.consumed_macros += 300
            self.macro_rollout.clear()
            self.ppo_updates += 1


class ClockMeter:
    """No environment, task generation, radio or network; only orchestration clocks."""

    def __init__(self, cfg):
        self.reset_count = self.integration_steps = self.cost_sum = self.total_power_energy = 0
        self.env = SimpleNamespace(deployment_resource_frames=0, service_resource_frames=20,
                                   continuous_mission=True, resource_frame_prepare_count=0, mobility_update_count=0)
        self.carried_queue = 0
        self.macro_calls = 0

    def reset(self, scene, macros):
        if macros != 90:
            raise AssertionError("A training scene must span90 macros")
        self.reset_count += 1
        self.scene = scene

    def macro_observation(self):
        return np.full(67, self.integration_steps/10000, dtype=np.float32)

    def prepare_service(self):
        return np.zeros(68, dtype=np.float32)

    def start_macro(self, action):
        self.macro_calls += 1

    def step_frame(self, action, gamma):
        self.integration_steps += 5
        self.cost_sum += .5
        self.total_power_energy += .7
        self.carried_queue += 1
        self.env.resource_frame_prepare_count += 1
        self.env.mobility_update_count += 1
        return dict(reward=-.5, route=np.full((2, 3, 3), 1/3))

    def advance_macro_clock(self):
        pass

    def metrics(self):
        return dict(time_s=self.integration_steps/10, queue_total_mbit=self.carried_queue)


class Stage4ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.saved, _ = load_source(load_spec())
        with isolated_rng():
            cls.sac = ReadOnlySAC(cls.saved)
        cls.cfg = stage_config(cls.saved["config"], 201)

    def test_selective_restore_fresh_seeds_zero_means_and_empty_optimizers(self):
        before = capture_rng()
        with frozen_sac_guard(self.sac), patch.object(Memory, "__init__", side_effect=AssertionError("No replay constructor")):
            agents = [MissionMacroPPO(stage_config(self.saved["config"], s), s, self.sac) for s in PPO_SEEDS]
        self.assertTrue(state_equal(before, capture_rng()))
        for agent in agents:
            self.assertFalse(hasattr(agent, "replay"))
            self.assertFalse(agent.macro_rollout)
            self.assertEqual(agent.ppo_updates, 0)
            self.assertFalse(agent.ppo_actor_optimizer.state_dict()["state"])
            self.assertFalse(agent.ppo_critic_optimizer.state_dict()["state"])
            self.assertFalse(state_equal(agent.initial_actor, self.saved["macro_actor"]))
            self.assertFalse(state_equal(agent.initial_critic, self.saved["macro_critic"]))
            raw, _ = agent.macro_action(np.ones(67, dtype=np.float32), deterministic=True)
            np.testing.assert_array_equal(raw, np.zeros(6))
            with agent.exploration_rng():
                sampled, _ = agent.macro_action(np.ones(67, dtype=np.float32), deterministic=False)
            self.assertTrue(np.any(sampled != 0))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(state_equal(agents[a].initial_actor, agents[b].initial_actor))
            self.assertFalse(state_equal(agents[a].initial_critic, agents[b].initial_critic))
        self.assertTrue(all(self.sac.assert_unchanged().values()))

    def test_protocol_separates_scenes_and_preserves_all_other_configuration(self):
        p = protocol()
        self.assertFalse(p["formal_training_authorized"])
        self.assertEqual(p["full_updates_per_seed"], 12)
        self.assertEqual(p["monitor_windows"], [0, 30, 60, 90, 120])
        for seed in PPO_SEEDS:
            cfg = stage_config(self.saved["config"], seed)
            cfg["training"]["seed"] = 101
            cfg["training"]["joint_windows"] = 0
            self.assertEqual(cfg, mission_config(self.saved["config"]))
        for seed in (101, 204, True):
            with self.assertRaises(ValueError):
                stage_config(self.saved["config"], seed)
        for scene in (33600, 33601, *range(33700, 33710)):
            with self.assertRaises(ValueError):
                collect_stream(self.cfg, None, scene, train=True, macros=2)
        for scene in (72000, 33610, *range(33700, 33710)):
            with self.assertRaises(ValueError):
                collect_stream(self.cfg, None, scene, train=False, macros=2)

    def test_clock_double_full120_budget_no_reset_at300_or600(self):
        policy = FixedMacro(self.cfg)
        for index, scene in enumerate(TRAIN_SCENES):
            summary, meter, _, _, windows, _ = collect_stream(
                self.cfg, policy, scene, train=True, macros=90, macro_offset=index*90,
                meter_factory=ClockMeter, record=False)
            self.assertEqual(meter.reset_count, 1)
            self.assertEqual(meter.macro_calls, 90)
            self.assertEqual(summary["sac_decisions"], 1800)
            self.assertEqual([r["time_s"] for r in windows], [300, 600, 900])
            self.assertEqual([r["queue_total_mbit"] for r in windows], [600, 1200, 1800])
            self.assertEqual(summary["extra_tail_actions"] if "extra_tail_actions" in summary else 0, 0)
        self.assertEqual(policy.collected_macros, 3600)
        self.assertEqual(policy.consumed_macros, 3600)
        self.assertEqual(policy.ppo_updates, 12)
        self.assertFalse(policy.macro_rollout)
        self.assertEqual(len(policy.flags), 3600)
        self.assertTrue(all(not v for v in policy.flags))
        self.assertEqual(len(policy.micro_flags), 72000)
        self.assertTrue(all(policy.micro_flags))
        transitions = [r for batch in policy.batches for r in batch]
        for index, transition in enumerate(transitions):
            self.assertEqual(transition[-1], float(index % 90 != 89))
        expected_reward = -sum(.99**i*.5 for i in range(20))/20
        self.assertTrue(all(r[3] == expected_reward for r in transitions))

    def test_raw_action_projection_logging_is_observational(self):
        meter = Stage4Meter(self.cfg)
        meter.reset(33600, 90)
        raw = np.array([2., -2., .2, .3, .6, .8])
        before = raw.copy()
        info = meter.start_macro(raw)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_allclose(info["actual_displacement_m"][0], [20/np.sqrt(2), -20/np.sqrt(2)], atol=1e-10)
        np.testing.assert_allclose(info["actual_displacement_m"][1], [4., 6.], atol=1e-10)
        np.testing.assert_array_equal([r["raw_x"] for r in meter.commands], raw.reshape(3, 2)[:, 0])
        self.assertEqual([r["raw_clip_triggered"] for r in meter.commands], [1, 0, 0])
        self.assertEqual([r["disk_projection_triggered"] for r in meter.commands], [1, 0, 0])
        self.assertEqual([r["actual_at_20m"] for r in meter.commands], [1, 0, 1])
        self.assertEqual(set(r["role"] for r in meter.commands), {"heavy", "medium", "light"})

    def test_real20s_original_rollout_reward_transition_equivalence(self):
        policy = FixedMacro(self.cfg)
        new, meter, _, obs, _, _ = collect_stream(self.cfg, policy, 72000, train=True, macros=2)
        legacy_meter = Stage4Meter(self.cfg)
        # The legacy runner may attempt storage, but this test double never owns a replay.
        legacy = SimpleNamespace(macro_action=policy.macro_action, sac_actor=UniformRouting(),
                                 store=lambda *a: None, update_sac=lambda: None, update_ppo=lambda *a: None,
                                 macro_rollout=[], stage="test", sac_updates=0, sac_actor_updates=0,
                                 sac_critic_gradient_steps=0, sac_target_updates=0, ppo_updates=0, last_sac_stats={},
                                 entropy=SimpleNamespace(log_alpha=torch.tensor(-3.)))
        with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", return_value=legacy_meter):
            old, _, _, _, _ = rollout(self.cfg, legacy, 72000, 2, train=True, record=True, complete_tail=False)
        self.assertTrue(state_equal(obs, legacy.macro_rollout))
        self.assertEqual(meter.readings, legacy_meter.readings)
        for key in ("J", "power_w", "queue_total_mbit", "task_delay_weighted_ms"):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(len(meter.micro_states), 40)
        self.assertTrue(all(policy.micro_flags))

    def test_original_full_batch_update_and_logging_equivalence(self):
        logged = MissionMacroPPO(self.cfg, 201, self.sac, journal=True)
        plain = MissionMacroPPO(self.cfg, 201, self.sac, journal=False)
        rng = np.random.default_rng(10101)
        samples = []
        with logged.exploration_rng():
            for index in range(300):
                state = rng.normal(0, .1, 67).astype(np.float32)
                action, logprob = logged.macro_action(state, deterministic=False)
                samples.append((state, action, logprob, -float(np.square(action).sum()),
                                state.copy(), float(index not in (89, 179, 269, 299))))
        # Align exploration RNG only for a controlled logging-on/off equality test.
        plain.rng = copy.deepcopy(logged.rng)
        for policy in (logged, plain):
            policy.macro_rollout = copy.deepcopy(samples)
            policy.collected_macros = 300
            with frozen_sac_guard(self.sac):
                policy.update_ppo(300/3600)
            self.assertEqual(policy.ppo_updates, 1)
            self.assertEqual(policy.consumed_macros, 300)
            self.assertGreater(policy.ppo_gradient_steps, 0)
            before = policy.ppo_state()
            policy.update_ppo(300/3600, force=True)
            self.assertTrue(state_equal(before, policy.ppo_state()))
        self.assertTrue(state_equal(logged.ppo_state(), plain.ppo_state()))
        self.assertEqual(len(logged.batch_log), 1)
        self.assertEqual(logged.batch_log[0]["macro_discount"], .99**20)
        np.testing.assert_array_equal(logged.batch_log[0]["raw_action"], [s[1] for s in samples])
        self.assertFalse(logged.update_log[0]["forced"])
        self.assertAlmostEqual(logged.update_log[0]["std_after"], np.exp(.95*np.log(1/3)+.05*np.log(1/30)), places=7)
        self.assertTrue(all(self.sac.assert_unchanged().values()))

    def test_sac_guard_rejects_learning_replay_and_detects_mutation(self):
        with frozen_sac_guard(self.sac):
            for call in (lambda: Memory(1, 1), self.sac.critic.soft_update,
                         self.sac.optimizers[0].step, lambda: self.sac.entropy.learn(None)):
                with self.assertRaises(AssertionError):
                    call()
        param = next(self.sac.sac_actor.action_net.parameters())
        original = param.detach().clone()
        try:
            with self.assertRaisesRegex(AssertionError, "Frozen SAC"):
                with frozen_sac_guard(self.sac), torch.no_grad():
                    param.add_(1)
        finally:
            with torch.no_grad():
                param.copy_(original)
        self.assertTrue(all(self.sac.assert_unchanged().values()))

    def test_short_smoke_uses_full_historical_schedule_and_initial_hover(self):
        policy = MissionMacroPPO(self.cfg, 201, self.sac)
        historical = json.loads((ROOT / "results/mission_rpgm_v1_pilot/hover_33600/crn.json").read_text())[:40]
        reference, ref_meter, _, _ = fixed_short(self.cfg, self.sac, "hover")
        row, meter, _, obs, _, _ = evaluate_short(policy)
        assert_crn_equal(historical, crn_records(ref_meter))
        assert_crn_equal(historical, crn_records(meter))
        self.assertEqual(meter.readings, ref_meter.readings)
        self.assertEqual(row["J"], reference["J"])
        np.testing.assert_array_equal([r[1] for r in obs], np.zeros((2, 6)))


if __name__ == "__main__":
    unittest.main()
