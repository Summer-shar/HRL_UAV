"""The combined candidate-radio / deployment-CPU training boundary path."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hrl_uav.experiment import rollout
from hrl_uav.wireless_check import AuditMeter, ENERGY, PSD
from tests.test_deployment_compute import queued_parent


class WirelessPilotBoundaryTest(unittest.TestCase):
    def test_candidate_radio_cpu_real_training_rollout_and_tail(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "configs/hrl_ppo_only_unlock.json").read_text())
        cfg["environment"].update(PSD, wireless_model="psd_per_uav_budget",
                                  compute_during_deployment=True,
                                  pause_service_during_flight=True, u2u_bandwidth_mhz=6)
        cfg["training"]["window_macros"] = 1
        gamma = cfg["training"]["gamma_per_resource_frame"]
        meter = AuditMeter(cfg)
        meter.check_local = False
        env = meter.env
        frames, stored, parents, prepared, decisions, ends = [], [], [], {}, [], []
        real_reset, real_step, real_prepare = meter.reset, meter.step_frame, meter.prepare_service

        def reset(seed, macros):
            real_reset(seed, macros)
            parents.append(queued_parent(env, "high", [(0, 100.6, "compute")])[0])
            parents.append(queued_parent(env, "low", [(0, 100.8, "compute")])[0])
            parents.append(queued_parent(env, "low", [(1, .2, "tx")])[0])
            return meter.micro_observation()

        def step(action, discount):
            frame = real_step(action, discount)
            frames.append(copy.deepcopy(frame))
            return frame

        def prepare():
            state = real_prepare()
            prepared[round(env._current_time(), 6)] = state.copy()
            return state

        def choose(state, deterministic=False):
            self.assertEqual(env.macro_service_pause_steps_remaining, 0)
            decisions.append(state.copy())
            return np.zeros(18, dtype=np.float32)

        def store(state, action, route, reward, next_state, discount):
            stored.append((state.copy(), reward, next_state.copy(), discount, env._current_time()))

        def window(row, mdp, pending, last):
            self.assertIs(mdp, meter)
            self.assertIsNotNone(pending)
            ends.append((row, meter.total_power_energy, meter.cost_sum))

        entropy = Mock()
        entropy.log_alpha.exp.return_value.item.return_value = .05
        agent = SimpleNamespace(
            stage="boundary-test", entropy=entropy, macro_rollout=[], last_sac_stats={},
            sac_updates=0, sac_actor_updates=0, sac_critic_gradient_steps=0,
            sac_target_updates=0, ppo_updates=0,
            sac_actor=SimpleNamespace(choose_action=Mock(side_effect=choose)),
            macro_action=Mock(return_value=(np.zeros(6), 0.0)),
            store=Mock(side_effect=store), update_sac=Mock(), update_ppo=Mock(),
        )
        with patch("hrl_uav.experiment.QueueMDP", return_value=meter), \
             patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("no networks")), \
             patch.object(meter, "reset", side_effect=reset), \
             patch.object(meter, "step_frame", side_effect=step), \
             patch.object(meter, "prepare_service", side_effect=prepare), \
             patch.object(env, "_generate_traffic", return_value={}):
            summary, rows, _, _, returned = rollout(
                cfg, agent, 51, 2, macro_enabled=False, train=True,
                complete_tail=True, on_window=window)

        self.assertIs(returned, meter)
        self.assertEqual(meter.reset_count, 1)
        self.assertEqual(env.wireless_model, "psd_per_uav_budget")
        self.assertTrue(env.compute_during_deployment)
        self.assertEqual(len(stored), 36)
        self.assertEqual(summary["boundary_transitions"], 2)
        self.assertEqual(summary["final_pending_dropped"], 0)
        self.assertAlmostEqual(summary["boundary_discount_min"], gamma**3)
        self.assertAlmostEqual(parents[0].finish_time, 10.1)
        self.assertAlmostEqual(parents[1].finish_time, 20.2)
        self.assertIsNotNone(parents[2].finish_time)
        for idx, last_frame, next_time, queue, count in (
                (17, 19, 11.0, 91.4, 1), (35, 39, 21.0, 0, 0)):
            expected = (frames[last_frame]["reward"] + gamma*frames[last_frame+1]["reward"]
                        + gamma**2*frames[last_frame+2]["reward"])
            transition = stored[idx]
            self.assertAlmostEqual(transition[1], expected)
            self.assertAlmostEqual(transition[3], gamma**3)
            self.assertEqual(transition[4], next_time)
            self.assertEqual(sum(np.isclose(t[4], next_time) for t in stored), 1)
            np.testing.assert_array_equal(transition[2], prepared[next_time])
            for feature, values in (("compute_queue", [0, 0, 0, queue, 0, 0]),
                                    ("unfinished_tasks", [0, 0, 0, count, 0, 0])):
                start, end = meter.micro_feature_slices[feature]
                np.testing.assert_allclose(np.expm1(transition[2][start:end])*10, values, atol=2e-5)
        for idx, transition in enumerate(stored):
            np.testing.assert_array_equal(transition[0], decisions[idx])
            if idx not in (17, 35):
                self.assertAlmostEqual(transition[3], gamma)

        self.assertEqual(len(rows), 2)
        for key, value in ends[-1][0].items():
            if key in summary:
                self.assertEqual(summary[key], value, key)
        self.assertEqual(summary["time_s"], 20)
        self.assertEqual(env._current_time(), 21)
        self.assertEqual(summary["task_completed_low_count"], 1)
        self.assertEqual(env.completed_priority_count["low"], 2)
        self.assertAlmostEqual(summary["power_w"]*20, ends[-1][1])
        self.assertAlmostEqual(summary["J"]*200, ends[-1][2])
        prefix = meter.readings[:40]
        self.assertAlmostEqual(sum(r[k] for r in prefix for k in ENERGY), ends[-1][1])
        self.assertGreater(meter.total_power_energy, ends[-1][1])
        self.assertGreater(sum(r["energy_u2u_j"] for r in prefix), 0)
        for frame, reading in zip(frames, meter.readings):
            if frame["phase"] == "deployment":
                self.assertEqual(reading["energy_u2u_j"], 0)
                self.assertEqual(reading["energy_upload_j"], 0)
                for info in frame["infos"]:
                    self.assertEqual(np.sum(info.get("tx_service_matrix_mbits", 0)), 0)
        # The metered link energy is the same actual radio energy used in reward.
        self.assertAlmostEqual(sum(r["tx_energy_j"] for r in meter.link_readings),
                               sum(r["energy_u2u_j"] for r in meter.readings))
        self.assertEqual(len(decisions), 36)
        self.assertEqual(agent.update_sac.call_count, 36 // cfg["training"]["sac_update_frames"])
        agent.update_ppo.assert_not_called()
        self.assertEqual(agent.macro_rollout, [])


if __name__ == "__main__":
    unittest.main()
