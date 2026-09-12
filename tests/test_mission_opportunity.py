"""Bounded nonlearning checks for Mission Stage3A, never training tests."""

import copy
import gzip
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import isolated_rng, rollout
from hrl_uav.mission_opportunity import (
    ROOT, SCENES, TIMES, MissionPilotMeter, MissionSnapshot, ObservedPolicy,
    actions, branch_metrics, crn_records, initial_counts, macro_step, mission_config,
    objective_values, physical_fields, physical_state, protocol, semantic_hash, validate_branch,
)
from hrl_uav.mission_opportunity_analysis import summarize
from hrl_uav.mission_smoke import load_source, load_spec
from hrl_uav.resumption import state_equal
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.trajectory_check import ReadOnlySAC, forbidden, no_learning


class MissionOpportunityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = mission_config(json.loads((ROOT/"configs/sac_only_wireless6_pilot.json").read_text()))
        self.fields = physical_fields(self.cfg)
        self.sac = SimpleNamespace(sac_actor=UniformRouting(), optimizers=(), assert_unchanged=lambda: {"test_double": True})
        for target in ("torch.autograd.backward", "torch.autograd.grad", "hrl_uav.learning.DirectHRL.__init__",
                       "hrl_uav.agents.ppo_agent.ActorNet.__init__", "hrl_uav.agents.ppo_agent.CriticNet.__init__"):
            context = patch(target, forbidden)
            context.start()
            self.addCleanup(context.stop)

    def test_matrix_limits(self):
        self.assertEqual(len(SCENES)*len(TIMES)*len(actions()), 208)
        self.assertEqual(protocol()["total_branch_macros"], 1248)
        for action in actions():
            validate_branch(33600, 100, action)
        for s, t, a, h in ((33610,100,actions()[0],6), (33700,100,actions()[0],6),
                            (33600,90,actions()[0],6), (33600,100,actions()[0],7),
                            (33600,100,dict(actions()[1], delta=[.1,0,0,0,0,0]),6)):
            with self.assertRaises(ValueError):
                validate_branch(s, t, a, h)

    def test_manual_twenty_seconds_matches_native_rollout(self):
        native = MissionPilotMeter(self.cfg)
        policy = ObservedPolicy(self.sac, "predictive_centroid")
        policy.meter = native
        with isolated_rng(), no_learning(self.sac), patch("hrl_uav.experiment.QueueMDP", return_value=native):
            summary, _, _, _, _ = rollout(self.cfg, policy, 33600, 2, macro_enabled=True,
                                           train=False, record=True, complete_tail=False)
        meter = MissionPilotMeter(self.cfg)
        meter.reset(33600, 2)
        macros = [macro_step(meter, self.sac, np.zeros(6)) for _ in range(2)]
        self.assertEqual(semantic_hash(physical_state(meter, self.fields)), semantic_hash(physical_state(native, self.fields)))
        assert_crn_equal(crn_records(native), crn_records(meter))
        values = objective_values(macros, .99)
        self.assertAlmostEqual(values["C60"]/200, summary["J"], places=12)
        self.assertAlmostEqual(values["full_energy_j"]/20, summary["power_w"], places=9)

    def test_snapshot_real_sac_twenty_second_continuation_and_perturbation(self):
        saved, _ = load_source(load_spec())
        with isolated_rng():
            sac = ReadOnlySAC(saved)
        cfg = mission_config(saved["config"])
        with isolated_rng(), no_learning(sac), torch.no_grad():
            original = MissionPilotMeter(cfg)
            original.reset(33600, 4)
            macro_step(original, sac, np.zeros(6))
            snap = MissionSnapshot(original, self.fields)
            with tempfile.TemporaryDirectory() as directory:
                file = Path(directory)/"snapshot.pkl.gz"
                snap.save(file)
                self.assertEqual(semantic_hash(pickle.loads(gzip.decompress(file.read_bytes()))), snap.initial_hash)
            expected = [macro_step(original, sac, np.zeros(6)) for _ in range(2)]
            original_hash = semantic_hash(physical_state(original, self.fields))
            zero = snap.restore()
            actual = [macro_step(zero, sac, np.zeros(6)) for _ in range(2)]
            self.assertEqual(semantic_hash(physical_state(zero, self.fields)), original_hash)
            self.assertEqual(objective_values(expected, .99), objective_values(actual, .99))
            self.assertIsNot(zero.env.parent_tasks, original.env.parent_tasks)
            self.assertIsNot(zero.env.np_random, original.env.np_random)
            self.assertIs(zero.feature_slices, zero.micro_feature_slices)
            changed = snap.restore()
            pert = macro_step(changed, sac, actions()[1]["delta"])
            macro_step(changed, sac, np.zeros(6))
            assert_crn_equal(crn_records(zero), crn_records(changed))
            self.assertTrue(state_equal(zero.env.np_random.get_state(), changed.env.np_random.get_state()))
            np.testing.assert_array_equal(pert["control"]["a_raw"], pert["control"]["a_pred"]+actions()[1]["delta"])
            self.assertLessEqual(np.linalg.norm(pert["control"]["executed_displacement_m"], axis=1).max(), 20+1e-8)
            self.assertEqual(changed.reset_count, 0)
            self.assertEqual(semantic_hash(physical_state(original, self.fields)), original_hash)
        self.assertTrue(all(sac.assert_unchanged().values()))

    def test_sixty_second_metrics_conservation_and_reward(self):
        meter = MissionPilotMeter(self.cfg)
        meter.reset(33600, 6)
        initial = initial_counts(meter)
        macros = [macro_step(meter, self.sac, np.zeros(6)) for _ in range(6)]
        result = branch_metrics(meter, macros, initial)
        self.assertEqual(result["sac_decisions"], 120)
        self.assertEqual(len(meter.positions), 601)
        self.assertEqual(result["replay_samples"], 0)
        self.assertEqual(result["replay_writes"], 0)
        for p in ("high", "low"):
            self.assertEqual(result[f"arrived_{p}"]-result[f"completed_{p}"], result[f"unfinished_{p}"])
        costs = np.arange(1., 601.)/1000
        for i, macro in enumerate(macros):
            block = costs[i*100:(i+1)*100]
            for row, cost in zip(macro["physical_rows"], block):
                row.update(cost=cost, holding=.9*cost, energy=.1*cost)
            frames = [-sum(.99**(j/5)*block[f*5+j] for j in range(5)) for f in range(20)]
            macro["macro_reward"] = sum(.99**f*r for f, r in enumerate(frames))/20
        result = objective_values(macros, .99)
        self.assertAlmostEqual(result["ppo_return60"], -sum(.99**(i/5)*c for i, c in enumerate(costs))/20, places=10)

    def test_effective_oracle_fixed_direction_and_conflict_analysis(self):
        states, rows = [], []
        for seed in SCENES:
            for t in TIMES:
                feature = dict(seed=seed, start_s=t, roles=["heavy", "medium", "light"],
                    transfer_queue=np.zeros((3,3)).tolist(), current_load=[12,5,3], compute_queue=[1,2,3],
                    upload_queue=[0,0,0], u2g_mean_rate_mbps=[4,4,4], u2u_reference_quality=np.ones((3,3)).tolist(),
                    centroid_offset=np.zeros((3,2)).tolist(), centroid_velocity=np.ones((3,2)).tolist())
                states.append(feature)
                for a in actions():
                    # An ineffective direction is neutral; each state's best direction changes with time.
                    delta = 0 if a["dimension"] in (0,6) else (-2 if a["dimension"] == (t//100-1)%5+1 else -1)
                    rows.append(dict(seed=seed, start_s=t, action=a["name"], dimension=a["dimension"], sign=a["sign"],
                                     C60=10+delta, ppo_return60=-1+(-delta if a["sign"] == 1 else delta),
                                     delta_C60=delta, delta_C60_holding=delta, delta_C60_energy=0,
                                     relative_C60_percent=10*delta, delta_ppo_return60=-delta if a["sign"] == 1 else delta,
                                     ineffective=a["dimension"] == 6, component_clip_uavs=0, unit_disk_projection_uavs=0,
                                     map_projection_uavs=0, safety_projection_uavs=0))
        result, per_state, fixed, _, _ = summarize(rows, states)
        self.assertEqual(result["ineffective_nonzero"], 32)
        self.assertEqual(result["improving_states"], 16)
        self.assertGreater(result["oracle_extra_gain_C60"], 0)
        self.assertEqual(result["conflict_states"], 16)
        self.assertEqual(len(per_state), 16)
        self.assertEqual(len(fixed), 12)


if __name__ == "__main__":
    unittest.main()
