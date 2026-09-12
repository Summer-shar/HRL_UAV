"""Short no-network tests for full snapshots and exact native reward chronology."""
import gzip
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import rollout
from hrl_uav.local_counterfactual import (
    ROOT, SCENES, TIMES, FullSnapshot, ResidualMeter, actions, physical_fields,
    physical_state, semantic_hash, macro_step, objective_values, validate_branch,
)
from hrl_uav.resumption import state_equal
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.trajectory_check import crn_records


class LocalCounterfactualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = json.loads((ROOT/'configs/sac_only_wireless6_pilot.json').read_text())
        self.fields = physical_fields(self.cfg)
        self.sac = UniformRouting()

    def test_fixed_matrix_and_reject_other_states_epsilon_horizon(self):
        self.assertEqual(len(SCENES)*len(TIMES)*len(actions()), 520)
        for action in actions():
            validate_branch(33600,100,action)
            self.assertAlmostEqual(np.linalg.norm(action['delta']), .05 if action['dimension'] else 0)
        for scene,time,action,horizon in ((33700,100,actions()[0],6),
                (33600,90,actions()[0],6),(33600,100,actions()[0],7),
                (33600,100,dict(actions()[1],delta=[.1,0,0,0,0,0]),6)):
            with self.assertRaises(ValueError):
                validate_branch(scene,time,action,horizon)

    def test_manual_chronology_equals_real_rollout_without_learning(self):
        with patch('hrl_uav.experiment.QueueMDP',ResidualMeter):
            summary,_,_,_,native = rollout(self.cfg,self.sac,33600,2,
                macro_enabled=False,train=False,record=True,complete_tail=False)
        manual = ResidualMeter(self.cfg)
        manual.reset(33600,2)
        macros = [macro_step(manual,self.sac,np.zeros(6)) for _ in range(2)]
        self.assertEqual(semantic_hash(physical_state(native,self.fields)),
                         semantic_hash(physical_state(manual,self.fields)))
        assert_crn_equal(crn_records(native),crn_records(manual))
        values = objective_values(macros,self.cfg['training']['gamma_per_resource_frame'])
        self.assertAlmostEqual(values['C60']/200,summary['J'],places=12)
        self.assertAlmostEqual(values['full_energy_j']/20,summary['power_w'],places=10)
        self.assertAlmostEqual(sum(values[k] for k in
            ('energy_compute_j','energy_upload_j','energy_u2u_j','energy_propulsion_j')),
            values['full_energy_j'],places=8)

    def test_full_snapshot_hooks_zero_continuation_and_CRN(self):
        original = ResidualMeter(self.cfg)
        original.reset(33600,3)
        macro_step(original,self.sac,np.zeros(6))
        snap = FullSnapshot(original,self.fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.pkl.gz'
            snap.save(path)
            loaded = pickle.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(semantic_hash(loaded),snap.initial_hash)
        expected = macro_step(original,self.sac,np.zeros(6))
        original_hash = semantic_hash(physical_state(original,self.fields))
        clone = snap.restore()
        actual = macro_step(clone,self.sac,np.zeros(6))
        self.assertEqual(semantic_hash(physical_state(clone,self.fields)),original_hash)
        self.assertEqual(expected['cost'],actual['cost'])
        self.assertEqual(len(clone.readings),20)
        self.assertEqual(len(original.readings),40)
        self.assertIsNot(clone.env.parent_tasks,original.env.parent_tasks)
        self.assertIsNot(clone.env.np_random,original.env.np_random)
        # Different coordinates may change service but not arrivals/mobility/load RNG.
        perturbed = snap.restore()
        macro_step(perturbed,self.sac,np.array([.05,0,0,0,0,0]))
        self.assertEqual(semantic_hash(physical_state(original,self.fields)),original_hash)
        assert_crn_equal(crn_records(clone),crn_records(perturbed))
        self.assertTrue(state_equal(clone.env.np_random.get_state(),perturbed.env.np_random.get_state()))
        self.assertGreater(np.linalg.norm(clone.env.uav_coords-perturbed.env.uav_coords),0)
        self.assertEqual(perturbed.reset_calls,0)
        np.testing.assert_array_equal(perturbed.controls[0]['sampled_delta'],[.05,0,0,0,0,0])

    def test_discount_exact_two_levels_not_mean_physical_cost(self):
        gamma=.99
        costs=np.arange(1.,601.)/1000
        macros=[]
        for k in range(6):
            block=costs[k*100:(k+1)*100]
            frames=[-sum(gamma**(j/5)*block[f*5+j] for j in range(5)) for f in range(20)]
            rows=[dict(cost=c,holding=.9*c,energy=.1*c,full_energy_j=1,
                       energy_compute_j=.2,energy_upload_j=.3,energy_u2u_j=.1,energy_propulsion_j=.4)
                  for c in block]
            macros.append(dict(macro_reward=sum(gamma**f*frames[f] for f in range(20))/20,
                               physical_rows=rows))
        result=objective_values(macros,gamma)
        self.assertAlmostEqual(result['ppo_return60'],
            -sum(gamma**(i/5)*costs[i] for i in range(600))/20,places=11)
        self.assertAlmostEqual(result['C60'],costs.sum(),places=10)
        self.assertNotAlmostEqual(result['ppo_return60'],-result['C60']/20,places=2)

    def test_offline_ranking_conflict_sign_and_ties(self):
        from scripts.analyze_local_counterfactual import ranking
        rows=[dict(action='zero',C60=10.,ppo_return60=-1.),
              dict(action='plus',C60=9.,ppo_return60=-2.),
              dict(action='minus',C60=10.+1e-10,ppo_return60=-1.)]
        result=ranking(rows,1e-8,1e-10)
        self.assertEqual(result['conflict_indices'],[1])
        self.assertFalse(result['best_sets_overlap'])
        self.assertEqual(result['pair_discordant'],2)
        self.assertEqual(result['pair_ties'],1)

    def test_six_macro_metrics_and_energy_ledger(self):
        from hrl_uav.local_counterfactual import branch_metrics, branch_record, save_branch
        meter=ResidualMeter(self.cfg)
        meter.reset(33600,6)
        completed=dict(meter.env.completed_priority_count)
        delays=dict(meter.env.completed_priority_delay_sum)
        macros=[macro_step(meter,self.sac,np.zeros(6)) for _ in range(6)]
        result=branch_metrics(meter,macros,completed,delays)
        self.assertEqual(result['physical_steps'],600)
        self.assertEqual(result['micro_routing_decisions'],108)
        self.assertEqual(result['end_s'],60)
        self.assertAlmostEqual(result['C60'],meter.cost_sum,places=9)
        self.assertAlmostEqual(result['full_energy_j'],meter.total_power_energy,places=8)
        record=branch_record(33600,actions()[0],'test-state',result,result)
        self.assertEqual(record['start_s'],0)
        self.assertEqual(record['delta_C60'],0)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'zero'
            save_branch(target,meter,macros,record)
            self.assertEqual(json.loads((target/'metrics.json').read_text())['start_s'],0)
            self.assertTrue((target/'controls.npz').is_file())


if __name__=='__main__':
    unittest.main()
