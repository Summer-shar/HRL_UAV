import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav import trajectory_check as base
from hrl_uav.target_tracking_check import RULES,evaluate,pairing,tracking_action,numeric_rows
from tests.test_trajectory_check import FixedDummy


ROOT=Path(__file__).resolve().parents[1]


class TargetTrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=json.loads((ROOT/"configs/sac_only_wireless6_pilot.json").read_text())
        self.centroids=np.array([[300,300],[300,600],[600,450]],float)
        self.current=self.centroids+np.array([[6,-4],[-3,5],[8,2]])
        self.loads=np.array([14.375,6.325,4.025])

    def test_coefficient_one_recovers_original_raw_actions(self):
        for rule in RULES.values():
            raw,d=tracking_action(self.current,self.centroids,self.loads,rule['offsets_m'],60,1)
            np.testing.assert_array_equal(raw,base.role_action(self.loads,rule['offsets_m'],60))
            np.testing.assert_allclose(d['feasible_target'],d['original_target'],atol=1e-10,rtol=0)

    def test_half_is_world_tracking_not_half_offset(self):
        raw,d=tracking_action(self.current,self.centroids,self.loads,RULES['light_xp']['offsets_m'],60,.5)
        np.testing.assert_allclose(d['feasible_target'],self.current+.5*(d['original_target']-self.current),atol=1e-10,rtol=0)
        self.assertGreater(np.linalg.norm(raw-.5*base.role_action(self.loads,RULES['light_xp']['offsets_m'],60)),.01)
        self.assertFalse(d['projected'].any())

    def test_projection_records_unreachable_target_without_expanding_action(self):
        far=self.centroids+np.array([[200,0],[0,180],[-190,0]])
        raw,d=tracking_action(far,self.centroids,self.loads,RULES['zero']['offsets_m'],60,.5)
        self.assertTrue(d['projected'].all())
        self.assertTrue(np.all(np.linalg.norm(raw.reshape(3,2),axis=1)<=1+1e-12))
        self.assertTrue(np.all(np.linalg.norm(d['feasible_target']-d['interpolated_target'],axis=1)>1))

    def test_real_short_rollout_mapping_energy_and_pairing(self):
        dummy=FixedDummy()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for label,target in (('A','zero'),('C','light_xp')):
                native,_=base.evaluate_rule(self.cfg,dummy,RULES[target],33600,2,root/(label+'native'),'smoke')
                native.update(policy=label,source_path=str(root/(label+'native')))
                one,_=evaluate(self.cfg,dummy,target,1,33600,2,root/(label+'one'),label+'one')
                self.assertTrue(pairing(native,one,same=True)['crn_equal'])
                half,meter=evaluate(self.cfg,dummy,target,.5,33600,2,root/(label+'half'),'B' if label=='A' else 'D')
                self.assertTrue(pairing(native,half)['crn_equal'])
                if label=='A':
                    self.assertGreater(half['mean_position_delta_vs_zero_m'],0)
                self.assertEqual(len(meter.tracking),6)
                self.assertLessEqual(half['max_speed_mps'],20+1e-8)
                np.testing.assert_allclose(half['energy_steady_j']+half['energy_transient_j']+half['energy_hover_j'],half['energy_propulsion_j'],atol=1e-8)
                self.assertTrue(all(x['energy_u2u_j']==0 for x in meter.readings if x['service_steps']==0))

    def test_scope_is_closed(self):
        for coefficient in (.25,.75,2):
            with self.assertRaises(ValueError):
                tracking_action(self.current,self.centroids,self.loads,RULES['zero']['offsets_m'],60,coefficient)
        for seed in (33610,33611,33612,33700,33709):
            with self.assertRaises(ValueError):
                evaluate(self.cfg,FixedDummy(),'zero',.5,seed,90,ROOT/'never_created','B')

    def test_corrected_numeric_source_reader(self):
        rows=numeric_rows(ROOT/'results/trajectory_opportunity_sac101/per_seed_corrected.csv')
        selected=[r for r in rows if r['policy']=='zero' and r['seed']==33600]
        self.assertEqual(len(selected),1)
        self.assertIsInstance(selected[0]['J'],float)


if __name__ == '__main__':
    unittest.main()
