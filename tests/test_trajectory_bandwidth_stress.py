"""Short no-network tests. No trained performance measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.local_counterfactual import (
    ROOT,FullSnapshot,ResidualMeter,physical_fields,physical_state,semantic_hash,macro_step,
    branch_metrics,branch_record,save_branch,actions,
)
from hrl_uav.resumption import state_equal
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.trajectory_check import crn_records
from hrl_uav.trajectory_bandwidth_stress import (
    switch_bandwidth,augmented_metrics,read_csv,resource_response,verify_reused_geometry,
)


class BandwidthStressTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=json.loads((ROOT/'configs/sac_only_wireless6_pilot.json').read_text())
        self.fields=physical_fields(self.cfg)
        self.sac=UniformRouting()

    def test_switch_preserves_every_other_state_and_reference_observations(self):
        meter=ResidualMeter(self.cfg)
        meter.reset(33600,6)
        macro_step(meter,self.sac,np.zeros(6))
        obs=meter.micro_observation().copy()
        reference=meter.env.compute_u2u_reference_quality().copy()
        before=semantic_hash(physical_state(meter,self.fields))
        old_cfg=copy.deepcopy(meter.config)
        audit=switch_bandwidth(meter,3,self.fields)
        self.assertTrue(audit['initial_state_excluding_bandwidth_equal'])
        self.assertEqual(meter.env.B_u2u_total,3e6)
        self.assertEqual(meter.env.noise_psd_w_per_hz,1.95e-19)
        np.testing.assert_array_equal(obs,meter.micro_observation())
        np.testing.assert_array_equal(reference,meter.env.compute_u2u_reference_quality())
        switch_bandwidth(meter,6,self.fields)
        self.assertEqual(meter.config,old_cfg)
        self.assertEqual(before,semantic_hash(physical_state(meter,self.fields)))
        for invalid in (1,4,True,float('nan')):
            with self.assertRaises(ValueError):
                switch_bandwidth(meter,invalid,self.fields)

    def test_same_snapshot_six_macro_output_schema_geometry_and_CRN(self):
        original=ResidualMeter(self.cfg)
        original.reset(33600,7)
        macro_step(original,self.sac,np.zeros(6))
        snap=FullSnapshot(original,self.fields)
        reference_hash=semantic_hash(physical_state(original,self.fields))
        old=zero3=links3=None
        with tempfile.TemporaryDirectory() as directory:
            for bw,action in ((6,actions()[0]),(3,actions()[0]),(3,actions()[1])):
                meter=snap.restore()
                switch_bandwidth(meter,bw,self.fields)
                done=dict(meter.env.completed_priority_count)
                delays=dict(meter.env.completed_priority_delay_sum)
                macros=[macro_step(meter,self.sac,np.array(action['delta']) if k==0 else np.zeros(6)) for k in range(6)]
                values=branch_metrics(meter,macros,done,delays)
                if bw==6:
                    old=values.copy()
                    links6=copy.deepcopy(meter.link_readings)
                    crn6=crn_records(meter)
                    rng6=copy.deepcopy(meter.env.np_random.get_state())
                if bw==3 and action['name']=='zero':
                    zero3=values.copy()
                    links3=copy.deepcopy(meter.link_readings)
                    verify_reused_geometry(meter,macros,Path(directory)/'b6_zero')
                    self.assertNotAlmostEqual(values['C60'],old['C60'],places=5)
                zero=old if bw==6 else zero3
                links=links6 if bw==6 else links3
                expanded=augmented_metrics(values,meter.link_readings,zero,links)
                record=dict(expanded,seed=33600,action=action['name'],dimension=action['dimension'],
                            sign=action['sign'],initial_state_sha256=snap.initial_hash)
                record.update(bandwidth_mhz=bw,reused=False,source_path='test',source_metrics_sha256='test')
                dest=Path(directory)/f'b{bw}_{action["name"]}'
                save_branch(dest,meter,macros,record)
                assert_crn_equal(crn6,crn_records(meter))
                assert_crn_equal(crn6,read_csv(dest/'crn.csv'))
                self.assertTrue(state_equal(rng6,meter.env.np_random.get_state()))
                self.assertAlmostEqual(record['J60'],record['C60']/600)
                self.assertEqual(record['micro_routing_decisions'],108)
                self.assertEqual(record['end_s'],70)
                self.assertEqual(meter.reset_calls,0)
                self.assertEqual(reference_hash,semantic_hash(physical_state(original,self.fields)))
                if action['name']=='zero':
                    self.assertEqual(record['rate_response_abs_mbps_service'],0)

    def test_actual_rate_sensitivity_uses_service_denominator_and_matching_frames(self):
        row=dict(time_s=1.5,source=0,destination=1,service_steps=5,rate_mbps_sum=10,
                 served_mbit=1,tx_time_s=.1,active_steps=5)
        zero=[dict(row),dict(row,destination=2,rate_mbps_sum=0,served_mbit=0,tx_time_s=0,active_steps=0)]
        candidate=[dict(row,rate_mbps_sum=15),dict(zero[1],rate_mbps_sum=5,active_steps=1)]
        r=resource_response(candidate,zero)
        self.assertEqual(r['rate_response_abs_mbps_service'],1)
        self.assertEqual(r['active_pattern_response_link_seconds'],.1)
        candidate[1]['time_s']=2
        with self.assertRaises(AssertionError):
            resource_response(candidate,zero)


if __name__=='__main__':
    unittest.main()
