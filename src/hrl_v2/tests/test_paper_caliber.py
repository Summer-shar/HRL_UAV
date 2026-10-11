"""Paper-caliber numerical contracts, with no research training."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.env import UAV_MEC_Env
from hrl_uav.industrial_baselines import GreedyRouting
from hrl_uav.industrial_deployment_env import IndustrialDeploymentEnv
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.baselines import centroid, legacy_view
from hrl_v2.checkpoint import identity, load
from hrl_v2.control_diagnostics import Script
from hrl_v2.paths import source_root
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.policy import Policy
from hrl_v2.randomness import Streams
from hrl_v2.run import read_config
from hrl_v2.tests.test_physical_contract import quiet_scene, step, LOCAL, admit

ROOT = source_root().parent
CONFIG = source_root()/'hrl_v2/configs/v2_3_0_lite.json'
SCENE = json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text())


class PaperCaliberTests(unittest.TestCase):
    def test_u2u_numbers_power_noise_and_reference_quality(self):
        p = HotspotIndustrialEnv(SCENE).physics
        active = ~np.eye(3, dtype=bool)
        six = p.allocate_u2u_resources(active)
        np.testing.assert_allclose(six['u2u_rate_bps']/1e6,
            [[0,3.404,3.245],[3.404,0,3.245],[3.245,3.245,0]], rtol=0,atol=.0005)
        np.testing.assert_array_equal(six['u2u_link_power_w'][active],np.ones(6))
        np.testing.assert_array_equal(six['u2u_link_noise_w'][active],np.full(6,1e-13))
        np.testing.assert_array_equal(six['u2u_link_bandwidth_hz'][active],np.full(6,5e5))
        one=np.zeros((3,3),bool); one[0,1]=True
        single=p.allocate_u2u_resources(one)
        self.assertAlmostEqual(single['u2u_rate_bps'][0,1]/1e6,20.426356372992135,places=12)
        p.u2u_reference_power_w=1.
        p.u2u_reference_bandwidth_hz=512820.5128205128
        reference=UAV_MEC_Env.compute_u2u_reference_quality(p)
        np.testing.assert_allclose(reference,np.log2(1+p._u2u_channel_gain()/1e-13),rtol=0,atol=2e-15)
        self.assertAlmostEqual(1.95e-19*512820.5/1e-13,1.,delta=3e-8)
        self.assertFalse(p.allocate_u2u_resources(np.eye(3,dtype=bool))['u2u_rate_bps'].any())

    def test_legacy_u2u_branch_is_bitwise_unchanged(self):
        old=json.loads(subprocess.check_output(['git','show','4f8198a:src/hrl_v2/configs/physical_scene.json'],cwd=ROOT))
        p=HotspotIndustrialEnv(old).physics
        for mask in (np.zeros((3,3),bool),~np.eye(3,dtype=bool),np.array([[0,1,1],[1,0,0],[0,0,0]],bool)):
            a=p.allocate_u2u_resources(mask)
            bandwidth=mask*6e6/max(mask.sum(),1)
            count=mask.sum(1)
            power=np.divide(mask*1.,count[:,None],out=np.zeros((3,3)),where=count[:,None]>0)
            noise=1.95e-19*bandwidth
            snr=np.divide(power*p._u2u_channel_gain(),noise,out=np.zeros((3,3)),where=mask)
            np.testing.assert_array_equal(a['u2u_rate_bps'],bandwidth*np.log2(1+snr))
            np.testing.assert_array_equal(a['u2u_link_power_w'],power)

    def test_propulsion_segment_hover_command_independent_and_motion(self):
        for request in (None,np.zeros(6),np.array([1,0,.5,0,0,0])):
            env=HotspotIndustrialEnv(quiet_scene())
            if request is not None: env.start_motion(request)
            velocities=[0.,0.,0.] if request is None else [20*request[0],20*request[2],0.]
            expected=env.physics._compute_tcom_flying_power(np.asarray(velocities)).sum()
            readings=[step(env) for _ in range(50)]
            self.assertAlmostEqual(sum(r['energy_j']['propulsion'] for r in readings),expected,places=10)
            self.assertTrue(all(r['energy_j']['propulsion']==0. for r in readings[10:]))
            self.assertAlmostEqual(step(env)['energy_j']['propulsion'],
                                   3*env.physics._compute_tcom_flying_power(np.array([0.]))[0]*.1)
        p=env.physics._compute_tcom_flying_power(np.array([20.,10.,0.]))
        np.testing.assert_allclose(p,[205.2,152.5,194.8],rtol=0,atol=.05)
        self.assertAlmostEqual(3*180*p[2],105172.3,delta=.05)

    def test_cost_uses_propulsion_weight_once_and_retains_raw_joules(self):
        env=HotspotIndustrialEnv(quiet_scene())
        admit(env,bits=1e8,cycles=1e12,priority=0)
        admit(env,bits=1e8,cycles=1e12,priority=1)
        env.prepare_frame()
        with patch.object(env,'_upload_step',return_value=2.),patch.object(env,'_compute_step',return_value=3.),\
             patch.object(env,'_transfer_step',return_value=5.),\
             patch.object(env.physics,'_compute_tcom_flying_power',return_value=np.full(3,100.)):
            r=env.micro_step(LOCAL)
        self.assertEqual(r['holding_task_s'],1.1)
        self.assertEqual(r['energy_j'],dict(upload=2.,cpu=3.,u2u=5.,propulsion=30.))
        self.assertAlmostEqual(r['cost'],.9*1.1/10+.1*(2+3+5+.005*30)/14,places=14)

    def report_at_distance(self,distance,scene=SCENE):
        adapter=PhysicsAdapter(scene)
        adapter.reset(arrival_seed=1,layout_seed=2,load_seed=3)
        r=adapter.report()
        terminals=tuple(replace(t,position=(r.uavs[t.source].position[0]+distance,
                            r.uavs[t.source].position[1]),upload=((0.,1.,0.,0.),(0.,0.,0.,0.))) for t in r.terminals)
        return adapter,replace(r,terminals=terminals)

    def test_centroid_20m_step_does_not_overshoot(self):
        for distance,expected in ((8.,8.),(30.,20.)):
            adapter,r=self.report_at_distance(distance)
            policy=Policy(None,None,None,Streams(1),mode='centroid',deployment=SCENE['deployment'])
            action,_=policy.upper(r,[])
            moved=adapter.start_macro(action)['displacement_m']
            np.testing.assert_allclose(moved,np.tile([expected,0.],(3,1)),rtol=0,atol=1e-12)
            script=Script(r,'toward','equal',deployment=SCENE['deployment'])
            np.testing.assert_array_equal(script.upper(r,[])[0],action)

    def test_centroid_old_10m_actions_match_original_bytes(self):
        # Run the actual archived function so weights/window/prediction also match.
        text=subprocess.check_output(['git','show','4f8198a:src/hrl_v2/baselines.py'],cwd=ROOT).decode()
        code=text[text.index('def centroid('):text.index('\n\ndef legacy_view')]
        namespace={'np':np}; exec(code,namespace)
        _,r=self.report_at_distance(8.)
        history=np.random.default_rng(83).uniform(size=(31,39,6))
        for distance in (8.,30.):
            _,r=self.report_at_distance(distance)
            r=replace(r,history_cursor=31)
            expected=namespace['centroid'](r,history)
            actual=centroid(r,history,max_target_step_m=10.)
            self.assertEqual(actual.tobytes(),expected.tobytes())
        old_script=Script(r,'toward','equal',deployment=dict(max_target_step_m=10.))
        for macro in (0,1,2,3):
            request=old_script.upper(replace(r,tick=50*macro),history)[0]
            expected=old_script.heading*np.clip(old_script.distance-10*macro,0.,10.)[:,None]/10
            self.assertEqual(request.tobytes(),expected.ravel().tobytes())

    def test_greedy_estimates_priority_inbound_and_candidate_link(self):
        _,r=self.report_at_distance(8.)
        view=legacy_view(r)
        view.compute_cycles=np.array([[1,2,3],[9,8,7]],float)*1e10
        view.transfer_cycles=np.zeros((2,3,3));view.transfer_cycles[0,2,1]=4e10
        view.transfer_cycles[1,1,0]=5e10
        view.transfer_bits=np.zeros((2,3,3));view.transfer_bits[1,0,1]=1e6
        view.ready_tasks=(SimpleNamespace(priority=0,source=0,input_bits=2e6),)
        rule=GreedyRouting(SCENE['base_scene']); times=rule.estimates(view)
        self.assertEqual(times[0,0,0],1.)
        self.assertEqual(times[1,0,0],15.)
        rate=3e6*np.log2(1+1e-6/300**2/1e-13)
        self.assertAlmostEqual(times[0,0,1],6.+3e6/rate)
        self.assertAlmostEqual(times[1,0,1],14.+1e6/rate)
        route=rule(view).reshape(2,3,3)
        np.testing.assert_array_equal(route.sum(2),np.ones((2,3)))
        self.assertTrue(np.isin(route,[0.,1.]).all())

    def test_new_config_only_changes_declared_cost_and_status(self):
        new=read_config(CONFIG,ROOT)
        old=json.loads((CONFIG.parent/'v2_2_1_lite.json').read_text())
        new=copy.deepcopy(new);new['status']=old['status']
        self.assertEqual(new['objective'].pop('propulsion_weight'),.005)
        self.assertEqual(new['objective'].pop('energy_mode'),'deployment_segment_propulsion')
        self.assertEqual(new,old)
        with self.assertRaisesRegex(ValueError,'physical objective'):
            read_config(CONFIG.parent/'v2_2_1_lite.json',ROOT)

    def test_scene_changes_are_confined_to_the_approved_parameters(self):
        old=json.loads(subprocess.check_output(['git','show','4f8198a:src/hrl_v2/configs/physical_scene.json'],cwd=ROOT))
        new=copy.deepcopy(SCENE)
        for key in ('schema','status'): new[key]=old[key]
        new['deployment']['max_target_step_m']=old['deployment']['max_target_step_m']
        b,prior=new['base_scene'],old['base_scene']
        for key in ('status','parameter_sources'): b[key]=prior[key]
        b['motion']['max_speed_mps']=prior['motion']['max_speed_mps']
        b['resources']['u2u_bandwidth_hz']=prior['resources']['u2u_bandwidth_hz']
        b['resources'].pop('u2u_model'); b['resources'].pop('u2u_noise_w')
        b['objective']['energy_mode']=prior['objective']['energy_mode']
        b['objective'].pop('propulsion_weight')
        self.assertEqual(new,old)

    def test_greedy_is_causal_and_does_not_consume_policy_randomness(self):
        from hrl_v2.baselines import Routing
        _,r=self.report_at_distance(8.)
        rule=Routing('greedy',SCENE,4101)
        a,ids=rule(r,np.zeros((5,39,6)))
        b,other=rule(r,np.full((50,39,6),1e20))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(ids,other)

    def test_old_checkpoint_identity_rejected_before_payload_read(self):
        binding=identity(ROOT,CONFIG)
        old=copy.deepcopy(binding);old['scene_sha256']='old-scene'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'old.pt'
            p.with_suffix('.json').write_text(json.dumps(dict(schema=binding['schema'],identity=old)))
            with patch('hrl_v2.checkpoint.torch.load',side_effect=AssertionError('must not deserialize')):
                with self.assertRaisesRegex(ValueError,'identity mismatch'): load(p,binding)

    def test_active_motion_method_is_deployment_override(self):
        adapter=PhysicsAdapter(SCENE)
        self.assertIs(type(adapter.env).start_motion,IndustrialDeploymentEnv.start_motion)
        self.assertNotIn('_reference',__import__('inspect').getfile(type(adapter)))
        self.assertEqual(adapter.report().max_speed,20.)


if __name__=='__main__': unittest.main()
