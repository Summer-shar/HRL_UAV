"""Bounded Lite candidate checks; no research run or old checkpoint."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import torch

from hrl_v2.collector import collect
from hrl_v2.lite_observation import LiteObserver, physical_summaries, LOWER_DIM, UPPER_DIM, TAUS, SLICES
from hrl_v2.lite_batch import LiteBatch
from hrl_v2.replay import Replay, Sample, Episode
from hrl_v2.randomness import Streams
from hrl_v2.runtime import Learner, configure
from hrl_v2.tests.test_rollout_semantics import adapter, RandomScript

ROOT=Path(__file__).parents[3]
CONFIG=json.loads((Path(__file__).parents[1]/'configs/v2_4_0_lite.json').read_text())


class LiteScript(RandomScript):
    def __init__(self):
        super().__init__()
        self.observer=LiteObserver()
    def begin_episode(self,identity):
        super().begin_episode(identity)
        self.observer.reset()
    def upper(self,report,history):
        request,aux=super().upper(report,history)
        # These are only synthetic PPO behavior values for optimizer plumbing.
        aux.update(pre_tanh=np.zeros(6),mean=np.zeros(6),logstd=np.full(6,np.log(.15)),
                   logprob=float(-6*(np.log(.15)+.5*np.log(2*np.pi))))
        return request,aux


class LiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device=configure(test_cpu=True)
        cls.world=adapter(10)
        cls.reports=[]
        cls.policy=LiteScript()
        cls.episode=collect(cls.world,cls.policy,upper_version='lite-test',lower_version=0,
                            identity={},gamma_lower=.99,on_report=lambda kind,r:cls.reports.append((kind,r)))

    def test_dimensions_determinism_and_reset(self):
        a,b=LiteObserver(self.world.arrivals),LiteObserver(self.world.arrivals)
        reports=sorted([r for _,r in self.reports],key=lambda r:(r.tick,r.history_cursor,r.remaining_macro))
        # At a shared tick, uncommitted upper precedes committed lower.
        for r in reports:
            x,y=a.update(r),b.update(r)
            self.assertEqual(x.lower.tobytes(),y.lower.tobytes())
            self.assertEqual(x.upper.tobytes(),y.upper.tobytes())
        self.assertEqual(x.lower.shape,(110,))
        self.assertEqual(x.upper.shape,(729,))
        print('Lite dimensions: lower=110, upper=39*17+66=729',flush=True)
        a.reset(self.world.arrivals)
        self.assertEqual(a.cursor,0)
        self.assertFalse(a.ema.any())

    def test_upper_boundary_and_idle_potential(self):
        for kind,r in self.reports:
            if kind not in ('upper','upper_successor'): continue
            self.assertEqual(r.remaining_macro,0)
            self.assertEqual(r.remaining_adjustment,0)
            for u in r.uavs: np.testing.assert_allclose(u.target,u.position,rtol=0,atol=1e-12)
        r=adapter(5).report()
        self.assertTrue(all(link.rate==0 and link.all_active_rate>0 for link in r.links))
        x=LiteObserver([]).update(r)
        np.testing.assert_array_equal(x.lower[SLICES['links']][:6],np.zeros(6))
        self.assertTrue((x.lower[SLICES['links']][6:]>0).all())

    def test_upper_pooling_is_within_source_and_parameters_are_independent(self):
        learner=Learner(815,self.device,config=CONFIG)
        x=torch.tensor(self.episode.upper_obs.copy())
        y=x.clone(); terminals=y[:,:663].reshape(-1,3,13,17)
        terminals[:]=terminals.flip(-2).clone()
        for net in (learner.ppo.actor,learner.ppo.value):
            torch.testing.assert_close(net.encoder(x),net.encoder(y),rtol=2e-5,atol=2e-5)
        self.assertFalse(set(map(id,learner.ppo.actor.parameters())) & set(map(id,learner.ppo.value.parameters())))

    def test_incremental_ema_and_future_causality(self):
        history=[v.copy() for v in self.world.arrivals]
        observer=LiteObserver(history)
        for kind,r in self.reports:
            if kind not in ('lower','lower_successor'): continue
            actual=observer.update(r)
            expected=np.zeros((39,2,2,2),dtype=np.float64)
            # Independent frame/terminal/priority calculation, with no observer call.
            for frame in history[:r.history_cursor]:
                for k,tau in enumerate(TAUS):
                    decay=np.exp(-.5/tau)
                    expected[:,:,:,k]=decay*expected[:,:,:,k]+(1-decay)*frame.reshape(39,2,3)[:,:,:2]/.5
            np.testing.assert_allclose(observer.ema,expected,rtol=0,atol=1e-6)
            mutated=[v.copy() for v in history]
            for v in mutated[r.history_cursor:]: v[:]=1e20
            other=LiteObserver(mutated).update(r)
            np.testing.assert_array_equal(actual.lower,other.lower)
            np.testing.assert_array_equal(actual.upper,other.upper)

    def test_conservation_queue_and_parent_counts(self):
        for _,r in self.reports:
            raw=physical_summaries(r)
            self.assertEqual(raw['parents'].sum(),len(r.parents))
            fs={f.key:f for f in r.fragments}
            for kind,field,width in [('cpu','compute_remaining',3),('transfer','transfer_remaining',2)]:
                queues=[q for q in r.queues if q.kind==kind]
                for q,row in zip(queues,raw[kind].reshape(-1,width)):
                    total=sum(getattr(fs[f],field) for f in q.items)
                    self.assertAlmostEqual(row[0],total)
                    self.assertEqual(row[1],len(q.items))
            np.testing.assert_allclose(raw['upload'].sum(),sum(p.upload_remaining for p in r.parents),rtol=1e-14,atol=1e-6)
            np.testing.assert_allclose(raw['ready'][:,:,1].sum(),sum(p.cycles for p in r.parents if p.stage=='ready'),rtol=1e-14,atol=1e-6)
            self.assertEqual(raw['ready'][:,:,2].sum(),sum(p.stage=='ready' for p in r.parents))

    def test_age_geometry_clocks_and_no_episode_countdown(self):
        r=[r for kind,r in self.reports if kind=='lower'][-1]
        observer=LiteObserver(self.world.arrivals)
        original=observer.update(r)
        # Shift absolute time and all birth/entry times together. No absolute
        # tick or rollout fraction may leak into the fixed observation.
        shifted=replace(r,tick=r.tick+100,parents=tuple(replace(p,birth=p.birth+10,
            routed_at=None if p.routed_at is None else p.routed_at+10) for p in r.parents),
            fragments=tuple(replace(f,entered=f.entered+10) for f in r.fragments))
        other=LiteObserver(self.world.arrivals).update(shifted)
        np.testing.assert_allclose(original.lower,other.lower,rtol=0,atol=1e-7)
        np.testing.assert_allclose(original.upper,other.upper,rtol=0,atol=1e-7)
        np.testing.assert_array_equal(original.lower[SLICES['clocks']],np.log1p([r.remaining_macro,r.remaining_adjustment]).astype(np.float32))
        expected=np.asarray([[*np.asarray(u.position)/r.map_size,*np.asarray(u.target)/r.map_size] for u in r.uavs],dtype=np.float32).ravel()
        np.testing.assert_array_equal(original.lower[SLICES['uav']],expected)

    def test_parent_count_is_actual_step_start_holding_basis(self):
        world=adapter(5)
        world.start_macro(np.zeros(6));report=world.prepare()
        counts=physical_summaries(report)['parents']
        observed=LiteObserver(world.arrivals).update(report).lower[SLICES['parents']]
        np.testing.assert_allclose(np.expm1(observed).reshape(2,3),counts,atol=1e-6,rtol=1e-6)
        reading=world.step(np.tile(np.eye(3),(2,1,1)).astype(np.float64))
        self.assertAlmostEqual(reading['holding_task_s'],.1*(10*counts[0].sum()+counts[1].sum()),places=12)

    def test_replay_real_successors_and_no_reports(self):
        e=self.episode
        self.assertEqual(e.lower_obs.shape,(21,LOWER_DIM))
        self.assertEqual(e.upper_obs.shape,(3,UPPER_DIM))
        self.assertEqual(e.history,b'')
        self.assertEqual(e.reports,[])
        samples=[Sample(e,i,1.) for i in range(20)]
        a,b=LiteBatch.replay(samples)
        np.testing.assert_array_equal(a.tensor('cpu').numpy(),e.lower_obs[:-1])
        np.testing.assert_array_equal(b.tensor('cpu').numpy(),e.lower_obs[1:])
        self.assertFalse(np.array_equal(e.lower_obs[-1],e.lower_obs[-2]))
        self.assertTrue(e.lower_obs[-1].any())
        last=[r for kind,r in self.reports if kind=='lower_successor'][0]
        np.testing.assert_array_equal(e.lower_obs[-1],LiteObserver(self.world.arrivals).update(last).lower)
        self.assertTrue(e.truncated[-1]);self.assertFalse(any(e.terminated))
        upper=[r for kind,r in self.reports if kind=='upper']+[r for kind,r in self.reports if kind=='upper_successor']
        for index,report in enumerate(upper):
            np.testing.assert_array_equal(e.upper_obs[index],LiteObserver(self.world.arrivals).update(report).upper)

    def test_observation_does_not_change_physics_reward_or_actions(self):
        world=adapter(10)
        episode=collect(world,RandomScript(),upper_version='lite-test',lower_version=0,identity={},gamma_lower=.99)
        np.testing.assert_array_equal(episode.actions,self.episode.actions)
        np.testing.assert_array_equal(episode.supports,self.episode.supports)
        np.testing.assert_array_equal(episode.rewards,self.episode.rewards)
        np.testing.assert_array_equal(world.arrivals,self.world.arrivals)
        self.assertEqual(world.report(),self.world.report())
        self.assertGreater(episode.metrics['max_report_bytes'],0)
        self.assertIsNone(self.episode.metrics['max_report_bytes'])
        self.assertEqual({k:v for k,v in episode.metrics.items() if k!='max_report_bytes'},
                         {k:v for k,v in self.episode.metrics.items() if k!='max_report_bytes'})

    def test_batch_rows_are_independent_and_selection_order_exact(self):
        learner=Learner(819,self.device,config=CONFIG)
        states=LiteBatch(self.episode.lower_obs.copy())
        with torch.no_grad():
            logits,kappa=learner.sac.actor.batch(states)
            self.assertEqual(logits.shape,(len(states),6,7))
            self.assertEqual(kappa.shape,(len(states),6,7,3))
            self.assertEqual(logits.dtype,torch.float64)
            self.assertEqual(kappa.dtype,torch.float64)
            reverse=states.select(list(range(len(states)-1,-1,-1)))
            a,b=learner.sac.actor.batch(reverse)
            np.testing.assert_array_equal(a.numpy(),logits.flip(0).numpy())
            np.testing.assert_array_equal(b.numpy(),kappa.flip(0).numpy())
            # Q has nonzero initialized outputs, unlike zero policy heads.
            actions=torch.zeros(len(states),36)
            q=learner.sac.q1
            all_values=q.value_features(q.encode_batch(states),actions)
            one_values=torch.cat([q.value_features(q.encode_batch(states[i:i+1]),actions[i:i+1]) for i in range(len(states))])
            torch.testing.assert_close(all_values,one_values,rtol=2e-5,atol=2e-5)

    def test_sampling_matches_baseline_rng(self):
        replay=Replay();replay.set_upper_version('lite-test')
        for _ in range(5):replay.add(copy.deepcopy(self.episode))
        a,b=np.random.default_rng(417),np.random.default_rng(417)
        samples=replay.sample(a)
        expected=[]
        from hrl_v2.replay import probabilities
        # Baseline 0420f12 sampler, including exact RNG call sizes/order.
        for episode_index in b.integers(len(replay.episodes),size=4):
            e=replay.episodes[int(episode_index)];q=probabilities(e.severe)
            for i in b.choice(len(e.rewards),size=32,replace=True,p=q):
                expected.append((id(e),int(i),float((1/len(e.rewards))/q[i])))
        self.assertEqual([(id(s.episode),s.index,s.weight) for s in samples],expected)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_config_and_protected_source_identity(self):
        original=json.loads((Path(__file__).parents[1]/'configs/v2_4_0.json').read_text())
        reduced={k:v for k,v in CONFIG.items() if k not in ('encoder','observation_version')}
        self.assertEqual(original,reduced)
        self.assertEqual(hashlib.sha256((Path(__file__).parents[1]/'configs/v2_1_0.json').read_bytes()).hexdigest(),
                         '7966b4de8e03ddba346d070819c265599e9ae192736a8aaba1fdfd85088076e4')
        protected=['src/hrl_v2/'+name for name in ('networks.py','features.py','batching.py','physics_adapter.py',
            'reports.py','hybrid_simplex.py','sac.py','schedule.py','configs/physical_scene.json')]
        protected += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/hrl_uav').glob('*.py')]
        # Explicit physical-contract changes approved for lite_v2_3_0, plus
        # industrial_hotspot_env.py for lite_v2_4_0 terminal mobility.
        allowed={'src/hrl_v2/physics_adapter.py','src/hrl_v2/configs/physical_scene.json'}
        allowed.update('src/hrl_uav/'+name for name in ('env.py','industrial_env.py',
            'industrial_scene.py','industrial_hotspot_scene.py','industrial_baselines.py',
            'industrial_hotspot_env.py'))
        for name in protected:
            if name in allowed: continue
            baseline=subprocess.check_output(['git','show','0420f12:'+name],cwd=ROOT)
            self.assertEqual((ROOT/name).read_bytes().replace(b'\r\n',b'\n'),baseline.replace(b'\r\n',b'\n'),name)

    def test_graph_dispatch_initialization_identical(self):
        config=copy.deepcopy(CONFIG);config.pop('encoder')
        original=Learner(811,self.device,config=config)
        config['encoder']='graph'
        explicit=Learner(811,self.device,config=config)
        from hrl_v2.runtime import parameter_hash
        for prefix,names in [('sac',('actor','q1','q2','target1','target2')),('ppo',('actor','value'))]:
            for name in names:
                self.assertEqual(parameter_hash(getattr(getattr(original,prefix),name)),parameter_hash(getattr(getattr(explicit,prefix),name)))

    def test_real_optimizer_steps_without_report_hot_path(self):
        learner=Learner(812,self.device,config=CONFIG)
        e=copy.deepcopy(self.episode)
        samples=[Sample(e,i%len(e.rewards),1.) for i in range(128)]
        def values(net):return torch.cat([p.detach().flatten().clone() for p in net.parameters()])
        def changed(before,net):
            after=values(net);self.assertTrue(torch.isfinite(after).all());self.assertFalse(torch.equal(before,after))
        streams=Streams(812)
        with patch.object(Episode,'report',side_effect=AssertionError('Report in learning hot path')), \
             patch('hrl_v2.features.build',side_effect=AssertionError('Graph build in Lite fit')), \
             patch('hrl_v2.batching.encode',side_effect=AssertionError('Graph encoder in Lite fit')), \
             patch('hrl_v2.replay.unpack',side_effect=AssertionError('History in Lite fit')):
            before=[values(q) for q in (learner.sac.q1,learner.sac.q2)]
            learner.sac._critic_fit(samples,.0005,streams)
            for old,q in zip(before,(learner.sac.q1,learner.sac.q2)):changed(old,q)
            old=values(learner.sac.actor)
            learner.sac._actor_fit(samples,.0005,streams);changed(old,learner.sac.actor)
            old=values(learner.ppo.value)
            learner.ppo.update([copy.deepcopy(e) for _ in range(5)],streams)
            changed(old,learner.ppo.value)
            # Explicitly witness the actor optimizer update before any allowed
            # epoch KL rollback can restore its parameters.
            rows=[(e,h,float(i+1)) for i,h in enumerate(e.high)]
            old=values(learner.ppo.actor)
            learner.ppo._minibatch(LiteBatch.upper(rows),rows,np.asarray([-1.,1.]),True)
            changed(old,learner.ppo.actor)
        self.assertEqual(learner.sac.counts['critic_fits'],1)
        self.assertEqual(learner.sac.counts['actor_updates'],1)
        self.assertEqual(learner.ppo.counts['value_steps'],7)
        self.assertFalse(set(map(id,learner.sac.q1.parameters())) & set(map(id,learner.sac.q2.parameters())))


if __name__=='__main__':unittest.main()
