"""Task 4 storage changes must preserve observations, RNG and committed logs."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hrl_v2.archive import (canonical_rows, publish_checkpoint, recover_metric,
    save_network_snapshot, should_checkpoint, supersede_episodes, _equal_state)
from hrl_v2.checkpoint import atomic_json, digest, load
from hrl_v2.lite_observation import LOWER_DIM, UPPER_DIM
from hrl_v2.replay import Episode, LiteReportKey
from hrl_v2.runtime import Learner, configure
from hrl_v2.tests.test_paper_baselines import CONFIG


def episode():
    e=Episode('upper',0,{},encoder='lite',gamma_lower=.99)
    def key(t):
        return e.add_report(SimpleNamespace(tick=t,history_cursor=t//5,phase='hold',
                           remaining_macro=0.,remaining_adjustment=0.))
    a,b,c,d=key(0),key(5),key(5),key(10)
    e.observations=[a,c];e.successors=[b,d]
    e.actions=[np.ones((2,3,3))/3]*2;e.supports=[np.full(6,6)]*2
    e.rewards=[-1.,-2.];e.terminated=[False]*2;e.truncated=[False,True];e.severe=[False]*2
    e.high=[dict(report=a,successor=b,reward_physical=-1.,reward_learning=-1.,terminated=False,truncated=False),
            dict(report=c,successor=d,reward_physical=-2.,reward_learning=-2.,terminated=False,truncated=True)]
    e.lower_obs=np.zeros((3,LOWER_DIM),np.float32);e.upper_obs=np.zeros((3,UPPER_DIM),np.float32)
    return e


class LiteKeyTests(unittest.TestCase):
    def test_keys_are_immutable_and_add_report_never_pickles(self):
        with patch('hrl_v2.replay.pack',side_effect=AssertionError('Report serialized')):
            e=episode()
        self.assertIsInstance(e.reports[0],LiteReportKey)
        with self.assertRaises(FrozenInstanceError): e.reports[0].tick=9
        with patch.object(Episode,'report',side_effect=AssertionError('Lite report decoded')):
            e.seal([0,0,0],dict(cost_sum=300.))
        self.assertTrue(e.complete)
        self.assertEqual(e.reports,[])
        self.assertFalse(e.lower_obs.flags.writeable)

    def test_seal_rejects_tampered_successor_and_upper_successor(self):
        for mutate in ('lower','upper','stride','history','span','physical_reward','learning_reward','shape','finite'):
            e=episode()
            if mutate=='lower': e.reports[1]=replace(e.reports[1],phase='adjustment')
            elif mutate=='upper': e.high[0]['successor']=e.add_report(SimpleNamespace(tick=5,history_cursor=1,phase='adjustment',remaining_macro=0.,remaining_adjustment=0.))
            elif mutate=='stride': e.reports[3]=replace(e.reports[3],tick=11)
            elif mutate=='history': e.reports[3]=replace(e.reports[3],history_cursor=100)
            elif mutate=='span': e.high[1]['report']=0
            elif mutate=='physical_reward': e.high[0]['reward_physical']=-9.
            elif mutate=='learning_reward': e.high[0]['reward_learning']=-9.
            elif mutate=='shape': e.lower_obs=e.lower_obs[:-1]
            elif mutate=='finite': e.lower_obs[0,0]=np.nan
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                e.seal([0,0,0],dict(cost_sum=300.))

    def test_training_skips_byte_count_but_evaluation_still_counts(self):
        from hrl_v2.tests.test_paper_baselines import world,SCENE
        from hrl_v2.upper_baselines import UpperLearner,fixed_upper_policy
        from hrl_v2.randomness import Streams
        from hrl_v2.evaluate import gate_manifest
        from hrl_v2.collector import collect,collect_metrics
        device=configure(test_cpu=True)
        learner=UpperLearner(4101,device,CONFIG)
        def setup():
            a=world(gate_manifest(20)['scenarios'][0],5)
            p=fixed_upper_policy(learner.ppo,SCENE,Streams(4101),'equal',stochastic=False)
            return a,p
        a,p=setup();seen=[]
        with patch.object(a,'report_bytes',side_effect=AssertionError('Redundant serialization')):
            e=collect(a,p,upper_version='test',lower_version='equal',identity={},on_report=lambda kind,r:seen.append((kind,r)))
        self.assertIsNone(e.metrics['max_report_bytes'])
        self.assertTrue(any(kind=='lower_successor' for kind,r in seen))
        a,p=setup()
        with patch.object(a,'report_bytes',wraps=a.report_bytes) as counter:
            metrics=collect_metrics(a,p,identity={})
        self.assertEqual(counter.call_count,10)
        self.assertGreater(metrics['max_report_bytes'],0)
        self.assertEqual({k:v for k,v in e.metrics.items() if k!='max_report_bytes'},
                         {k:v for k,v in metrics.items() if k!='max_report_bytes'})


class ArchiveTests(unittest.TestCase):
    def test_checkpoint_cadence_and_mandatory_nodes(self):
        for every in (1,3,10):
            expected={n for n in range(601) if n%every==0}|{0,300,400,450,500,550,600}
            actual={n for n in range(601) if should_checkpoint(n,every,pending=n if n in (300,400,450,500,550,600) else None)}
            self.assertEqual(actual,expected)
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError): should_checkpoint(1,value)

    def test_upper_anchors_survive_complete_retention(self):
        from hrl_v2.tests.test_paper_baselines import SCENE
        from hrl_v2.upper_baselines import UpperLearner,upper_phases
        from hrl_v2.train_upper import _run
        from hrl_v2.schedule import phase
        config=copy.deepcopy(CONFIG);config['task']['rollout_seconds']=5
        learner=UpperLearner(4101,configure(test_cpu=True),config)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'upper'
            _run(learner,SCENE,config,4101,root,out,'NC',{},
                 phases=[phase(400),*upper_phases()[:4]],checkpoint_every=1)
            self.assertTrue((out/'checkpoints/episode-0001-anchor.pt').exists())
            self.assertTrue((out/'checkpoints/episode-0005-anchor.pt').exists())
            self.assertTrue((out/'checkpoints/episode-0000-initial.pt').exists())
            self.assertEqual(len(list((out/'checkpoints').glob('*-complete.pt'))),2)
            self.assertEqual(len(canonical_rows(out/'train_metrics.jsonl')),5)

    def test_snapshot_preserves_rng_and_learner_and_cannot_resume(self):
        configure(test_cpu=True);learner=Learner(4101,torch.device('cpu'),CONFIG)
        before=copy.deepcopy(learner.state_dict())
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        py=random.getstate();np_state=pickle.dumps(np.random.get_state())
        with tempfile.TemporaryDirectory() as directory:
            path=save_network_snapshot(directory,learner,25,{'fixture':'task4'})
            sha=digest(path)
            self.assertEqual(save_network_snapshot(directory,learner,25,{'fixture':'task4'}),path)
            self.assertEqual(digest(path),sha)
            self.assertTrue(_equal_state(before,learner.state_dict()))
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
            self.assertEqual(py,random.getstate());self.assertEqual(np_state,pickle.dumps(np.random.get_state()))
            with self.assertRaises(ValueError): load(path,{'fixture':'task4'})
            with torch.no_grad(): next(learner.sac.actor.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError,'differs'): save_network_snapshot(directory,learner,25,{'fixture':'task4'})
            self.assertEqual(digest(path),sha)

    def test_superseded_and_canonical_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir()
            checkpoint=out/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
            publish_checkpoint(root,out,checkpoint,6,'first')
            metric=out/'train_metrics.jsonl';learning=out/'learning.jsonl'
            rows=[dict(global_episode=i,attempt_id='first',value=i) for i in range(1,9)]
            metric.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            learning.write_text(''.join(json.dumps(dict(r,event=j))+'\n' for r in rows for j in (1,2)),encoding='utf-8')
            for row in rows: atomic_json(out/'episodes'/f"episode-{row['global_episode']:04d}.json",dict(row,train_metric=row))
            self.assertEqual(len(canonical_rows(metric)),6)
            moved=supersede_episodes(out,6)
            self.assertEqual(moved,['episodes/superseded/first/episode-0007.json','episodes/superseded/first/episode-0008.json'])
            self.assertTrue((out/'episodes/episode-0006.json').exists())
            with metric.open('a',encoding='utf-8') as f:
                for i in (7,8,9): f.write(json.dumps(dict(global_episode=i,attempt_id='second',value=i))+'\n')
            publish_checkpoint(root,out,checkpoint,9,'second')
            actual=canonical_rows(metric)
            self.assertEqual([r['global_episode'] for r in actual],list(range(1,10)))
            self.assertEqual([r['attempt_id'] for r in actual],['first']*6+['second']*3)
            self.assertEqual(len(canonical_rows(learning)),12)
            with metric.open('a',encoding='utf-8') as f: f.write('{incomplete')
            self.assertEqual(canonical_rows(metric),actual)

    def test_recover_committed_metric_does_not_admit_old_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir();cp=out/'cp.pt';cp.write_bytes(b'cp')
            publish_checkpoint(root,out,cp,1,'new')
            metric=out/'train_metrics.jsonl'
            metric.write_text(json.dumps(dict(global_episode=1,attempt_id='old'))+'\n',encoding='utf-8')
            row=dict(global_episode=1,attempt_id='new',reward=3.)
            atomic_json(out/'episodes/episode-0001.json',dict(attempt_id='new',train_metric=row))
            recover_metric(out,1);recover_metric(out,1)
            self.assertEqual(len(canonical_rows(metric)),1)
            self.assertEqual(canonical_rows(metric)[0]['reward'],3.)


if __name__=='__main__': unittest.main()
