"""Independent upper baselines: tiny real PPO updates, no research training."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_v2 import SCHEMA
from hrl_v2.baseline_protocol import PAPER_COMPARISONS, UPPER_BASELINES, labeled_results
from hrl_v2.baselines import Routing
from hrl_v2.checkpoint import identity, load, save
from hrl_v2.collector import collect_metrics
from hrl_v2.evaluate import gate_manifest, manifest, panel as proposed_panel
from hrl_v2.paper_evaluation import comparison_policy, load_checkpoints
from hrl_v2.randomness import Streams, derive
from hrl_v2.replay import Replay
from hrl_v2.runtime import Learner, configure, parameter_hash
from hrl_v2.schedule import phase
from hrl_v2.tests.test_paper_baselines import (CONFIG, CONFIG_PATH, SCENE, ROOT,
                                            world, old_module, arrival_bytes)
from hrl_v2.timing import ClockConfig
from hrl_v2.train_upper import _run, collect_upper, expected_budget, training_manifest
from hrl_v2.upper_baselines import (UpperLearner, baseline_tag, fixed_upper_policy,
                                   load_typed, upper_phases)


class UpperBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = configure(test_cpu=True)
        cls.binding = identity(ROOT, CONFIG_PATH)
        cls.tiny = copy.deepcopy(CONFIG)
        cls.tiny['task']['rollout_seconds'] = 5

    def checkpoint_set(self, directory):
        paths = {}
        for index, name in enumerate(('Proposed','NC','TOO','TFOO')):
            learner = (Learner if name == 'Proposed' else UpperLearner)(4101, self.device, CONFIG)
            with torch.no_grad(): learner.ppo.actor.mean.bias.fill_(.1*(index+1))
            tag = {} if name == 'Proposed' else baseline_tag(name)
            path = Path(directory)/(name+'.pt')
            save(path, dict(schema=SCHEMA, safe_episode_boundary=True, identity=self.binding,
                           completed_episode=0, pending_evaluation=None, seed=4101,
                           learner=learner.state_dict(), **tag))
            paths[name] = path
        return paths

    def test_main_labels_and_reference_supplement(self):
        self.assertEqual([(p,n) for p,n,_,_ in PAPER_COMPARISONS],
                         [('NC','moving_local'),('CA','hover_equal'),('TOO','moving_equal'),
                          ('TFOO','moving_cvx'),('Proposed','moving')])
        labeled = labeled_results({name:{} for name in ('hover_local','hover_equal','hover_cvx',
                                                       'shared_upper_equal','shared_upper_cvx')})
        self.assertEqual([r['paper_name'] for r in labeled['main_comparison']], ['CA'])
        self.assertEqual({r['internal_name'] for r in labeled['supplementary']},
                         {'hover_local','hover_cvx','shared_upper_equal','shared_upper_cvx'})

    def test_budget_and_every_external_seed_match_proposed_upper_phases(self):
        budget = expected_budget(CONFIG)
        self.assertEqual((budget['episodes'], budget['ppo_batches'], budget['physical_steps']), (200,40,1800000))
        entries = training_manifest(4101)
        proposed = [phase(n) for n in range(1,601) if phase(n).layer == 'upper']
        self.assertEqual(upper_phases(), proposed)
        self.assertEqual(len(entries),200)
        for entry,p in zip(entries,proposed):
            self.assertEqual(entry['proposed_global_episode'],p.global_episode)
            for purpose in ('arrival','layout','load'):
                self.assertEqual(entry[purpose+'_seed'], derive(4101,p.stage,p.cycle,p.block_episode,purpose=purpose))
            a=Streams(4101,p.stage,p.cycle,p.block_episode)
            b=Streams(4101,entry['stage'],entry['cycle'],entry['block_episode'])
            for key,value in a.state_dict()['generators'].items():
                self.assertEqual(value.numpy().tobytes(),b.state_dict()['generators'][key].numpy().tobytes())

    def test_upper_initialization_and_ppo_hyperparameters_match_proposed(self):
        proposed = Learner(4101,self.device,CONFIG)
        upper = UpperLearner(4101,self.device,CONFIG)
        self.assertFalse(hasattr(upper,'sac'))
        for name in ('actor','value'):
            self.assertEqual(parameter_hash(getattr(proposed.ppo,name)),parameter_hash(getattr(upper.ppo,name)))
        for name in ('actor_optimizer','value_optimizer'):
            self.assertEqual(getattr(proposed.ppo,name).state_dict(),getattr(upper.ppo,name).state_dict())
        self.assertEqual((proposed.ppo.gamma,proposed.ppo.gae_lambda),(upper.ppo.gamma,upper.ppo.gae_lambda))

    def test_upper_collection_and_update_never_touch_sac_or_replay(self):
        learner = Learner(4101,self.device,self.tiny)
        names = ('actor','q1','q2','target1','target2')
        before = {n:parameter_hash(getattr(learner.sac,n)) for n in names}
        counts = dict(learner.sac.counts)
        version = learner.sac.version
        upper_before = parameter_hash(learner.ppo.value)
        with ExitStack() as stack:
            for obj,name in [(learner.sac.actor,'forward'),(learner.sac,'event'),
                             (learner.sac,'_critic_fit'),(learner.sac,'_actor_fit'),
                             (Replay,'__init__'),(Replay,'add')]:
                stack.enter_context(patch.object(obj,name,side_effect=AssertionError('SAC/replay was accessed')))
            episodes = []
            for p in upper_phases()[:5]:
                e,streams=collect_upper(learner.ppo,SCENE,self.tiny,4101,p,'equal')
                episodes.append(e)
            learner.ppo.update(episodes,streams)
        self.assertEqual(learner.ppo.counts['batches'],1)
        self.assertEqual(learner.sac.counts,counts)
        self.assertEqual(learner.sac.version,version)
        self.assertEqual(before,{n:parameter_hash(getattr(learner.sac,n)) for n in names})
        self.assertNotEqual(upper_before,parameter_hash(learner.ppo.value))

    def test_fixed_routes_match_existing_overrides_byte_for_byte(self):
        learner=UpperLearner(4101,self.device,CONFIG)
        for route in ('local','equal','cvx'):
            with self.subTest(route=route):
                adapter=world(gate_manifest(20)['scenarios'][0])
                policy=fixed_upper_policy(learner.ppo,SCENE,Streams(4101),route,stochastic=True)
                oracle=None if route=='equal' else Routing(route,SCENE,4101)
                adapter.start_macro(np.zeros(6))
                for _ in range(20):
                    report=adapter.prepare()
                    a,ids=policy.lower(report,adapter.arrivals)
                    b,bids=(np.full((2,3,3),1/3,dtype=np.float64),np.full(6,6)) if oracle is None else oracle(report,adapter.arrivals)
                    self.assertEqual(a.tobytes(),b.tobytes())
                    self.assertEqual(ids.tobytes(),bids.tobytes())
                    adapter.step(a)
                if route=='cvx':
                    self.assertGreater(len(policy.lower_override.rule.solver.records),0)

    def test_tiny_training_budget_and_typed_checkpoints_for_all_three_routes(self):
        for baseline in UPPER_BASELINES:
            with self.subTest(baseline=baseline),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); output=root/baseline
                upper=UpperLearner(4101,self.device,self.tiny)
                with patch.object(Replay,'add',side_effect=AssertionError('SAC replay write')), \
                     patch('hrl_v2.sac.SAC.__init__',side_effect=AssertionError('SAC constructed')):
                    result=_run(upper,SCENE,self.tiny,4101,root,output,baseline,self.binding,phases=upper_phases()[:5])
                self.assertEqual(result['actual_counts']['episodes'],5)
                self.assertEqual(result['actual_counts']['physical_steps'],250)
                self.assertEqual(result['actual_counts']['ppo_batches'],1)
                self.assertEqual(result['ppo']['batches'],1)
                self.assertEqual(result['ppo']['value_steps'],6)
                latest=json.loads((output/'latest.json').read_text(encoding='utf-8'))
                saved=load_typed(root/latest['path'],self.binding,baseline)
                self.assertEqual(saved['baseline_type'],baseline)
                self.assertEqual(saved['lower_override'],UPPER_BASELINES[baseline][1])
                self.assertNotIn('replay',saved)
                self.assertNotIn('sac',saved['learner'])
                self.assertEqual(saved['ppo_batch'],[])
                metrics=[json.loads(s) for s in (output/'train_metrics.jsonl').read_text().splitlines()]
                self.assertEqual(len(metrics),5)
                self.assertTrue(all(m['sac_forwards']==m['sac_updates']==m['sac_replay_writes']==0 for m in metrics))
                with self.assertRaises(FileExistsError):
                    _run(upper,SCENE,self.tiny,4101,root,output,baseline,self.binding,phases=upper_phases()[:5])

    def test_checkpoint_type_and_lower_override_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.checkpoint_set(directory)
            for actual,path in paths.items():
                for expected in paths:
                    if actual != expected:
                        with self.subTest(actual=actual,expected=expected),self.assertRaisesRegex(ValueError,'type mismatch'):
                            load_typed(path,self.binding,expected)
            value=load(paths['NC'],self.binding)
            value['lower_override']='equal'
            corrupt=Path(directory)/'wrong_route.pt'; save(corrupt,value)
            with self.assertRaisesRegex(ValueError,'type mismatch'):
                load_typed(corrupt,self.binding,'NC')
            invalid=dict(paths,NC=paths['Proposed'])
            with patch('hrl_v2.paper_evaluation.UpperLearner',side_effect=AssertionError('Constructed before type validation')):
                with self.assertRaisesRegex(ValueError,'type mismatch'):
                    load_checkpoints(invalid,self.binding,CONFIG,self.device)

    def test_resume_preserves_pending_ppo_batch_parameters_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            continuous=UpperLearner(4101,self.device,self.tiny)
            resumed=UpperLearner(4101,self.device,self.tiny)
            phases=upper_phases()[:5]
            _run(continuous,SCENE,self.tiny,4101,root,root/'continuous','TOO',self.binding,phases=phases)
            calls=[]
            def interrupted(*args,**kwargs):
                if len(calls)==2: raise RuntimeError('test interruption')
                calls.append(1)
                return collect_upper(*args,**kwargs)
            with patch('hrl_v2.train_upper.collect_upper',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'test interruption'):
                    _run(resumed,SCENE,self.tiny,4101,root,root/'resumed','TOO',self.binding,phases=phases)
            latest=json.loads((root/'resumed/latest.json').read_text())
            saved=load_typed(root/latest['path'],self.binding,'TOO')
            self.assertEqual(saved['completed_episode'],2)
            self.assertEqual(len(saved['ppo_batch']),2)
            resumed=UpperLearner(4101,self.device,self.tiny)
            _run(resumed,SCENE,self.tiny,4101,root,root/'resumed','TOO',self.binding,phases=phases,saved=saved)
            from hrl_v2.tests.test_learning import fingerprint
            self.assertEqual(fingerprint(continuous.state_dict()),fingerprint(resumed.state_dict()))
            self.assertEqual(resumed.ppo.counts['batches'],1)

    def test_independent_panel_actions_use_each_own_checkpoint_and_share_arrivals(self):
        with tempfile.TemporaryDirectory() as directory:
            learners,ids,seed=load_checkpoints(self.checkpoint_set(directory),self.binding,CONFIG,self.device)
            self.assertEqual(len(set(ids.values())),4)
            tapes=[]; actions=[]
            scenario=gate_manifest(20)['scenarios'][8]
            for index,(owner,name) in enumerate((('Proposed','moving'),('NC','moving_local'),
                                                 ('TOO','moving_equal'),('TFOO','moving_cvx'))):
                adapter=world(scenario)
                streams=Streams(seed,episode=scenario['scenario'],split='dev',policy_stream=1)
                policy=comparison_policy(learners[owner],SCENE,streams,name,stochastic=True)
                action,_=policy.upper(adapter.report(),adapter.arrivals)
                expected=torch.full((6,),.1*(index+1),dtype=torch.float32).tanh().numpy()
                self.assertEqual(action.tobytes(),expected.tobytes())
                actions.append(action.tobytes())
                collect_metrics(adapter,policy,identity={})
                tapes.append(arrival_bytes(adapter))
            self.assertEqual(len(set(actions)),4)
            self.assertEqual(tapes,[tapes[0]]*4)
            with self.assertRaisesRegex(ValueError,'completed checkpoint'):
                load_checkpoints(self.checkpoint_set(directory),self.binding,CONFIG,self.device,final_test=True)

    def test_proposed_training_modules_are_unchanged_and_legacy_panel_bytes_match(self):
        paths=['run.py','ppo.py','sac.py','runtime.py','schedule.py','policy.py','randomness.py',
               'collector.py','checkpoint.py','gates.py']
        output=subprocess.check_output(['git','diff','e914db2','--']+['src/hrl_v2/'+p for p in paths],cwd=ROOT)
        self.assertEqual(output,b'')
        learner=Learner(4101,self.device,CONFIG)
        old=old_module('evaluate').panel
        with tempfile.TemporaryDirectory() as directory:
            dirs=[Path(directory)/str(i) for i in range(2)]
            for fn,path in zip((old,proposed_panel),dirs):
                fn(learner,SCENE,4101,manifest('dev'),path,'same-checkpoint',stochastic=True,
                   timing=ClockConfig.from_scene(SCENE,5))
            a={p.name:p.read_bytes() for p in dirs[0].iterdir()}
            b={p.name:p.read_bytes() for p in dirs[1].iterdir()}
            self.assertEqual(a,b)

    def test_entry_is_dry_by_default_and_has_production_budget(self):
        result=subprocess.check_output([str(ROOT/'.venv/Scripts/python.exe'),'-m','hrl_v2.train_upper',
                                        '--baseline','TOO'],cwd=ROOT)
        result=json.loads(result)
        self.assertEqual(result['status'],'dry_run')
        self.assertEqual(result['physical_steps'],0)
        self.assertEqual(result['budget']['ppo_batches'],40)
        self.assertEqual(result['budget']['episodes'],200)


if __name__ == '__main__': unittest.main()
