"""Independent parent-time accounting and strict paired gate checks."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from hrl_v2.metrics import stage_residence,pool
from hrl_v2.collector import collect_metrics
from hrl_v2.tests.test_rollout_semantics import adapter,RandomScript
from hrl_v2.gates import g1,g2
from hrl_v2.evaluate import gate_manifest,manifest


class DelayGateTests(unittest.TestCase):
    def test_censoring_and_stage_partition_from_parent_timestamps(self):
        world=adapter(10)
        result=collect_metrics(world,RandomScript(),identity={})
        selected=[p for p in world.env.parents.values() if p.arrival.birth_s<10]
        weights=[10 if p.arrival.priority==0 else 1 for p in selected]
        delays=[min(10,p.finish_s if p.finish_s is not None else 10)-p.arrival.birth_s for p in selected]
        expected=sum(w*d for w,d in zip(weights,delays))/sum(weights)
        self.assertAlmostEqual(result['truncated_delay_s'],expected,places=10)
        self.assertAlmostEqual(sum(result['stage_delay_s'].values()),expected,places=10)
        self.assertGreater(result['unfinished_parents'],0)
        for row in result['parents']:
            self.assertAlmostEqual(sum(row['stage_residence_s'].values()),row['observed_residence_s'],places=10)
        pooled=pool([result,result])
        self.assertAlmostEqual(pooled['truncated_delay_s'],expected,places=10)
        for a,b in zip(result['priority'],pooled['priority']):
            expected_p95=float(np.percentile(a['completed_delays_s']*2,95)) if a['completed_delays_s'] else None
            self.assertEqual(expected_p95,b['completed_p95_s'])

    def test_parallel_fragments_use_last_branch_and_clip_window(self):
        p=SimpleNamespace(arrival=SimpleNamespace(birth_s=0.),upload_finish_s=2.,route_time_s=3.,finish_s=10.,fragment_ids=[0,1])
        e=SimpleNamespace(fragments={0:SimpleNamespace(fragment_id=0,entered_s=3.,finish_s=5.,stage='complete'),
                                    1:SimpleNamespace(fragment_id=1,entered_s=7.,finish_s=10.,stage='complete')})
        self.assertEqual(stage_residence(e,p,0,10),dict(upload=2.,ready=1.,u2u=4.,cpu=3.))
        self.assertEqual(stage_residence(e,p,4,8),dict(upload=0.,ready=0.,u2u=3.,cpu=1.))
        p.finish_s=None;e.fragments[1].finish_s=None;e.fragments[1].stage='u2u'
        self.assertEqual(stage_residence(e,p,0,6),dict(upload=2.,ready=1.,u2u=3.,cpu=0.))

    def test_gates_exact_thresholds_and_no_streams_as_scenarios(self):
        ref={str(i):dict(truncated_delay_s=10.,unfinished_parents=3) for i in range(20)}
        a=copy.deepcopy(ref)
        for i in range(12):a[str(i)]['truncated_delay_s']=9.
        baseline=dict(hover_equal=ref,hover_local=ref)
        self.assertTrue(g1(a,baseline,12)['passed'])
        a['11']['truncated_delay_s']=10.
        self.assertFalse(g1(a,baseline,12)['passed'])
        a['11']['truncated_delay_s']=9.;a['0']['unfinished_parents']=4
        self.assertFalse(g1(a,baseline,12)['passed'])
        self.assertTrue(g2(ref,ref)['weak_upper_signal'])
        a['0']['unfinished_parents']=3
        self.assertFalse(g2(a,ref)['weak_upper_signal'])
        with self.assertRaises(ValueError):g1({'0':a['0']},baseline,12)

    def test_additional_dev_manifest_never_uses_test_seeds(self):
        dev=gate_manifest(20); test=manifest('test')
        self.assertEqual(len(dev['scenarios']),20)
        self.assertEqual(dev['scenarios'][:6],manifest('dev')['scenarios'])
        self.assertTrue(set(s['layout_seed'] for s in dev['scenarios']).isdisjoint(s['layout_seed'] for s in test['scenarios']))


if __name__=='__main__': unittest.main()
