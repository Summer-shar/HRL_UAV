"""Production Policy/collector equivalence on disposable bounded fixtures."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_v2.collector import collect, collect_metrics
from hrl_v2.hybrid_simplex import action_features
from hrl_v2.metrics import PooledMetrics, pool
from hrl_v2.networks import LowerActor, UpperActor, ValueNetwork
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.policy import Policy
from hrl_v2.randomness import Streams


ROOT=Path(__file__).absolute().parents[2]


class RuntimeOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(927)
            cls.lower=LowerActor()
            cls.upper=UpperActor()
            cls.value=ValueNetwork()
            # Nonconstant heads expose arrival/graph discrepancies that zero
            # initialized policy heads would hide.
            for head in (cls.lower.support,cls.lower.concentration,cls.upper.mean,cls.upper.std):
                torch.nn.init.normal_(head[-1].weight,std=.015)
        scene=json.loads((ROOT/"hrl_v2/configs/physical_scene.json").read_text())
        adapter=PhysicsAdapter(scene)
        adapter.reset(arrival_seed=7,layout_seed=8,load_seed=9)
        adapter.start_macro(np.zeros(6))
        cls.first=adapter.prepare()
        for _ in range(5): adapter.step(np.tile(np.eye(3),(2,1,1)).astype(np.float64))
        cls.second=adapter.prepare()
        cls.history=adapter.arrivals

    def compare_output(self,actual,expected):
        np.testing.assert_allclose(actual[0],expected[0],atol=2e-6,rtol=2e-5)
        if isinstance(actual[1],dict):
            self.assertEqual(actual[1].keys(),expected[1].keys())
            for key in actual[1]:
                np.testing.assert_allclose(actual[1][key],expected[1][key],atol=2e-6,rtol=2e-5)
        else: np.testing.assert_array_equal(actual[1],expected[1])

    def test_policy_really_extends_each_independent_encoder(self):
        reference=Policy(self.lower,self.upper,self.value,Streams(4101))
        optimized=Policy(self.lower,self.upper,self.value,Streams(4101))
        expected=[]
        encoders=[network.encoder for network in (self.lower,self.upper,self.value)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for encoder in encoders:
                stack.enter_context(patch.object(encoder,"infer",side_effect=lambda r,h,e,encoder=encoder: encoder(r,h)))
            for report in (self.first,self.second,self.second):
                expected.append((reference.upper(report,self.history),reference.lower(report,self.history)))
        lengths=[[] for _ in encoders]
        with ExitStack() as stack:
            for encoder,observed in zip(encoders,lengths):
                hook=encoder.arrival.register_forward_pre_hook(lambda m,args,observed=observed: observed.append(len(args[0])))
                stack.callback(hook.remove)
            for report,wanted in zip((self.first,self.second,self.second),expected):
                self.compare_output(optimized.upper(report,self.history),wanted[0])
                self.compare_output(optimized.lower(report,self.history),wanted[1])
        self.assertEqual(lengths,[[1,1],[1,1],[1,1]])
        for key,state in optimized.streams.state_dict()["generators"].items():
            torch.testing.assert_close(state,reference.streams.state_dict()["generators"][key],atol=0,rtol=0)

    def test_policy_episode_and_parameter_versions_invalidate_history(self):
        policy=Policy(self.lower,self.upper,None,Streams(4101),stochastic=False)
        encoder=self.lower.encoder
        lengths=[]
        hook=encoder.arrival.register_forward_pre_hook(lambda m,args: lengths.append(len(args[0])))
        try:
            policy.lower(self.first,self.history)
            policy.lower(self.second,self.history)
            policy.begin_episode({"episode":1})
            policy.lower(self.second,self.history)
            policy.begin_episode({"episode":1})
            policy.lower(self.second,self.history)
            saved=copy.deepcopy(encoder.arrival.state_dict())
            encoder.arrival.load_state_dict(saved)
            actual=policy.lower(self.second,self.history)
            self.assertEqual(lengths,[1,1,2,2,2])
            with patch.object(encoder,"infer",side_effect=lambda r,h,e: encoder(r,h)):
                expected=policy.lower(self.second,self.history)
            self.compare_output(actual,expected)
        finally: hook.remove()

    def test_action_cast_features_and_gradients_match_original(self):
        for device in [torch.device("cpu")]+([torch.device("cuda:0")] if torch.cuda.is_available() else []):
            raw=torch.linspace(.001,.999,2*42*6*3,dtype=torch.float64).reshape(2,42,6,3).requires_grad_()
            old=raw.to(device=device,dtype=torch.float32)
            expected=torch.cat((old.flatten(-2),(old.detach()>0).float().flatten(-2)),-1)
            expected_error=(old.detach().to("cpu",dtype=torch.float64)-raw.detach()).abs().max().item()
            actual,error=action_features(raw,device)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertEqual(error,expected_error)
            weights=torch.linspace(-1,1,actual.numel(),device=device).reshape_as(actual)
            a=torch.autograd.grad((actual*weights).sum(),raw,retain_graph=True)[0]
            b=torch.autograd.grad((expected*weights).sum(),raw)[0]
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        with self.assertRaises(FloatingPointError):
            action_features(torch.full((1,6,3),torch.finfo(torch.float64).tiny,dtype=torch.float64),"cpu")

    def test_run_policy_skips_value_only_for_lower_collection(self):
        from hrl_v2.run import collection_policy
        learner=SimpleNamespace(sac=SimpleNamespace(actor=self.lower),
                                ppo=SimpleNamespace(actor=self.upper,value=self.value))
        called=[]
        hook=self.value.register_forward_pre_hook(lambda m,args: called.append(True))
        try:
            lower=collection_policy(learner,Streams(4101),SimpleNamespace(layer="lower",stage=2))
            _,aux=lower.upper(self.first,self.history)
            self.assertEqual(called,[])
            self.assertEqual(aux["value"],0.)
            upper=collection_policy(learner,Streams(4101),SimpleNamespace(layer="upper",stage=2))
            _,aux=upper.upper(self.first,self.history)
            self.assertEqual(called,[True])
            self.assertTrue(np.isfinite(aux["value"]))
        finally: hook.remove()

    def test_streamed_collection_matches_stored_without_episode_allocation(self):
        # Exercise the complete production collector clock on an inexpensive
        # adapter; only five real physical ticks are used in setUpClass.
        class Adapter:
            def __init__(self):
                self.env=SimpleNamespace(step_index=0)
                self.arrivals=[]
                self.cost=0.
            def report(self): return SimpleNamespace(tick=self.env.step_index,severe=False)
            def prepare(self):
                self.arrivals.append(np.zeros((39,6)))
                return self.report()
            def start_macro(self,request): return {"displacement_m":np.zeros((3,2))}
            def step(self,route):
                self.env.step_index+=1
                cost=self.env.step_index/1000
                self.cost+=cost
                return {"cost":cost}
            @staticmethod
            def report_bytes(report): return 128
        class Scripted:
            def begin_episode(self,identity): self.identity=identity
            def upper(self,report,history): return np.zeros(6),{}
            def lower(self,report,history):
                return np.tile(np.eye(3),(2,1,1)),np.asarray([0,1,2,0,1,2])
        def finish(adapter):
            self.assertEqual(adapter.env.step_index,9000)
            return {"cost_sum":adapter.cost,"seconds":900.,"parents":[]}
        expected_ticks=[]; actual_ticks=[]
        with patch("hrl_v2.collector.finish",side_effect=finish):
            episode=collect(Adapter(),Scripted(),upper_version="test",lower_version=0,identity={},
                            on_tick=lambda a,r: expected_ticks.append((a.env.step_index,r["cost"])))
            with patch("hrl_v2.collector.Episode",side_effect=AssertionError("Evaluation allocated replay")):
                actual=collect_metrics(Adapter(),Scripted(),identity={},
                                       on_tick=lambda a,r: actual_ticks.append((a.env.step_index,r["cost"])))
        self.assertEqual(actual,episode.metrics)
        self.assertEqual(actual_ticks,expected_ticks)
        self.assertEqual(len(episode.rewards),1800)

    def test_streaming_pool_matches_legacy_sums_exactly(self):
        results=[dict(cost_sum=1000.3+i/11,seconds=900.,parents=[{"unused":i}],priority=[
            dict(arrived=10+i,completed=9+i,unfinished=1,delay_sum_s=3.5+i/7),
            dict(arrived=21+i,completed=18+i,unfinished=3,delay_sum_s=12.4+i/13)]) for i in range(6)]
        priorities=[]
        for p in range(2):
            row={key:sum(r["priority"][p][key] for r in results)
                 for key in ("arrived","completed","unfinished","delay_sum_s")}
            row["unfinished_rate"]=row["unfinished"]/row["arrived"]
            priorities.append(row)
        expected=dict(cost_per_s=sum(r["cost_sum"] for r in results)/sum(r["seconds"] for r in results),
            pooled_delay_ms=1000*sum(w*p["delay_sum_s"] for w,p in zip((10,1),priorities))/sum(w*p["completed"] for w,p in zip((10,1),priorities)),
            priority=priorities,episodes=6)
        accumulator=PooledMetrics()
        for result in results: accumulator.add(result)
        self.assertEqual(accumulator.result(),expected)
        self.assertEqual(pool(iter(results)),expected)
        self.assertNotIn("parents",accumulator.__dict__)


if __name__=="__main__": unittest.main()
