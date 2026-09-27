"""Bounded integration checks against the pinned V2 source and real networks.

No research run/checkpoint is loaded. The Git baseline is read-only, and every
optimizer below belongs to a temporary model constructed by the test.
"""
import copy
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_v2.networks import LowerActor, QNetwork, UpperActor, ValueNetwork
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.ppo import PPO, logprob
from hrl_v2.prepared import PreparedBatch
from hrl_v2.randomness import Streams
from hrl_v2.replay import Episode, Sample, pack
from hrl_v2.runtime import configure
from hrl_v2.sac import SAC


ROOT=Path(__file__).absolute().parents[2]


def pinned_baseline():
    """Load real 137d2f1 modules without an untracked checkout dependency."""
    name="_hrl_v2_optimizer_reference_137d2f1"
    if name in sys.modules: return sys.modules[name]
    package=types.ModuleType(name)
    package.__path__=[str(ROOT/"hrl_v2")]
    package.__package__=name
    sys.modules[name]=package
    for leaf in ("features","hybrid_simplex","replay","batching","networks","sac"):
        path=f"hrl_v2/{leaf}.py"
        reference_root=ROOT/"hrl_v2/_reference"
        manifest=json.loads((reference_root/"manifest.json").read_text(encoding="utf-8"))
        raw=(reference_root/path).read_bytes()
        if __import__("hashlib").sha256(raw).hexdigest()!=manifest["sha256"][path]:
            raise RuntimeError("Pinned local baseline source changed: "+path)
        source=raw.decode("utf-8")
        module=types.ModuleType(f"{name}.{leaf}")
        module.__package__=name
        module.__file__=f"137d2f1:{path}"
        sys.modules[module.__name__]=module
        exec(compile(source,module.__file__,"exec"),module.__dict__)
        setattr(package,leaf,module)
    return package


def maximum_tensor_difference(a,b):
    if torch.is_tensor(a): return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict): return max((maximum_tensor_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(tuple,list)): return max((maximum_tensor_difference(x,y) for x,y in zip(a,b)),default=0.)
    if a!=b: raise AssertionError((a,b))
    return 0.


class OptimizerReuseTests(unittest.TestCase):
    measurements={}
    @classmethod
    def setUpClass(cls):
        requested=os.environ.get("V2_EQ_DEVICE","cpu")
        if requested not in ("cpu","cuda:0"): raise ValueError("V2_EQ_DEVICE must be cpu or cuda:0")
        cls.device=configure(test_cpu=requested=="cpu")
        torch.set_num_threads(2)
        scene=json.loads((ROOT/"hrl_v2/configs/physical_scene.json").read_text())
        adapter=PhysicsAdapter(scene)
        adapter.reset(arrival_seed=12,layout_seed=22,load_seed=32)
        adapter.start_macro(np.zeros(6))
        first=adapter.prepare()
        for _ in range(5): adapter.step(np.tile(np.eye(3),(2,1,1)).astype(np.float64))
        second=adapter.prepare()
        history=np.asarray(adapter.arrivals)
        # Distinct prefixes share one real episode, including a zero cursor.
        cls.states=PreparedBatch([(replace(first,history_cursor=0),history),(first,history),(second,history)])
        episode=Episode("bounded",0,{})
        episode.reports=[pack(first),pack(second),pack(replace(second,tick=9000,remaining_ticks=0))]
        episode.observations=[0,1]
        episode.successors=[1,2]
        episode.rewards=[-.2,-.3]
        episode.terminated=[False,True]
        episode.actions=[np.full((2,3,3),1/3,dtype=np.float64)]*2
        episode.history=pack(history)
        cls.samples=[Sample(episode,i%2,1.+.1*(i%3)) for i in range(17)]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(912)
            cls.seed_sac=SAC(LowerActor().to(cls.device),QNetwork().to(cls.device),QNetwork().to(cls.device))
            # Exercise the actor encoder on the first fit: zero-initialized
            # output heads otherwise make its first history gradient zero.
            with torch.no_grad():
                cls.seed_sac.actor.support[-1].weight.normal_(0,.01)
                cls.seed_sac.actor.concentration[-1].weight.normal_(0,.01)
            cls.seed_ppo=PPO(UpperActor().to(cls.device),ValueNetwork().to(cls.device))
            with torch.no_grad():
                cls.seed_ppo.actor.mean[-1].weight.normal_(0,.01)
                cls.seed_ppo.actor.std[-1].weight.normal_(0,.01)

    def baseline_sac(self):
        baseline=pinned_baseline()
        reference=baseline.sac.SAC(baseline.networks.LowerActor().to(self.device),
                                  baseline.networks.QNetwork().to(self.device),baseline.networks.QNetwork().to(self.device))
        for name in ("actor","q1","q2","target1","target2"):
            getattr(reference,name).load_state_dict(getattr(self.seed_sac,name).state_dict())
        return reference

    def assert_optimizer_close(self,actual,expected,networks,optimizer):
        # The split first linear layer and sum order are float32 changes, so
        # output/gradient closeness rather than byte identity is required.
        measurements=self.measurements.setdefault(self._testMethodName,{})
        for name in networks:
            left,right=getattr(actual,name),getattr(expected,name)
            measurements[name+"_parameter_max_abs"]=maximum_tensor_difference(left.state_dict(),right.state_dict())
            measurements[name+"_gradient_max_abs"]=max((float((a.grad-b.grad).abs().max())
                for a,b in zip(left.parameters(),right.parameters()) if a.grad is not None),default=0.)
            largest_name=max(left.state_dict(),key=lambda key:float((left.state_dict()[key]-right.state_dict()[key]).abs().max()))
            a=dict(left.named_parameters())[largest_name]; b=dict(right.named_parameters())[largest_name]
            position=(a.detach()-b.detach()).abs().flatten().argmax()
            measurements[name+"_largest_parameter_difference"]=dict(name=largest_name,flat_index=int(position),
                actual=float(a.detach().flatten()[position]),reference=float(b.detach().flatten()[position]),
                actual_clipped_gradient=None if a.grad is None else float(a.grad.flatten()[position]),
                reference_clipped_gradient=None if b.grad is None else float(b.grad.flatten()[position]))
            for key,value in left.state_dict().items():
                torch.testing.assert_close(value,right.state_dict()[key],atol=3e-5,rtol=2e-5,msg=lambda s:f"{name}.{key}: {s}")
            for (key,a),(_,b) in zip(left.named_parameters(),right.named_parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-3,msg=lambda s:f"gradient {name}.{key}: {s}")
        left=getattr(actual,optimizer).state_dict(); right=getattr(expected,optimizer).state_dict()
        measurements[optimizer+"_state_max_abs"]=maximum_tensor_difference(left,right)
        self.assertEqual(left["param_groups"],right["param_groups"])
        for key,state in left["state"].items():
            for field,value in state.items():
                torch.testing.assert_close(value,right["state"][key][field],atol=2e-7,rtol=2e-3)

    def test_critic_matches_pinned_baseline_and_backpropagates_history_once(self):
        reference=self.baseline_sac(); actual=copy.deepcopy(self.seed_sac)
        expected=reference._critic_fit(self.samples,.005,Streams(4101))
        forward={name:0 for name in ("actor","q1","q2","target1","target2")}
        backward={name:0 for name in ("q1","q2")}
        hooks=[]
        def count(table,name):
            def increment(*args): table[name]+=1
            return increment
        for name in forward:
            hooks.append(getattr(actual,name).encoder.arrival.register_forward_hook(count(forward,name)))
        for name in backward:
            hooks.append(getattr(actual,name).encoder.arrival.register_full_backward_hook(count(backward,name)))
        try: result=actual._critic_fit(self.samples,.005,Streams(4101))
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(forward,dict.fromkeys(forward,1))
        self.assertEqual(backward,dict.fromkeys(backward,1))
        for key in result: np.testing.assert_allclose(result[key],expected[key],rtol=3e-5,atol=3e-6)
        self.assert_optimizer_close(actual,reference,("q1","q2"),"critic_optimizer")
        self.assertEqual(actual.counts,reference.counts)

    def test_actor_matches_pinned_baseline_and_keeps_independent_q_encoders(self):
        reference=self.baseline_sac(); actual=copy.deepcopy(self.seed_sac)
        before=[copy.deepcopy(q.state_dict()) for q in (actual.q1,actual.q2)]
        reference_streams=Streams(4101); actual_streams=Streams(4101)
        expected=reference._actor_fit(self.samples,.005,reference_streams)
        result=actual._actor_fit(self.samples,.005,actual_streams)
        for key in result: np.testing.assert_allclose(result[key],expected[key],rtol=3e-5,atol=3e-6)
        self.assert_optimizer_close(actual,reference,("actor",),"actor_optimizer")
        self.assertEqual(maximum_tensor_difference(actual_streams.state_dict(),reference_streams.state_dict()),0.)
        for q,old in zip((actual.q1,actual.q2),before): self.assertEqual(maximum_tensor_difference(q.state_dict(),old),0.)
        self.assertEqual(actual.counts,reference.counts)

    def test_real_ppo_minibatch_matches_unshared_full_history(self):
        reference=copy.deepcopy(self.seed_ppo); actual=copy.deepcopy(self.seed_ppo)
        reference.microbatch_size=actual.microbatch_size=2
        with torch.no_grad():
            mean,std=actual.actor.batch(self.states)
            z=mean+mean.new_tensor(np.linspace(-.02,.02,18).reshape(3,6))
            old=logprob(z,mean,std)
        rows=[(None,dict(pre_tanh=z[i].cpu().numpy(),logprob=old[i].item()),-.3-.1*i) for i in range(3)]
        advantages=np.asarray([-.4,.1,.3])
        with patch("hrl_v2.ppo.network_history_scope",side_effect=lambda *args:nullcontext()):
            reference._minibatch(self.states,rows,advantages,True)
        actual._minibatch(self.states,rows,advantages,True)
        self.assert_optimizer_close(actual,reference,("actor",),"actor_optimizer")
        self.assert_optimizer_close(actual,reference,("value",),"value_optimizer")
        self.assertEqual(actual.counts,reference.counts)
        self.assertIsNone(getattr(actual.actor.encoder,"_history_tape",None))
        self.assertIsNone(getattr(actual.value.encoder,"_history_tape",None))

    def test_graph_preprocessing_matches_pinned_outputs_and_all_gradients(self):
        baseline=pinned_baseline()
        reference=baseline.networks.QNetwork().to(self.device); actual=copy.deepcopy(self.seed_sac.q1)
        reference.load_state_dict(actual.state_dict())
        states=self.states[1:]
        expected=reference.encode_batch(states)
        expected.square().mean().backward()
        output=actual.encode_batch(states)
        output.square().mean().backward()
        torch.testing.assert_close(output,expected,atol=2e-5,rtol=2e-5)
        largest=0.
        for (name,a),(_,b) in zip(actual.named_parameters(),reference.named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:
                largest=max(largest,float((a.grad-b.grad).abs().max()))
                torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-4,msg=lambda s:f"{name}: {s}")
        self.measurements[self._testMethodName]=dict(output_max_abs=float((output-expected).abs().max()),gradient_max_abs=largest)


if __name__=="__main__": unittest.main()
