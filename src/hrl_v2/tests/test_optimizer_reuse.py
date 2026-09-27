"""Bounded integration checks against the pinned V2 source and real networks.

No research run/checkpoint is loaded. The Git baseline is read-only, and every
optimizer below belongs to a temporary model constructed by the test.
"""
import copy
from contextlib import ExitStack, nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import hrl_v2.sac as sac_module
from hrl_v2.hybrid_simplex import SUPPORTS
from hrl_v2.networks import LowerActor, QNetwork, UpperActor, ValueNetwork
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.ppo import PPO, logprob
from hrl_v2.prepared import PreparedBatch, history_scope
from hrl_v2.randomness import Streams
from hrl_v2.replay import Episode, Sample, pack
from hrl_v2.runtime import configure
from hrl_v2.sac import SAC


ROOT=Path(__file__).absolute().parents[2]
CORRECTNESS_COMMIT="4fa6e650fe57fbdbfba4d7962ed52776ed9bfdf0"
CORRECTNESS_SAC_SHA256="f77c7567d82e46d764ca092da4ef0e178aa25d9af5e6a30b8b4c3d3901c7122b"


def correctness_sac_baseline():
    """Freeze this change's original SAC control flow, without Git/output dependencies.

    The file is the exact committed source, not a second test implementation.
    Its imports intentionally use the current unchanged network/prepared modules
    so the comparison isolates moving targets ahead of current-Q computation.
    """
    name="hrl_v2._sac_reference_4fa6e65"
    if name in sys.modules: return sys.modules[name]
    path=Path(__file__).with_name("_sac_reference_4fa6e65.py")
    raw=path.read_bytes()
    if __import__("hashlib").sha256(raw).hexdigest()!=CORRECTNESS_SAC_SHA256:
        raise RuntimeError("Correctness-baseline SAC source changed")
    module=types.ModuleType(name)
    module.__package__="hrl_v2"
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(raw.decode("utf-8"),str(path),"exec"),module.__dict__)
    return module


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
        if leaf=="features":
            # Compare optimizer execution on the SAME current causal inputs.
            # The archived report clock intentionally has different semantics;
            # it is not a source of the current environment specification.
            from hrl_v2.features import build
            module.build=build
        setattr(package,leaf,module)
    return package


def maximum_tensor_difference(a,b):
    if torch.is_tensor(a): return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict): return max((maximum_tensor_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(tuple,list)): return max((maximum_tensor_difference(x,y) for x,y in zip(a,b)),default=0.)
    if a!=b: raise AssertionError((a,b))
    return 0.


def tensor_error_summary(actual,reference):
    actual=actual.detach().cpu().double(); reference=reference.detach().cpu().double()
    difference=actual-reference
    reference_norm=float(reference.norm())
    difference_norm=float(difference.norm())
    return dict(max_abs=float(difference.abs().max()) if difference.numel() else 0.,
                difference_l2=difference_norm,reference_l2=reference_norm,
                actual_l2=float(actual.norm()),relative_l2=difference_norm/max(reference_norm,1e-30))


def gradient_error_summary(actual,reference):
    """Normwise diagnostics keep cancellation near zero out of the denominator."""
    squares=dict(difference_l2=0.,reference_l2=0.,actual_l2=0.)
    maximum=0.; presence=[]
    for name,gradient in actual.items():
        other=reference[name]
        if (gradient is None)!=(other is None): presence.append(name)
        if gradient is None or other is None: continue
        error=tensor_error_summary(gradient,other)
        maximum=max(maximum,error["max_abs"])
        for key in squares: squares[key]+=error[key]**2
    norms={key:math.sqrt(value) for key,value in squares.items()}
    return dict(norms,max_abs=maximum,presence_mismatches=presence,
                relative_l2=norms["difference_l2"]/max(norms["reference_l2"],1e-30))


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
        episode.reports=[pack(first),pack(second),pack(replace(second,tick=10))]
        episode.observations=[0,1]
        episode.successors=[1,2]
        episode.rewards=[-.2,-.3]
        episode.terminated=[False,True]
        episode.truncated=[False,False]
        episode.actions=[np.full((2,3,3),1/3,dtype=np.float64)]*2
        episode.history=pack(history)
        cls.samples=[Sample(episode,i%2,1.+.1*(i%3)) for i in range(17)]
        # Reuse the same physical reports: this batch adds no simulation steps.
        # Distinct episode histories, zero/repeated cursors, terminal holes and
        # a truncated successor exercise both packing and the sampling order.
        terminal=copy.deepcopy(episode)
        terminal.reports[0]=pack(replace(first,history_cursor=0))
        terminal.reports[2]=pack(replace(second,tick=1000000))
        terminal.actions[0]=np.tile(np.eye(3),(2,1,1)).astype(np.float64)
        continuing=copy.deepcopy(episode)
        continuing.terminated=[False,False]
        continuing.truncated=[False,True]
        continuing.rewards=[-.4,-.1]
        continuing.actions[1]=np.tile(np.array([.2,.8,0.]),(2,3,1))
        other_history=history.copy(); other_history[:,0,0]*=1.25
        continuing.history=pack(other_history)
        cls.critic_samples=[]
        for i in range(128):
            e,index=(terminal,1) if i<16 else ((terminal,continuing)[(i//2)%2],i%2)
            cls.critic_samples.append(Sample(e,index,.8+.1*(i%5)))
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

    def optimizer_differences(self,actual,expected,networks,optimizer):
        """Collect every network/Adam diagnostic before any tolerance assertion."""
        measurements={}
        for name in networks:
            left,right=getattr(actual,name),getattr(expected,name)
            measurements[name+"_parameter_max_abs"]=maximum_tensor_difference(left.state_dict(),right.state_dict())
            gradients=gradient_error_summary({key:p.grad for key,p in left.named_parameters()},
                                             {key:p.grad for key,p in right.named_parameters()})
            measurements[name+"_gradient_max_abs"]=gradients["max_abs"]
            measurements[name+"_clipped_gradient_error"]=gradients
            largest_name=max(left.state_dict(),key=lambda key:float((left.state_dict()[key]-right.state_dict()[key]).abs().max()))
            a=dict(left.named_parameters())[largest_name]; b=dict(right.named_parameters())[largest_name]
            position=(a.detach()-b.detach()).abs().flatten().argmax()
            largest=dict(name=largest_name,flat_index=int(position),
                actual=float(a.detach().flatten()[position]),reference=float(b.detach().flatten()[position]),
                actual_clipped_gradient=None if a.grad is None else float(a.grad.flatten()[position]),
                reference_clipped_gradient=None if b.grad is None else float(b.grad.flatten()[position]))
            left_optimizer=getattr(actual,optimizer); right_optimizer=getattr(expected,optimizer)
            if (a.grad is not None and b.grad is not None and
                    int(left_optimizer.state.get(a,{}).get("step",0))==1 and
                    int(right_optimizer.state.get(b,{}).get("step",0))==1):
                group=next(group for group in left_optimizer.param_groups if any(p is a for p in group["params"]))
                ga,gb=largest["actual_clipped_gradient"],largest["reference_clipped_gradient"]
                lr,eps=group["lr"],group["eps"]
                predicted=abs(lr*(ga/(abs(ga)+eps)-gb/(abs(gb)+eps)))
                observed=abs(largest["actual"]-largest["reference"])
                largest["adam_first_step_sensitivity"]=dict(lr=lr,eps=eps,
                    max_gradient_over_eps=max(abs(ga),abs(gb))/eps,
                    predicted_parameter_difference=predicted,observed_parameter_difference=observed,
                    prediction_residual=abs(predicted-observed),difference_over_lr=observed/lr,
                    explanation="For the shared initial parameter and empty Adam slots, update is lr*g/(abs(g)+eps). Diagnostic only; does not waive tolerance.")
            measurements[name+"_largest_parameter_difference"]=largest
        left=getattr(actual,optimizer).state_dict(); right=getattr(expected,optimizer).state_dict()
        measurements[optimizer+"_state_max_abs"]=maximum_tensor_difference(left,right)
        return measurements

    def assert_optimizer_close(self,actual,expected,networks,optimizer):
        # The split first linear layer and sum order are float32 changes, so
        # output/gradient closeness rather than byte identity is required.
        measurements=self.measurements.setdefault(self._testMethodName,{})
        measurements.update(self.optimizer_differences(actual,expected,networks,optimizer))
        for name in networks:
            left,right=getattr(actual,name),getattr(expected,name)
            for key,value in left.state_dict().items():
                torch.testing.assert_close(value,right.state_dict()[key],atol=3e-5,rtol=2e-5,msg=lambda s:f"{name}.{key}: {s}")
            for (key,a),(_,b) in zip(left.named_parameters(),right.named_parameters()):
                self.assertEqual(a.grad is None,b.grad is None,f"gradient presence {name}.{key}")
                if a.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-3,msg=lambda s:f"gradient {name}.{key}: {s}")
        left=getattr(actual,optimizer).state_dict(); right=getattr(expected,optimizer).state_dict()
        self.assertEqual(left["param_groups"],right["param_groups"])
        self.assertEqual(left["state"].keys(),right["state"].keys())
        for key,state in left["state"].items():
            for field,value in state.items():
                torch.testing.assert_close(value,right["state"][key][field],atol=2e-7,rtol=2e-3)

    def test_critic_matches_pinned_baseline_and_backpropagates_history_once(self):
        reference=self.baseline_sac(); actual=copy.deepcopy(self.seed_sac)
        # The pinned baseline predates the explicit discount (implicit gamma=1).
        # This test checks computational equivalence, so match its discount.
        actual.gamma=1.
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

    def trace_critic_fit(self,sac,samples):
        """Observe production calls without replacing sampling or gradients."""
        trace=dict(samples=[],successors=[],targets=None,losses=[],gradients={},
                   encoded={name:[] for name in ("q1","q2")},
                   forward=dict.fromkeys(("actor","q1","q2","target1","target2"),0),
                   backward=dict.fromkeys(("q1","q2"),0))
        module=sys.modules[type(sac).__module__]
        original_sample=module.sample
        original_encode=module.encode_batch
        original_tensor=torch.tensor
        target_functions=[sac._critic_fit]
        if hasattr(sac,"_critic_targets"): target_functions.append(sac._critic_targets)
        target_codes={getattr(function,"__wrapped__",function).__code__ for function in target_functions}
        target_tensors=[]
        original_actor=sac.actor.batch
        original_backward=torch.Tensor.backward
        original_clip=torch.nn.utils.clip_grad_norm_
        parameters={f"{name}.{key}":p for name in ("q1","q2")
                    for key,p in getattr(sac,name).named_parameters()}
        def cpu(value): return value.detach().cpu().clone()
        def sampled(logits,kappa,streams,stream=0):
            output=original_sample(logits,kappa,streams,stream)
            trace["samples"].append(dict(logits=cpu(logits),kappa=cpu(kappa),
                                         output=tuple(cpu(x) for x in output),stream=stream))
            return output
        def encoded(network,states):
            output=original_encode(network,states)
            for name in trace["encoded"]:
                if network is getattr(sac,name):
                    trace["encoded"][name].append((states.indices,cpu(output)))
            return output
        def tensor(*args,**kwargs):
            # Observe the actual reward tensor and its later in-place bootstrap.
            # Caller code filters out all graph/distribution tensors. No target
            # equation or baseline implementation is duplicated in this test.
            caller=sys._getframe(1).f_code
            output=original_tensor(*args,**kwargs)
            if caller in target_codes and kwargs.get("dtype") is torch.float64:
                target_tensors.append(output)
            return output
        def actor(states):
            self.assertFalse(any(r.tick==1000000 for r,_ in states),"A terminated successor reached the actor")
            trace["successors"].append(states.indices)
            return original_actor(states)
        def backward(value,*args,**kwargs):
            self.assertEqual(value.ndim,0)
            trace["losses"].append(cpu(value))
            return original_backward(value,*args,**kwargs)
        def clipped(items,*args,**kwargs):
            items=list(items)
            self.assertEqual([id(p) for p in items],[id(p) for p in parameters.values()])
            self.assertEqual(args,(1.,)); self.assertEqual(kwargs,dict(error_if_nonfinite=True))
            trace["gradients"]={name:None if p.grad is None else cpu(p.grad) for name,p in parameters.items()}
            norm=original_clip(items,*args,**kwargs)
            trace["joint_unclipped_norm"]=float(norm)
            return norm
        def count(table,name):
            def increment(*args): table[name]+=1
            return increment
        streams=Streams(4101)
        with ExitStack() as stack:
            for name in trace["forward"]:
                hook=getattr(sac,name).encoder.arrival.register_forward_hook(count(trace["forward"],name))
                stack.callback(hook.remove)
            for name in trace["backward"]:
                hook=getattr(sac,name).encoder.arrival.register_full_backward_hook(count(trace["backward"],name))
                stack.callback(hook.remove)
            stack.enter_context(patch.object(module,"sample",side_effect=sampled))
            stack.enter_context(patch.object(module,"encode_batch",side_effect=encoded))
            stack.enter_context(patch.object(torch,"tensor",new=tensor))
            stack.enter_context(patch.object(sac.actor,"batch",side_effect=actor))
            stack.enter_context(patch.object(torch.Tensor,"backward",new=backward))
            clip=stack.enter_context(patch("torch.nn.utils.clip_grad_norm_",side_effect=clipped))
            step=stack.enter_context(patch.object(sac.critic_optimizer,"step",wraps=sac.critic_optimizer.step))
            trace["result"]=sac._critic_fit(samples,.005,streams)
            self.assertEqual(clip.call_count,1); self.assertEqual(step.call_count,1)
        trace["streams"]=streams.state_dict()
        trace["loss"]=torch.stack(trace["losses"]).sum()
        self.assertEqual(sum(value.numel() for value in target_tensors),len(samples))
        trace["targets"]=torch.cat([cpu(value) for value in target_tensors])
        return trace

    @torch.no_grad()
    def post_update_q(self,sac,samples):
        """Compare updated functions using identical 16-state evaluation chunks."""
        states,_=PreparedBatch.replay(samples)
        values={name:[] for name in ("q1","q2")}
        device=next(sac.q1.parameters()).device
        with ExitStack() as scopes:
            for name in values: scopes.enter_context(history_scope(getattr(sac,name).encoder,states))
            for start in range(0,len(samples),16):
                selected=samples[start:start+16]
                current=states[start:start+len(selected)]
                actions=torch.as_tensor(np.asarray([s.episode.actions[s.index] for s in selected]).reshape(-1,6,3))
                features=sac_module.shared_action_features(actions,(sac.q1,sac.q2),device)
                for name in values:
                    q=getattr(sac,name)
                    values[name].append(q.value_features(q.encode_batch(current),features).cpu())
        return {name:torch.cat(rows) for name,rows in values.items()}

    def check_critic_chunks_against_correctness_baseline(self,chunks):
        samples=self.critic_samples
        baseline=correctness_sac_baseline()
        reference=baseline.SAC(*(copy.deepcopy(getattr(self.seed_sac,name)) for name in ("actor","q1","q2")),
                               gamma=self.seed_sac.gamma)
        expected=self.trace_critic_fit(reference,samples)
        expected_values=self.post_update_q(reference,samples)
        self.assertEqual(reference.counts,dict(triggers=0,skipped=0,events=0,critic_fits=1,
                                              actor_updates=0,target_updates=0))
        expected_slots=[]
        for start in range(0,len(samples),16):
            slots=tuple(len(samples)+i for i in range(start,min(start+16,len(samples)))
                        if not samples[i].episode.terminated[samples[i].index])
            if slots: expected_slots.append(slots)
        self.assertEqual(expected["successors"],expected_slots)
        self.assertEqual(expected["forward"],dict.fromkeys(expected["forward"],2))
        self.assertEqual(expected["backward"],dict.fromkeys(expected["backward"],2))
        self.assertEqual(len(expected["losses"]),8)
        for row in expected["samples"]:
            actions,ids,_=row["output"]
            allowed=torch.zeros_like(actions,dtype=torch.bool)
            for index,support in enumerate(SUPPORTS):
                mask=torch.tensor([j in support for j in range(3)],dtype=torch.bool)
                allowed|=(ids==index)[...,None]&mask
            self.assertTrue(torch.equal(actions>0,allowed))
            torch.testing.assert_close(actions.sum(-1),torch.ones_like(actions[...,0]),atol=1e-12,rtol=0)
            self.assertEqual(row["stream"],200)
        terminal=torch.tensor([s.episode.terminated[s.index] for s in samples])
        rewards=torch.tensor([s.episode.rewards[s.index] for s in samples],dtype=torch.float64)
        torch.testing.assert_close(expected["targets"][terminal],rewards[terminal],atol=0,rtol=0)
        truncated=torch.tensor([s.episode.truncated[s.index] for s in samples])
        self.assertTrue(torch.all(expected["targets"][truncated]!=rewards[truncated]))
        measurements=self.measurements.setdefault(self._testMethodName,{})
        measurements["reference"]=dict(commit=CORRECTNESS_COMMIT,sha256=CORRECTNESS_SAC_SHA256,
                                        effective_batch=128,gamma=reference.gamma)
        # Express the raw-gradient absolute bound in the same units as the
        # unchanged post-clip bound. A separate normwise gate detects scaling
        # drift; this does not relax the parameter or Adam acceptance limits.
        raw_atol=2e-6*max(1.,expected["joint_unclipped_norm"])
        for chunk in chunks:
            with self.subTest(critic_microbatch_size=chunk):
                actual=copy.deepcopy(self.seed_sac); actual.critic_microbatch_size=chunk
                result=self.trace_critic_fit(actual,samples)
                updated_values=self.post_update_q(actual,samples)
                # Commit every diagnostic before any candidate tolerance gate.
                # The original limits below remain unchanged during diagnosis.
                errors=dict(target_max_abs=maximum_tensor_difference(result["targets"],expected["targets"]),
                            sample_max_abs=maximum_tensor_difference(result["samples"],expected["samples"]),
                            rng_max_abs=maximum_tensor_difference(result["streams"],expected["streams"]),
                            loss_abs=float(abs(result["loss"]-expected["loss"])),
                            unclipped_gradient_error=gradient_error_summary(result["gradients"],expected["gradients"]),
                            joint_unclipped_norm_actual=result["joint_unclipped_norm"],
                            joint_unclipped_norm_reference=expected["joint_unclipped_norm"],
                            post_update_q={name:tensor_error_summary(updated_values[name],expected_values[name]) for name in updated_values},
                            post_update_q_note="Diagnostic only; both updated networks use evaluation chunk 16 with no stochastic sampling.",
                            encoding_max_abs=0.,counts=dict(actual.counts),
                            history_forward=dict(result["forward"]),history_backward=dict(result["backward"]),
                            tolerances=dict(loss=dict(atol=3e-6,rtol=3e-5),
                                            unclipped_gradient=dict(atol=raw_atol,rtol=2e-3,joint_relative_l2=1e-5,
                                                basis="2e-6 post-clip absolute tolerance times max(1, reference joint unclipped norm)"),
                                            clipped_gradient=dict(atol=2e-6,rtol=2e-3),
                                            parameter=dict(atol=3e-5,rtol=2e-5),
                                            adam=dict(atol=2e-7,rtol=2e-3)))
                errors["unclipped_gradient_max_abs"]=errors["unclipped_gradient_error"]["max_abs"]
                for name in ("q1","q2"):
                    errors[name+"_unclipped_gradient_error"]=gradient_error_summary(
                        {key:value for key,value in result["gradients"].items() if key.startswith(name+".")},
                        {key:value for key,value in expected["gradients"].items() if key.startswith(name+".")})
                    value=torch.cat([value for _,value in result["encoded"][name]])
                    other=torch.cat([value for _,value in expected["encoded"][name]])
                    errors["encoding_max_abs"]=max(errors["encoding_max_abs"],maximum_tensor_difference(value,other))
                errors.update(self.optimizer_differences(actual,reference,("q1","q2"),"critic_optimizer"))
                measurements[f"chunk_{chunk}"]=errors
                self.assertEqual(result["successors"],expected_slots)
                self.assertEqual(result["forward"],expected["forward"])
                self.assertEqual(result["backward"],expected["backward"])
                self.assertEqual(len(result["losses"]),128//chunk)
                self.assertEqual(len(result["samples"]),len(expected["samples"]))
                self.assertEqual(maximum_tensor_difference(result["samples"],expected["samples"]),0.)
                self.assertEqual(maximum_tensor_difference(result["streams"],expected["streams"]),0.)
                torch.testing.assert_close(result["targets"],expected["targets"],atol=0,rtol=0)
                torch.testing.assert_close(result["loss"],expected["loss"],atol=3e-6,rtol=3e-5)
                self.assertLessEqual(errors["unclipped_gradient_error"]["relative_l2"],1e-5)
                for name,gradient in result["gradients"].items():
                    other=expected["gradients"][name]
                    self.assertEqual(gradient is None,other is None,name)
                    if gradient is not None:
                        torch.testing.assert_close(gradient,other,atol=raw_atol,rtol=2e-3,msg=lambda s:f"{name}: {s}")
                for name in ("q1","q2"):
                    self.assertEqual(tuple(i for ids,_ in result["encoded"][name] for i in ids),tuple(range(128)))
                    value=torch.cat([value for _,value in result["encoded"][name]])
                    other=torch.cat([value for _,value in expected["encoded"][name]])
                    torch.testing.assert_close(value,other,atol=2e-5,rtol=2e-5)
                for key in result["result"]:
                    np.testing.assert_allclose(result["result"][key],expected["result"][key],rtol=3e-5,atol=3e-6)
                self.assert_optimizer_close(actual,reference,("q1","q2"),"critic_optimizer")
                self.assertEqual(actual.counts,reference.counts)
                self.assertEqual(actual.counts["critic_fits"],1)
                self.assertEqual(actual.version,self.seed_sac.version)
                self.assertEqual(actual.microbatch_size,16)
                self.assertEqual(actual.actor_optimizer.state_dict(),self.seed_sac.actor_optimizer.state_dict())
                for name in ("actor","target1","target2"):
                    self.assertEqual(maximum_tensor_difference(getattr(actual,name).state_dict(),getattr(self.seed_sac,name).state_dict()),0.)
                modules=[getattr(actual,name) for name in ("actor","q1","q2","target1","target2")]
                storages=[{p.data_ptr() for p in module.parameters()} for module in modules]
                for i,storage in enumerate(storages):
                    for other in storages[i+1:]: self.assertFalse(storage&other)
                for module in modules: self.assertIsNone(getattr(module.encoder,"_history_tape",None))

    def test_critic_default_chunk_matches_correctness_baseline(self):
        self.assertEqual(self.seed_sac.critic_microbatch_size,16)
        self.check_critic_chunks_against_correctness_baseline((16,))

    @unittest.skipUnless(os.environ.get("V2_AUDIT_CANDIDATE_CHUNKS")=="1",
        "64/128 were rejected by the unchanged parameter tolerance; set V2_AUDIT_CANDIDATE_CHUNKS=1 to audit them explicitly")
    def test_critic_candidate_chunks_require_explicit_numerical_audit(self):
        # These are rejected candidates, not part of the supported default.
        # Opting into their audit retains the real failing numerical gates.
        self.check_critic_chunks_against_correctness_baseline((64,128))

    @torch.no_grad()
    def test_graph_batch_matches_single_graphs_without_cross_graph_messages(self):
        q=copy.deepcopy(self.seed_sac.q1)
        samples=[self.critic_samples[i] for i in (0,16,18,19)]
        states,_=PreparedBatch.replay(samples)
        order=[3,1,0,2]
        with history_scope(q.encoder,states):
            batched=q.encode_batch(states)
            single=torch.cat([q.encode_batch(states[i:i+1]) for i in range(len(states))])
            permuted=q.encode_batch(states.select(order))
        measurements=dict(single_vs_batch=tensor_error_summary(batched,single),
                          permutation=tensor_error_summary(permuted,batched[order]))
        self.measurements[self._testMethodName]=measurements
        torch.testing.assert_close(batched,single,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(permuted,batched[order],atol=2e-5,rtol=2e-5)

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
