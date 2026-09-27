import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch

from hrl_v2.runtime import configure,Learner
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams,derive,initialization
from hrl_v2.replay import Episode,Replay,Sample,pack
from hrl_v2.networks import LowerActor
from hrl_v2.sac import actor_batch
from hrl_v2.policy import Policy
from hrl_v2.checkpoint import save,load,identity
from hrl_v2 import SCHEMA

ROOT=Path(__file__).absolute().parents[2]
SCENE=json.loads((ROOT/"hrl_v2/configs/physical_scene.json").read_text())


def fingerprint(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode()); visit(x[k])
        elif isinstance(x,(tuple,list)):
            for item in x: visit(item)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device=configure()
        cls.learner=Learner(4101,cls.device)
        cls.adapter=PhysicsAdapter(SCENE)
        cls.adapter.reset(arrival_seed=1,layout_seed=2,load_seed=3)
        cls.adapter.start_macro(np.zeros(6)); cls.report=cls.adapter.prepare()
        for _ in range(5): cls.adapter.step(np.tile(np.eye(3),(2,1,1)).astype(np.float64))
        cls.successor=cls.adapter.prepare()
        cls.history=np.asarray(cls.adapter.arrivals)

    def test_01_initialization_and_no_parameter_sharing(self):
        l=self.learner
        modules=(l.sac.actor,l.sac.q1,l.sac.q2,l.sac.target1,l.sac.target2,l.ppo.actor,l.ppo.value)
        sets=[{p.data_ptr() for p in m.parameters()} for m in modules]
        for i,a in enumerate(sets):
            for b in sets[i+1:]: self.assertFalse(a&b)
        with initialization(derive(4101,purpose="actor")): same=LowerActor()
        with initialization(derive(4102,purpose="actor")): different=LowerActor()
        self.assertEqual(fingerprint(same.state_dict()),fingerprint(l.sac.actor.state_dict()))
        self.assertNotEqual(fingerprint(different.state_dict()),fingerprint(l.sac.actor.state_dict()))

    def test_02_cuda_float64_distribution_gradient(self):
        l=self.learner
        states=[(self.report,self.history),(self.successor,self.history)]
        logits,kappa=actor_batch(l.sac.actor,states)
        self.assertEqual(logits.device.type,"cpu"); self.assertEqual(logits.dtype,torch.float64)
        torch.testing.assert_close(logits,torch.zeros_like(logits),atol=0,rtol=0)
        torch.testing.assert_close(kappa,torch.full_like(kappa,2.),atol=1e-7,rtol=0)
        from hrl_v2.hybrid_simplex import sample
        a,_,_=sample(logits,kappa,Streams(4101))
        l.sac.actor.zero_grad(); l.sac.q1.zero_grad()
        encoded=torch.stack([l.sac.q1.encode(r,h) for r,h in states])
        l.sac.q1.value(encoded,a).sum().backward()
        self.assertGreater(l.sac.actor.concentration[-1].weight.grad.abs().sum().item(),0.)

    def test_03_sac_update_terminal_mask_and_upper_freeze(self):
        l=self.learner
        episode=Episode("unit",0,{})
        episode.reports=[pack(self.report),pack(self.successor),pack(replace(self.successor,tick=9000,remaining_ticks=0))]
        episode.observations=[0,0]; episode.successors=[1,2]
        episode.rewards=[-.2,-.3]; episode.terminated=[False,True]
        episode.actions=[np.full((2,3,3),1/3,dtype=np.float64)]*2
        episode.history=pack(self.history)
        samples=[Sample(episode,i%2,1.) for i in range(4)]
        class SmallReplay:
            episodes=[episode]
            def sample(self,rng): return samples
        upper=fingerprint(l.ppo.actor.state_dict()); upper_opt=fingerprint(l.ppo.actor_optimizer.state_dict())
        old=copy.deepcopy(l.sac.target1.state_dict())
        original=l.sac.actor.batch
        def reject_terminal(states):
            if any(r.remaining_ticks==0 for r,_ in states): raise AssertionError("Terminal actor was evaluated")
            return original(states)
        l.sac.counts["events"]=1
        before=dict(l.sac.counts)
        with patch.object(l.sac.actor,"batch",side_effect=reject_terminal):
            row=l.sac.event(SmallReplay(),.005,True,Streams(4101))
        self.assertEqual(l.sac.counts["critic_fits"]-before["critic_fits"],2)
        self.assertEqual(l.sac.counts["actor_updates"]-before["actor_updates"],1)
        self.assertTrue(np.isfinite(row["actor"]["objective"]))
        self.assertEqual(upper,fingerprint(l.ppo.actor.state_dict())); self.assertEqual(upper_opt,fingerprint(l.ppo.actor_optimizer.state_dict()))
        for key,value in l.sac.target1.state_dict().items():
            torch.testing.assert_close(value,old[key].lerp(l.sac.q1.state_dict()[key],.005),atol=0,rtol=0)

    def test_04_complete_checkpoint_restore_and_rng(self):
        l=self.learner; streams=Streams(4101); replay=Replay(); replay.set_upper_version("unit")
        from hrl_v2.tests.test_physics import PhysicsTests
        fixture=getattr(PhysicsTests,"episode",None)
        if fixture is not None:
            replay.set_upper_version(fixture.upper_version); replay.add(fixture)
        binding=identity(ROOT,ROOT/"hrl_v2/configs/v2_1_0.json")
        before=fingerprint(l.state_dict()); stream_state=copy.deepcopy(streams.state_dict())
        artifact_root=ROOT/"hrl_v2/test_artifacts"; artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            path=Path(directory)/"checkpoint.pt"
            save(path,dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding,
                completed_episode=1,pending_evaluation=300,learner=l.state_dict(),replay=replay,
                ppo_batch=[fixture] if fixture is not None else [],streams=stream_state))
            with torch.no_grad(): next(l.sac.actor.parameters()).add_(1)
            loaded=load(path,binding); l.load_state_dict(loaded["learner"])
            self.assertEqual(before,fingerprint(l.state_dict()))
            self.assertEqual(loaded["pending_evaluation"],300)
            self.assertEqual(loaded["replay"].upper_version,replay.upper_version)
            if fixture is not None:
                restored_episode=loaded["replay"].episodes[0]
                self.assertEqual(restored_episode.reports,fixture.reports)
                self.assertEqual(restored_episode.history,fixture.history)
                np.testing.assert_array_equal(restored_episode.actions,fixture.actions)
                self.assertEqual(len(loaded["ppo_batch"]),1)
            restored=Streams(4101); restored.load_state_dict(loaded["streams"])
            torch.testing.assert_close(torch.rand(5,generator=streams.generator("support")),
                                       torch.rand(5,generator=restored.generator("support")),rtol=0,atol=0)
            with self.assertRaises(ValueError): load(path,{**binding,"schema":"old"})

    def test_05_evaluation_policy_zero_updates_and_isolated_rng(self):
        l=self.learner; before=fingerprint(l.state_dict()); training=Streams(4101)
        random_before=fingerprint(training.state_dict())
        evaluator=Policy(l.sac.actor,l.ppo.actor,None,Streams(4101,episode=1,split="test",policy_stream=1))
        global_before=torch.get_rng_state().clone()
        evaluator.upper(self.report,self.history); evaluator.lower(self.report,self.history)
        self.assertEqual(before,fingerprint(l.state_dict()))
        self.assertEqual(random_before,fingerprint(training.state_dict()))
        torch.testing.assert_close(global_before,torch.get_rng_state(),atol=0,rtol=0)

    def test_06_history_prefix_recompute_preserves_gradient(self):
        encoder=self.learner.sac.actor.encoder
        states=[(self.report,self.history),(self.successor,self.history)]
        encoder.zero_grad()
        batched=encoder.histories(states)
        loss=sum(x.square().sum() for x in batched); loss.backward()
        combined=encoder.arrival.weight_ih_l0.grad.clone()
        encoder.zero_grad()
        independent=[encoder.histories([s])[0] for s in states]
        sum(x.square().sum() for x in independent).backward()
        torch.testing.assert_close(combined,encoder.arrival.weight_ih_l0.grad,atol=2e-6,rtol=2e-5)

    def test_07_disjoint_graph_batch_matches_independent_outputs_and_gradients(self):
        from hrl_v2.batching import encode
        q=self.learner.sac.q1
        states=[(self.report,self.history),(self.successor,self.history)]
        q.zero_grad(); separate=torch.stack([q.encode(r,h) for r,h in states])
        separate.square().mean().backward()
        expected={k:p.grad.clone() for k,p in q.encoder.named_parameters() if p.grad is not None}
        q.zero_grad(); together=q.encode_batch(states)
        torch.testing.assert_close(together,separate,atol=2e-5,rtol=2e-5)
        together.square().mean().backward()
        for name,p in q.encoder.named_parameters():
            if name in expected: torch.testing.assert_close(p.grad,expected[name],atol=2e-5,rtol=2e-4)
        actor=self.learner.sac.actor
        sequential=[actor(r,h) for r,h in states]; logits,kappa=actor.batch(states)
        torch.testing.assert_close(logits,torch.stack([x[0] for x in sequential]),atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(kappa,torch.stack([x[1] for x in sequential]),atol=2e-6,rtol=2e-5)


if __name__=="__main__": unittest.main()
