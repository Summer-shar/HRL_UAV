"""V2 data-cache and full-history VJP checks on bounded, disposable models."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from hrl_v2.batching import prepare
from hrl_v2.features import arrival_array
from hrl_v2.hybrid_simplex import action_features
from hrl_v2.networks import Encoder, QNetwork
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.prepared import PreparedBatch, history_scope
from hrl_v2.replay import Episode, Sample, pack
from hrl_v2.runtime import configure


class PreparedOptimizationTests(unittest.TestCase):
    def setUp(self):
        # Random ReLU fixtures must not depend on other tests consuming RNG.
        # Near a kink, two float32 algebra orders can have matching outputs
        # but different derivatives. Keep this regression reproducible without
        # changing production precision or relaxing its gradient tolerances.
        scope=torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
        scope.__enter__()
        self.addCleanup(scope.__exit__,None,None,None)
        torch.manual_seed(927)

    @classmethod
    def setUpClass(cls):
        cls.device=configure()
        scene=json.loads((Path(__file__).parents[1]/"configs/physical_scene.json").read_text())
        adapter=PhysicsAdapter(scene)
        adapter.reset(arrival_seed=61,layout_seed=62,load_seed=63)
        adapter.start_macro(np.zeros(6))
        cls.report=adapter.prepare()
        route=np.tile(np.eye(3),(2,1,1)).astype(np.float64)
        for _ in range(5): adapter.step(route)
        cls.successor=adapter.prepare()
        cls.history=np.asarray(adapter.arrivals)

    def test_replay_decompresses_each_episode_and_report_once(self):
        from hrl_v2 import replay, prepared
        episode=Episode("fixture",0,{})
        episode.reports=[pack(self.report),pack(self.successor)]
        episode.observations=[0,1]; episode.successors=[1,1]
        episode.history=pack(self.history)
        samples=[Sample(episode,i%2,1.) for i in range(16)]
        with patch.object(replay,"unpack",wraps=replay.unpack) as unpack:
            current,following=PreparedBatch.replay(samples)
            self.assertEqual(unpack.call_count,3)  # One history and two reports.
        self.assertIs(current[1][0],following[0][0])
        self.assertIs(current[0][1],following[0][1])
        prepared.fixed_graph.cache_clear()
        with patch.object(prepared,"build",wraps=prepared.build) as build:
            a=prepare(current[:4],self.device)
            self.assertIs(a,prepare(current[:4],self.device))
            prepare(following[:4],self.device)
            self.assertEqual(build.call_count,2)
            # Separate sampled batches retain the baseline's fixed-data LRU.
            repeated=PreparedBatch([(copy.deepcopy(self.report),self.history)])
            prepare(repeated,self.device)
            self.assertEqual(build.call_count,2)
        self.assertEqual(prepared.fixed_graph.cache_info().maxsize,256)
        self.assertFalse(any(t.requires_grad for t in a.features.values()))

    def test_fixed_history_inputs_extend_by_suffix(self):
        from hrl_v2 import prepared
        batch=PreparedBatch([(self.report,self.history)])
        with patch.object(prepared,"arrival_array",wraps=arrival_array) as convert:
            first=batch.arrivals(self.history,1,self.device)
            full=batch.arrivals(self.history,len(self.history),self.device)
            self.assertEqual(full.data_ptr(),batch.arrivals(self.history,len(self.history),self.device).data_ptr())
            self.assertEqual([c.args[1] for c in convert.call_args_list],[1,len(self.history)-1])
        torch.testing.assert_close(first,full[:1],atol=0,rtol=0)
        torch.testing.assert_close(full,arrival_array(self.history,len(self.history),self.device),atol=0,rtol=0)

    def test_batch_cache_released_without_waiting_for_cyclic_gc(self):
        batch=PreparedBatch([(self.report,self.history)])
        view=batch[:1]
        root=weakref.ref(batch)
        cached=prepare(view,self.device)
        tensor=weakref.ref(cached.features["terminal"])
        del batch,cached
        self.assertIsNotNone(root())
        del view
        self.assertIsNone(root())
        self.assertIsNone(tensor())

    def test_long_incremental_history_and_optimizer_invalidation(self):
        encoder=Encoder().to(self.device)
        rng=np.random.default_rng(20)
        history=rng.random((1800,39,6))*[1e6,1e9,1,1e6,1e9,1]
        episode=object()
        cache=encoder._incremental_history
        with torch.no_grad():
            for cursor in (0,1,20,37,900,1800,1800,900):
                report=replace(self.report,history_cursor=cursor)
                actual=cache.get(encoder,report,history,episode)
                expected=encoder.histories([(report,history)])[0]
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        optimizer=torch.optim.SGD(encoder.arrival.parameters(),lr=.001)
        for p in encoder.arrival.parameters(): p.grad=torch.ones_like(p)
        optimizer.step()
        with torch.no_grad(),patch.object(encoder.arrival,"forward",wraps=encoder.arrival.forward) as forward:
            actual=cache.get(encoder,report,history,episode)
            self.assertEqual(forward.call_args.args[0].shape[0],900)
            expected=encoder.histories([(report,history)])[0]
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

    def test_step_tape_full_bptt_matches_repeated_prefix_gradients(self):
        encoder=Encoder().to(self.device)
        # Two independent episode histories, repeated cursors, and a zero cursor.
        rng=np.random.default_rng(19)
        histories=[rng.random((12,39,6))*[1e6,1e9,1,1e6,1e9,1] for _ in range(2)]
        states=PreparedBatch([(replace(self.report,history_cursor=c),histories[e])
                              for e,c in ((0,0),(0,3),(1,7),(0,12),(1,3),(0,3))])
        expected=[]
        for index in range(len(states)):
            hidden=encoder.histories(states[index:index+1])[0]
            expected.append(hidden.detach().clone())
            if hidden.requires_grad: ((index+1)*hidden.square().mean()).backward()
        gradients={n:p.grad.clone() for n,p in encoder.arrival.named_parameters()}
        encoder.zero_grad(set_to_none=True)
        with patch.object(encoder.arrival,"forward",wraps=encoder.arrival.forward) as forward:
            with history_scope(encoder,states) as tape:
                self.assertEqual(forward.call_count,2)
                self.assertEqual(sorted(c.args[0].shape[0] for c in forward.call_args_list),[7,12])
                for index in range(len(states)):
                    hidden=encoder.histories(states[index:index+1])[0]
                    torch.testing.assert_close(hidden,expected[index],atol=2e-6,rtol=2e-5)
                    if hidden.requires_grad: ((index+1)*hidden.square().mean()).backward()
                self.assertIsNone(encoder.arrival.weight_ih_l0.grad)
                tape.backward()
                with self.assertRaises(RuntimeError): tape.backward()
            self.assertIsNone(encoder._history_tape)
            with self.assertRaises(RuntimeError): tape.get(states)
        for name,p in encoder.arrival.named_parameters():
            torch.testing.assert_close(p.grad,gradients[name],atol=2e-6,rtol=2e-5)

    def test_tape_rejects_parameter_change_and_releases_after_failure(self):
        encoder=Encoder().to(self.device)
        batch=PreparedBatch([(self.report,self.history)])
        with self.assertRaises(RuntimeError):
            with history_scope(encoder,batch) as tape:
                encoder.histories(batch)[0].sum().backward()
                with torch.no_grad(): encoder.arrival.weight_ih_l0.add_(.001)
                tape.backward()
        self.assertIsNone(encoder._history_tape)
        self.assertTrue(tape.closed)
        self.assertEqual(tape.outputs,{})

    def test_q_factored_head_matches_candidate_outputs_and_all_gradients(self):
        q=QNetwork().to(self.device)
        reference=copy.deepcopy(q)
        for module in reference.modules():
            if isinstance(module,torch.nn.GRU): module.flatten_parameters()
        state=torch.randn(2,512,device=self.device,requires_grad=True)
        actions=torch.rand(2,42,6,3,dtype=torch.float64,requires_grad=True)
        with torch.no_grad(): actions.div_(actions.sum(-1,keepdim=True))
        old_state=state.detach().clone().requires_grad_()
        old_actions=actions.detach().clone().requires_grad_()
        features,_=action_features(actions,self.device)
        old_features,_=action_features(old_actions,self.device)
        actual=q.value_features(state,features,projected=q.state_projection(state))
        expected=reference.head(torch.cat((old_state[:,None].expand(-1,42,-1),old_features),-1)).squeeze(-1)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        actual.square().mean().backward(); expected.square().mean().backward()
        torch.testing.assert_close(state.grad,old_state.grad,atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(actions.grad,old_actions.grad,atol=2e-6,rtol=2e-4)
        for (_,a),(_,b) in zip(q.head.named_parameters(),reference.head.named_parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-4)


if __name__=="__main__": unittest.main()
