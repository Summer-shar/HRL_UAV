from dataclasses import asdict,FrozenInstanceError,replace
import json
from pathlib import Path
import unittest
import numpy as np
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect
from hrl_v2.features import build
from hrl_v2.replay import Replay,unpack,resolve

SCENE=json.loads((Path(__file__).parents[1]/"configs/physical_scene.json").read_text())


def request(tick):
    return .2*np.sin(np.arange(6)+tick/50)


def routes(tick):
    # Rotate exact-zero faces, vertices and interior actions every 0.5 seconds.
    values=np.asarray([[.3,.5,.2],[.6,.4,0],[0,1,0],[.4,0,.6],[0,.3,.7],[1,0,0]],dtype=np.float64)
    return np.roll(values,tick//5,axis=0).reshape(2,3,3)


class Scripted:
    def upper(self,r,h): return request(r.tick),{}
    def lower(self,r,h):
        from hrl_v2.hybrid_simplex import SUPPORTS
        a=routes(r.tick)
        return a,np.asarray([SUPPORTS.index(tuple(np.flatnonzero(row))) for row in a.reshape(6,3)])


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter=PhysicsAdapter(SCENE)
        cls.adapter.reset(arrival_seed=1,layout_seed=2,load_seed=3)
        cls.legacy=HotspotIndustrialEnv(SCENE)
        cls.legacy.reset(1,2,3)
        cls.events=0
        def on_event(): cls.events+=1
        def on_tick(adapter,actual):
            e=cls.legacy; tick=e.step_index
            if tick%50==0: e.start_motion(request(tick))
            if tick%5==0: e.prepare_frame()
            expected=e.micro_step(routes(tick//5*5))
            if actual!=expected: raise AssertionError(f"Physical reading mismatch at tick {tick}")
            np.testing.assert_array_equal(e.physics.uav_coords,adapter.env.physics.uav_coords)
            if e.ready!=adapter.env.ready or e.upload!=adapter.env.upload or e.compute!=adapter.env.compute or e.transfer!=adapter.env.transfer:
                raise AssertionError(f"Queue order mismatch at tick {tick}")
            if e._active_parents!=adapter.env._active_parents or e._active_fragments!=adapter.env._active_fragments:
                raise AssertionError(f"Active per-task residual mismatch at tick {tick}")
            for name in ("generated_bits","generated_cycles","upload_served_bits","tx_served_bits","cpu_served_cycles"):
                if getattr(e,name)!=getattr(adapter.env,name): raise AssertionError(f"Conservation ledger mismatch: {name}")
        cls.episode=collect(cls.adapter,Scripted(),upper_version="script",lower_version=0,identity={},
                            on_event=on_event,on_tick=on_tick)

    def test_T01_900_second_physics_identity(self):
        a,b=self.adapter.env,self.legacy
        self.assertEqual(a.metrics(),b.metrics())
        self.assertEqual({k:asdict(v) for k,v in a.parents.items()},{k:asdict(v) for k,v in b.parents.items()})
        self.assertEqual({k:asdict(v) for k,v in a.fragments.items()},{k:asdict(v) for k,v in b.fragments.items()})

    def test_T06_true_macro_successors_and_no_terminal_arrivals(self):
        e=self.episode
        self.assertEqual(len(self.adapter.arrivals),1800); self.assertEqual(self.events,90)
        self.assertEqual(self.adapter.env._scene._frame,1799)
        for i in (0,8,9,10,1798):
            r=e.report(e.successors[i]); nxt=e.report(e.observations[i+1])
            self.assertEqual(r,nxt); self.assertEqual(r.tick,(i+1)*5)
            if i%10==9: self.assertEqual(r.phase,"adjustment")
        self.assertEqual(e.report(e.successors[-1]).tick,9000)
        with self.assertRaises(RuntimeError): self.adapter.prepare()
        with self.assertRaises(RuntimeError): self.adapter.start_macro(np.zeros(6))

    def test_T02_reports_detached_from_hidden_generator(self):
        e=self.adapter.env; report=self.adapter.report()
        previous=e._scene._busy.copy(); e._scene._busy=~previous
        self.assertEqual(report,self.adapter.report()); e._scene._busy=previous
        with self.assertRaises(FrozenInstanceError): report.tick=0
        self.assertEqual(len(report.queues),58)
        self.assertNotIn("busy",asdict(report)); self.assertNotIn("scene",asdict(report))

    def test_T03_fifo_and_parent_links(self):
        r=next(self.episode.report(i) for i in self.episode.observations if any(len(q.items)>1 for q in self.episode.report(i).queues))
        queues=list(r.queues); index=next(i for i,q in enumerate(queues) if len(q.items)>1)
        queues[index]=replace(queues[index],items=tuple(reversed(queues[index].items)))
        changed=replace(r,queues=tuple(queues))
        self.assertNotEqual(build(r).queues,build(changed).queues)
        self.assertNotEqual(build(r).edges,build(changed).edges)

    def test_T03_sibling_relationships_not_lost(self):
        for index in self.episode.observations:
            r=self.episode.report(index)
            choices=[(a,b) for a in r.fragments for b in r.fragments
                     if a.parent!=b.parent and a.priority==b.priority]
            if choices: break
        a,b=choices[0]
        fs=tuple(replace(f,parent=b.parent) if f.key==a.key else replace(f,parent=a.parent)
                 if f.key==b.key else f for f in r.fragments)
        ps=tuple(replace(p,children=tuple(b.key if x==a.key else x for x in p.children)) if p.key==a.parent else
                 replace(p,children=tuple(a.key if x==b.key else x for x in p.children)) if p.key==b.parent else p
                 for p in r.parents)
        changed=replace(r,parents=ps,fragments=fs)
        self.assertEqual(sum(f.compute_remaining for f in r.fragments),sum(f.compute_remaining for f in changed.fragments))
        self.assertNotEqual(build(r).edges["p_f"],build(changed).edges["p_f"])

    def test_T08_complete_replay_and_version_flush(self):
        replay=Replay(); replay.set_upper_version("script"); replay.add(self.episode)
        sample=replay.sample(np.random.default_rng(1)); self.assertEqual(len(sample),128)
        states=resolve(sample[:2]); self.assertEqual(states[0][0],self.episode.report(self.episode.observations[sample[0].index]))
        replay.set_upper_version("different"); self.assertFalse(replay.episodes)
        with self.assertRaises(ValueError): replay.add(self.episode)

    def test_T10_censoring_and_lossless_storage(self):
        e=self.episode; result=e.metrics
        self.assertLess(abs(result["holding_identity_error"]),1e-6)
        self.assertTrue(any(x["censored"] for x in result["parents"]))
        self.assertTrue(all(x["delay_s"] is None for x in result["parents"] if x["censored"]))
        np.testing.assert_array_equal(unpack(e.history),np.asarray(self.adapter.arrivals))
        self.assertAlmostEqual(sum(e.rewards),-result["cost_sum"]/100,places=9)


if __name__=="__main__": unittest.main()
