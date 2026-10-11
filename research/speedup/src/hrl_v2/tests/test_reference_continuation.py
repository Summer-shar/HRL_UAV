"""Deterministic external workload: 60 s equals 30 s + 30 s without reset.

Only the arrival source and rollout length change. Production geometry,
radio, service capacities, movement, costs, and queue rules stay intact.
The shared fixture executes exactly 1,200 physical steps when the tests run.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.industrial_scene import Arrival
from hrl_v2.collector import collect
from hrl_v2.hybrid_simplex import SUPPORTS
from hrl_v2.metrics import finish as physical_finish
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.replay import unpack
from hrl_v2.timing import ClockConfig


SCENE=json.loads((Path(__file__).parents[1]/"configs/physical_scene.json").read_text())
SEEDS=dict(arrival_seed=91,layout_seed=92,load_seed=93)


def generator_state(scene):
    """Read hidden generator state only in the test oracle, never the policy."""
    return pickle.dumps(dict(
        streams=[[copy.deepcopy(r.bit_generator.state) for r in getattr(scene,name)]
                 for name in ("_activity_rng","_arrival_rng","_mark_rng")],
        busy=scene._busy.copy(),frame=scene._frame,task_id=scene._task_id,
        local_event_index=scene._local_event_index.copy()),protocol=5)


class ArrivalTape:
    """A fixed external script; calling it never calls the scene generator."""
    def __init__(self,terminals,lower_interval):
        users={source:[t for t in terminals if t.source==source] for source in range(3)}
        # Tiny high-priority tasks complete under normal resources. A large
        # low-priority compute task and a huge upload survive both cutoffs.
        specifications=(
            (0,0,0,0,1e3,1e6),
            (0,0,1,1,1e3,1e13),
            (59,1,0,1,1e12,1e14),
            (59,2,0,0,1e3,1e10),
            (60,1,1,0,1e3,1e6),
            (90,2,1,0,1e3,1e6),
            (119,0,2,0,1e3,1e6),
            (120,2,2,0,1e3,1e6),
        )
        self.by_frame={}
        self.arrivals=[]
        self.calls=[]
        for task_id,(frame,source,slot,priority,bits,cycles) in enumerate(specifications):
            terminal=users[source][slot]
            arrival=Arrival(task_id,terminal.user_id,source,terminal.task_type,priority,
                            frame*lower_interval,bits,cycles)
            self.by_frame.setdefault(frame,[]).append(arrival)
            self.arrivals.append(arrival)

    def __call__(self,frame):
        if frame!=len(self.calls):
            raise AssertionError("Arrival frame was skipped or prepared twice")
        self.calls.append(frame)
        return tuple(self.by_frame.get(frame,()))


class FixedPolicy:
    def __init__(self,clock):
        self.clock=clock
        self.begins=0
        self.upper_calls=[]
        self.lower_calls=[]

    def begin_episode(self,identity): self.begins+=1

    def upper(self,report,history):
        macro=report.tick//self.clock.physical_per_upper
        sign=1. if macro%2==0 else -1.
        request=sign*np.asarray([.2,.1,-.1,.2,.1,-.2],dtype=np.float64)
        self.upper_calls.append((report.tick,request.tolist()))
        return request,dict(value=report.time/10.)

    def lower(self,report,history):
        frame=report.tick//self.clock.physical_per_lower
        route=np.empty((2,3,3),dtype=np.float64)
        for priority in range(2):
            for source in range(3):
                route[priority,source]=np.roll([.3,.5,.2],(source+priority+frame)%3)
        support=np.asarray([SUPPORTS.index(tuple(np.flatnonzero(row))) for row in route.reshape(6,3)])
        self.lower_calls.append((report.tick,route.tolist()))
        return route,support


class RecordedAdapter(PhysicsAdapter):
    def __init__(self,seconds):
        super().__init__(SCENE,ledger=True,timing=ClockConfig.from_scene(SCENE,seconds))
        self.reset(**SEEDS)
        self.prepare_calls=[]
        self.commit_calls=[]
        self.trace=[]
        self.tape=ArrivalTape(self.env._scene.terminals,self.timing.lower_interval)
        self.original_generator_state=generator_state(self.env._scene)
        self.env._scene.frame=self.tape

    def prepare(self):
        self.prepare_calls.append(self.env.step_index)
        return super().prepare()

    def start_macro(self,request):
        self.commit_calls.append((self.env.step_index,np.asarray(request).tolist()))
        return super().start_macro(request)

    def step(self,route):
        reading=super().step(route)
        env=self.env
        self.trace.append(dict(tick=env.step_index,action=route.tolist(),reading=copy.deepcopy(reading),
            positions=env.physics.uav_coords.tolist(),velocity=env.velocity.tolist(),target=env._target.tolist(),
            upload=[tuple(queue) for queue in env.upload],ready=tuple(env.ready),
            cpu=[[tuple(queue) for queue in priority] for priority in env.compute],
            transfer=[[[tuple(queue) for queue in source] for source in priority] for priority in env.transfer],
            parents={key:asdict(parent) for key,parent in env.parents.items()},
            fragments={key:asdict(fragment) for key,fragment in env.fragments.items()}))
        return reading


def window(adapter,policy):
    return collect(adapter,policy,upper_version="fixed-external-script",lower_version=0,identity={})


class ReferenceContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.whole=RecordedAdapter(60)
        cls.split=RecordedAdapter(30)
        cls.whole_policy=FixedPolicy(cls.whole.timing)
        cls.split_policy=FixedPolicy(cls.split.timing)
        cls.cutoff_reads=[]
        def record_cutoff(adapter,start):
            result=physical_finish(adapter,start)
            cls.cutoff_reads.append(dict(tick=adapter.env.step_index,
                prepared_frame=adapter.env._prepared_frame,parent_ids=tuple(adapter.env.parents),
                result=copy.deepcopy(result)))
            return result
        with patch("hrl_v2.collector.finish",side_effect=record_cutoff):
            cls.full=window(cls.whole,cls.whole_policy)
            cls.first=window(cls.split,cls.split_policy)
            cls.boundary_successor=cls.first.report(cls.first.successors[-1])
            cls.second=window(cls.split,cls.split_policy)

    def test_each_physical_step_matches_without_reset(self):
        self.assertEqual(len(self.whole.trace)+len(self.split.trace),1200)
        self.assertEqual(self.whole.trace,self.split.trace)
        self.assertEqual(self.whole.env.ledger,self.split.env.ledger)
        self.assertEqual(self.whole.report(),self.split.report())
        self.assertEqual(self.whole_policy.upper_calls,self.split_policy.upper_calls)
        self.assertEqual(self.whole_policy.lower_calls,self.split_policy.lower_calls)
        np.testing.assert_array_equal(self.whole.arrivals,self.split.arrivals)
        np.testing.assert_array_equal(self.full.actions,self.first.actions+self.second.actions)
        np.testing.assert_array_equal(self.full.rewards,self.first.rewards+self.second.rewards)

    def test_no_duplicate_prepare_macro_commit_or_hidden_rng_draws(self):
        for adapter,policy in ((self.whole,self.whole_policy),(self.split,self.split_policy)):
            self.assertEqual(adapter.prepare_calls,list(range(0,601,5)))
            self.assertEqual(adapter.tape.calls,list(range(121)))
            self.assertEqual([tick for tick,_ in adapter.commit_calls],list(range(0,601,50)))
            self.assertEqual(adapter.commit_calls,policy.upper_calls)
            self.assertEqual(policy.begins,1)
            self.assertEqual(generator_state(adapter.env._scene),adapter.original_generator_state)
        self.assertEqual(self.whole.env.config,self.split.env.config)
        self.assertEqual(self.whole.env.scene_wrapper,SCENE)

    def test_cutoff_statistics_precede_next_arrival_and_survivors_persist(self):
        self.assertEqual([(row["tick"],row["prepared_frame"]) for row in self.cutoff_reads],
                         [(600,119),(300,59),(600,119)])
        for row in self.cutoff_reads:
            cutoff=row["tick"]*self.whole.env.dt
            expected=tuple(a.task_id for a in self.whole.tape.arrivals if a.birth_s<cutoff)
            self.assertEqual(row["parent_ids"],expected)
        self.assertNotIn(4,{p["parent"] for p in self.first.metrics["parents"]})
        self.assertIn(4,{p["parent"] for p in self.second.metrics["parents"]})
        self.assertNotIn(7,{p["parent"] for p in self.full.metrics["parents"]})
        self.assertIn(7,self.whole.env.parents)  # Prepared for the real successor only.
        for metrics in (self.first.metrics,self.second.metrics,self.full.metrics):
            unfinished={p["parent"] for p in metrics["parents"] if p["censored"]}
            self.assertTrue({1,2}<=unfinished)
            self.assertTrue(any(p["completed"] for p in metrics["parents"]))
        self.assertGreater(self.split.env.parents[2].upload_remaining,0.)
        self.assertIsNone(self.split.env.parents[1].finish_s)
        first_crossing=next(p for p in self.first.metrics["parents"] if p["parent"]==3)
        second_crossing=next(p for p in self.second.metrics["parents"] if p["parent"]==3)
        self.assertTrue(first_crossing["censored"])
        self.assertTrue(second_crossing["carry_in"])
        self.assertTrue(second_crossing["completed"])
        self.assertFalse(second_crossing["censored"])
        self.assertGreater(second_crossing["finish_s"],30.)
        self.assertLess(second_crossing["finish_s"],60.)

    def test_real_successor_and_independent_reset(self):
        self.assertEqual(self.boundary_successor,self.second.report(self.second.observations[0]))
        self.assertEqual(self.first.report(self.first.high[-1]["successor"]),
                         self.second.report(self.second.high[0]["report"]))
        self.assertEqual(self.boundary_successor.tick,300)
        self.assertEqual(self.boundary_successor.history_cursor,61)
        self.assertEqual(len(unpack(self.first.history)),61)
        for episode in (self.first,self.second,self.full):
            self.assertFalse(any(episode.terminated))
            self.assertEqual(episode.truncated,[False]*(len(episode.rewards)-1)+[True])
            self.assertFalse(episode.metrics["terminated"])
            self.assertTrue(episode.metrics["truncated"])
        reset=PhysicsAdapter(SCENE,timing=ClockConfig.from_scene(SCENE,30))
        reset.reset(**SEEDS)  # Zero physical service steps.
        self.assertEqual(reset.env.step_index,0)
        self.assertEqual(reset.arrivals,[])
        self.assertEqual(reset.env.parents,{})
        self.assertIsNone(reset._continuation)
        self.assertNotEqual(reset.report(),self.boundary_successor)
        self.assertEqual(self.split.env.step_index,600)

    def test_holding_cost_and_censoring_from_independent_birth_completion_ledger(self):
        dt=self.whole.env.dt
        completions={row["parent_id"]:round(row["time_s"]/dt) for row in self.whole.env.ledger
                     if row["kind"]=="parent_complete"}
        self.assertEqual(len(completions),sum(row["kind"]=="parent_complete" for row in self.whole.env.ledger))
        objective=SCENE["base_scene"]["objective"]
        for episode,begin,end in ((self.full,0,600),(self.first,0,300),(self.second,300,600)):
            metrics=episode.metrics
            population=[]
            holding=0.
            for arrival in self.whole.tape.arrivals:
                birth=round(arrival.birth_s/dt)
                finish=completions.get(arrival.task_id)
                if birth>=end or (finish is not None and finish<=begin):
                    continue
                residence=(min(end,finish if finish is not None else end)-max(begin,birth))*dt
                holding+=objective["priority_weights"][arrival.priority]*residence
                population.append((arrival,birth,finish,residence))
            self.assertEqual({p["parent"] for p in metrics["parents"]},
                             {a.task_id for a,_,_,_ in population})
            np.testing.assert_allclose(metrics["holding_task_s"],holding,rtol=1e-12,atol=1e-8)
            for row in metrics["parents"]:
                arrival,birth,finish,residence=next(p for p in population if p[0].task_id==row["parent"])
                censored=finish is None or finish>end
                self.assertEqual(row["carry_in"],birth<begin)
                self.assertEqual(row["censored"],censored)
                self.assertEqual(row["observed_residence_s"],residence)
                self.assertEqual(row["delay_s"],None if censored else (finish-birth)*dt)
                self.assertEqual(row["cutoff_age_s"],(end-birth)*dt if censored else None)
            energy={name:sum(record["reading"]["energy_j"][name] for record in self.whole.trace[begin:end])
                    for name in ("upload","u2u","cpu","propulsion")}
            for name,value in energy.items():
                np.testing.assert_allclose(metrics["energy_j"][name],value,rtol=1e-12,atol=1e-8)
            cost=objective["holding_weight"]*holding/objective["holding_reference_task_s"]+(
                1-objective["holding_weight"])*(energy['upload']+energy['cpu']+energy['u2u']+
                objective['propulsion_weight']*energy['propulsion'])/objective["energy_reference_j"]
            np.testing.assert_allclose(metrics["cost_sum"],cost,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(sum(episode.rewards),-cost/episode.reward_scale,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(sum(row["reward_physical"] for row in episode.high),
                                       -cost/episode.reward_scale,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(self.first.metrics["cost_sum"]+self.second.metrics["cost_sum"],
                                   self.full.metrics["cost_sum"],rtol=1e-12,atol=1e-8)


if __name__=="__main__": unittest.main()
