"""Short physical windows verify continuation without formal RL training."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from hrl_v2.collector import collect
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.ppo import generalized_advantage
from hrl_v2.replay import Replay, unpack
from hrl_v2.timing import ClockConfig


SCENE=json.loads((Path(__file__).parents[1]/"configs/physical_scene.json").read_text())


class RandomScript:
    def __init__(self,seed=53):
        self.rng=np.random.default_rng(seed)
        self.begins=0
        self.upper_calls=0

    def begin_episode(self,identity): self.begins+=1

    def upper(self,report,history):
        self.upper_calls+=1
        return self.rng.uniform(-.25,.25,6),dict(value=report.time/7)

    def lower(self,report,history):
        route=self.rng.dirichlet([2,2,2],size=6).reshape(2,3,3)
        return route,np.full(6,6)


def adapter(seconds):
    result=PhysicsAdapter(SCENE,timing=ClockConfig.from_scene(SCENE,seconds))
    result.reset(arrival_seed=41,layout_seed=42,load_seed=43)
    return result


def window(world,policy):
    return collect(world,policy,upper_version="script",lower_version=0,identity={})


def clone_boundary(boundary):
    """An in-memory state copy, not a durable research checkpoint format.

    hashlib digests expose copy() but cannot be pickled. Research persistence
    remains gated until a complete simulator/RNG snapshot format is designed.
    """
    scene=boundary[0].env._scene
    memo={id(digest):digest.copy() for digest in (scene.event_digest,scene.canonical_event_digest)}
    return copy.deepcopy(boundary,memo)


class RolloutSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=adapter(5)
        cls.policy=RandomScript()
        cls.first=window(cls.world,cls.policy)
        cls.boundary=clone_boundary((cls.world,cls.policy))
        cls.second=window(cls.world,cls.policy)
        cls.long_world=adapter(10)
        cls.long_policy=RandomScript()
        cls.long=window(cls.long_world,cls.long_policy)

    def test_cutoff_is_truncation_and_successor_is_next_decision(self):
        first,second=self.first,self.second
        self.assertFalse(any(first.terminated))
        self.assertEqual(first.truncated,[False]*9+[True])
        self.assertFalse(first.high[-1]["terminated"])
        self.assertTrue(first.high[-1]["truncated"])
        successor=first.report(first.successors[-1])
        self.assertEqual(successor,second.report(second.observations[0]))
        self.assertEqual(successor.tick,50)
        self.assertEqual(successor.history_cursor,11)
        self.assertEqual(successor.phase,"adjustment")
        self.assertEqual(first.report(first.high[-1]["successor"]),
                         second.report(second.high[0]["report"]))
        self.assertAlmostEqual(first.high[-1]["next_value"],5/7)
        self.assertNotEqual(successor,adapter(5).report())

    def test_split_windows_preserve_queues_actions_arrivals_and_rng(self):
        combined_actions=self.first.actions+self.second.actions
        np.testing.assert_array_equal(combined_actions,self.long.actions)
        np.testing.assert_array_equal(self.world.arrivals,self.long_world.arrivals)
        self.assertEqual(self.world.report(),self.long_world.report())
        self.assertEqual(self.policy.rng.bit_generator.state,self.long_policy.rng.bit_generator.state)
        self.assertEqual(self.policy.begins,1)
        self.assertEqual(self.policy.upper_calls,3)
        self.assertEqual(self.policy.upper_calls,self.long_policy.upper_calls)
        for split,whole in zip(self.first.high+self.second.high,self.long.high):
            np.testing.assert_array_equal(split["request"],whole["request"])
        np.testing.assert_allclose(self.first.rewards+self.second.rewards,self.long.rewards,rtol=0,atol=0)
        self.assertAlmostEqual(self.first.metrics["cost_sum"]+self.second.metrics["cost_sum"],
                               self.long.metrics["cost_sum"],places=9)
        self.assertEqual(self.first.metrics["seconds"],5)
        self.assertEqual(self.second.metrics["seconds"],5)
        self.assertTrue(set(self.first.metrics["parents"][i]["parent"]
                            for i in range(len(self.first.metrics["parents"]))) <= set(self.world.env.parents))

    def test_restored_in_memory_boundary_replays_exact_continuation(self):
        restored,policy=clone_boundary(self.boundary)
        result=window(restored,policy)
        self.assertEqual(result.reports,self.second.reports)
        self.assertEqual(result.history,self.second.history)
        np.testing.assert_array_equal(result.actions,self.second.actions)
        self.assertEqual(result.metrics,self.second.metrics)

    def test_cutoff_metrics_precede_boundary_arrivals(self):
        final=self.first.report(self.first.successors[-1])
        self.assertEqual(len(unpack(self.first.history)),final.history_cursor)
        self.assertTrue(all(row["birth_s"]<5 for row in self.first.metrics["parents"]))
        self.assertEqual(self.first.metrics["priority"][0]["arrived"]+
                         self.first.metrics["priority"][1]["arrived"],
                         int(np.asarray(unpack(self.first.history))[:-1,:,2::3].sum()))

    def test_replay_rejects_reset_report_as_truncated_successor(self):
        episode=copy.deepcopy(self.first)
        episode.successors[-1]=episode.add_report(adapter(5).report())
        with self.assertRaisesRegex(ValueError,"timeline"):
            episode.seal(unpack(self.first.history),self.first.metrics)

    def test_configurable_sixty_second_rollout_and_replay_counts(self):
        world=adapter(60)
        episode=window(world,RandomScript())
        self.assertEqual(world.env.step_index,600)
        self.assertEqual(len(episode.rewards),120)
        self.assertEqual(len(episode.high),12)
        self.assertEqual(len(episode.successors),120)
        self.assertEqual(episode.metrics["seconds"],60)
        replay=Replay(); replay.set_upper_version("script"); replay.add(episode)
        self.assertEqual(replay.writes,120)
        self.assertTrue(all(s.index<120 for s in replay.sample(np.random.default_rng(8))))

    def test_reset_removes_pending_decision(self):
        world,policy=clone_boundary(self.boundary)
        world.reset(arrival_seed=41,layout_seed=42,load_seed=43)
        fresh=window(world,RandomScript())
        self.assertEqual(fresh.reports,self.first.reports)

    def test_pending_behavior_version_cannot_be_relabelled(self):
        world,policy=clone_boundary(self.boundary)
        with self.assertRaisesRegex(ValueError,"upper policy version"):
            collect(world,policy,upper_version="changed",lower_version=0,identity={})

    def test_collection_event_clock_does_not_restart_at_cutoff(self):
        world,policy=clone_boundary(self.boundary)
        ticks=[]
        collect(world,policy,upper_version="script",lower_version=0,identity={},
                on_event=lambda:ticks.append(world.env.step_index))
        self.assertEqual(ticks,[100])

    def test_clock_divisibility(self):
        for args in ((.1,.55,5,60),(.1,.5,5,61),(.1,.5,4.9,60)):
            with self.assertRaises(ValueError): ClockConfig(*args)


class AdvantageBoundaryTests(unittest.TestCase):
    def test_truncation_bootstraps_but_breaks_gae_trace(self):
        # The reward 100 belongs to a different sampled window. It must not
        # flow through the preceding truncation, whose successor value is 7.
        advantage,target=generalized_advantage([1,2,100],[3,4,5],[4,7,11],
            [False,False,False],[False,True,True],gamma=.5,gae_lambda=.8)
        np.testing.assert_allclose(advantage,[.6,1.5,100.5],rtol=0,atol=1e-14)
        np.testing.assert_allclose(target,[3.6,5.5,105.5],rtol=0,atol=1e-14)

    def test_true_termination_suppresses_bootstrap_and_trace(self):
        advantage,target=generalized_advantage([1,2,100],[3,4,5],[4,np.nan,11],
            [False,True,False],[False,False,True],gamma=.5,gae_lambda=.8)
        np.testing.assert_allclose(advantage,[-.8,-2,100.5],rtol=0,atol=1e-14)
        self.assertEqual(target[1],2)

    def test_unbroken_trace_and_explicit_test_discounts(self):
        advantage,target=generalized_advantage([1,2],[3,4],[4,7],
            [False,False],[False,True],gamma=1.,gae_lambda=1.)
        np.testing.assert_array_equal(target,[10,9])
        np.testing.assert_array_equal(advantage,[7,5])


if __name__=="__main__": unittest.main()
