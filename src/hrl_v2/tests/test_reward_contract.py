"""Pure collector/replay algebra: no production environment or physical steps."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hrl_v2.collector import collect
from hrl_v2.timing import ClockConfig


class RewardAdapter:
    """Synthetic costs and timestamps, with two lower decisions per macro."""
    def __init__(self):
        self.timing=ClockConfig(.1,.5,1.,2.)
        self.env=SimpleNamespace(step_index=0)
        self.arrivals=[]
        self.cost=0.

    def report(self):
        return SimpleNamespace(tick=self.env.step_index,history_cursor=len(self.arrivals),severe=False)

    def prepare(self):
        self.arrivals.append(np.zeros((39,6)))
        return self.report()

    def start_macro(self,request):
        return {"displacement_m":np.zeros((3,2))}

    def step(self,route):
        cost=(self.env.step_index//self.timing.physical_per_lower+1)/self.timing.physical_per_lower
        self.env.step_index+=1
        self.cost+=cost
        return {"cost":cost}

    @staticmethod
    def report_bytes(report):
        return 128


class RewardPolicy:
    def upper(self,report,history):
        return np.zeros(6),{"value":float(report.tick)}

    def lower(self,report,history):
        return np.tile(np.eye(3),(2,1,1)),np.asarray([0,1,2,0,1,2])


def collect_rewards(**kwargs):
    adapter=RewardAdapter()
    with patch("hrl_v2.collector.snapshot",side_effect=lambda a:{"cost":a.cost}), \
         patch("hrl_v2.collector.finish",side_effect=lambda a,start:{"cost_sum":a.cost-start["cost"]}):
        episode=collect(adapter,RewardPolicy(),upper_version="reward-test-upper",lower_version=0,
                        identity={},reward_scale=1.,**kwargs)
    return episode,adapter


class RewardContractTests(unittest.TestCase):
    def test_discounted_macro_reward_is_separate_from_physical_cost(self):
        episode,_=collect_rewards(gamma_lower=.5)
        np.testing.assert_allclose(episode.rewards,[-1.,-2.,-3.,-4.],rtol=0,atol=1e-14)
        np.testing.assert_allclose([h["reward_physical"] for h in episode.high],[-3.,-7.],rtol=0,atol=1e-14)
        np.testing.assert_allclose([h["reward_learning"] for h in episode.high],[-2.,-5.],rtol=0,atol=1e-14)
        self.assertAlmostEqual(sum(h["reward_physical"] for h in episode.high),-episode.metrics["cost_sum"])
        self.assertEqual(episode.gamma_lower,.5)
        self.assertTrue(all("reward" not in h for h in episode.high))

    def test_default_undiscounted_scripted_collection_remains_compatible(self):
        episode,_=collect_rewards()
        self.assertEqual(episode.gamma_lower,1.)
        np.testing.assert_allclose([h["reward_learning"] for h in episode.high],
                                   [h["reward_physical"] for h in episode.high],rtol=0,atol=1e-14)

    def test_seal_rejects_wrong_learning_reward_without_changing_physical_total(self):
        episode,adapter=collect_rewards(gamma_lower=.5)
        episode.high[0]["reward_learning"]=episode.high[0]["reward_physical"]
        with self.assertRaisesRegex(ValueError,"learning reward"):
            episode.seal(adapter.arrivals,episode.metrics)

    def test_seal_checks_each_physical_macro_even_if_window_total_matches(self):
        episode,adapter=collect_rewards(gamma_lower=.5)
        episode.high[0]["reward_physical"]+=1.
        episode.high[1]["reward_physical"]-=1.
        with self.assertRaisesRegex(ValueError,"physical / micro reward"):
            episode.seal(adapter.arrivals,episode.metrics)

    def test_seal_requires_both_explicit_reward_fields(self):
        episode,adapter=collect_rewards(gamma_lower=.5)
        for field in ("reward_physical","reward_learning"):
            broken=copy.deepcopy(episode)
            del broken.high[0][field]
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"reward"):
                broken.seal(adapter.arrivals,broken.metrics)

    def test_discount_validation_happens_before_collection(self):
        for discount in (-.1,1.1,np.nan,np.inf):
            adapter=RewardAdapter()
            with self.subTest(discount=discount),self.assertRaisesRegex(ValueError,"discount"):
                collect(adapter,RewardPolicy(),upper_version="test",lower_version=0,identity={},gamma_lower=discount)
            self.assertEqual(adapter.env.step_index,0)
            self.assertEqual(adapter.arrivals,[])


if __name__=="__main__": unittest.main()
