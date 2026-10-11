import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from hrl_v2.ppo import PPO,logprob
from hrl_v2.randomness import Streams
from hrl_v2.replay import pack
from hrl_v2.tests.test_learning import fingerprint


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__(); self.mean=nn.Parameter(torch.zeros(6)); self.std=nn.Parameter(torch.full((6,),-1.9))
    def forward(self,report,history): return self.mean,self.std
    def batch(self,states): return self.mean.expand(len(states),-1),self.std.expand(len(states),-1)


class TinyValue(nn.Module):
    def __init__(self):
        super().__init__(); self.bias=nn.Parameter(torch.tensor(0.))
    def forward(self,report,history): return self.bias
    def batch(self,states): return self.bias.expand(len(states))


class ToyEpisode:
    complete=True
    lower_version=7
    upper_version="sampled-upper-3"
    history=pack(np.zeros((1,39,6)))
    def __init__(self,actor):
        mean,std=actor(None,None)
        self.high=[]
        for i in range(180):
            z=torch.full((6,),(i%7-3)*.03)
            self.high.append(dict(report=0,reward_physical=-1.,reward_learning=-1.,value=0.,next_value=0.,
                terminated=False,truncated=i==179,successor=0,pre_tanh=z.numpy(),
                logprob=logprob(z,mean,std).item(),mean=mean.detach().numpy().copy(),logstd=std.detach().numpy().copy()))
    def report(self,index): return None


class PPOTests(unittest.TestCase):
    def test_epoch_rollback_restores_adam_and_value_continues(self):
        ppo=PPO(TinyActor(),TinyValue()); batch=[ToyEpisode(ppo.actor) for _ in range(5)]
        actor=fingerprint(ppo.actor.state_dict()); optimizer=fingerprint(ppo.actor_optimizer.state_dict())
        with patch("hrl_v2.ppo.gaussian_kl",return_value=torch.tensor(1.)):
            result=ppo.update(batch,Streams(4101))
        self.assertEqual(fingerprint(ppo.actor.state_dict()),actor)
        self.assertEqual(fingerprint(ppo.actor_optimizer.state_dict()),optimizer)
        self.assertEqual(ppo.counts["actor_steps_accepted"],0)
        self.assertEqual(ppo.counts["actor_steps_attempted"],6)
        self.assertEqual(ppo.counts["rollbacks"],1)
        self.assertEqual(ppo.counts["value_steps"],36)
        self.assertNotEqual(ppo.value.bias.item(),0.)
        self.assertFalse(result[0]["accepted"])

    def test_accepted_epochs_and_lower_version_rejection(self):
        ppo=PPO(TinyActor(),TinyValue()); batch=[ToyEpisode(ppo.actor) for _ in range(5)]
        with patch("hrl_v2.ppo.gaussian_kl",return_value=torch.tensor(0.)):
            ppo.update(batch,Streams(4101))
        self.assertEqual(ppo.counts["actor_steps_accepted"],36)
        self.assertEqual(ppo.counts["value_steps"],36)
        batch[-1].lower_version=8
        with self.assertRaises(ValueError): ppo.update(batch,Streams(4101))

    def test_short_rollouts_use_actual_macro_count(self):
        ppo=PPO(TinyActor(),TinyValue(),gamma=.8,gae_lambda=.6)
        batch=[ToyEpisode(ppo.actor) for _ in range(5)]
        for episode in batch:
            episode.high=episode.high[:2]
            episode.high[-1].update(truncated=True,next_value=7.)
        with patch("hrl_v2.ppo.gaussian_kl",return_value=torch.tensor(0.)):
            ppo.update(batch,Streams(4101))
        self.assertEqual(ppo.counts["actor_steps_accepted"],6)
        self.assertEqual(ppo.counts["value_steps"],6)

    def test_mixed_sampled_upper_versions_are_rejected_before_update(self):
        ppo=PPO(TinyActor(),TinyValue())
        batch=[SimpleNamespace(complete=True,lower_version=7,upper_version="sampled-upper-3") for _ in range(5)]
        batch[-1].upper_version="sampled-upper-4"
        with self.assertRaisesRegex(ValueError,"crosses upper behavior policy"):
            ppo.update(batch,Streams(4101))
        self.assertEqual(ppo.counts["batches"],0)

    def test_missing_sampled_versions_are_rejected_even_when_all_match(self):
        ppo=PPO(TinyActor(),TinyValue())
        for field in ("upper_version","lower_version"):
            for missing in ("absent",None,""):
                with self.subTest(field=field,missing=missing):
                    batch=[SimpleNamespace(complete=True,lower_version=7,upper_version="sampled-upper-3") for _ in range(5)]
                    for episode in batch:
                        if missing=="absent": delattr(episode,field)
                        else: setattr(episode,field,missing)
                    with self.assertRaisesRegex(ValueError,"missing .* versions"):
                        ppo.update(batch,Streams(4101))
        self.assertEqual(ppo.counts["batches"],0)

    def test_gae_targets_use_learning_reward_and_real_truncated_successor(self):
        ppo=PPO(TinyActor(),TinyValue(),gamma=.8,gae_lambda=.6)
        batch=[ToyEpisode(ppo.actor) for _ in range(5)]
        for episode in batch:
            episode.high=episode.high[:2]
            episode.high[0].update(reward_physical=-30.,reward_learning=-3.)
            episode.high[1].update(reward_physical=-50.,reward_learning=-5.,truncated=True,next_value=7.)
        # Isolate target construction from optimization; no network update is
        # needed to distinguish the two reward fields or verify bootstrapping.
        with patch.object(ppo,"_minibatch") as minibatch,patch.object(ppo,"_kl",return_value=0.):
            ppo.update(batch,Streams(4101))
        for call in minibatch.call_args_list:
            targets=sorted(row[2] for row in call.args[1])
            np.testing.assert_allclose(targets,[-2.712]*5+[.6]*5,rtol=0,atol=1e-12)
