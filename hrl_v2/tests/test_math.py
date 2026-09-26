import itertools
import math
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hrl_v2 import hybrid_simplex as hybrid
from hrl_v2.design import reference_hybrid_simplex as reference
from hrl_v2.randomness import Streams,derive
from hrl_v2.replay import probabilities
from hrl_v2.ppo import returns,logprob,gaussian_kl
from hrl_v2.schedule import phase,expected_counts


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.logits=torch.zeros(8,6,7,dtype=torch.float64,requires_grad=True)
        self.raw=torch.zeros(8,6,7,3,dtype=torch.float64,requires_grad=True)
        self.kappa=hybrid.concentration_from_raw(self.raw)
        self.streams=Streams(4101)

    def test_all_supports(self):
        faces,_=hybrid.all_face_samples(self.kappa)
        for i,support in enumerate(hybrid.SUPPORTS):
            self.assertTrue((faces[:,:,i,list(support)]>0).all())
            outside=[j for j in range(3) if j not in support]
            self.assertTrue((faces[:,:,i,outside]==0).all())
        torch.testing.assert_close(faces.sum(-1),torch.ones_like(faces[...,0]),atol=1e-12,rtol=0)

    def test_density_recomputation(self):
        a,ids,ell=hybrid.sample(self.logits,self.kappa,self.streams)
        torch.testing.assert_close(ell,hybrid.log_relative_density(self.logits,self.kappa,a,ids),rtol=0,atol=1e-12)

    def test_uniform_reference_density(self):
        k=torch.ones_like(self.kappa)
        a,ids,ell=hybrid.sample(self.logits,k,self.streams)
        torch.testing.assert_close(ell,torch.zeros_like(ell),rtol=0,atol=1e-12)

    def test_uniform_face_normalization(self):
        _,ell=hybrid.all_face_samples(torch.ones_like(self.kappa))
        torch.testing.assert_close(ell,torch.zeros_like(ell),rtol=0,atol=1e-12)

    def test_density_rejects_wrong_support(self):
        a,ids,_=hybrid.sample(self.logits,self.kappa,self.streams)
        ids[:]=0; a[:]=1/3
        with self.assertRaises(ValueError): hybrid.log_relative_density(self.logits,self.kappa,a,ids)

    def test_deterministic_decode(self):
        a=hybrid.deterministic_decode(self.logits,self.kappa)
        self.assertTrue((a[...,0]==1).all()); self.assertTrue((a[...,1:]==0).all())
        logits=self.logits.detach().clone(); logits[:,:,3]=1
        a=hybrid.deterministic_decode(logits,self.kappa)
        self.assertTrue((a[...,:2]==.5).all()); self.assertTrue((a[...,2]==0).all())

    def test_continuous_gradient(self):
        faces,_=hybrid.all_face_samples(self.kappa)
        loss=faces[:,:,6,0].square().mean(); loss.backward()
        self.assertGreater(self.raw.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(self.raw.grad).all())

    def test_surrogate_gradients_finite(self):
        def critic(a): return -(a[...,0,0]*a[...,1,1]+a.square().sum((-1,-2)))
        loss=hybrid.surrogate(self.logits,self.kappa,critic,.005,self.streams)
        loss.backward()
        for gradient in (self.raw.grad,self.logits.grad):
            self.assertTrue(torch.isfinite(gradient).all()); self.assertGreater(gradient.abs().sum(),0)

    def test_nonadditive_discrete_gradient_exact_enumeration(self):
        # Integrate all 49 choices of two coupled rows. Conditional faces are
        # fixed to their means so this isolates the exact categorical gradient.
        base=torch.linspace(-.6,.8,42,dtype=torch.float64).reshape(6,7).requires_grad_()
        combos=list(itertools.product(range(7),repeat=2)); n=len(combos)
        ids=torch.zeros(n,6,dtype=torch.long)
        ids[:,:2]=torch.tensor(combos)
        faces=torch.zeros(n,6,7,3,dtype=torch.float64)
        for m,support in enumerate(hybrid.SUPPORTS): faces[:,:,m,list(support)]=1/len(support)
        def critic(a):
            x=a[...,0,0]; y=a[...,1,1]
            return .3*x*y+(x-2*y).square()+torch.sin(x+y)
        q=base.softmax(-1)
        weights=q[0,ids[:,0]]*q[1,ids[:,1]]
        action=faces.gather(2,ids[...,None,None].expand(-1,-1,1,3)).squeeze(2)
        alpha=.013
        expected=-(weights*critic(action)).sum()+alpha*(q*(base.log_softmax(-1)+math.log(7))).sum()
        exact=torch.autograd.grad(expected,base)[0]
        with patch.object(hybrid,"all_face_samples",return_value=(faces,torch.zeros(n,6,7,dtype=torch.float64))), \
             patch.object(torch,"multinomial",return_value=ids.flatten()[:,None]):
            estimated=hybrid.surrogate(base[None].expand(n,-1,-1),torch.ones_like(faces),critic,
                                      alpha,self.streams,weights.detach()*n)
        actual=torch.autograd.grad(estimated,base)[0]
        torch.testing.assert_close(actual,exact,rtol=1e-10,atol=1e-12)

    def test_importance_weights_restore_uniform(self):
        severe=np.array([False,True,False,True,False])
        q=probabilities(severe); w=(1/len(q))/q; values=np.array([2.,9.,1.,20.,-5.])
        self.assertAlmostEqual(np.sum(q*w*values),np.mean(values))
        np.testing.assert_array_equal(probabilities([False]*5),np.full(5,.2))

    def test_reward_and_terminal_identity(self):
        costs=torch.arange(9000,dtype=torch.float64).reshape(1800,5)
        micro=hybrid.physical_frame_reward(costs)
        self.assertEqual(micro.sum().item(),micro.reshape(180,10).sum(1).sum().item())
        r=torch.tensor([1.,2.]); done=torch.tensor([True,False]); nxt=torch.tensor([float("nan"),4.])
        torch.testing.assert_close(hybrid.finite_horizon_target(r,done,nxt),torch.tensor([1.,6.]))

    def test_ppo_full_returns_and_tanh_density(self):
        np.testing.assert_array_equal(returns([1,2,3]),[6,5,3])
        z=torch.tensor([100.,-100.,0.,.3,1.,2.]); mean=torch.zeros(6); std=torch.full((6,),-.7)
        self.assertTrue(torch.isfinite(logprob(z,mean,std)))
        self.assertEqual(gaussian_kl(mean,std,mean,std).item(),0)

    def test_stream_separation_and_restore(self):
        before=torch.get_rng_state().clone(); saved=self.streams.state_dict()
        a=hybrid.sample(self.logits,self.kappa,self.streams)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        self.streams.load_state_dict(saved); b=hybrid.sample(self.logits,self.kappa,self.streams)
        for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(derive(4101,episode=1,purpose="layout",split="test"),derive(4105,episode=1,purpose="layout",split="test"))
        self.assertNotEqual(derive(4101,purpose="actor"),derive(4102,purpose="actor"))

    def test_fixed_schedule_counts(self):
        effective=0; actor=0; upper=0
        for i in range(1,601):
            p=phase(i)
            if p.layer=="upper": upper+=1
            elif p.block_episode!=1:
                for _ in range(90):
                    effective+=1
                    if p.actor_enabled and effective%2==0: actor+=1
        target=expected_counts()
        self.assertEqual(effective,target["sac_events"]); self.assertEqual(actor,target["sac_actor_updates"])
        self.assertEqual(upper//5,target["ppo_batches"])


if __name__=="__main__": unittest.main()
