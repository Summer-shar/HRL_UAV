"""Hybrid SAC with finite terminal targets and two independent critic fits."""
import copy
import numpy as np
import torch
from torch import nn
from .hybrid_simplex import sample, surrogate
from .replay import resolve


def actor_batch(actor,states):
    return actor.batch(states)


def encode_batch(network,states):
    return network.encode_batch(states)


class SAC:
    def __init__(self,actor,q1,q2):
        self.actor,self.q1,self.q2=actor,q1,q2
        self.target1,self.target2=copy.deepcopy(q1),copy.deepcopy(q2)
        for target in (self.target1,self.target2):
            for module in target.modules():
                if isinstance(module,nn.GRU): module.flatten_parameters()
        self.target1.requires_grad_(False); self.target2.requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
        self.critic_optimizer=torch.optim.Adam(list(q1.parameters())+list(q2.parameters()),lr=3e-4,betas=(.9,.999),eps=1e-8)
        self.counts=dict(triggers=0,skipped=0,events=0,critic_fits=0,actor_updates=0,target_updates=0)
        self.version=0

    def _critic_fit(self,samples,alpha,streams):
        self.critic_optimizer.zero_grad(set_to_none=True)
        device=next(self.q1.parameters()).device
        residuals=[[],[]]; target_values=[]
        for start in range(0,len(samples),16):
            batch=samples[start:start+16]
            states=resolve(batch)
            target=torch.tensor([s.episode.rewards[s.index] for s in batch],device=device,dtype=torch.float64)
            live=[i for i,s in enumerate(batch) if not s.episode.terminated[s.index]]
            with torch.no_grad():
                if live:
                    successors=resolve([batch[i] for i in live],next_state=True)
                    logits,kappa=actor_batch(self.actor,successors)
                    actions,_,ell=sample(logits,kappa,streams,200)
                    a=self.target1.value(encode_batch(self.target1,successors),actions)
                    b=self.target2.value(encode_batch(self.target2,successors),actions)
                    target[live]+=torch.minimum(a,b).double()-alpha*ell.sum(-1).to(device)
            if not torch.isfinite(target).all(): raise FloatingPointError("Invalid Bellman target")
            actions=torch.as_tensor(np.asarray([s.episode.actions[s.index] for s in batch]).reshape(-1,6,3))
            weights=target.new_tensor([s.weight for s in batch])
            loss=target.new_zeros(())
            for j,q in enumerate((self.q1,self.q2)):
                error=q.value(encode_batch(q,states),actions)-target
                loss=loss+(weights*error.square()).sum()/len(samples)
                residuals[j].extend(error.detach().cpu().tolist())
            loss.backward()
            target_values.extend(target.cpu().tolist())
        norms=[float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in q.parameters() if p.grad is not None])))
               for q in (self.q1,self.q2)]
        nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),1.,error_if_nonfinite=True)
        self.critic_optimizer.step(); self.counts["critic_fits"]+=1
        return dict(residual_mean=[float(np.mean(x)) for x in residuals],
                    residual_mse=[float(np.mean(np.square(x))) for x in residuals],
                    gradient_norm=norms,target_mean=float(np.mean(target_values)),target_variance=float(np.var(target_values)))

    def _actor_fit(self,samples,alpha,streams):
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.q1.requires_grad_(False); self.q2.requires_grad_(False)
        objective=[]; surrogate_value=0.
        try:
            for start in range(0,len(samples),16):
                batch=samples[start:start+16]; states=resolve(batch)
                with torch.no_grad():
                    e1=encode_batch(self.q1,states); e2=encode_batch(self.q2,states)
                def critic(actions):
                    return torch.minimum(self.q1.value(e1,actions),self.q2.value(e2,actions)).to("cpu",dtype=torch.float64)
                logits,kappa=actor_batch(self.actor,states)
                weights=logits.new_tensor([s.weight for s in batch])
                loss=surrogate(logits,kappa,critic,alpha,streams,weights)*len(batch)/len(samples)
                loss.backward(); surrogate_value+=loss.item()
                with torch.no_grad():
                    action,_,ell=sample(logits.detach(),kappa.detach(),streams,100)
                    value=alpha*ell.sum(-1)-critic(action)
                    objective.extend((weights*value).tolist())
            norm=float(nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True))
            self.actor_optimizer.step(); self.counts["actor_updates"]+=1; self.version+=1
        finally:
            self.q1.requires_grad_(True); self.q2.requires_grad_(True)
        return dict(objective=float(np.mean(objective)),surrogate=surrogate_value,gradient_norm=norm)

    def event(self,replay,alpha,actor_enabled,streams):
        self.counts["triggers"]+=1
        if not replay.episodes:
            self.counts["skipped"]+=1
            return dict(skipped=True)
        self.counts["events"]+=1
        fits=[self._critic_fit(replay.sample(streams.numpy),alpha,streams) for _ in range(2)]
        actor=None
        if actor_enabled and self.counts["events"]%2==0:
            actor=self._actor_fit(replay.sample(streams.numpy),alpha,streams)
        with torch.no_grad():
            for target,current in ((self.target1,self.q1),(self.target2,self.q2)):
                for a,b in zip(target.parameters(),current.parameters()): a.lerp_(b,.005)
        self.counts["target_updates"]+=1
        return dict(skipped=False,critics=fits,actor=actor)
