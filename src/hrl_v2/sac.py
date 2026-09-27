"""Hybrid SAC with finite terminal targets and two independent critic fits."""
import copy
from contextlib import ExitStack
import numpy as np
import torch
from torch import nn
from .hybrid_simplex import sample, surrogate, action_features
from .prepared import PreparedBatch, history_scope
from .replay import resolve


def actor_batch(actor,states):
    return actor.batch(states)


def encode_batch(network,states):
    return network.encode_batch(states)


def shared_action_features(actions,networks,device):
    """Validate/cast an action once for independent Q heads on one device."""
    features,error=action_features(actions,device)
    for network in networks:
        network.last_action_cast_error=max(network.last_action_cast_error,error)
    return features


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
        # Execution chunk only: replay remains 128 samples and updates unchanged.
        self.microbatch_size=16

    def _critic_fit(self,samples,alpha,streams):
        self.critic_optimizer.zero_grad(set_to_none=True)
        device=next(self.q1.parameters()).device
        states,successors=PreparedBatch.replay(samples)
        live_states=successors.select([i for i,s in enumerate(samples) if not s.episode.terminated[s.index]])
        residuals=[[],[]]; target_values=[]
        with ExitStack() as scopes:
            tapes=[scopes.enter_context(history_scope(q.encoder,states)) for q in (self.q1,self.q2)]
            with torch.no_grad():
                for network in (self.actor,self.target1,self.target2):
                    scopes.enter_context(history_scope(network.encoder,live_states))
            for start in range(0,len(samples),self.microbatch_size):
                batch=samples[start:start+self.microbatch_size]
                current=states[start:start+len(batch)]
                target=torch.tensor([s.episode.rewards[s.index] for s in batch],device=device,dtype=torch.float64)
                live=[i for i,s in enumerate(batch) if not s.episode.terminated[s.index]]
                with torch.no_grad():
                    if live:
                        following=successors.select([start+i for i in live])
                        logits,kappa=actor_batch(self.actor,following)
                        actions,_,ell=sample(logits,kappa,streams,200)
                        features=shared_action_features(actions,(self.target1,self.target2),device)
                        a=self.target1.value_features(encode_batch(self.target1,following),features)
                        b=self.target2.value_features(encode_batch(self.target2,following),features)
                        target[live]+=torch.minimum(a,b).double()-alpha*ell.sum(-1).to(device)
                actions=torch.as_tensor(np.asarray([s.episode.actions[s.index] for s in batch]).reshape(-1,6,3))
                features=shared_action_features(actions,(self.q1,self.q2),device)
                weights=target.new_tensor([s.weight for s in batch])
                loss=target.new_zeros(())
                for j,q in enumerate((self.q1,self.q2)):
                    error=q.value_features(encode_batch(q,current),features)-target
                    loss=loss+(weights*error.square()).sum()/len(samples)
                    residuals[j].append(error.detach())
                loss.backward()
                target_values.append(target)
            for tape in tapes:
                if tape is not None: tape.backward()
        target_values=torch.cat(target_values)
        if not torch.isfinite(target_values).all(): raise FloatingPointError("Invalid Bellman target")
        residuals=[torch.cat(values) for values in residuals]
        norms=[torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in q.parameters() if p.grad is not None]))
               for q in (self.q1,self.q2)]
        stats=torch.stack([*(x.mean() for x in residuals),*(x.square().mean() for x in residuals),
                           *norms,target_values.mean(),target_values.var(unbiased=False)]).detach().cpu().tolist()
        nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),1.,error_if_nonfinite=True)
        self.critic_optimizer.step(); self.counts["critic_fits"]+=1
        return dict(residual_mean=stats[:2],residual_mse=stats[2:4],
                    gradient_norm=stats[4:6],target_mean=stats[6],target_variance=stats[7])

    def _actor_fit(self,samples,alpha,streams):
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.q1.requires_grad_(False); self.q2.requires_grad_(False)
        objective=[]; surrogate_values=[]
        try:
            states=resolve(samples)
            with ExitStack() as scopes:
                tape=scopes.enter_context(history_scope(self.actor.encoder,states))
                with torch.no_grad():
                    for q in (self.q1,self.q2): scopes.enter_context(history_scope(q.encoder,states))
                for start in range(0,len(samples),self.microbatch_size):
                    batch=samples[start:start+self.microbatch_size]; current=states[start:start+len(batch)]
                    with torch.no_grad():
                        e1=encode_batch(self.q1,current); e2=encode_batch(self.q2,current)
                        p1=self.q1.state_projection(e1); p2=self.q2.state_projection(e2)
                    def critic(actions):
                        features=shared_action_features(actions,(self.q1,self.q2),e1.device)
                        return torch.minimum(self.q1.value_features(e1,features,p1),
                                             self.q2.value_features(e2,features,p2)).to("cpu",dtype=torch.float64)
                    logits,kappa=actor_batch(self.actor,current)
                    weights=logits.new_tensor([s.weight for s in batch])
                    loss=surrogate(logits,kappa,critic,alpha,streams,weights)*len(batch)/len(samples)
                    loss.backward(); surrogate_values.append(loss.detach())
                    with torch.no_grad():
                        action,_,ell=sample(logits.detach(),kappa.detach(),streams,100)
                        value=alpha*ell.sum(-1)-critic(action)
                        objective.append(weights*value)
                if tape is not None: tape.backward()
            norm=float(nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True))
            self.actor_optimizer.step(); self.counts["actor_updates"]+=1; self.version+=1
        finally:
            self.q1.requires_grad_(True); self.q2.requires_grad_(True)
        return dict(objective=torch.cat(objective).mean().item(),
                    surrogate=torch.stack(surrogate_values).sum().item(),gradient_norm=norm)

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
