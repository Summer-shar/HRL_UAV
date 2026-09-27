"""PPO with explicit continuing-service bootstrap and GAE boundaries."""
import copy
import json
from contextlib import ExitStack, nullcontext
import math
import numpy as np
import torch
from torch import nn
from .replay import unpack
from .prepared import PreparedBatch, history_scope


def network_history_scope(network,states):
    # Also support the small duck-typed policies used by protocol unit tests.
    return history_scope(network.encoder,states) if hasattr(network,"encoder") else nullcontext()


def logprob(z,mean,logstd):
    normal=-.5*((z-mean)/logstd.exp()).square()-logstd-.5*math.log(2*math.pi)
    correction=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
    return (normal-correction).sum(-1)


def gaussian_kl(old_mean,old_logstd,mean,logstd):
    return (logstd-old_logstd+(old_logstd.mul(2).exp()+(old_mean-mean).square())/
            (2*logstd.mul(2).exp())-.5).sum(-1)


def returns(rewards):
    """Finite return helper retained for the algebra/reference tests only."""
    return np.cumsum(np.asarray(rewards,dtype=np.float64)[::-1])[::-1].copy()


def generalized_advantage(rewards,values,next_values,terminated,truncated,*,gamma,gae_lambda):
    """Bootstrap on truncation, but never propagate GAE through a boundary."""
    arrays=[np.asarray(x,dtype=np.float64) for x in (rewards,values,next_values)]
    rewards,values,next_values=arrays
    terminated=np.asarray(terminated,dtype=bool)
    truncated=np.asarray(truncated,dtype=bool)
    if any(x.ndim!=1 or len(x)!=len(rewards) for x in arrays+[terminated,truncated]):
        raise ValueError("GAE inputs must be equally sized one-dimensional arrays")
    if not 0<=gamma<=1 or not 0<=gae_lambda<=1:
        raise ValueError("Discount and GAE lambda must lie in [0, 1]")
    residual=rewards-values+gamma*np.where(terminated,0.,next_values)
    if not np.isfinite(residual).all(): raise ValueError("GAE needs finite nonterminal successor values")
    advantage=np.empty_like(rewards)
    following=0.
    for i in range(len(rewards)-1,-1,-1):
        following=residual[i]+gamma*gae_lambda*(not (terminated[i] or truncated[i]))*following
        advantage[i]=following
    return advantage,advantage+values


class PPO:
    def __init__(self,actor,value,*,gamma=None,gae_lambda=None):
        self.actor,self.value=actor,value
        if gamma is None or gae_lambda is None:
            from .paths import default_config
            settings=json.loads(default_config().read_text(encoding="utf-8"))["ppo"]
            gamma=settings["gamma"] if gamma is None else gamma
            gae_lambda=settings["gae_lambda"] if gae_lambda is None else gae_lambda
        self.gamma,self.gae_lambda=float(gamma),float(gae_lambda)
        if not 0<=self.gamma<=1 or not 0<=self.gae_lambda<=1:
            raise ValueError("PPO discount and GAE lambda must lie in [0, 1]")
        self.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
        self.value_optimizer=torch.optim.Adam(value.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8)
        self.counts=dict(batches=0,actor_steps_attempted=0,actor_steps_accepted=0,value_steps=0,rollbacks=0)
        self.version=0
        self.microbatch_size=16

    def _minibatch(self,states,rows,advantages,active):
        """One unchanged effective minibatch, independent full-history tapes."""
        if active: self.actor_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        with ExitStack() as scopes:
            actor_tape=scopes.enter_context(network_history_scope(self.actor,states)) if active else None
            value_tape=scopes.enter_context(network_history_scope(self.value,states))
            for offset in range(0,len(states),self.microbatch_size):
                selected=rows[offset:offset+self.microbatch_size]
                current=states[offset:offset+len(selected)]
                if active:
                    mean,std=self.actor.batch(current)
                    z=mean.new_tensor(np.asarray([h["pre_tanh"] for _,h,_ in selected]))
                    old=mean.new_tensor([h["logprob"] for _,h,_ in selected])
                    ratio=(logprob(z,mean,std)-old).exp()
                    advantage=mean.new_tensor(advantages[offset:offset+len(selected)])
                    actor_loss=-torch.minimum(ratio*advantage,ratio.clamp(.85,1.15)*advantage).sum()/len(states)
                    actor_loss.backward()
                value=self.value.batch(current)
                target=value.new_tensor([G for _,_,G in selected])
                (.5*(value-target).square().sum()/len(states)).backward()
            for tape in (actor_tape,value_tape):
                if tape is not None: tape.backward()
        if active:
            nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
            self.actor_optimizer.step(); self.counts["actor_steps_attempted"]+=1
        nn.utils.clip_grad_norm_(self.value.parameters(),1.,error_if_nonfinite=True)
        self.value_optimizer.step(); self.counts["value_steps"]+=1

    def _kl(self,states,rows):
        """Exact epoch KL, sharing only a no-gradient arrival prefix."""
        divergences=[]
        with torch.no_grad(),network_history_scope(self.actor,states):
            for start in range(0,len(rows),self.microbatch_size):
                selected=rows[start:start+self.microbatch_size]
                mean,std=self.actor.batch(states[start:start+len(selected)])
                old_mean=mean.new_tensor(np.asarray([h["mean"] for _,h,_ in selected]))
                old_std=mean.new_tensor(np.asarray([h["logstd"] for _,h,_ in selected]))
                divergences.append(gaussian_kl(old_mean,old_std,mean,std).reshape(-1))
        return torch.cat(divergences).double().mean().item()

    def update(self,episodes,streams):
        if len(episodes)!=5 or any(not e.complete for e in episodes):
            raise ValueError("PPO needs five complete episodes")
        if len({e.lower_version for e in episodes})!=1:
            raise ValueError("PPO batch crosses lower policy versions")
        histories={id(e):unpack(e.history) for e in episodes}
        rows=[]
        raw_advantages=[]
        for e in episodes:
            advantage,targets=generalized_advantage(
                [h["reward"] for h in e.high],[h["value"] for h in e.high],
                [h["next_value"] for h in e.high],[h["terminated"] for h in e.high],
                [h["truncated"] for h in e.high],gamma=self.gamma,gae_lambda=self.gae_lambda)
            raw_advantages.extend(advantage)
            for h,G in zip(e.high,targets):
                rows.append((e,h,float(G)))
        if not rows: raise ValueError("PPO needs nonempty rollout windows")
        reports={}
        states=[]
        for e,h,_ in rows:
            key=(id(e),h["report"])
            if key not in reports: reports[key]=e.report(h["report"])
            states.append((reports[key],histories[id(e)]))
        states=PreparedBatch(states)
        advantages=np.asarray(raw_advantages)
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        active=True
        logs=[]
        for epoch in range(6):
            actor_start=copy.deepcopy(self.actor.state_dict()) if active else None
            optimizer_start=copy.deepcopy(self.actor_optimizer.state_dict()) if active else None
            order=torch.randperm(len(rows),generator=streams.generator("replay",400)).tolist()
            accepted=0
            for start in range(0,len(rows),150):
                ids=order[start:start+150]
                self._minibatch(states.select(ids),[rows[index] for index in ids],advantages[ids],active)
                if active: accepted+=1
            if active:
                divergence=self._kl(states,rows)
                if not math.isfinite(divergence): raise FloatingPointError("Nonfinite PPO KL")
                if divergence>.02:
                    self.actor.load_state_dict(actor_start); self.actor_optimizer.load_state_dict(optimizer_start)
                    self.counts["rollbacks"]+=1; active=False
                else: self.counts["actor_steps_accepted"]+=accepted
                logs.append(dict(epoch=epoch+1,kl=divergence,accepted=active))
        self.counts["batches"]+=1; self.version+=1
        return logs
