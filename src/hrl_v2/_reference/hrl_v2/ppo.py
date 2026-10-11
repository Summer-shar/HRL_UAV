"""Undiscounted episodic PPO with epoch-atomic actor/Adam KL rollback."""
import copy
import math
import numpy as np
import torch
from torch import nn
from .replay import unpack


def logprob(z,mean,logstd):
    normal=-.5*((z-mean)/logstd.exp()).square()-logstd-.5*math.log(2*math.pi)
    correction=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
    return (normal-correction).sum(-1)


def gaussian_kl(old_mean,old_logstd,mean,logstd):
    return (logstd-old_logstd+(old_logstd.mul(2).exp()+(old_mean-mean).square())/
            (2*logstd.mul(2).exp())-.5).sum(-1)


def returns(rewards):
    return np.cumsum(np.asarray(rewards,dtype=np.float64)[::-1])[::-1].copy()


class PPO:
    def __init__(self,actor,value):
        self.actor,self.value=actor,value
        self.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
        self.value_optimizer=torch.optim.Adam(value.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8)
        self.counts=dict(batches=0,actor_steps_attempted=0,actor_steps_accepted=0,value_steps=0,rollbacks=0)
        self.version=0

    def update(self,episodes,streams):
        if len(episodes)!=5 or any(not e.complete for e in episodes):
            raise ValueError("PPO needs five complete episodes")
        if len({e.lower_version for e in episodes})!=1:
            raise ValueError("PPO batch crosses lower policy versions")
        histories={id(e):unpack(e.history) for e in episodes}
        rows=[]
        for e in episodes:
            for h,G in zip(e.high,returns([x["reward"] for x in e.high])):
                rows.append((e,h,float(G)))
        advantages=np.asarray([G-h["value"] for _,h,G in rows])
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        active=True
        logs=[]
        device=next(self.actor.parameters()).device
        for epoch in range(6):
            actor_start=copy.deepcopy(self.actor.state_dict()) if active else None
            optimizer_start=copy.deepcopy(self.actor_optimizer.state_dict()) if active else None
            order=torch.randperm(900,generator=streams.generator("replay",400)).tolist()
            accepted=0
            for start in range(0,900,150):
                ids=order[start:start+150]
                if active: self.actor_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                for offset in range(0,150,16):
                    chosen=ids[offset:offset+16]; selected=[rows[index] for index in chosen]
                    states=[(e.report(h["report"]),histories[id(e)]) for e,h,_ in selected]
                    if active:
                        mean,std=self.actor.batch(states)
                        z=mean.new_tensor(np.asarray([h["pre_tanh"] for _,h,_ in selected]))
                        old=mean.new_tensor([h["logprob"] for _,h,_ in selected])
                        ratio=(logprob(z,mean,std)-old).exp()
                        advantage=mean.new_tensor(advantages[chosen])
                        actor_loss=-torch.minimum(ratio*advantage,ratio.clamp(.85,1.15)*advantage).sum()/150
                        actor_loss.backward()
                    value=self.value.batch(states)
                    target=value.new_tensor([G for _,_,G in selected])
                    (.5*(value-target).square().sum()/150).backward()
                if active:
                    nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
                    self.actor_optimizer.step(); accepted+=1
                    self.counts["actor_steps_attempted"]+=1
                nn.utils.clip_grad_norm_(self.value.parameters(),1.,error_if_nonfinite=True)
                self.value_optimizer.step(); self.counts["value_steps"]+=1
            if active:
                divergences=[]
                with torch.no_grad():
                    for start in range(0,900,16):
                        selected=rows[start:start+16]
                        states=[(e.report(h["report"]),histories[id(e)]) for e,h,_ in selected]
                        mean,std=self.actor.batch(states)
                        old_mean=mean.new_tensor(np.asarray([h["mean"] for _,h,_ in selected]))
                        old_std=mean.new_tensor(np.asarray([h["logstd"] for _,h,_ in selected]))
                        divergences.extend(gaussian_kl(old_mean,old_std,mean,std).cpu().reshape(-1).tolist())
                divergence=float(np.mean(divergences))
                if not math.isfinite(divergence): raise FloatingPointError("Nonfinite PPO KL")
                if divergence>.02:
                    self.actor.load_state_dict(actor_start); self.actor_optimizer.load_state_dict(optimizer_start)
                    self.counts["rollbacks"]+=1; active=False
                else: self.counts["actor_steps_accepted"]+=accepted
                logs.append(dict(epoch=epoch+1,kl=divergence,accepted=active))
        self.counts["batches"]+=1; self.version+=1
        return logs
