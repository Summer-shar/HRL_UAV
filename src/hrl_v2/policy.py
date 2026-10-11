"""Execution policies accept causal report data, never the simulator."""
import numpy as np
import torch
from .hybrid_simplex import sample,deterministic_decode
from .ppo import logprob


class Policy:
    def __init__(self,lower,upper,value,streams,*,mode="moving",stochastic=True,deployment=None,upper_stochastic=None):
        self.lower_actor,self.upper_actor,self.value=lower,upper,value
        self.streams,self.mode,self.stochastic=streams,mode,stochastic
        self.upper_stochastic=stochastic if upper_stochastic is None else upper_stochastic
        if mode == "centroid" and deployment is None:
            raise ValueError("Centroid policy requires the scene deployment configuration")
        self.deployment=deployment
        self.lower_override=None
        self.observer=None
        if lower is not None and getattr(lower.encoder,"kind",None)=="lite":
            from .lite_observation import LiteObserver
            self.observer=LiteObserver()
        self.begin_episode(None)

    def begin_episode(self,identity):
        # Identity is owned by this collection, not a value such as a repeated
        # scenario number. Parameter versions are checked by each encoder.
        self.episode_identity=identity
        self.entropy_sum=0.
        self.entropy_count=0
        self._episode=object()
        if self.observer is not None: self.observer.reset()
        for network in (self.lower_actor,self.upper_actor,self.value):
            if network is not None: network.encoder.reset_inference()

    @torch.no_grad()
    def upper(self,report,history):
        observation=self.observer.update(report,history) if self.observer is not None else None
        if self.mode=="hover": return np.zeros(6),{}
        if self.mode=="centroid":
            from .baselines import centroid
            return centroid(report,history,max_target_step_m=self.deployment["max_target_step_m"]),{}
        state=observation.upper if observation is not None else report
        mean,std=self.upper_actor(state,history,episode=self._episode)
        noise=torch.randn(6,generator=self.streams.generator("upper",self.streams.policy_stream)).to(mean.device) if self.upper_stochastic else torch.zeros_like(mean)
        z=mean+std.exp()*noise
        value=self.value(state,history,episode=self._episode).item() if self.value is not None else 0.
        return z.tanh().cpu().numpy(),dict(pre_tanh=z.cpu().numpy(),mean=mean.cpu().numpy(),
            logstd=std.cpu().numpy(),logprob=logprob(z,mean,std).item(),value=value)

    @torch.no_grad()
    def lower(self,report,history):
        observation=self.observer.update(report,history) if self.observer is not None else None
        if self.lower_override is not None: return self.lower_override(report,history)
        logits,kappa=self.lower_actor(observation.lower if observation is not None else report,history,episode=self._episode)
        if self.stochastic:
            actions,ids,ell=sample(logits[None],kappa[None],self.streams,self.streams.policy_stream)
            # Reuse the executed sample: no extra draw or parameter update.
            self.entropy_sum-=float(ell.sum())
            self.entropy_count+=1
        else:
            actions=deterministic_decode(logits[None],kappa[None]); ids=logits.argmax(-1)[None]
        return actions[0].numpy().reshape(2,3,3),ids[0].numpy()
