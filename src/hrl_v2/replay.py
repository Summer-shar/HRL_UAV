"""Losslessly compressed rollout windows; sampling returns exact IS weights."""
from dataclasses import dataclass, field
import pickle
import zlib
import numpy as np


def pack(value):
    return zlib.compress(pickle.dumps(value,protocol=5),level=1)


def unpack(value):
    return pickle.loads(zlib.decompress(value))


@dataclass(frozen=True)
class LiteReportKey:
    """Immutable boundary identity; collected Lite vectors hold observations."""
    tick: int
    history_cursor: int
    phase: str
    remaining_macro: float
    remaining_adjustment: float


@dataclass
class Episode:
    upper_version: str
    lower_version: int
    identity: dict
    reports: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminated: list = field(default_factory=list)
    truncated: list = field(default_factory=list)
    severe: list = field(default_factory=list)
    high: list = field(default_factory=list)
    history: bytes = b""
    metrics: dict = field(default_factory=dict)
    complete: bool = False
    expected_lower_steps: int = None
    expected_upper_steps: int = None
    reward_scale: float = 100.
    gamma_lower: float = 1.
    encoder: str = "graph"
    lower_obs: object = field(default_factory=list)
    upper_obs: object = field(default_factory=list)

    def report(self,index):
        return unpack(self.reports[index])

    def add_report(self,report):
        self.reports.append(LiteReportKey(report.tick, report.history_cursor, report.phase,
            report.remaining_macro, report.remaining_adjustment) if self.encoder=="lite" else pack(report))
        return len(self.reports)-1

    def seal(self,history,metrics):
        if not np.isfinite(self.reward_scale) or self.reward_scale<=0:
            raise ValueError("Reward scale must be finite and positive")
        if not np.isfinite(self.gamma_lower) or not 0<=self.gamma_lower<=1:
            raise ValueError("Lower reward discount must be finite and lie in [0, 1]")
        count=len(self.rewards)
        fields=(self.observations,self.successors,self.actions,self.supports,
                self.terminated,self.truncated,self.severe)
        if not count or any(len(field)!=count for field in fields):
            raise ValueError("A sealed rollout requires every transition and its real successor")
        if ((self.expected_lower_steps is not None and count!=self.expected_lower_steps) or
            (self.expected_upper_steps is not None and len(self.high)!=self.expected_upper_steps)):
            raise ValueError("Rollout counts differ from the configured clocks")
        indices=self.observations+self.successors+[h["report"] for h in self.high]+[h["successor"] for h in self.high]
        if any(not isinstance(i,(int,np.integer)) or not 0<=i<len(self.reports) for i in indices):
            raise ValueError("Rollout contains an invalid observation or successor reference")
        if any(self.terminated) or sum(self.truncated)!=1 or not self.truncated[-1]:
            raise ValueError("Continuing service truncates only the final transition; it never terminates")
        if not self.high or any(h["terminated"] for h in self.high) or \
            sum(h["truncated"] for h in self.high)!=1 or not self.high[-1]["truncated"]:
            raise ValueError("Upper rollout boundaries must match continuing-service semantics")
        # Validate the stored data, not just the collector's intent. In
        # particular a reset report must never become a truncated successor.
        cached={}
        def read(index):
            if not isinstance(index,(int,np.integer)) or not 0<=index<len(self.reports):
                raise ValueError("Invalid rollout report reference")
            if index not in cached: cached[index]=self.reports[index] if self.encoder=="lite" else self.report(index)
            return cached[index]
        stride=None
        for index,(source,target) in enumerate(zip(self.observations,self.successors)):
            before,after=read(source),read(target)
            delta=after.tick-before.tick
            if delta<=0 or (stride is not None and delta!=stride):
                raise ValueError("A successor must advance the same physical timeline, never reset")
            stride=delta
            if hasattr(before,"history_cursor") and not (0<=before.history_cursor<=after.history_cursor<=len(history)):
                raise ValueError("Successor history must be a causal continuation")
            if index+1<count and after!=read(self.observations[index+1]):
                raise ValueError("Stored successor differs from the next actual observation")
        for index,high in enumerate(self.high):
            before,after=read(high["report"]),read(high["successor"])
            if after.tick<=before.tick:
                raise ValueError("Upper successor must advance the physical timeline")
            if index+1<len(self.high) and after!=read(self.high[index+1]["report"]):
                raise ValueError("Upper successor differs from the next decision report")
        if not np.isclose(sum(self.rewards),-metrics["cost_sum"]/self.reward_scale,rtol=1e-11,atol=1e-9):
            raise ValueError("Micro reward / physical cost identity failed")
        if count%len(self.high):
            raise ValueError("Each upper transition must contain the same number of lower transitions")
        lower_per_upper=count//len(self.high)
        discounts=self.gamma_lower**np.arange(lower_per_upper,dtype=np.float64)
        for index,high in enumerate(self.high):
            begin=index*lower_per_upper
            lower=np.asarray(self.rewards[begin:begin+lower_per_upper],dtype=np.float64)
            if (read(high["report"]).tick!=read(self.observations[begin]).tick or
                read(high["successor"]).tick!=read(self.successors[begin+lower_per_upper-1]).tick):
                raise ValueError("Upper reward span differs from its lower transitions")
            if not np.isclose(high.get("reward_physical",np.nan),lower.sum(),rtol=1e-11,atol=1e-9):
                raise ValueError("Macro physical / micro reward identity failed")
            if not np.isclose(high.get("reward_learning",np.nan),discounts@lower,rtol=1e-11,atol=1e-9):
                raise ValueError("Macro learning reward differs from discounted lower rewards")
        self.history=pack(np.asarray(history,dtype=np.float64))
        self.metrics=metrics
        if self.encoder=="lite":
            from .lite_observation import LOWER_DIM, terminals_in_upper, upper_dim
            self.lower_obs=np.asarray(self.lower_obs,dtype=np.float32)
            self.upper_obs=np.asarray(self.upper_obs,dtype=np.float32)
            width=self.upper_obs.shape[-1] if self.upper_obs.ndim==2 else -1
            if (self.lower_obs.shape!=(count+1,LOWER_DIM) or self.upper_obs.ndim!=2 or
                self.upper_obs.shape!=(len(self.high)+1,upper_dim(terminals_in_upper(width))) or
                not np.isfinite(self.lower_obs).all() or not np.isfinite(self.upper_obs).all()):
                raise ValueError("Lite replay requires all collected observations and true successors")
            self.lower_obs.setflags(write=False); self.upper_obs.setflags(write=False)
            # Boundary keys have validated physical/reward checks above.
            # They remain available to collection audit hooks, not in replay.
            self.reports=[]; self.history=b""
        elif self.encoder!="graph":
            raise ValueError("Unknown replay encoder")
        self.complete=True

    @property
    def storage_bytes(self):
        return (len(self.history)+sum(map(len,self.reports))+
                (self.lower_obs.nbytes+self.upper_obs.nbytes if self.encoder=="lite" else 0))


@dataclass
class Sample:
    episode: Episode
    index: int
    weight: float


def probabilities(severe):
    severe=np.asarray(severe,dtype=bool)
    if not len(severe): raise ValueError("Empty episode")
    if not severe.any(): return np.full(len(severe),1/len(severe))
    return .9/len(severe)+.1*severe/severe.sum()


class Replay:
    def __init__(self):
        self.episodes=[]
        self.upper_version=None
        self.writes=0
        self.flushes=0

    def set_upper_version(self,version):
        if version!=self.upper_version:
            self.episodes.clear()
            self.upper_version=version
            self.flushes+=1

    def add(self,episode):
        if not episode.complete or episode.upper_version!=self.upper_version:
            raise ValueError("Incomplete episode or upper policy version mismatch")
        self.episodes.append(episode)
        if len(self.episodes)>55: self.episodes.pop(0)
        self.writes+=len(episode.rewards)

    def sample(self,rng):
        if not self.episodes: raise ValueError("No completed episode to sample")
        result=[]
        for e in rng.integers(len(self.episodes),size=4):
            episode=self.episodes[int(e)]
            q=probabilities(episode.severe)
            count=len(episode.rewards)
            for i in rng.choice(count,size=32,replace=True,p=q):
                result.append(Sample(episode,int(i),float((1/count)/q[i])))
        return result


def resolve(samples, *, next_state=False):
    """Materialize a sample as a PreparedBatch.

    The batch owns decompressed episode histories and report graph caches for
    the duration of an optimizer step, so current/next encoders can reuse the
    same immutable Python objects and fixed graph tensors.
    """
    if samples and getattr(samples[0].episode,"encoder","graph")=="lite":
        from .lite_batch import LiteBatch
        return LiteBatch.replay(samples)[int(next_state)]
    if any(getattr(s.episode,"encoder","graph")!="graph" for s in samples):
        raise ValueError("Cannot mix graph and Lite replay")
    from .prepared import PreparedBatch
    histories={}
    reports={}
    result=[]
    for s in samples:
        key=id(s.episode)
        if key not in histories: histories[key]=unpack(s.episode.history)
        indices=s.episode.successors if next_state else s.episode.observations
        report_key=(key, indices[s.index])
        if report_key not in reports:
            reports[report_key]=s.episode.report(indices[s.index])
        result.append((reports[report_key],histories[key]))
    return PreparedBatch(result)
