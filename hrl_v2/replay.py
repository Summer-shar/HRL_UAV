"""Losslessly compressed whole episodes; sampling returns exact IS weights."""
from dataclasses import dataclass, field
import pickle
import zlib
import numpy as np


def pack(value):
    return zlib.compress(pickle.dumps(value,protocol=5),level=1)


def unpack(value):
    return pickle.loads(zlib.decompress(value))


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
    severe: list = field(default_factory=list)
    high: list = field(default_factory=list)
    history: bytes = b""
    metrics: dict = field(default_factory=dict)
    complete: bool = False

    def report(self,index):
        return unpack(self.reports[index])

    def add_report(self,report):
        self.reports.append(pack(report))
        return len(self.reports)-1

    def seal(self,history,metrics):
        if len(self.rewards)!=1800 or len(self.high)!=180 or len(self.successors)!=1800:
            raise ValueError("Only full 900-second episodes can enter replay")
        if sum(self.terminated)!=1 or not self.terminated[-1]:
            raise ValueError("Episode must terminate exactly at its final transition")
        if not np.isclose(sum(self.rewards),-metrics["cost_sum"]/100,rtol=1e-11,atol=1e-9):
            raise ValueError("Micro reward / physical cost identity failed")
        if not np.isclose(sum(h["reward"] for h in self.high),sum(self.rewards),rtol=1e-11,atol=1e-9):
            raise ValueError("Macro / micro reward identity failed")
        self.history=pack(np.asarray(history,dtype=np.float64))
        self.metrics=metrics
        self.complete=True

    @property
    def storage_bytes(self):
        return len(self.history)+sum(map(len,self.reports))


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
        self.writes+=1800

    def sample(self,rng):
        if not self.episodes: raise ValueError("No completed episode to sample")
        result=[]
        for e in rng.integers(len(self.episodes),size=4):
            episode=self.episodes[int(e)]
            q=probabilities(episode.severe)
            for i in rng.choice(1800,size=32,replace=True,p=q):
                result.append(Sample(episode,int(i),float((1/1800)/q[i])))
        return result


def resolve(samples, *, next_state=False):
    """Materialize a sample as a PreparedBatch.

    The batch owns decompressed episode histories and report graph caches for
    the duration of an optimizer step, so current/next encoders can reuse the
    same immutable Python objects and fixed graph tensors.
    """
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
