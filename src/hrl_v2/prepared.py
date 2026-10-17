"""Fixed data owned by one sampled batch; never caches learned encodings."""
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
import numpy as np
import torch
from .features import build, arrival_array


@lru_cache(maxsize=256)
def fixed_graph(report):
    """Preserve the baseline's bounded cache of immutable CPU graph features.

    Batch-local identity lookup avoids repeating the structural report hash
    for every independent encoder. Neither cache contains learned tensors.
    """
    return build(report)


class PreparedBatch:
    def __init__(self, states, *, owner=None, indices=None):
        self.states=list(states)
        self._owner=owner
        self.indices=tuple(range(len(self.states))) if indices is None else tuple(indices)
        if owner is None:
            self.graphs={}
            self.inputs={}
            self.tensors=OrderedDict()

    @property
    def owner(self):
        # A root batch must not own itself: that cycle retains its device
        # tensors until cyclic GC, across otherwise completed optimizer steps.
        return self if self._owner is None else self._owner

    def __len__(self): return len(self.states)
    def __iter__(self): return iter(self.states)
    def __getitem__(self, key):
        if isinstance(key,slice): return self.select(list(range(len(self)))[key])
        return self.states[key]

    def select(self, indices):
        return PreparedBatch([self.states[i] for i in indices],owner=self.owner,
                             indices=[self.indices[i] for i in indices])

    def graph(self, report):
        key=id(report)
        if key not in self.owner.graphs: self.owner.graphs[key]=fixed_graph(report)
        return self.owner.graphs[key]

    def arrivals(self, history, cursor, device):
        if cursor < 0 or cursor > len(history):
            raise ValueError("History cursor outside observed arrivals")
        key=(id(history),str(device))
        cached=self.owner.inputs.get(key)
        if cached is None or len(cached)<cursor:
            # A successor may extend an already normalized prefix by one frame.
            # Normalize/transfer only that suffix; raw inputs carry no gradients.
            begin=0 if cached is None else len(cached)
            delta=arrival_array(history[begin:cursor],cursor-begin,device)
            cached=delta if cached is None else torch.cat((cached,delta))
            self.owner.inputs[key]=cached
        return cached[:cursor]

    @classmethod
    def replay(cls,samples):
        if samples and getattr(samples[0].episode,"encoder","graph")=="lite":
            from .lite_batch import LiteBatch
            return LiteBatch.replay(samples)
        if any(getattr(s.episode,"encoder","graph")!="graph" for s in samples):
            raise ValueError("Cannot mix graph and Lite replay")
        from .replay import unpack
        histories={}; reports={}; states=[]; successors=[]
        for sample in samples:
            e=sample.episode; key=id(e)
            if key not in histories: histories[key]=unpack(e.history)
            for ids,dest in ((e.observations,states),(e.successors,successors)):
                report_key=(key,ids[sample.index])
                if report_key not in reports: reports[report_key]=e.report(ids[sample.index])
                dest.append((reports[report_key],histories[key]))
        current=cls(states)
        following=cls(successors,owner=current,indices=range(len(states),2*len(states)))
        return current,following


class HistoryTape:
    """One GRU graph per episode/encoder/optimizer step, one final history VJP.

    Microbatch graphs consume detached leaves. Their accumulated leaf adjoints
    are propagated through the full, untruncated GRU graph only in backward().
    """
    def __init__(self,encoder,states):
        if not isinstance(states,PreparedBatch): states=PreparedBatch(states)
        self.encoder=encoder; self.outputs={}; self.leaves={}; self.closed=False
        self.versions=tuple((id(p),p._version) for p in encoder.arrival.parameters())
        self.train=torch.is_grad_enabled() and any(p.requires_grad for p in encoder.arrival.parameters())
        device=next(encoder.parameters()).device
        groups={}
        for r,h in states:
            key=id(h)
            if key not in groups or r.history_cursor>groups[key][1]: groups[key]=(h,r.history_cursor)
        for key,(h,cursor) in groups.items():
            if not cursor: continue
            output=encoder.arrival(states.arrivals(h,cursor,device))[0]
            self.outputs[key]=output
            self.leaves[key]=output.detach().requires_grad_(self.train)
        self.zero=next(encoder.parameters()).new_zeros(39,16)

    def get(self,states):
        if self.closed: raise RuntimeError("History tape already consumed")
        for r,h in states:
            if r.history_cursor < 0 or (r.history_cursor and
                (id(h) not in self.leaves or r.history_cursor > len(self.leaves[id(h)]))):
                raise ValueError("State is outside this optimizer-step history tape")
        return [self.leaves[id(h)][r.history_cursor-1] if r.history_cursor else self.zero for r,h in states]

    def backward(self):
        if self.closed: raise RuntimeError("History tape may be backpropagated only once")
        if self.versions != tuple((id(p),p._version) for p in self.encoder.arrival.parameters()):
            raise RuntimeError("Arrival parameters changed within a history tape")
        outputs=[]; adjoints=[]
        if self.train:
            for key,leaf in self.leaves.items():
                if leaf.grad is not None: outputs.append(self.outputs[key]); adjoints.append(leaf.grad)
            if outputs: torch.autograd.backward(outputs,adjoints)
        self.closed=True
        self.outputs.clear(); self.leaves.clear()


@contextmanager
def history_scope(encoder,states):
    if getattr(encoder,"kind",None)=="lite":
        yield None
        return
    if getattr(encoder,"_history_tape",None) is not None: raise RuntimeError("Nested history scope")
    tape=HistoryTape(encoder,states)
    encoder._history_tape=tape
    try: yield tape
    finally:
        encoder._history_tape=None
        tape.closed=True
        tape.outputs.clear(); tape.leaves.clear()


class IncrementalHistory:
    """Inference-only cache with strong episode/history identity and versions."""
    def __init__(self): self.clear()
    def clear(self):
        self.episode=None; self.history=None; self.signature=None; self.cursor=0; self.hidden=None

    @torch.no_grad()
    def get(self,encoder,report,history,episode):
        parameter=next(encoder.parameters())
        # _version changes on Adam, Polyak, load_state_dict, rollback and manual
        # in-place edits. Include every encoder parameter, not only GRU weights.
        signature=(str(parameter.device),parameter.dtype,tuple((id(p),p._version) for p in encoder.parameters()))
        cursor=report.history_cursor
        if cursor<0 or cursor>len(history): raise ValueError("History cursor outside observed arrivals")
        if (episode is not self.episode or history is not self.history or signature!=self.signature
                or cursor<self.cursor):
            self.clear(); self.episode=episode; self.history=history; self.signature=signature
        if cursor>self.cursor:
            delta=arrival_array(history[self.cursor:cursor],cursor-self.cursor,parameter.device)
            _,self.hidden=encoder.arrival(delta,self.hidden)
            self.cursor=cursor
        return self.hidden[0] if self.hidden is not None else parameter.new_zeros(39,16)
