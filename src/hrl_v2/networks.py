"""Independent report/history encoders and shared-per-UAV/action-row heads."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .features import build, arrival_array, RELATIONS, EDGE_WIDTHS
from .hybrid_simplex import SUPPORTS, concentration_from_raw, action_features
from .prepared import IncrementalHistory


def mlp(*widths):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(widths)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def initialize(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.orthogonal_(m.weight,math.sqrt(2))
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.GRU):
            for name,p in m.named_parameters():
                if "bias" in name: nn.init.zeros_(p)
                else:
                    for gate in p.chunk(3):
                        if "weight_ih" in name: nn.init.xavier_uniform_(gate,gain=1)
                        else: nn.init.orthogonal_(gate,gain=1)
        elif isinstance(m,nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,0,.02)


class TypedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(64,64,bias=False)
        self.keys=nn.ModuleList(nn.Linear(64,64,bias=False) for _ in RELATIONS)
        self.values=nn.ModuleList(nn.Linear(64,64,bias=False) for _ in RELATIONS)
        self.edge=nn.ModuleDict({str(i):nn.Linear(d,64,bias=False) for i,d in enumerate(EDGE_WIDTHS) if d})
        self.bias=nn.Parameter(torch.zeros(len(RELATIONS),4))
        self.output=nn.Linear(64,64)
        self.norm1=nn.LayerNorm(64,eps=1e-5); self.norm2=nn.LayerNorm(64,eps=1e-5)
        self.ff=mlp(64,128,64)

    def forward(self,h,graph):
        from .batching import prepare_attention
        if not hasattr(graph,"relations"):
            graph=prepare_attention(graph.edges,graph.neighbors,h.device)
        keys,values,biases=[],[],[]
        for i,(source,edge_values) in enumerate(graph.relations):
            if source.numel()==0: continue
            k=self.keys[i](h[source])
            if str(i) in self.edge: k=k+self.edge[str(i)](edge_values)
            keys.append(k); values.append(self.values[i](h[source]))
            biases.append(self.bias[i].expand(source.numel(),-1))
        keys=torch.cat(keys).view(-1,4,16)
        values=torch.cat(values).view(-1,4,16)
        biases=torch.cat(biases)
        query=self.query(h).view(-1,4,16)
        outputs=[]
        for nodes,ids,mask in graph.buckets:
            score=(query[nodes,None]*keys[ids]).sum(-1)/4 + biases[ids]
            probability=score.masked_fill(~mask[:,:,None],-torch.inf).softmax(1)
            outputs.append((probability[:,:,:,None]*values[ids]).sum(1).reshape(-1,64))
        output=torch.cat(outputs)[graph.inverse]
        h=self.norm1(h+self.output(output))
        return self.norm2(h+self.ff(h))


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(64,64); self.k=nn.Linear(64,64); self.v=nn.Linear(64,64)

    def forward(self,query,items):
        if items.shape[0]==0: return query.new_zeros(64)
        weights=(self.k(items)*self.q(query)).sum(-1).div(8).softmax(0)
        return (weights[:,None]*self.v(items)).sum(0)


class UAVAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(128,128); self.k=nn.Linear(128,128); self.v=nn.Linear(128,128)
        self.out=nn.Linear(128,128); self.ff=mlp(128,256,128)
        self.n1=nn.LayerNorm(128,eps=1e-5); self.n2=nn.LayerNorm(128,eps=1e-5)

    def forward(self,x):
        q,k,v=[f(x).reshape(*x.shape[:-1],4,32).transpose(-3,-2) for f in (self.q,self.k,self.v)]
        a=(q@k.transpose(-1,-2)/math.sqrt(32)).softmax(-1)
        y=(a@v).transpose(-3,-2).reshape(x.shape)
        x=self.n1(x+self.out(y))
        return self.n2(x+self.ff(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.arrival=nn.GRU(6,16)
        self.nodes=nn.ModuleDict({k:mlp(w,64,64) for k,w in dict(terminal=27,parent=14,fragment=16,uav=24).items()})
        self.queues=nn.ModuleDict({"queue_"+k:nn.GRU(64,64) for k in ("upload","ready","transfer","cpu")})
        self.queue_projection=nn.Linear(128,64); self.queue_norm=nn.LayerNorm(64,eps=1e-5)
        self.graph=nn.ModuleList(TypedAttention() for _ in range(3))
        self.readouts=nn.ModuleList(Readout() for _ in range(3))
        self.uav_projection=mlp(261,256,128)
        self.geo_q=nn.Linear(128,128); self.geo_k=nn.Linear(64,128); self.geo_v=nn.Linear(64,128)
        self.geo_bias=mlp(6,32,1); self.geo_merge=mlp(256,128,128)
        self.uav_attention=nn.ModuleList(UAVAttention() for _ in range(2))
        self.global_projection=mlp(258,256,128)
        # Inference-only state.  It is deliberately per encoder: independently
        # trainable actor/critic encoders must never share hidden tensors.
        self._incremental_history = IncrementalHistory()

    def histories(self,states):
        """Recompute each observed prefix once per microbatch, with full BPTT."""
        tape=getattr(self,"_history_tape",None)
        if tape is not None: return tape.get(states)
        device=next(self.parameters()).device
        groups={}
        for r,h in states: groups.setdefault(id(h),[]).append((r,h))
        outputs={}
        for key,items in groups.items():
            cursor=max(r.history_cursor for r,_ in items)
            arrivals=arrival_array(items[0][1],cursor,device)
            outputs[key]=self.arrival(arrivals)[0] if cursor else None
        zero=next(self.parameters()).new_zeros(39,16)
        return [outputs[id(h)][r.history_cursor-1] if r.history_cursor else zero for r,h in states]

    def forward(self,report,history,arrival_hidden=None):
        from .batching import encode
        v,g,ready=encode(self,[(report,history)],None if arrival_hidden is None else [arrival_hidden])
        return v[0],g[0],ready[0]

    @torch.no_grad()
    def infer(self, report, history, episode):
        """Run one online observation, extending the arrival GRU by its delta."""
        hidden = self._incremental_history.get(self, report, history, episode)
        return self.forward(report, history, hidden)

    def reset_inference(self):
        self._incremental_history.clear()


class LowerActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder(); self.priority=nn.Embedding(2,16)
        self.row=mlp(275,256,128); self.support=mlp(386,128,1); self.concentration=mlp(386,128,1)
        initialize(self)
        nn.init.zeros_(self.support[-1].weight); nn.init.zeros_(self.support[-1].bias)
        nn.init.zeros_(self.concentration[-1].weight)
        nn.init.constant_(self.concentration[-1].bias,math.log(math.expm1(1)))

    def forward(self,report,history,arrival_hidden=None,episode=None):
        v,g,ready=(self.encoder.infer(report,history,episode) if episode is not None and arrival_hidden is None
                   else self.encoder(report,history,arrival_hidden))
        logits,kappa=self.decode(v[None],g[None],ready[None])
        return logits[0],kappa[0]

    def batch(self,states):
        from .batching import encode
        return self.decode(*encode(self.encoder,states))

    def decode(self,v,g,ready):
        batch=v.shape[0]
        sources=torch.arange(6,device=v.device)%3
        priorities=torch.arange(6,device=v.device)//3
        rows=self.row(torch.cat((g[:,None].expand(-1,6,-1),v[:,sources],
            self.priority(priorities)[None].expand(batch,-1,-1),ready),-1))
        support_mask=v.new_tensor([[float(j in m) for j in range(3)] for m in SUPPORTS])
        count=support_mask.sum(-1,keepdim=True)
        sums=support_mask@v
        means=sums/count
        complement=(1-support_mask)@v/(3-count).clamp_min(1)
        size=(count/3)[None,None].expand(batch,6,-1,-1)
        inclusion=support_mask[:,sources].T[None,:,:,None].expand(batch,-1,-1,-1)
        logits=self.support(torch.cat((rows[:,:,None].expand(-1,-1,7,-1),sums[:,None].expand(-1,6,-1,-1),
            complement[:,None].expand(-1,6,-1,-1),size,inclusion),-1)).squeeze(-1)
        same_source=(sources[:,None]==torch.arange(3,device=v.device)[None]).float()
        raw=self.concentration(torch.cat((rows[:,:,None,None].expand(-1,-1,7,3,-1),v[:,None,None].expand(-1,6,7,-1,-1),
            means[:,None,:,None].expand(-1,6,-1,3,-1),size[:,:,:,None].expand(-1,-1,-1,3,-1),
            same_source[None,:,None,:,None].expand(batch,-1,7,-1,-1)),-1)).squeeze(-1)
        return logits.to(device="cpu",dtype=torch.float64),concentration_from_raw(raw.to("cpu"))


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.head=mlp(548,256,256,1)
        initialize(self); nn.init.orthogonal_(self.head[-1].weight,1.)
        self.last_action_cast_error=0.

    def encode(self,report,history,arrival_hidden=None):
        v,g,_=self.encoder(report,history,arrival_hidden)
        return torch.cat((v.flatten(),g))

    def encode_batch(self,states):
        from .batching import encode
        v,g,_=encode(self.encoder,states)
        return torch.cat((v.flatten(1),g),1)

    def value(self,encoded,actions):
        features,error=action_features(actions,encoded.device)
        self.last_action_cast_error=max(self.last_action_cast_error,error)
        return self.value_features(encoded,features)

    def state_projection(self,encoded):
        return torch.nn.functional.linear(encoded,self.head[0].weight[:,:512],self.head[0].bias)

    def value_features(self,encoded,features,projected=None):
        projected=self.state_projection(encoded) if projected is None else projected
        if features.ndim==3: projected=projected[:,None]
        action=torch.nn.functional.linear(features,self.head[0].weight[:,512:])
        return self.head[1:](projected+action).squeeze(-1)


class UpperActor(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.mean=mlp(128,64,2); self.std=mlp(128,64,2)
        initialize(self)
        nn.init.zeros_(self.mean[-1].weight); nn.init.zeros_(self.mean[-1].bias)
        nn.init.zeros_(self.std[-1].weight)
        nn.init.constant_(self.std[-1].bias,math.atanh((math.log(.15)+2.5)/2.5))

    def forward(self,report,history,arrival_hidden=None,episode=None):
        v,_,_= (self.encoder.infer(report,history,episode) if episode is not None and arrival_hidden is None
                else self.encoder(report,history,arrival_hidden))
        return self.mean(v).flatten(),(-2.5+2.5*self.std(v).tanh()).flatten()

    def batch(self,states):
        from .batching import encode
        v,_,_=encode(self.encoder,states)
        return self.mean(v).flatten(1),(-2.5+2.5*self.std(v).tanh()).flatten(1)


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.head=mlp(128,128,1)
        initialize(self); nn.init.orthogonal_(self.head[-1].weight,1.)

    def forward(self,report,history,arrival_hidden=None,episode=None):
        _,g,_=(self.encoder.infer(report,history,episode) if episode is not None and arrival_hidden is None
               else self.encoder(report,history,arrival_hidden))
        return self.head(g).squeeze(-1)

    def batch(self,states):
        from .batching import encode
        _,g,_=encode(self.encoder,states)
        return self.head(g).squeeze(-1)
