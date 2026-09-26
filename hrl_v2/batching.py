"""Disjoint-graph batching: identical edges and queues, no state truncation."""
from functools import lru_cache
import math
from types import SimpleNamespace
import torch
from torch.nn.utils.rnn import pad_sequence
from .features import build,RELATIONS


@lru_cache(maxsize=256)
def graph_for(report):
    # Only fixed, non-learned features are cached. No learned history hidden
    # state or network output survives an optimizer step.
    return build(report)


def encode(encoder,states):
    if not states: raise ValueError("Cannot encode an empty batch")
    device=next(encoder.parameters()).device
    graphs=[graph_for(r) for r,_ in states]
    hidden=encoder.histories(states)
    kinds=("terminal","parent","fragment","uav")
    widths=(11,14,16,24)
    typed=[]; typed_indices={}; typed_offset=0
    for kind,width in zip(kinds,widths):
        rows=[row for g in graphs for row in g.features[kind]]
        x=torch.tensor(rows,device=device,dtype=torch.float32).reshape(-1,width)
        if kind=="terminal": x=torch.cat((x,torch.cat(hidden)),1)
        typed.append(encoder.nodes[kind](x))
        cursor=typed_offset
        for i,g in enumerate(graphs):
            n=len(g.features[kind]); typed_indices[i,kind]=list(range(cursor,cursor+n)); cursor+=n
        typed_offset=cursor
    # Convert type-major embedding order to disjoint graph-major node order.
    order=[n for i in range(len(graphs)) for kind in kinds for n in typed_indices[i,kind]]
    h=torch.cat(typed)[torch.tensor(order,device=device)]
    queues={k:[] for k in ("upload","ready","transfer","cpu")}
    edges={k:[] for k in RELATIONS}
    pools=[]; geometry=[]; uavs=[]; clocks=[]
    offset=0
    for g in graphs:
        for kind,qs in g.queues.items(): queues[kind].extend([[v+offset for v in q] for q in qs])
        for kind,es in g.edges.items(): edges[kind].extend([(s+offset,d+offset,x) for s,d,x in es])
        pools.extend([[[v+offset for v in pool] for pool in ps] for ps in g.pools])
        geometry.extend(g.geometry); uavs.extend(v+offset for v in g.uavs); clocks.extend([g.clock]*3)
        offset+=len(g.neighbors)
    neighbors=[[] for _ in range(offset)]; edge_offset=0
    for kind in RELATIONS:
        for i,(_,d,_) in enumerate(edges[kind]): neighbors[d].append(edge_offset+i)
        edge_offset+=len(edges[kind])
    all_ids=[]; all_contexts=[]
    for kind,qs in queues.items():
        qs=[q for q in qs if q]
        if not qs: continue
        padded=pad_sequence([h[q] for q in qs])
        out,_=encoder.queues["queue_"+kind](padded)
        for i,q in enumerate(qs):
            all_ids.extend(q); all_contexts.append(out[:len(q),i])
    if all_ids:
        ids=torch.tensor(all_ids,device=device)
        updated=encoder.queue_norm(encoder.queue_projection(torch.cat((h[ids],torch.cat(all_contexts)),1)))
        h=h.index_copy(0,ids,updated)
    merged=SimpleNamespace(edges=edges,neighbors=neighbors)
    for layer in encoder.graph: h=layer(h,merged)
    uavs=torch.tensor(uavs,device=device)
    own=h[uavs]; read=[]
    for k in range(3):
        layer=encoder.readouts[k]
        ps=[p[k] for p in pools]; width=max(1,max(map(len,ps)))
        ids=torch.tensor([p+[0]*(width-len(p)) for p in ps],device=device)
        counts=torch.tensor([len(p) for p in ps],device=device)
        mask=torch.arange(width,device=device)[None]<counts[:,None]
        scores=(layer.q(own)[:,None]*layer.k(h)[ids]).sum(-1)/8
        # An empty set gets a dummy finite score; its entire output is masked
        # back to zero. No empty softmax is ever evaluated.
        valid=mask.clone(); valid[counts==0,0]=True
        probability=scores.masked_fill(~valid,-torch.inf).softmax(1)*mask
        read.append((probability[:,:,None]*layer.v(h)[ids]).sum(1))
    extra=h.new_tensor([[math.log1p(len(p)) for p in ps]+clock for ps,clock in zip(pools,clocks)])
    v=encoder.uav_projection(torch.cat((own,*read,extra),1))
    ts=torch.tensor([p[0] for p in pools],device=device)
    terminals=h[ts]
    scores=(encoder.geo_q(v)[:,None]*encoder.geo_k(terminals)).sum(-1)/math.sqrt(128)
    scores=scores+encoder.geo_bias(h.new_tensor(geometry)).squeeze(-1)
    context=(scores.softmax(1)[:,:,None]*encoder.geo_v(terminals)).sum(1)
    v=encoder.geo_merge(torch.cat((v,context),1)).reshape(len(states),3,128)
    for layer in encoder.uav_attention: v=layer(v)
    clock=h.new_tensor([g.clock for g in graphs])
    global_state=encoder.global_projection(torch.cat((v.mean(1),v.max(1).values,clock),1))
    ready=h.new_tensor([g.ready for g in graphs])
    return v,global_state,ready
