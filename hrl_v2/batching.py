"""Disjoint-graph batching: identical edges and queues, no state truncation."""
import math
from types import SimpleNamespace
import torch
from torch.nn.utils.rnn import pad_sequence
from .features import build,RELATIONS
from .prepared import PreparedBatch


def graph_for(report):
    return build(report)


def prepare(states,device):
    if not isinstance(states,PreparedBatch): states=PreparedBatch(states)
    key=(states.indices,str(device))
    cache=states.owner.tensors
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    graphs=[states.graph(r) for r,_ in states]
    kinds=("terminal","parent","fragment","uav")
    widths=(11,14,16,24)
    features={}; typed_indices={}; typed_offset=0
    for kind,width in zip(kinds,widths):
        rows=[row for g in graphs for row in g.features[kind]]
        features[kind]=torch.tensor(rows,device=device,dtype=torch.float32).reshape(-1,width)
        cursor=typed_offset
        for i,g in enumerate(graphs):
            n=len(g.features[kind]); typed_indices[i,kind]=list(range(cursor,cursor+n)); cursor+=n
        typed_offset=cursor
    order=torch.tensor([n for i in range(len(graphs)) for kind in kinds for n in typed_indices[i,kind]],device=device)
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
    queue_data={}
    for kind,qs in queues.items():
        qs=[q for q in qs if q]
        if not qs: continue
        lengths=[len(q) for q in qs]; width=max(lengths)
        # Sequence-major padding, gathered in a single operation.
        padded=torch.tensor([q+[0]*(width-len(q)) for q in qs],device=device).T
        mask=torch.arange(width,device=device)[:,None]<torch.tensor(lengths,device=device)[None,:]
        ids=torch.tensor([i for q in qs for i in q],device=device)
        queue_data[kind]=(padded,mask,ids,lengths)
    attention=prepare_attention(edges,neighbors,device)
    uavs=torch.tensor(uavs,device=device)
    pool_data=[]
    for k in range(3):
        ps=[p[k] for p in pools]; width=max(1,max(map(len,ps)))
        ids=torch.tensor([p+[0]*(width-len(p)) for p in ps],device=device)
        counts=torch.tensor([len(p) for p in ps],device=device)
        mask=torch.arange(width,device=device)[None]<counts[:,None]
        valid=mask.clone(); valid[counts==0,0]=True
        pool_data.append((ids,mask,valid))
    tensor=lambda data:torch.tensor(data,device=device,dtype=torch.float32)
    prepared=SimpleNamespace(features=features,order=order,queues=queue_data,attention=attention,uavs=uavs,
        pools=pool_data,extra=tensor([[math.log1p(len(p)) for p in ps]+clock for ps,clock in zip(pools,clocks)]),
        terminals=torch.tensor([p[0] for p in pools],device=device),geometry=tensor(geometry),
        clock=tensor([g.clock for g in graphs]),ready=tensor([g.ready for g in graphs]),count=len(states))
    cache[key]=prepared
    # Two device microbatches suffice for current/next state reuse; CPU graphs
    # and normalized histories remain batch-owned for the full optimizer step.
    while len(cache)>2: cache.popitem(last=False)
    return prepared


def prepare_attention(edges,neighbors,device):
    relations=[]
    for kind in RELATIONS:
        rows=edges[kind]
        source=torch.tensor([r[0] for r in rows],device=device,dtype=torch.long)
        values=torch.tensor([r[2] for r in rows],device=device,dtype=torch.float32)
        relations.append((source,values))
    buckets={}
    for i,ns in enumerate(neighbors): buckets.setdefault(1<<(len(ns)-1).bit_length(),[]).append(i)
    prepared=[]; order=[]
    for width,nodes in buckets.items():
        ids=torch.tensor([neighbors[n]+[0]*(width-len(neighbors[n])) for n in nodes],device=device)
        mask=torch.arange(width,device=device)[None,:]<torch.tensor([len(neighbors[n]) for n in nodes],device=device)[:,None]
        prepared.append((torch.tensor(nodes,device=device),ids,mask)); order.extend(nodes)
    return SimpleNamespace(relations=relations,buckets=prepared,inverse=torch.tensor(order,device=device).argsort())


def encode(encoder,states,hidden=None):
    if not states: raise ValueError("Cannot encode an empty batch")
    if not isinstance(states,PreparedBatch): states=PreparedBatch(states)
    device=next(encoder.parameters()).device
    g=prepare(states,device)
    hidden=encoder.histories(states) if hidden is None else hidden
    typed=[]
    for kind in ("terminal","parent","fragment","uav"):
        x=g.features[kind]
        if kind=="terminal": x=torch.cat((x,torch.cat(hidden)),1)
        typed.append(encoder.nodes[kind](x))
    h=torch.cat(typed)[g.order]
    ids=[]; contexts=[]
    for kind,(padded,mask,index,lengths) in g.queues.items():
        out,_=encoder.queues["queue_"+kind](h[padded]*mask[:,:,None])
        contexts.extend(out[:n,i] for i,n in enumerate(lengths)); ids.append(index)
    if ids:
        ids=torch.cat(ids)
        updated=encoder.queue_norm(encoder.queue_projection(torch.cat((h[ids],torch.cat(contexts)),1)))
        h=h.index_copy(0,ids,updated)
    for layer in encoder.graph: h=layer(h,g.attention)
    own=h[g.uavs]; read=[]
    for layer,(ids,mask,valid) in zip(encoder.readouts,g.pools):
        scores=(layer.q(own)[:,None]*layer.k(h)[ids]).sum(-1)/8
        probability=scores.masked_fill(~valid,-torch.inf).softmax(1)*mask
        read.append((probability[:,:,None]*layer.v(h)[ids]).sum(1))
    v=encoder.uav_projection(torch.cat((own,*read,g.extra),1))
    terminals=h[g.terminals]
    scores=(encoder.geo_q(v)[:,None]*encoder.geo_k(terminals)).sum(-1)/math.sqrt(128)
    scores=scores+encoder.geo_bias(g.geometry).squeeze(-1)
    context=(scores.softmax(1)[:,:,None]*encoder.geo_v(terminals)).sum(1)
    v=encoder.geo_merge(torch.cat((v,context),1)).reshape(g.count,3,128)
    for layer in encoder.uav_attention: v=layer(v)
    global_state=encoder.global_projection(torch.cat((v.mean(1),v.max(1).values,g.clock),1))
    return v,global_state,g.ready
