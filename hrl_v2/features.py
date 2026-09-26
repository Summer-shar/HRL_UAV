"""Fixed feature order and sparse typed relationships from ReportV2 only."""
from dataclasses import dataclass
import math
import numpy as np
import torch

RELATIONS = ("t_u","u_t","p_t","t_p","p_u","u_p","p_f","f_p",
             "f_s","s_f","f_d","d_f","prev","next","link","self")
EDGE_WIDTHS = (4,4,0,0,0,0,0,0,0,0,0,0,0,0,9,0)


def L(x, scale=1.):
    a = np.asarray(x,dtype=np.float64)
    if not np.isfinite(a).all() or (a<0).any():
        raise ValueError("Features contain negative or nonfinite physical values")
    return np.log1p(a/scale)


def one(index, size):
    return [float(i==index) for i in range(size)]


@dataclass
class Graph:
    features: dict
    queues: dict
    edges: dict
    neighbors: list
    uavs: list
    pools: list
    geometry: list
    ready: object
    clock: list


def build(report):
    r = report
    parent = {p.key:p for p in r.parents}
    fragment = {f.key:f for f in r.fragments}
    t_index = {t.key:i for i,t in enumerate(r.terminals)}
    p_index = {p.key:39+i for i,p in enumerate(r.parents)}
    f_index = {f.key:39+len(parent)+i for i,f in enumerate(r.fragments)}
    u_index = [39+len(parent)+len(fragment)+i for i in range(3)]
    nodes = u_index[-1]+1
    features = dict(terminal=[],parent=[],fragment=[],uav=[])
    for t in r.terminals:
        features["terminal"].append([*np.asarray(t.position)/r.map_size,L(t.rate,1e6).item(),
            *[value for p in range(2) for value in (
                L(t.upload[p][0],1e6).item(),L(t.upload[p][1],1e9).item(),
                L(t.upload[p][2]).item(),L(t.upload[p][3],10).item())]])
    for p in r.parents:
        valid = p.routed_at is not None
        features["parent"].append([*one(p.priority,2),L(p.bits,1e6).item(),L(p.cycles,1e9).item(),
            L(p.upload_remaining,1e6).item(),L(r.time-p.birth,10).item(),
            *one(("ground","ready","routed").index(p.stage),3),float(valid),
            L(len(p.children)).item(),L(p.completed_children).item(),
            L(r.time-p.routed_at,10).item() if valid else 0.,float(valid)])
    queues = {k:[] for k in ("upload","ready","transfer","cpu")}
    queue_positions = {}
    edges = {k:[] for k in RELATIONS}
    def edge(kind, source, destination, values=()):
        edges[kind].append((source,destination,tuple(values)))
    for q in r.queues:
        index = p_index if q.kind in ("upload","ready") else f_index
        ids = [index[k] for k in q.items]
        queues[q.kind].append(ids)
        ahead = 0.
        for rank,key in enumerate(q.items):
            if q.kind in ("cpu","transfer"):
                queue_positions[key] = (rank,ahead,q)
                f = fragment[key]
                ahead += f.compute_remaining if q.kind=="cpu" else f.transfer_remaining
        for a,b in zip(ids,ids[1:]):
            edge("prev",a,b)
            edge("next",b,a)
    high_work = {}
    for q in r.queues:
        if q.kind in ("cpu","transfer") and q.owner[0]==0:
            high_work[q.kind,q.owner[1:]] = sum(fragment[k].compute_remaining if q.kind=="cpu"
                else fragment[k].transfer_remaining for k in q.items)
    for f in r.fragments:
        p = parent[f.parent]
        rank,ahead,q = queue_positions[f.key]
        scale = 1e9 if f.stage=="cpu" else 1e6
        high = high_work[q.kind,q.owner[1:]] if f.priority else 0.
        features["fragment"].append([*one(f.priority,2),*one(("transfer","cpu").index(f.stage),2),
            f.ratio,L(f.bits,1e6).item(),L(f.cycles,1e9).item(),L(f.transfer_remaining,1e6).item(),
            L(f.compute_remaining,1e9).item(),L(r.time-f.entered,10).item(),L(r.time-p.birth,10).item(),
            L(rank).item(),L(ahead,scale).item(),L(high,scale).item(),L(len(p.children)-1).item(),
            float(len(p.children)==1)])
    for u in r.uavs:
        features["uav"].append([*np.asarray(u.position)/r.map_size,*np.asarray(u.velocity)/r.max_speed,
            *np.asarray(u.target)/r.map_size,r.remaining_macro/5,r.remaining_adjustment,
            L(u.cpu_hz,1e9).item(),*L(np.asarray(u.compute)/u.cpu_hz,10),
            *L(np.asarray(u.in_transit)/u.cpu_hz,10),*L(u.recent_service,1e9),
            *L([u.ready[p][0] for p in range(2)],1e6),*L([u.ready[p][1] for p in range(2)],1e9),
            *L([u.ready[p][2] for p in range(2)]),*one(("uncommitted","adjustment","hold").index(r.phase),3)])
    for t in r.terminals:
        delta = (np.asarray(t.position)-r.uavs[t.source].position)/r.map_size
        vals = [*delta,np.linalg.norm(delta),L(t.rate,1e6).item()]
        edge("t_u",t_index[t.key],u_index[t.source],vals)
        edge("u_t",u_index[t.source],t_index[t.key],[-delta[0],-delta[1],*vals[2:]])
    for p in r.parents:
        edge("p_t",p_index[p.key],t_index[p.terminal]); edge("t_p",t_index[p.terminal],p_index[p.key])
        edge("p_u",p_index[p.key],u_index[p.source]); edge("u_p",u_index[p.source],p_index[p.key])
        for child in p.children:
            edge("p_f",p_index[p.key],f_index[child]); edge("f_p",f_index[child],p_index[p.key])
    for f in r.fragments:
        edge("f_s",f_index[f.key],u_index[f.source]); edge("s_f",u_index[f.source],f_index[f.key])
        edge("f_d",f_index[f.key],u_index[f.destination]); edge("d_f",u_index[f.destination],f_index[f.key])
    for link in r.links:
        distance = np.linalg.norm(np.asarray(r.uavs[link.source].position)-r.uavs[link.destination].position)/r.map_size
        edge("link",u_index[link.source],u_index[link.destination],
             [L(link.rate,1e6).item(),L(link.all_active_rate,1e6).item(),float(link.active),
              *L(link.bits,1e6),*L(link.cycles,1e9),L(link.power).item(),distance])
    for i in range(nodes):
        edge("self",i,i)
    neighbors = [[] for _ in range(nodes)]
    offset = 0
    for kind in RELATIONS:
        for i,(_,dest,_) in enumerate(edges[kind]):
            neighbors[dest].append(offset+i)
        offset += len(edges[kind])
    pools,geometry = [],[]
    for u in r.uavs:
        ts = [t_index[t.key] for t in r.terminals if t.source==u.key]
        pools.append([ts,[p_index[p.key] for p in r.parents if p.source==u.key],
                      [f_index[f.key] for f in r.fragments if f.destination==u.key]])
        rows = []
        for i in ts:
            t = r.terminals[i]
            delta = (np.asarray(t.position)-u.position)/r.map_size
            rows.append([*delta,np.linalg.norm(delta),L(t.upload[0][1],1e9).item(),
                         L(t.upload[1][1],1e9).item(),L(t.upload[0][0]+t.upload[1][0],1e6).item()])
        geometry.append(rows)
    ready = [[L(u.ready[p][0],1e6).item(),L(u.ready[p][1],1e9).item(),L(u.ready[p][2]).item()]
             for p in range(2) for u in r.uavs]
    return Graph(features,queues,edges,neighbors,u_index,pools,geometry,ready,[r.tick/9000,r.remaining_ticks/9000])


def arrival_array(history, cursor, device):
    if cursor>len(history):
        raise ValueError("History cursor extends beyond observed arrivals")
    data = np.asarray(history[:cursor],dtype=np.float64).reshape(cursor,39,6)
    data = L(data,np.asarray([1e6,1e9,1,1e6,1e9,1]))
    return torch.as_tensor(data,dtype=torch.float32,device=device)
