"""Unchanged routing rules exposed through the same causal report contract."""
from types import SimpleNamespace
import numpy as np
from .hybrid_simplex import SUPPORTS
from hrl_uav.industrial_baselines import local_scores,RandomRouting,CVXRouting,GreedyRouting
from hrl_uav.routing import project_rows_to_simplex


def centroid(report,history,*,max_target_step_m):
    r=report
    n=len(r.terminals)
    observed=np.asarray(history[max(0,r.history_cursor-20):r.history_cursor],dtype=float).reshape(-1,n,6)
    elapsed=min(10.,r.history_cursor*.5)
    rates=observed.sum(0)/elapsed if elapsed else np.zeros((n,6))
    result=[]
    for u in r.uavs:
        terminals=[t for t in r.terminals if t.source==u.key]
        weights=np.asarray([10*(t.upload[0][1]+5*rates[t.key,1])+
                            t.upload[1][1]+5*rates[t.key,4] for t in terminals])
        target=np.average([t.position for t in terminals],axis=0,weights=weights) if weights.sum() else np.asarray(u.position)
        delta=target-u.position
        result.extend(delta/max(max_target_step_m,np.linalg.norm(delta)))
    return np.asarray(result)


def legacy_view(report):
    cpu=np.asarray([u.compute for u in report.uavs]).T
    tx=np.zeros((2,3,3)); cycles=tx.copy(); rates=np.zeros((3,3)); power=rates.copy()
    for link in report.links:
        tx[:,link.source,link.destination]=link.bits
        cycles[:,link.source,link.destination]=link.cycles
        rates[link.source,link.destination]=link.all_active_rate
        power[link.source,link.destination]=link.all_active_power
    ready=tuple(SimpleNamespace(priority=p.priority,source=p.source,input_bits=p.bits,compute_cycles=p.cycles)
                for p in report.parents if p.stage=="ready")
    return SimpleNamespace(ready_tasks=ready,compute_cycles=cpu,transfer_bits=tx,transfer_cycles=cycles,
        cpu_hz=[u.cpu_hz for u in report.uavs],positions_m=[u.position for u in report.uavs],
        u2u_all_active_rate_bps=rates,u2u_all_active_power_w=power)


class Routing:
    def __init__(self,kind,scene,seed):
        self.kind=kind
        self.rule={"local":lambda:local_scores,"random":lambda:RandomRouting(seed),
                   "cvx":lambda:CVXRouting(scene["base_scene"]),
                   "greedy":lambda:GreedyRouting(scene["base_scene"])}[kind]()

    def __call__(self,report,history):
        # Preserve the original baseline's complete execution mapping. This
        # projection belongs only to the frozen baseline, never to V2 SAC.
        a=project_rows_to_simplex(np.asarray(self.rule(legacy_view(report)),dtype=np.float64).reshape(2,3,3)).reshape(6,3)
        ids=np.asarray([SUPPORTS.index(tuple(np.flatnonzero(row>0))) for row in a])
        return a.reshape(2,3,3),ids
