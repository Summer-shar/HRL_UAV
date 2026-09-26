"""Single owner of pending successors, committing the real next macro action."""
import numpy as np
from .replay import Episode
from .metrics import finish


def collect(adapter,policy,*,upper_version,lower_version,identity,on_event=None,on_tick=None):
    """Policy methods receive immutable reports and observed arrival arrays only.

    The adapter must already be reset. on_event samples *previous completed*
    episodes; the caller admits this episode only after collect returns.
    """
    if adapter.env.step_index!=0: raise ValueError("Collector requires reset")
    episode=Episode(upper_version,lower_version,identity)
    pending=False
    sealed=0
    max_report_bytes=0
    for macro in range(180):
        report=adapter.report()
        high_id=episode.add_report(report)
        request,aux=policy.upper(report,adapter.arrivals)
        motion=adapter.start_macro(request)
        high=dict(aux,report=high_id,reward=0.,request=np.asarray(request).copy(),
                  displacement=np.asarray(motion["displacement_m"]).copy())
        for frame in range(10):
            report=adapter.prepare()
            report_id=episode.add_report(report)
            max_report_bytes=max(max_report_bytes,adapter.report_bytes(report))
            if pending:
                episode.successors.append(report_id)
                sealed+=1
                if on_event is not None and sealed%20==0: on_event()
            route,support=policy.lower(report,adapter.arrivals)
            route=np.asarray(route,dtype=np.float64).reshape(2,3,3)
            from .hybrid_simplex import SUPPORTS
            ids=np.asarray(support).reshape(6)
            if any(not 0<=int(k)<7 or SUPPORTS[int(k)]!=tuple(np.flatnonzero(row>0))
                   for row,k in zip(route.reshape(6,3),ids)):
                raise ValueError("Recorded support differs from execution action")
            readings=[]
            for _ in range(5):
                readings.append(adapter.step(route))
                if on_tick is not None: on_tick(adapter,readings[-1])
            reward=-sum(x["cost"] for x in readings)/100
            episode.observations.append(report_id)
            episode.actions.append(route.copy())
            episode.supports.append(np.asarray(support,dtype=np.int64).copy())
            episode.rewards.append(reward)
            episode.terminated.append(adapter.env.step_index==9000)
            episode.severe.append(report.severe)
            high["reward"]+=reward
            pending=True
        episode.high.append(high)
    terminal=adapter.report()
    episode.successors.append(episode.add_report(terminal))
    sealed+=1
    if on_event is not None and sealed%20==0: on_event()
    metrics=finish(adapter)
    metrics["max_report_bytes"]=max_report_bytes
    episode.seal(adapter.arrivals,metrics)
    return episode
