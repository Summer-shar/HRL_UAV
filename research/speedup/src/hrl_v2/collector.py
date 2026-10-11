"""Continuing-service rollout windows with committed, real successors."""
import numpy as np
from .replay import Episode
from .metrics import finish, snapshot


def collect(adapter,policy,*,upper_version,lower_version,identity,on_event=None,on_tick=None,
            reward_scale=100.,gamma_lower=1.,on_report=None):
    """Collect a window without resetting the physical system.

    The final lower successor includes the actual next upper decision and the
    observed arrivals at that decision time. That already committed decision
    is retained on the adapter and consumed by the next call, never resampled.

    Upper records keep the undiscounted physical reward separately from the
    within-macro discounted learning reward. Production callers pass the
    configured lower discount explicitly; 1 preserves scripted-call defaults.
    """
    episode=Episode(upper_version,lower_version,identity,
                    expected_lower_steps=adapter.timing.lower_steps,
                    expected_upper_steps=adapter.timing.upper_steps,reward_scale=reward_scale,
                    gamma_lower=gamma_lower,encoder="lite" if getattr(policy,"observer",None) is not None else "graph")
    metrics=_collect(adapter,policy,identity=identity,episode=episode,on_event=on_event,
                     on_tick=on_tick,reward_scale=reward_scale,upper_version=upper_version,
                     gamma_lower=gamma_lower,on_report=on_report)
    episode.seal(adapter.arrivals,metrics)
    return episode


def collect_metrics(adapter,policy,*,identity,on_tick=None,reward_scale=100.):
    """Frozen evaluation uses the same transitions without replay allocation."""
    return _collect(adapter,policy,identity=identity,on_tick=on_tick,reward_scale=reward_scale)


def _commit_macro(adapter,policy):
    upper_report=adapter.report()
    request,aux=policy.upper(upper_report,adapter.arrivals)
    motion=adapter.start_macro(request)
    observer=getattr(policy,"observer",None)
    upper_obs=observer.update(upper_report,adapter.arrivals).upper if observer is not None else None
    return dict(upper_report=upper_report,upper_obs=upper_obs,request=np.asarray(request).copy(),aux=dict(aux),
                displacement=np.asarray(motion["displacement_m"]).copy(),
                lower_report=adapter.prepare())


def _collect(adapter,policy,*,identity,episode=None,on_event=None,on_tick=None,
             reward_scale=100.,upper_version=None,gamma_lower=1.,on_report=None):
    timing=adapter.timing
    if not np.isfinite(reward_scale) or reward_scale<=0:
        raise ValueError("Reward scale must be finite and positive")
    if not np.isfinite(gamma_lower) or not 0<=gamma_lower<=1:
        raise ValueError("Lower reward discount must be finite and lie in [0, 1]")
    continuation=getattr(adapter,"_continuation",None)
    if continuation is None:
        if adapter.env.step_index % timing.physical_per_upper:
            raise ValueError("A rollout must start at an upper decision boundary")
        if hasattr(policy,"begin_episode"): policy.begin_episode(identity)
        start=snapshot(adapter)
        block=_commit_macro(adapter,policy)
    else:
        if adapter.report()!=continuation["lower_report"]:
            raise ValueError("Physical state changed after the saved continuation")
        if upper_version is not None and continuation["upper_version"]!=upper_version:
            raise ValueError("A committed successor cannot change upper policy version")
        start=continuation["metrics_start"]
        block=continuation
        adapter._continuation=None
    sealed=0
    pending=False
    max_report_bytes=0
    total_reward=0.
    observer=getattr(policy,"observer",None)

    def seal_successor(report):
        nonlocal sealed
        if episode is not None: episode.successors.append(episode.add_report(report))
        sealed+=1
        # Keep the existing every-20-lower-decisions trigger on the physical
        # timeline; changing window length must not restart that counter.
        completed_lower=adapter.env.step_index//timing.physical_per_lower
        if on_event is not None and completed_lower%20==0: on_event()

    for macro in range(timing.upper_steps):
        if episode is not None:
            if observer is not None: episode.upper_obs.append(block["upper_obs"])
            if on_report is not None: on_report("upper",block["upper_report"])
            high=dict(block["aux"],report=episode.add_report(block["upper_report"]),
                      reward_physical=0.,reward_learning=0.,
                      request=block["request"],displacement=block["displacement"])
            if observer is not None: high["obs_index"]=macro
        for frame in range(timing.lower_per_upper):
            report=block["lower_report"] if frame==0 else adapter.prepare()
            observation=observer.update(report,adapter.arrivals) if observer is not None else None
            if episode is not None and observation is not None: episode.lower_obs.append(observation.lower)
            if on_report is not None: on_report("lower",report)
            if episode is not None and observer is not None:
                max_report_bytes=None
            else:
                max_report_bytes=max(max_report_bytes,adapter.report_bytes(report))
            if pending: seal_successor(report)
            report_id=episode.add_report(report) if episode is not None else None
            route,support=policy.lower(report,adapter.arrivals)
            route=np.asarray(route,dtype=np.float64).reshape(2,3,3)
            from .hybrid_simplex import SUPPORTS
            ids=np.asarray(support).reshape(6)
            if any(not 0<=int(k)<7 or SUPPORTS[int(k)]!=tuple(np.flatnonzero(row>0))
                   for row,k in zip(route.reshape(6,3),ids)):
                raise ValueError("Recorded support differs from execution action")
            cost=0.
            for _ in range(timing.physical_per_lower):
                reading=adapter.step(route)
                cost+=reading["cost"]
                if on_tick is not None: on_tick(adapter,reading)
            reward=-cost/reward_scale
            total_reward+=reward
            if episode is not None:
                episode.observations.append(report_id)
                episode.actions.append(route.copy())
                episode.supports.append(np.asarray(support,dtype=np.int64).copy())
                episode.rewards.append(reward)
                episode.terminated.append(False)
                episode.truncated.append(macro==timing.upper_steps-1 and frame==timing.lower_per_upper-1)
                episode.severe.append(report.severe)
                high["reward_physical"]+=reward
                high["reward_learning"]+=gamma_lower**frame*reward
            pending=True
        final=macro==timing.upper_steps-1
        if final:
            # The evaluation cutoff precedes arrivals prepared for the next
            # window. No queue is cleared and no physical time is advanced.
            metrics=finish(adapter,start)
            next_start=snapshot(adapter)
        following=_commit_macro(adapter,policy)
        if episode is not None:
            high.update(successor=episode.add_report(following["upper_report"]),
                        next_value=following["aux"].get("value"),terminated=False,truncated=final)
            episode.high.append(high)
        block=following
    if episode is not None and observer is not None:
        episode.lower_obs.append(observer.update(block["lower_report"],adapter.arrivals).lower)
        episode.upper_obs.append(block["upper_obs"])
    if on_report is not None:
        on_report("lower_successor",block["lower_report"])
        on_report("upper_successor",block["upper_report"])
    seal_successor(block["lower_report"])
    adapter._continuation=dict(block,metrics_start=next_start,upper_version=upper_version)
    metrics["max_report_bytes"]=max_report_bytes
    if not np.isclose(total_reward,-metrics["cost_sum"]/reward_scale,rtol=1e-11,atol=1e-9):
        raise ValueError("Lower reward / physical cost identity failed")
    return metrics
