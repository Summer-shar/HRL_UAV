"""Physical cost, pooled completed delay and explicit censored parents."""
import numpy as np


def finish(adapter):
    e=adapter.env
    if e.step_index!=9000: raise ValueError("Metrics require a complete finite episode")
    result=e.metrics()
    result["terminated"]=True
    result["queue_integrals_physical_grid"]=dict(adapter.queue_integrals)
    result["assigned_cycles"]=adapter.assigned_cycles.tolist()
    parents=[]
    holding=0.
    for p in e.parents.values():
        a=p.arrival
        end=900. if p.finish_s is None else p.finish_s
        age=(round(end/.1)-round(a.birth_s/.1))*.1
        holding+=(10 if a.priority==0 else 1)*age
        parents.append(dict(parent=a.task_id,priority=a.priority,birth_s=a.birth_s,
            completed=p.finish_s is not None,delay_s=None if p.finish_s is None else age,
            censored=p.finish_s is None,cutoff_age_s=900.-a.birth_s if p.finish_s is None else None,
            finish_s=p.finish_s))
    if not np.isclose(holding,e.holding,atol=1e-7,rtol=1e-10):
        raise ValueError("Holding cost / censored parent lifetime identity failed")
    result["holding_identity_error"]=holding-e.holding
    result["parents"]=parents
    for p,row in enumerate(result["priority"]):
        selected=[x for x in parents if x["priority"]==p]
        row["unfinished"]=sum(x["censored"] for x in selected)
        row["unfinished_rate"]=row["unfinished"]/row["arrived"] if row["arrived"] else None
        row["completed_delays_s"]=[x["delay_s"] for x in selected if x["completed"]]
        row["cutoff_ages_s"]=[x["cutoff_age_s"] for x in selected if x["censored"]]
    return result


def pool(results):
    if not results: raise ValueError("Cannot pool an empty panel")
    priorities=[]
    for p in range(2):
        rows=[r["priority"][p] for r in results]
        row={k:sum(x[k] for x in rows) for k in ("arrived","completed","unfinished","delay_sum_s")}
        row["unfinished_rate"]=row["unfinished"]/row["arrived"] if row["arrived"] else None
        priorities.append(row)
    denominator=sum(w*p["completed"] for w,p in zip((10,1),priorities))
    return dict(cost_per_s=sum(r["cost_sum"] for r in results)/sum(r["seconds"] for r in results),
        pooled_delay_ms=1000*sum(w*p["delay_sum_s"] for w,p in zip((10,1),priorities))/denominator if denominator else None,
        priority=priorities,episodes=len(results))
