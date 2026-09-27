"""Physical costs and censored parents in continuing-service windows.

A carry-in parent contributes holding only for its residence inside the
window; if completed here, its full lifetime delay is recorded.
"""
import numpy as np


def snapshot(adapter):
    """Capture counters before preparing arrivals at a window's left edge."""
    e = adapter.env
    return dict(start_tick=e.step_index, parent_ids=frozenset(e.parents),
        active_parent_ids=frozenset(e._active_parents), cost_sum=e.cost_sum,
        holding_task_s=e.holding, energy_j=dict(e.energy),
        queue_integrals=dict(adapter.queue_integrals),
        assigned_cycles=adapter.assigned_cycles.copy(),
        unfinished_parent_time=e.unfinished_parent_time_s_by_priority.copy(),
        arrivals_by_region=e._arrival_totals_by_region.copy())


def finish(adapter, start=None):
    """Read a cutoff before its new arrivals; never reset or flush service.

    With no snapshot, report the lifetime since reset. For adjacent windows
    snapshot before preparing the new left-edge arrivals. Rates use carry-in
    plus arrived as their denominator, and delays use completed parents only.
    """
    e = adapter.env
    if start is None:
        start = dict(start_tick=0, parent_ids=frozenset(), active_parent_ids=frozenset(),
            cost_sum=0., holding_task_s=0., energy_j={k: 0. for k in e.energy},
            queue_integrals={k: 0. for k in adapter.queue_integrals},
            assigned_cycles=np.zeros_like(adapter.assigned_cycles),
            unfinished_parent_time=np.zeros(2), arrivals_by_region=np.zeros((3, 2, 3)))
    begin, cutoff = start["start_tick"], e.step_index
    if not 0 <= begin < cutoff:
        raise ValueError("Metrics require a nonempty physical service window")
    seconds = (cutoff-begin)*e.dt
    result = e.metrics()
    result.update(terminated=False, truncated=True, window_start_s=begin*e.dt,
        window_end_s=cutoff*e.dt, seconds=seconds, physical_steps=cutoff-begin,
        cost_sum=e.cost_sum-start["cost_sum"], holding_task_s=e.holding-start["holding_task_s"],
        energy_j={k: v-start["energy_j"][k] for k, v in e.energy.items()},
        queue_integrals_physical_grid={k: v-start["queue_integrals"][k]
                                      for k, v in adapter.queue_integrals.items()},
        assigned_cycles=(adapter.assigned_cycles-start["assigned_cycles"]).tolist(),
        delay_scope="completed_in_window_full_lifetime; unfinished_at_cutoff_are_censored",
        rate_denominator="carry_in_plus_arrived", conservation_scope="since_reset")
    result["cost_per_s"] = result["mean_cost_per_s"] = result["cost_sum"]/seconds
    result["average_power_w"] = sum(result["energy_j"].values())/seconds
    integral = e.unfinished_parent_time_s_by_priority-start["unfinished_parent_time"]
    result["unfinished_parent_time_s_by_priority"] = integral.tolist()
    result["unfinished_parent_time_s"] = float(integral.sum())
    result["average_unfinished_parents"] = float(integral.sum()/seconds)
    regional = e._arrival_totals_by_region-start["arrivals_by_region"]
    result["arrivals_by_region"] = [dict(region=i, tasks=int(row[:, 2].sum()),
        input_bits=float(row[:, 0].sum()), compute_cycles=float(row[:, 1].sum()),
        priority=[dict(priority=p, tasks=int(v[2]), input_bits=float(v[0]),
                       compute_cycles=float(v[1])) for p, v in enumerate(row)])
        for i, row in enumerate(regional)]
    parents, holding = [], 0.
    weights = e.config["objective"]["priority_weights"]
    for key, p in e.parents.items():
        carried = key in start["active_parent_ids"]
        if key in start["parent_ids"] and not carried:
            continue
        a = p.arrival
        birth = round(a.birth_s/e.dt)
        end = cutoff if p.finish_s is None else round(p.finish_s/e.dt)
        residence = (min(end, cutoff)-max(birth, begin))*e.dt
        holding += weights[a.priority]*residence
        age = (end-birth)*e.dt
        parents.append(dict(parent=a.task_id, priority=a.priority, birth_s=a.birth_s,
            carry_in=carried, completed=p.finish_s is not None,
            delay_s=None if p.finish_s is None else age, censored=p.finish_s is None,
            cutoff_age_s=age if p.finish_s is None else None,
            observed_residence_s=residence, finish_s=p.finish_s))
    if not np.isclose(holding, result["holding_task_s"], atol=1e-7, rtol=1e-10):
        raise ValueError("Holding cost / censored parent window residence identity failed")
    result["holding_identity_error"] = holding-result["holding_task_s"]
    result["parents"] = parents
    result["priority"] = []
    for priority in range(2):
        selected = [p for p in parents if p["priority"] == priority]
        delays = [p["delay_s"] for p in selected if p["completed"]]
        ages = [p["cutoff_age_s"] for p in selected if p["censored"]]
        count = len(selected)
        carry_in = sum(p["carry_in"] for p in selected)
        result["priority"].append(dict(priority=priority, arrived=count-carry_in,
            carry_in=carry_in, served_population=count, completed=len(delays), unfinished=len(ages),
            completion_rate=len(delays)/count if count else None,
            unfinished_rate=len(ages)/count if count else None,
            delay_sum_s=sum(delays), mean_delay_ms=1000*np.mean(delays) if delays else None,
            completed_delays_s=delays, cutoff_ages_s=ages,
            oldest_unfinished_age_s=max(ages, default=0.)))
    denominator = sum(w*p["completed"] for w, p in zip(weights, result["priority"]))
    result["pooled_delay_ms"] = result["weighted_delay_ms"] = (
        1000*sum(w*p["delay_sum_s"] for w, p in zip(weights, result["priority"]))/denominator
        if denominator else None)
    result["completed_parents"] = sum(p["completed"] for p in result["priority"])
    result["arrived_parents"] = sum(p["arrived"] for p in result["priority"])
    result["carry_in_parents"] = sum(p["carry_in"] for p in result["priority"])
    result["served_population"] = len(parents)
    result["unfinished_rate"] = result["unfinished_parents"]/len(parents) if parents else None
    result["cutoff_backlog"] = dict(upload_bits=result["upload_pending_bits"],
        ready_cycles=result["ready_cycles"], transfer_bits=result["transfer_pending_bits"],
        cpu_cycles=result["cpu_pending_cycles"],
        upload_parents=sum(len(q) for q in e.upload), ready_parents=len(e.ready),
        transfer_fragments=sum(len(q) for p in e.transfer for s in p for q in s),
        cpu_fragments=sum(len(q) for p in e.compute for q in p))
    return result


class PooledMetrics:
    """Retain additive totals and cutoff summaries, without parent ledgers."""
    def __init__(self):
        self.episodes = 0
        self.cost_sum = self.seconds = self.holding = self.unfinished_time = 0.
        self.oldest = 0.
        self.priorities = [dict(arrived=0, carry_in=0, served_population=0,
            completed=0, unfinished=0, delay_sum_s=0) for _ in range(2)]
        self.energy = dict(upload=0., u2u=0., cpu=0., propulsion=0.)
        self.queue_integrals = {}
        self.cutoff_backlogs = {}

    def add(self, result):
        self.cost_sum += result["cost_sum"]
        self.seconds += result["seconds"]
        self.holding += result.get("holding_task_s", 0.)
        self.unfinished_time += result.get("unfinished_parent_time_s", 0.)
        self.oldest = max(self.oldest, result.get("oldest_unfinished_age_s", 0.))
        for total, row in zip(self.priorities, result["priority"]):
            for key in total:
                total[key] += row.get(key, row["arrived"] if key == "served_population" else 0)
        for total, key in ((self.energy, "energy_j"),
                           (self.queue_integrals, "queue_integrals_physical_grid"),
                           (self.cutoff_backlogs, "cutoff_backlog")):
            for name, value in result.get(key, {}).items():
                total[name] = total.get(name, 0.)+value
        self.episodes += 1

    def result(self):
        if not self.episodes or not self.seconds:
            raise ValueError("Cannot pool an empty panel")
        priorities = [dict(row,
            unfinished_rate=row["unfinished"]/row["served_population"] if row["served_population"] else None,
            completion_rate=row["completed"]/row["served_population"] if row["served_population"] else None,
            mean_delay_ms=1000*row["delay_sum_s"]/row["completed"] if row["completed"] else None)
            for row in self.priorities]
        denominator = sum(w*p["completed"] for w, p in zip((10, 1), priorities))
        population = sum(p["served_population"] for p in priorities)
        unfinished = sum(p["unfinished"] for p in priorities)
        return dict(cost_per_s=self.cost_sum/self.seconds, cost_sum=self.cost_sum, seconds=self.seconds,
            pooled_delay_ms=1000*sum(w*p["delay_sum_s"] for w, p in zip((10, 1), priorities))/denominator if denominator else None,
            priority=priorities, episodes=self.episodes, windows=self.episodes,
            completed_parents=sum(p["completed"] for p in priorities), unfinished_parents=unfinished,
            unfinished_rate=unfinished/population if population else None,
            oldest_unfinished_age_s=self.oldest, holding_task_s=self.holding,
            energy_j=dict(self.energy), average_power_w=sum(self.energy.values())/self.seconds,
            average_unfinished_parents=self.unfinished_time/self.seconds,
            queue_integrals_physical_grid=dict(self.queue_integrals),
            mean_cutoff_backlog={k: v/self.episodes for k, v in self.cutoff_backlogs.items()},
            censoring="unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
            rate_denominator="sum_of_window_service_populations")


def pool(results):
    accumulator = PooledMetrics()
    for result in results:
        accumulator.add(result)
    return accumulator.result()
