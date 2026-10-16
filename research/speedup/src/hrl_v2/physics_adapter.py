"""Causal exports around the continuing industrial physical service kernel."""
from collections import deque
import pickle
import numpy as np

from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from .reports import ReportV2, Terminal, Parent, Fragment, Queue, UAV, Link
from .timing import ClockConfig
from .baseline_protocol import is_paper_caliber


class PhysicsAdapter:
    def __init__(self, scene, *, ledger=False, timing=None):
        self.timing = ClockConfig.from_scene(scene) if timing is None else timing
        self.env = HotspotIndustrialEnv(scene, ledger_enabled=ledger)
        c = self.env.config
        o = c["objective"]
        if (self.env.dt, self.env.frame_steps, self.env.macro_steps) != (
                self.timing.physical_dt, self.timing.physical_per_lower,
                self.timing.physical_per_upper):
            raise ValueError("Physical scene and configured control clocks disagree")
        if not (np.allclose(o["priority_weights"], [10, 1], atol=0, rtol=0)
                and np.isclose(o["holding_weight"] / o["holding_reference_task_s"], .09)
                and np.isclose((1-o["holding_weight"]) / o["energy_reference_j"], 1/140)
                and ((o['energy_mode']=='full' and o.get('propulsion_weight',1.)==1.) or
                     is_paper_caliber(o))):
            raise ValueError("Original physical objective does not match V2")
        self._reset_reporting()

    def _reset_reporting(self):
        self._continuation = None
        self.arrivals = []
        self._recent_cpu = deque(maxlen=self.timing.physical_per_upper)
        self.queue_integrals = dict(upload_bits=0., ready_cycles=0., cpu_cycles=0., transfer_bits=0.)
        self.assigned_cycles = np.zeros((2, 3, 3))
        self._ratios = {}

    def reset(self, *, arrival_seed, layout_seed, load_seed):
        self.env.reset(int(arrival_seed), int(layout_seed), int(load_seed))
        self._reset_reporting()
        return self.report()

    def start_macro(self, request):
        return self.env.start_motion(request)

    def prepare(self):
        e = self.env
        if e.step_index // e.frame_steps == e._prepared_frame:
            return self.report()
        first = len(e.parents)
        e.prepare_frame()
        arrivals = np.zeros((len(e.upload), 6), dtype=np.float64)
        for pid in range(first, len(e.parents)):
            a = e.parents[pid].arrival
            arrivals[a.user_id, 3*a.priority:3*a.priority+3] += (a.input_bits, a.compute_cycles, 1)
        arrivals.setflags(write=False)
        self.arrivals.append(arrivals)
        return self.report()

    def step(self, route):
        e = self.env
        route = np.asarray(route)
        if route.dtype != np.float64:
            raise TypeError("Execution routes must preserve float64")
        pending = tuple(e.ready)
        before = {(p, i): sum(e.fragments[f].compute_remaining for f in e.compute[p][i])
                  for p in range(2) for i in range(3)}
        cpu_before = {f: e.fragments[f].compute_remaining for p in range(2) for i in range(3)
                      for f in e.compute[p][i]}
        ready_cycles = sum(e.parents[k].arrival.compute_cycles for k in pending)
        totals = dict(upload_bits=sum(p.upload_remaining for p in e._active_parents.values()),
                      ready_cycles=ready_cycles, cpu_cycles=sum(before.values()),
                      transfer_bits=sum(f.transfer_remaining for f in e._active_fragments.values()))
        # Account for locally routed work that can be served in this same tick.
        local = np.zeros((2, 3))
        for pid in pending:
            a = e.parents[pid].arrival
            local[a.priority, a.source] += a.compute_cycles*route[a.priority, a.source, a.source]
        result = e.micro_step(route)
        # Sum individual nonnegative service amounts; subtracting aggregate
        # queues would introduce negative roundoff into a nonnegative feature.
        service = np.zeros((2, 3))
        for fid, initial in cpu_before.items():
            f=e.fragments[fid]
            service[f.priority,f.destination] += initial-f.compute_remaining
        for pid in pending:
            for fid in e.parents[pid].fragment_ids:
                f=e.fragments[fid]
                self._ratios[fid]=float(route[f.priority,f.source,f.destination])
                if f.source==f.destination:
                    service[f.priority,f.destination] += f.compute_cycles-f.compute_remaining
        self._recent_cpu.append(service)
        for name, value in totals.items():
            self.queue_integrals[name] += value * e.dt
        for pid in pending:
            for fid in e.parents[pid].fragment_ids:
                f = e.fragments[fid]
                self.assigned_cycles[f.priority, f.source, f.destination] += f.compute_cycles
        return result

    def report(self):
        e = self.env
        view = e.view()
        parents = tuple(Parent(k, p.arrival.user_id, p.arrival.source, p.arrival.priority,
            p.arrival.birth_s, p.arrival.input_bits, p.arrival.compute_cycles, p.upload_remaining,
            "routed" if p.route_time_s is not None else "ground" if p.upload_finish_s is None else "ready",
            p.route_time_s, tuple(f for f in p.fragment_ids if e.fragments[f].finish_s is None),
            sum(e.fragments[f].finish_s is not None for f in p.fragment_ids))
            for k, p in e._active_parents.items())
        fragments = tuple(Fragment(k, f.parent_id, f.source, f.destination, f.priority,
            self._ratios[k], f.input_bits, f.compute_cycles,
            f.transfer_remaining, f.compute_remaining, "transfer" if f.stage == "u2u" else "cpu", f.entered_s)
            for k, f in e._active_fragments.items())
        terminals=[]
        for t in view.terminals:
            values=np.zeros((2,4))
            for pid in e.upload[t.user_id]:
                p=e.parents[pid]; a=p.arrival
                values[a.priority,:3]+=(p.upload_remaining,a.compute_cycles,1)
                values[a.priority,3]=max(values[a.priority,3],e.time_s-a.birth_s)
            terminals.append(Terminal(t.user_id,t.source,tuple(t.position_m),float(view.uplink_rate_bps[t.user_id]),
                tuple(tuple(row) for row in values)))
        terminals=tuple(terminals)
        queues = [Queue("upload", (i,), tuple(q)) for i, q in enumerate(e.upload)]
        queues += [Queue("ready", (), tuple(e.ready))]
        queues += [Queue("cpu", (p, i), tuple(e.compute[p][i])) for p in range(2) for i in range(3)]
        queues += [Queue("transfer", (p, i, j), tuple(e.transfer[p][i][j]))
                   for p in range(2) for i in range(3) for j in range(3) if i != j]
        recent = sum(self._recent_cpu, np.zeros((2, 3)))
        ready = np.zeros((3, 2, 3))
        for p in parents:
            if p.stage == "ready":
                ready[p.source, p.priority] += (p.bits, p.cycles, 1)
        uavs = tuple(UAV(i, tuple(e.physics.uav_coords[i]), tuple(e.velocity[i]), tuple(e._target[i]),
            view.cpu_hz[i], tuple(view.compute_cycles[p][i] for p in range(2)),
            tuple(sum(view.transfer_cycles[p][s][i] for s in range(3)) for p in range(2)),
            tuple(recent[:, i]), tuple(tuple(r) for r in ready[i])) for i in range(3))
        active = np.asarray(view.transfer_bits).sum(axis=0) > 0
        current = e.physics.allocate_u2u_resources(active)
        potential = e.physics.allocate_u2u_resources(~np.eye(3, dtype=bool))
        links = tuple(Link(i, j, float(current["u2u_rate_bps"][i,j]),
            float(potential["u2u_rate_bps"][i,j]), float(current["u2u_link_power_w"][i,j]),
            float(potential["u2u_link_power_w"][i,j]), bool(active[i,j]),
            tuple(view.transfer_bits[p][i][j] for p in range(2)),
            tuple(view.transfer_cycles[p][i][j] for p in range(2)))
            for i in range(3) for j in range(3) if i != j)
        remaining = max(0, e._macro_end_tick-e.step_index)*e.dt
        phase = "uncommitted" if remaining == 0 else "adjustment" if e.motion_ticks else "hold"
        return ReportV2(e.step_index, e.dt, phase, remaining, e.motion_ticks*e.dt,
            len(self.arrivals), e.config["map_size_m"], e.config["motion"]["max_speed_mps"],
            terminals, parents, fragments, uavs, links, tuple(queues))

    @staticmethod
    def report_bytes(report):
        return len(pickle.dumps(report, protocol=5))
