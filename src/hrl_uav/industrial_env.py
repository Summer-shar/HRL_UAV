"""Independent industrial service environment; no mutation of legacy mission code."""
from collections import deque
from dataclasses import dataclass, field
import copy
import numpy as np

from .env import UAV_MEC_Env
from .industrial_scene import IndustrialScene, Arrival, validate_config
from .industrial_observation import TerminalReport, ReadyTask, ControllerView, tuples


class IndustrialPhysics:
    """Reuse the audited public geometry, radio, power and path-safety formulas only."""
    compute_u2g_rate_bps = UAV_MEC_Env.compute_u2g_rate_bps
    _positive_wireless_parameter = staticmethod(UAV_MEC_Env._positive_wireless_parameter)
    allocate_u2u_link_bandwidth_hz = UAV_MEC_Env.allocate_u2u_link_bandwidth_hz
    _u2u_channel_gain = UAV_MEC_Env._u2u_channel_gain
    allocate_u2u_resources = UAV_MEC_Env.allocate_u2u_resources
    _compute_tcom_flying_power = UAV_MEC_Env._compute_tcom_flying_power
    _minimum_separation_along_joint_motion = UAV_MEC_Env._minimum_separation_along_joint_motion
    _project_joint_motion_to_safe_separation = UAV_MEC_Env._project_joint_motion_to_safe_separation

    def __init__(self, c, terminals):
        r = c["resources"]
        self.N, self.M = 3, 39
        self.H, self.beta_0 = c["height_m"], r["reference_gain"]
        self.B_gt_total, self.B_u2u_total = r["uplink_bandwidth_hz"], r["u2u_bandwidth_hz"]
        self.p_gt, self.u2u_tx_budget_w = r["terminal_power_w"], r["u2u_source_power_w"]
        self.noise_psd_w_per_hz = r["noise_psd_w_per_hz"]
        self.noise_power = self.noise_psd_w_per_hz * self.B_gt_total / self.M
        self.wireless_model, self.dynamic_u2u_bandwidth = "psd_per_uav_budget", True
        self.min_uav_separation_m = c["motion"]["min_separation_m"]
        self.user_coords = np.array([t.position_m for t in terminals])
        self.user_coords.setflags(write=False)
        self.user_allocations = np.array([t.source for t in terminals])
        self.uav_coords = np.array(c["uav_initial_positions_m"], dtype=float)


@dataclass
class Parent:
    arrival: Arrival
    upload_remaining: float
    upload_finish_s: object = None
    route_time_s: object = None
    fragment_ids: list = field(default_factory=list)
    finish_s: object = None


@dataclass
class Fragment:
    fragment_id: int
    parent_id: int
    source: int
    destination: int
    priority: int
    input_bits: float
    compute_cycles: float
    transfer_remaining: float
    compute_remaining: float
    stage: str
    entered_s: float
    finish_s: object = None


class IndustrialEnv:
    def __init__(self, config, *, ledger_enabled=False):
        self.config = validate_config(config)
        self.ledger_enabled = bool(ledger_enabled)
        self.dt = self.config["time"]["physical_s"]
        self.frame_steps = round(self.config["time"]["routing_s"] / self.dt)
        self.macro_steps = round(self.config["time"]["motion_s"] / self.dt)
        self.reset(0)

    def reset(self, seed, layout_seed=None):
        self._scene = IndustrialScene(self.config, seed, layout_seed)
        self.physics = IndustrialPhysics(self.config, self._scene.terminals)
        self.parents, self.fragments = {}, {}
        self._reset_active_state()
        self.upload = [deque() for _ in range(39)]
        self.compute = [[deque() for _ in range(3)] for _ in range(2)]
        self.transfer = [[[deque() for _ in range(3)] for _ in range(3)] for _ in range(2)]
        self.ready = deque()
        self._recent = deque()
        self._prepared_frame = -1
        self.step_index = self.revision = 0
        self.motion_ticks = 0
        self.velocity = np.zeros((3, 2))
        self._target = self.physics.uav_coords.copy()
        self.ledger = []
        self.holding = self.cost_sum = 0.
        self.energy = dict(upload=0., u2u=0., cpu=0., propulsion=0.)
        self.generated_bits = self.generated_cycles = self.upload_served_bits = self.tx_served_bits = self.cpu_served_cycles = 0.
        self._publish("reset_before_arrivals")
        return self.view()

    @property
    def time_s(self):
        return self.step_index * self.dt

    def _event(self, kind, **fields):
        if self.ledger_enabled:
            self.ledger.append(dict(kind=kind, **fields))

    def _reset_active_state(self):
        # Keep history for metrics/ledger queries, but visit only live work in
        # per-step costs and controller reports. Dict insertion order matches
        # the original history-filtered iteration and its floating-point sums.
        self._active_parents, self._active_fragments = {}, {}
        self._view_cache = None

    def _invalidate_view(self):
        # Service mutations use this boundary; deliberate diagnostic writes to
        # private task/physics fields must invalidate or publish their report.
        self._view_cache = None

    def _admit(self, arrival):
        if arrival.task_id in self.parents or abs(arrival.birth_s-self.time_s) > 1e-9:
            raise ValueError("Duplicate or non-current arrival")
        self._invalidate_view()
        self.parents[arrival.task_id] = Parent(arrival, arrival.input_bits)
        self._active_parents[arrival.task_id] = self.parents[arrival.task_id]
        self.upload[arrival.user_id].append(arrival.task_id)
        self._recent.append(arrival)
        self.generated_bits += arrival.input_bits
        self.generated_cycles += arrival.compute_cycles
        self._event("birth", time_s=self.time_s, parent_id=arrival.task_id, bits=arrival.input_bits, cycles=arrival.compute_cycles)

    def prepare_frame(self):
        if self.step_index % self.frame_steps:
            raise ValueError("Arrival preparation requires a routing boundary")
        frame = self.step_index // self.frame_steps
        if frame == self._prepared_frame:
            return self.view()
        for arrival in self._scene.frame(frame):
            self._admit(arrival)
        self._prepared_frame = frame
        self._publish("after_frame_arrivals")
        return self.view()

    def start_motion(self, action):
        if self.step_index % self.macro_steps or self.motion_ticks:
            raise ValueError("Motion commands require a completed macro boundary")
        raw = np.asarray(action, float)
        if raw.size != 6 or not np.isfinite(raw).all():
            raise ValueError("Expected six finite motion scores")
        scaled = np.clip(raw.reshape(3, 2), -1, 1)
        scaled /= np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1)
        duration = self.config["time"]["motion_s"]
        delta = scaled * self.config["motion"]["max_speed_mps"] * duration
        start = self.physics.uav_coords.copy()
        requested = np.clip(start + delta, 0, self.config["map_size_m"])
        actual, scale = self.physics._project_joint_motion_to_safe_separation(start, requested-start)
        self._target = start + actual
        self.velocity = actual / duration
        self.motion_ticks = self.macro_steps
        self._publish(self._phase)
        return dict(raw_action=raw.ravel().copy(), displacement_m=actual.copy(),
                    velocity_mps=self.velocity.copy(), duration_s=duration, safety_scale=scale,
                    minimum_separation_m=self.physics._minimum_separation_along_joint_motion(start, actual))

    def _publish(self, phase):
        self._invalidate_view()
        now, window = self.time_s, self.config["time"]["history_s"]
        start = max(0., now-window)
        while self._recent and self._recent[0].birth_s < start:
            self._recent.popleft()
        recent = np.zeros((39, 2, 3))
        for a in self._recent:
            if now > 0 and (a.birth_s > start or (start == 0 and now <= window)):
                recent[a.user_id, a.priority] += (a.input_bits, a.compute_cycles, 1)
        reports = []
        for terminal, queue in zip(self._scene.terminals, self.upload):
            values = np.zeros((2, 4))
            for pid in queue:
                p = self.parents[pid]
                v = values[p.arrival.priority]
                v[:3] += (p.upload_remaining, p.arrival.compute_cycles, 1)
                v[3] = max(v[3], now-p.arrival.birth_s)
            r = recent[terminal.user_id]
            reports.append(TerminalReport(terminal.user_id, terminal.source, terminal.position_m, now, True,
                tuple(values[:, 0]), tuple(values[:, 1]), tuple(values[:, 2]), tuple(values[:, 3]),
                tuple(r[:, 0]), tuple(r[:, 1]), tuple(r[:, 2]), now-start, now > start))
        self._terminal_reports = tuple(reports)
        self.revision += 1
        self._phase = phase

    def view(self):
        if self._view_cache is not None:
            return self._view_cache
        compute = np.zeros((2, 3))
        tx_bits, tx_cycles = np.zeros((2, 3, 3)), np.zeros((2, 3, 3))
        for f in self._active_fragments.values():
            if f.stage == "cpu":
                compute[f.priority, f.destination] += f.compute_remaining
            elif f.stage == "u2u":
                tx_bits[f.priority, f.source, f.destination] += f.transfer_remaining
                tx_cycles[f.priority, f.source, f.destination] += f.compute_remaining
        counts, ages = np.zeros((2, 3)), np.zeros((2, 3))
        for p in self._active_parents.values():
            if p.finish_s is None:
                key = (p.arrival.priority, p.arrival.source)
                counts[key] += 1
                ages[key] = max(ages[key], self.time_s-p.arrival.birth_s)
        ready = tuple(ReadyTask(pid, self.parents[pid].arrival.source, self.parents[pid].arrival.priority,
                      self.parents[pid].arrival.input_bits, self.parents[pid].arrival.compute_cycles,
                      self.time_s-self.parents[pid].arrival.birth_s) for pid in self.ready)
        current = self.physics.allocate_u2u_resources(tx_bits.sum(axis=0) > 0)
        potential = self.physics.allocate_u2u_resources(~np.eye(3, dtype=bool))
        self._view_cache = ControllerView(self.time_s, self.revision, self._phase, self._terminal_reports,
            tuples(self.physics.uav_coords), tuples(self.velocity), self.motion_ticks*self.dt, ready,
            tuples(compute), tuples(tx_bits), tuples(tx_cycles), tuples(counts), tuples(ages),
            tuple(self.config["resources"]["cpu_hz"]), tuple(self.physics.compute_u2g_rate_bps()),
            tuples(current["u2u_rate_bps"]), tuples(potential["u2u_rate_bps"]), tuples(potential["u2u_link_power_w"]))
        return self._view_cache

    def _route(self, ids, route):
        self._invalidate_view()
        for pid in ids:
            p = self.parents[pid]
            if p.route_time_s is not None:
                raise RuntimeError("A parent may only be routed once")
            a = p.arrival
            p.route_time_s = self.time_s
            for destination, ratio in enumerate(route[a.priority, a.source]):
                if ratio <= 0:
                    continue
                fid = len(self.fragments)
                local = destination == a.source
                f = Fragment(fid, pid, a.source, destination, a.priority, ratio*a.input_bits,
                             ratio*a.compute_cycles, 0. if local else ratio*a.input_bits,
                             ratio*a.compute_cycles, "cpu" if local else "u2u", self.time_s)
                self.fragments[fid] = f
                self._active_fragments[fid] = f
                p.fragment_ids.append(fid)
                (self.compute[a.priority][destination] if local else self.transfer[a.priority][a.source][destination]).append(fid)
                self._event("fragment_created", time_s=self.time_s, parent_id=pid, fragment_id=fid,
                            source=a.source, destination=destination, bits=f.input_bits, cycles=f.compute_cycles, stage=f.stage)
        self._invalidate_view()

    def _upload_step(self):
        self._invalidate_view()
        energy = 0.
        for user, rate in enumerate(self.physics.compute_u2g_rate_bps()):
            budget, elapsed = rate*self.dt, 0.
            queue = self.upload[user]
            while queue and budget > 0:
                p = self.parents[queue[0]]
                amount = min(p.upload_remaining, budget)
                duration = amount/rate
                self._event("upload_service", parent_id=p.arrival.task_id, start_s=self.time_s+elapsed,
                            duration_s=duration, bits=amount, energy_j=duration*self.physics.p_gt)
                p.upload_remaining = max(0., p.upload_remaining-amount)
                self.upload_served_bits += amount
                budget -= amount
                elapsed += duration
                energy += duration*self.physics.p_gt
                if p.upload_remaining == 0:
                    p.upload_remaining = 0.
                    p.upload_finish_s = self.time_s+elapsed
                    self.ready.append(queue.popleft())
                else:
                    break
        self._invalidate_view()
        return energy

    def _complete(self, fragment):
        self._invalidate_view()
        fragment.stage, fragment.finish_s = "complete", (self.step_index+1)*self.dt
        self._active_fragments.pop(fragment.fragment_id, None)
        p = self.parents[fragment.parent_id]
        if all(self.fragments[f].finish_s is not None for f in p.fragment_ids):
            p.finish_s = max(self.fragments[f].finish_s for f in p.fragment_ids)
            self._active_parents.pop(fragment.parent_id, None)
            self._event("parent_complete", parent_id=fragment.parent_id, time_s=p.finish_s)
        self._invalidate_view()

    def _compute_step(self):
        self._invalidate_view()
        r, energy = self.config["resources"], 0.
        for node, maximum in enumerate(r["cpu_hz"]):
            need = np.array([sum(self.fragments[f].compute_remaining for f in self.compute[p][node]) for p in range(2)])
            high = min(maximum, need[0]/self.dt)
            frequencies = (high, min(maximum-high, need[1]/self.dt))
            total = sum(frequencies)
            energy += r["cpu_kappa"] * total**3 * self.dt
            for priority, frequency in enumerate(frequencies):
                budget, elapsed = frequency*self.dt, 0.
                queue = self.compute[priority][node]
                # Every positive fragment must receive service, even when its
                # allocation is much smaller than an absolute epsilon.
                while queue and budget > 0:
                    f = self.fragments[queue[0]]
                    amount = min(f.compute_remaining, budget)
                    duration = amount/frequency
                    share = r["cpu_kappa"]*total**2*amount
                    self._event("cpu_service", fragment_id=f.fragment_id, parent_id=f.parent_id,
                                start_s=self.time_s+elapsed, duration_s=duration, cycles=amount,
                                frequency_hz=frequency, energy_j=share)
                    self.cpu_served_cycles += amount
                    f.compute_remaining = max(0., f.compute_remaining-amount)
                    elapsed += duration
                    budget -= amount
                    if f.compute_remaining == 0:
                        f.compute_remaining = 0.
                        queue.popleft()
                        self._complete(f)
                    else:
                        break
        self._invalidate_view()
        return energy

    def _transfer_step(self):
        self._invalidate_view()
        active = np.array([[any(self.transfer[p][i][j] for p in range(2)) for j in range(3)] for i in range(3)])
        resources = self.physics.allocate_u2u_resources(active)
        energy = 0.
        for source, destination in zip(*np.nonzero(active)):
            rate = resources["u2u_rate_bps"][source, destination]
            power = resources["u2u_link_power_w"][source, destination]
            budget, elapsed = rate*self.dt, 0.
            for priority in range(2):
                queue = self.transfer[priority][source][destination]
                while queue and budget > 0:
                    f = self.fragments[queue[0]]
                    amount = min(f.transfer_remaining, budget)
                    duration = amount/rate
                    self._event("u2u_service", fragment_id=f.fragment_id, parent_id=f.parent_id,
                                start_s=self.time_s+elapsed, duration_s=duration, bits=amount,
                                rate_bps=rate, power_w=power, energy_j=duration*power)
                    f.transfer_remaining = max(0., f.transfer_remaining-amount)
                    self.tx_served_bits += amount
                    budget -= amount
                    elapsed += duration
                    energy += duration*power
                    if f.transfer_remaining == 0:
                        f.transfer_remaining = 0.
                        queue.popleft()
                        f.stage, f.entered_s = "cpu", (self.step_index+1)*self.dt
                        self.compute[priority][destination].append(f.fragment_id)
                        self._event("u2u_complete", time_s=f.entered_s, fragment_id=f.fragment_id, parent_id=f.parent_id)
                    else:
                        break
        self._invalidate_view()
        return energy

    def micro_step(self, route):
        route = np.asarray(route, float)
        if route.shape != (2, 3, 3) or not np.isfinite(route).all() or np.any(route < 0) or not np.allclose(route.sum(axis=2), 1, atol=1e-10, rtol=0):
            raise ValueError("Expected finite nonnegative 2x3x3 simplex rows")
        if self._prepared_frame != self.step_index//self.frame_steps:
            raise RuntimeError("Prepare current arrivals before physical service")
        self._invalidate_view()
        o = self.config["objective"]
        holding = self.dt*sum(o["priority_weights"][p.arrival.priority] for p in self._active_parents.values() if p.finish_s is None)
        ready = tuple(self.ready)
        self.ready.clear()
        energy = dict(upload=self._upload_step())
        self._route(ready, route)
        energy["cpu"] = self._compute_step()
        energy["u2u"] = self._transfer_step()
        energy["propulsion"] = float(self.physics._compute_tcom_flying_power(np.linalg.norm(self.velocity, axis=1)).sum()*self.dt)
        if self.motion_ticks:
            self.physics.uav_coords += self.velocity*self.dt
            self.motion_ticks -= 1
            if not self.motion_ticks:
                self.physics.uav_coords = self._target.copy()
                self.velocity.fill(0.)
        self.step_index += 1
        self.holding += holding
        for name, value in energy.items():
            self.energy[name] += value
        cost = o["holding_weight"]*holding/o["holding_reference_task_s"] + (1-o["holding_weight"])*sum(energy.values())/o["energy_reference_j"]
        self.cost_sum += cost
        if self.step_index % self.frame_steps == 0:
            self._publish("frame_boundary_before_arrivals")
        self._invalidate_view()
        return dict(cost=cost, holding_task_s=holding, energy_j=energy, time_s=self.time_s)

    def conservation(self):
        unassigned = [p for p in self.parents.values() if p.route_time_s is None]
        return dict(upload_error_bits=self.generated_bits-self.upload_served_bits-sum(p.upload_remaining for p in self.parents.values()),
                    compute_error_cycles=self.generated_cycles-self.cpu_served_cycles-sum(p.arrival.compute_cycles for p in unassigned)-sum(f.compute_remaining for f in self.fragments.values()),
                    routing_error_bits=sum(p.arrival.input_bits for p in self.parents.values() if p.route_time_s is not None)-sum(f.input_bits for f in self.fragments.values()),
                    transfer_error_bits=sum(f.input_bits for f in self.fragments.values() if f.source != f.destination)-self.tx_served_bits-sum(f.transfer_remaining for f in self.fragments.values()))

    def metrics(self):
        priorities = []
        for priority in range(2):
            born = [p for p in self.parents.values() if p.arrival.priority == priority]
            done = [p for p in born if p.finish_s is not None]
            priorities.append(dict(arrived=len(born), completed=len(done),
                completion_rate=len(done)/len(born) if born else None,
                delay_sum_s=sum(p.finish_s-p.arrival.birth_s for p in done),
                mean_delay_ms=1000*np.mean([p.finish_s-p.arrival.birth_s for p in done]) if done else None))
        weights = self.config["objective"]["priority_weights"]
        denominator = sum(w*p["completed"] for w, p in zip(weights, priorities))
        view = self.view()
        pending = [p for p in self.parents.values() if p.finish_s is None]
        return dict(seconds=self.time_s, physical_steps=self.step_index, cost_sum=self.cost_sum,
            mean_cost_per_s=self.cost_sum/self.time_s if self.time_s else None,
            holding_task_s=self.holding, energy_j=copy.deepcopy(self.energy),
            average_power_w=sum(self.energy.values())/self.time_s if self.time_s else None,
            priority=priorities, weighted_delay_ms=1000*sum(w*p["delay_sum_s"] for w, p in zip(weights, priorities))/denominator if denominator else None,
            unfinished_parents=len(pending), oldest_unfinished_age_s=max((self.time_s-p.arrival.birth_s for p in pending), default=0.),
            upload_pending_bits=sum(p.upload_remaining for p in self.parents.values()),
            ground_pending_cycles=sum(p.arrival.compute_cycles for p in self.parents.values() if p.upload_remaining > 0),
            ready_cycles=sum(t.compute_cycles for t in view.ready_tasks),
            transfer_pending_bits=float(np.sum(view.transfer_bits)),
            transfer_future_cycles=float(np.sum(view.transfer_cycles)), cpu_pending_cycles=float(np.sum(view.compute_cycles)),
            event_sha256=self._scene.event_digest.hexdigest(), conservation=self.conservation(), terminated=False)
