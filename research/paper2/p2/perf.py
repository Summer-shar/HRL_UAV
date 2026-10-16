"""Bitwise-identical speed-ups of the simulator hot spots (opt-in, no source edits).

Profiling one 900 s episode of the v2.4.0 simulator (hover + equal split):
IndustrialEnv._publish took ~33 % of the time (39 terminal reports built with
many tiny numpy operations, ~3800 calls per episode) and
allocate_u2u_resources ~24 % (a pure function of the UAV positions and the
active-link mask, recomputed ~17 000 times although the UAVs hover 4 s of
every 5 s).

* ``_publish`` is re-implemented with plain-Python accumulation in exactly
  the original order (IEEE float64 addition is the same in Python and numpy),
  so every report value is bitwise identical.
* ``allocate_u2u_resources`` is memoised per physics instance on
  (mask, coordinates, whether coordinates were passed); copies are returned.

With a growing backlog (no cooperation, low U2U bandwidth, demand close to the
capacity, early M1 episodes) almost all time went into code that visits every
task in the system at every 0.1 s step or 0.5 s report: a 900 s episode with
~6000 waiting parents took 87 s instead of 7.6 s, 80 % of it in
PhysicsAdapter.report.  Added (design v3, 2026-10-17):

* ``PhysicsAdapter.report`` reuses the immutable Parent/Fragment records of
  tasks whose mutable attributes are the *same objects* as at the previous
  report (identity checks; a parent's children are tracked by a per-parent
  completion counter bumped in ``_complete``).  A reused record is exactly the
  record that would be rebuilt.  The ready-queue sums visit the ready parents
  in ascending id order, which is the original insertion order.
* ``IndustrialEnv.view`` accumulates in plain Python floats in the original
  order (same IEEE additions as the numpy element updates).
* ``IndustrialEnv.micro_step``: the holding term
  dt * sum(weight of every unfinished parent) equals dt * (w_high * n_high +
  w_low * n_low) exactly when both weights are integers (every partial sum is
  an exactly representable integer); the counts are the hotspot environment's
  alive counters.  Otherwise the original sum is used.
* ``IndustrialEnv._compute_step``: the CPU frequency of a priority class is
  min(available, queued cycles / dt).  The queued cycles are summed only until
  the prefix exceeds available * dt * (1 + 1e-6); then the minimum is the
  available frequency whatever the rounding of the full sum (nonnegative terms,
  relative rounding error < 1e-10).  Otherwise the original full sum is used.

The copied methods are guarded: apply() refuses to install a patch whose
source method no longer has the recorded SHA-256 (update the copy first).

``tests/test_perf_equivalence.py`` checks event digests, every report (== and
repr), every physical step reading and the full metrics dictionary over
several scenarios, controllers, loads and terminal counts, with and without
the patches.  Set P2_NO_PERF=1 to disable.
"""
import os

import numpy as np

from hrl_uav import industrial_env
from hrl_uav.env import UAV_MEC_Env
from hrl_uav.industrial_observation import TerminalReport

_ORIGINAL_ALLOCATE = UAV_MEC_Env.allocate_u2u_resources
_ZERO = np.zeros((2, 4))
_ZERO_RECENT = np.zeros((2, 3))
_ZERO.setflags(write=False)
_ZERO_RECENT.setflags(write=False)
_APPLIED = False


def _publish_fast(self, phase):
    self._invalidate_view()
    now, window = self.time_s, self.config["time"]["history_s"]
    start = max(0., now-window)
    recent_queue = self._recent
    while recent_queue and recent_queue[0].birth_s < start:
        recent_queue.popleft()
    recent = {}
    if now > 0:
        whole = start == 0 and now <= window
        for a in recent_queue:
            if a.birth_s > start or whole:
                key = (a.user_id, a.priority)
                acc = recent.get(key)
                if acc is None:
                    recent[key] = [0.+a.input_bits, 0.+a.compute_cycles, 1.]
                else:
                    acc[0] += a.input_bits
                    acc[1] += a.compute_cycles
                    acc[2] += 1
    parents = self.parents
    mobile = self._mobility is not None
    coords = self.physics.user_coords if mobile else None
    reports = []
    for terminal, queue in zip(self._scene.terminals, self.upload):
        if queue:
            rows = [[0., 0., 0., 0.], [0., 0., 0., 0.]]
            for pid in queue:
                p = parents[pid]
                arrival = p.arrival
                v = rows[arrival.priority]
                v[0] += p.upload_remaining
                v[1] += arrival.compute_cycles
                v[2] += 1
                v[3] = max(v[3], now-arrival.birth_s)
            values = np.array(rows)
        else:
            values = _ZERO
        uid = terminal.user_id
        high, low = recent.get((uid, 0)), recent.get((uid, 1))
        if high is None and low is None:
            r = _ZERO_RECENT
        else:
            r = np.array([high or (0., 0., 0.), low or (0., 0., 0.)])
        position = tuple(coords[uid].tolist()) if mobile else terminal.position_m
        reports.append(TerminalReport(uid, terminal.source, position, now, True,
            tuple(values[:, 0]), tuple(values[:, 1]), tuple(values[:, 2]), tuple(values[:, 3]),
            tuple(r[:, 0]), tuple(r[:, 1]), tuple(r[:, 2]), now-start, now > start))
    self._terminal_reports = tuple(reports)
    self.revision += 1
    self._phase = phase


def _allocate_cached(self, active_link_mask, uav_coords=None):
    coords = self.uav_coords if uav_coords is None else uav_coords
    key = (np.asarray(active_link_mask, dtype=bool).tobytes(),
           np.asarray(coords, dtype=np.float64).tobytes(), uav_coords is None)
    cache = self.__dict__.get('_p2_u2u_cache')
    if cache is None:
        cache = self.__dict__['_p2_u2u_cache'] = {}
    hit = cache.get(key)
    if hit is None:
        hit = _ORIGINAL_ALLOCATE(self, active_link_mask, uav_coords)
        if len(cache) >= 256:
            cache.clear()
        cache[key] = hit
    return {k: v.copy() for k, v in hit.items()}


def apply():
    """Install the patches once (idempotent). Returns True when active."""
    global _APPLIED
    if os.environ.get('P2_NO_PERF') == '1':
        return False
    if _APPLIED:
        return True
    _check_sources()
    cls = _env_class()
    cls._publish = _publish_fast
    industrial_env.IndustrialPhysics.allocate_u2u_resources = _allocate_cached
    cls.view = _view_fast
    cls.micro_step = _micro_step_fast
    cls._compute_step = _compute_step_fast
    cls._complete = _complete_tracked
    _PhysicsAdapter.report = _report_fast
    _APPLIED = True
    return True


def remove():
    global _APPLIED
    cls = _env_class()
    cls._publish = _ORIGINAL_PUBLISH_FN
    industrial_env.IndustrialPhysics.allocate_u2u_resources = _ORIGINAL_ALLOCATE
    for name, fn in _ORIGINAL_METHODS.items():
        setattr(cls, name, fn)
    _PhysicsAdapter.report = _ORIGINAL_REPORT
    _APPLIED = False


def _env_class():
    for name in dir(industrial_env):
        obj = getattr(industrial_env, name)
        if isinstance(obj, type) and '_publish' in obj.__dict__:
            return obj
    raise RuntimeError('Simulator class with _publish not found')


_ORIGINAL_PUBLISH_FN = _env_class().__dict__['_publish']


# ---------------------------------------------------------------------------
# Training-only adapter: skips diagnostics that no Lite observation, reward or
# physical transition reads (queue integrals and per-tick CPU service, which
# cost O(backlog) every 0.1 s and dominate overloaded episodes such as NC).
# Physics, rewards, reports used by Lite observations and assigned_cycles are
# unchanged; metrics['queue_integrals_physical_grid'] is NOT valid with it, and
# UAV.recent in reports stays zero.  Evaluation uses the full PhysicsAdapter.
# ---------------------------------------------------------------------------
from hrl_v2.physics_adapter import PhysicsAdapter as _PhysicsAdapter


class TrainingAdapter(_PhysicsAdapter):
    diagnostics = False

    def step(self, route):
        e = self.env
        route = np.asarray(route)
        if route.dtype != np.float64:
            raise TypeError("Execution routes must preserve float64")
        pending = tuple(e.ready)
        result = e.micro_step(route)
        ratios, assigned, parents, fragments = self._ratios, self.assigned_cycles, e.parents, e.fragments
        for pid in pending:
            for fid in parents[pid].fragment_ids:
                f = fragments[fid]
                ratios[fid] = float(route[f.priority, f.source, f.destination])
                assigned[f.priority, f.source, f.destination] += f.compute_cycles
        self._recent_cpu.append(_ZERO_SERVICE)
        return result


_ZERO_SERVICE = np.zeros((2, 3))
_ZERO_SERVICE.setflags(write=False)


# ---------------------------------------------------------------------------
# Backlog-independent hot spots (design v3).  Each copied method mirrors the
# v2.5.0 source line by line except for the marked change.
# ---------------------------------------------------------------------------
import hashlib
import inspect

from hrl_v2.reports import ReportV2, Terminal, Parent, Fragment, Queue, UAV, Link
from hrl_uav.industrial_observation import ReadyTask, ControllerView, tuples

_ENV = _env_class()
_ORIGINAL_METHODS = {name: _ENV.__dict__[name] for name in ('view', 'micro_step', '_compute_step', '_complete')}
_ORIGINAL_REPORT = _PhysicsAdapter.__dict__['report']
_SOURCE_SHA256 = {    # inspect.getsource of the v2.5.0 methods that are copied above
    'IndustrialEnv.view': '5f375138c52128486bbfb5049a2f0f5ca91774c28c7f4f15944ad8dd0d1fee47',
    'IndustrialEnv.micro_step': '961ade45e8914cfc9cf3c67f24ce9cde4d0f9e5f1313207ebe958fe9172a35be',
    'IndustrialEnv._compute_step': '8e1c24c8e16e1cef7828408c43bb956d88a99ee7a44c4ea04d537385eb6ca675',
    'PhysicsAdapter.report': 'e0448c22f6d55910222e15cedf27e6ac8bace2b7acdf268a55750466424d5107',
}
_EMPTY = {}


def _source_digest(fn):
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _check_sources():
    current = {'IndustrialEnv.view': _ORIGINAL_METHODS['view'],
               'IndustrialEnv.micro_step': _ORIGINAL_METHODS['micro_step'],
               'IndustrialEnv._compute_step': _ORIGINAL_METHODS['_compute_step'],
               'PhysicsAdapter.report': _ORIGINAL_REPORT}
    for name, fn in current.items():
        expected = _SOURCE_SHA256[name]
        if expected is not None and _source_digest(fn) != expected:
            raise RuntimeError(f'{name} changed in the simulator source; update its copy in p2/perf.py '
                               'and the recorded SHA-256 before using the perf patches')


def _complete_tracked(self, fragment):
    _ORIGINAL_METHODS['_complete'](self, fragment)
    versions = self._scene.__dict__.get('_p2_parent_version')
    if versions is None:
        versions = self._scene.__dict__['_p2_parent_version'] = {}
    versions[fragment.parent_id] = versions.get(fragment.parent_id, 0) + 1


def _view_fast(self):
    if self._view_cache is not None:
        return self._view_cache
    now = self.time_s
    # --- changed: plain-Python accumulation in the original visiting order ---
    compute = [[0., 0., 0.], [0., 0., 0.]]
    tx_bits = [[[0., 0., 0.] for _ in range(3)] for _ in range(2)]
    tx_cycles = [[[0., 0., 0.] for _ in range(3)] for _ in range(2)]
    for f in self._active_fragments.values():
        stage = f.stage
        if stage == "cpu":
            row = compute[f.priority]
            row[f.destination] += f.compute_remaining
        elif stage == "u2u":
            b, c = tx_bits[f.priority][f.source], tx_cycles[f.priority][f.source]
            d = f.destination
            b[d] += f.transfer_remaining
            c[d] += f.compute_remaining
    counts = [[0., 0., 0.], [0., 0., 0.]]
    ages = [[0., 0., 0.], [0., 0., 0.]]
    for p in self._active_parents.values():
        if p.finish_s is None:
            a = p.arrival
            pr, so = a.priority, a.source
            counts[pr][so] += 1
            ages[pr][so] = max(ages[pr][so], now-a.birth_s)
    compute, tx_bits, tx_cycles = np.array(compute), np.array(tx_bits), np.array(tx_cycles)
    counts, ages = np.array(counts), np.array(ages)
    # --- unchanged below ---
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


def _holding_sum(self, o):
    """sum(o['priority_weights'][p] for every unfinished parent), exactly."""
    alive = getattr(self, '_alive_counts', None)
    weights = o["priority_weights"]
    if (alive is not None and len(weights) == 2
            and all(isinstance(w, (int, float)) and not isinstance(w, bool) and float(w).is_integer() and abs(w) < 2**20
                    for w in weights)
            and int(alive[0]) + int(alive[1]) < 2**30):
        n0, n1 = int(alive[0]), int(alive[1])
        if n0 + n1 == 0:
            return 0
        if __debug__ and n0 + n1 != len(self._active_parents):
            raise AssertionError('alive counters disagree with the active parents')
        if isinstance(weights[0], float) or isinstance(weights[1], float):
            return float(weights[0])*n0 + float(weights[1])*n1
        return weights[0]*n0 + weights[1]*n1
    return sum(o["priority_weights"][p.arrival.priority] for p in self._active_parents.values() if p.finish_s is None)


def _micro_step_fast(self, route):
    route = np.asarray(route, float)
    if route.shape != (2, 3, 3) or not np.isfinite(route).all() or np.any(route < 0) or not np.allclose(route.sum(axis=2), 1, atol=1e-10, rtol=0):
        raise ValueError("Expected finite nonnegative 2x3x3 simplex rows")
    if self._prepared_frame != self.step_index//self.frame_steps:
        raise RuntimeError("Prepare current arrivals before physical service")
    self._invalidate_view()
    o = self.config["objective"]
    holding = self.dt*_holding_sum(self, o)          # --- changed: exact O(1) sum ---
    ready = tuple(self.ready)
    self.ready.clear()
    energy = dict(upload=self._upload_step())
    self._route(ready, route)
    energy["cpu"] = self._compute_step()
    energy["u2u"] = self._transfer_step()
    deployment_segment = self.step_index % self.macro_steps < getattr(self, "adjustment_steps", 0)
    charge_propulsion = o["energy_mode"] == "full" or deployment_segment
    energy["propulsion"] = (float(self.physics._compute_tcom_flying_power(
        np.linalg.norm(self.velocity, axis=1)).sum()*self.dt) if charge_propulsion else 0.)
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
    objective_energy = sum(energy[k] for k in ("upload", "cpu", "u2u")) + o.get("propulsion_weight", 1.)*energy["propulsion"]
    cost = o["holding_weight"]*holding/o["holding_reference_task_s"] + (1-o["holding_weight"])*objective_energy/o["energy_reference_j"]
    self.cost_sum += cost
    if self._mobility is not None and self.step_index % self._mobility.steps == 0:
        self.physics.user_coords = self._mobility.move(self.physics.user_coords)
    if self.step_index % self.frame_steps == 0:
        self._publish("frame_boundary_before_arrivals")
    self._invalidate_view()
    return dict(cost=cost, holding_task_s=holding, energy_j=energy, time_s=self.time_s)


def _queued_or_cap(fragments, queue, cap_cycles):
    """The original sum(compute_remaining over queue) when it can matter, or None
    when an exact prefix already exceeds cap_cycles * (1 + 1e-6)."""
    if cap_cycles >= 0:
        limit = cap_cycles*(1+1e-6)
        prefix = 0.
        for fid in queue:
            prefix += fragments[fid].compute_remaining
            if prefix > limit:
                return None
    return sum(fragments[f].compute_remaining for f in queue)


def _compute_step_fast(self):
    self._invalidate_view()
    r, energy = self.config["resources"], 0.
    for node, maximum in enumerate(r["cpu_hz"]):
        # --- changed: the queued cycles of a class are summed only while they can
        # be below the available frequency (min() then returns the same object) ---
        q0 = _queued_or_cap(self.fragments, self.compute[0][node], maximum*self.dt)
        if q0 is None:
            high = maximum
        else:
            high = min(maximum, np.array([q0, 0.])[0]/self.dt)
        rest = maximum-high
        if rest == 0 and not isinstance(rest, bool):
            # min(0, x >= 0) returns its first argument, the 0 of maximum-high
            low = rest
        else:
            q1 = _queued_or_cap(self.fragments, self.compute[1][node], rest*self.dt)
            low = rest if q1 is None else min(rest, np.array([0., q1])[1]/self.dt)
        frequencies = (high, low)
        # --- unchanged below ---
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


def _task_caches(env):
    d = env._scene.__dict__
    caches = d.get('_p2_report_cache')
    if caches is None:
        caches = d['_p2_report_cache'] = [{}, {}]
    return caches


def _report_fast(self):
    e = self.env
    view = e.view()
    caches = _task_caches(e)
    old_p, old_f = caches
    versions = e._scene.__dict__.get('_p2_parent_version', _EMPTY)
    all_fragments = e.fragments
    # --- changed: reuse records whose mutable attributes are the same objects ---
    new_p, parents = {}, []
    for k, p in e._active_parents.items():
        ur, rt, uf, n, v = p.upload_remaining, p.route_time_s, p.upload_finish_s, len(p.fragment_ids), versions.get(k, 0)
        hit = old_p.get(k)
        if hit is None or hit[0] is not ur or hit[1] is not rt or hit[2] is not uf or hit[3] != n or hit[4] != v:
            a = p.arrival
            hit = (ur, rt, uf, n, v, Parent(k, a.user_id, a.source, a.priority,
                a.birth_s, a.input_bits, a.compute_cycles, ur,
                "routed" if rt is not None else "ground" if uf is None else "ready",
                rt, tuple(f for f in p.fragment_ids if all_fragments[f].finish_s is None),
                sum(all_fragments[f].finish_s is not None for f in p.fragment_ids)))
        new_p[k] = hit
        parents.append(hit[5])
    parents = tuple(parents)
    ratios = self._ratios
    new_f, fragments = {}, []
    for k, f in e._active_fragments.items():
        tr, cr, st, en, ra = f.transfer_remaining, f.compute_remaining, f.stage, f.entered_s, ratios[k]
        hit = old_f.get(k)
        if hit is None or hit[0] is not tr or hit[1] is not cr or hit[2] is not st or hit[3] is not en or hit[4] is not ra:
            hit = (tr, cr, st, en, ra, Fragment(k, f.parent_id, f.source, f.destination, f.priority,
                ra, f.input_bits, f.compute_cycles, tr, cr, "transfer" if st == "u2u" else "cpu", en))
        new_f[k] = hit
        fragments.append(hit[5])
    fragments = tuple(fragments)
    caches[0], caches[1] = new_p, new_f
    # --- unchanged ---
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
    # --- changed: the ready parents (stage "ready" <=> waiting in env.ready), in
    # ascending id order = their order in `parents` ---
    for pid in sorted(e.ready):
        p = new_p[pid][5]
        if p.stage == "ready":
            ready[p.source, p.priority] += (p.bits, p.cycles, 1)
    # --- unchanged below ---
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
