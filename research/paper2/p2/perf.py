"""Bitwise-identical speed-ups of two simulator hot spots (opt-in, no source edits).

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

``tests/test_perf_equivalence.py`` checks event digests, every scalar metric
and every report over several scenarios with and without the patch.
Set P2_NO_PERF=1 to disable.
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
    cls = _env_class()
    cls._publish = _publish_fast
    industrial_env.IndustrialPhysics.allocate_u2u_resources = _allocate_cached
    _APPLIED = True
    return True


def remove():
    global _APPLIED
    cls = _env_class()
    cls._publish = _ORIGINAL_PUBLISH_FN
    industrial_env.IndustrialPhysics.allocate_u2u_resources = _ORIGINAL_ALLOCATE
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
