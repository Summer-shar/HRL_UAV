"""The perf patches must not change a single bit of the simulation.

Runs scripted controllers with and without p2.perf and compares every lower
report (== and repr, i.e. values and types), every Lite observation vector,
every physical step reading and the full metrics dictionary, for the nominal
scene, an overloaded scene, a heavy-backlog scene (U2U 1.5 MHz) and the
design-v3 terminal counts 7 and 19 per region.
Usage:  python tests/test_perf_equivalence.py [seconds]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np

from p2 import common as C
from p2 import perf
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.metrics import finish, snapshot
from hrl_v2.timing import ClockConfig


def controller(kind, rng):
    def upper(report):
        if kind == 'hover':
            return np.zeros(6)
        return rng.uniform(-1, 1, 6)

    def lower(report, step):
        if kind == 'hover':
            return np.full((2, 3, 3), 1/3)
        if kind == 'local':
            return np.tile(np.eye(3), (2, 1, 1))
        # varied supports incl. exact zeros and single-destination rows
        route = rng.dirichlet(np.ones(3), size=(2, 3))
        route[rng.random((2, 3)) < .3] = np.eye(3)[rng.integers(3)]
        return route
    return upper, lower


def run(kind, scenario, seconds, scene, adapter_class=PhysicsAdapter):
    from hrl_v2.lite_observation import LiteObserver
    rng = np.random.default_rng(scenario)
    upper, lower = controller(kind, rng)
    a = adapter_class(scene, timing=ClockConfig.from_scene(scene, rollout_seconds=seconds))
    a.reset(**C.scenario_seeds('test', scenario))
    start = snapshot(a)
    observer = LiteObserver()
    reports, readings, vectors = [], [], []

    def seen(r):
        reports.append(r)
        o = observer.update(r, a.arrivals)
        vectors.append((o.lower.copy(), o.upper.copy()))
    for macro in range(int(seconds/5)):
        r = a.report()
        seen(r)
        a.start_macro(upper(r))
        for frame in range(10):
            rep = a.prepare()
            seen(rep)
            route = lower(rep, frame)
            for _ in range(5):
                readings.append(a.step(route))
    return reports, readings, finish(a, start), vectors


def same(x, y, path='metrics'):
    if isinstance(x, dict):
        assert set(x) == set(y), path
        for k in x:
            same(x[k], y[k], f'{path}.{k}')
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), path
        for i, (a, b) in enumerate(zip(x, y)):
            same(a, b, f'{path}[{i}]')
    elif isinstance(x, np.ndarray):
        assert np.array_equal(x, y), path
    elif isinstance(x, float) and np.isnan(x):
        assert isinstance(y, float) and np.isnan(y), path
    else:
        assert x == y, f'{path}: {x!r} != {y!r}'


def main(seconds=300.):
    scenes = [C.load_scene(), C.scaled_scene(C.load_scene(), cpu_ghz=8., load=1.2),
              C.scaled_scene(C.load_scene(), u2u_mhz=1.5), C.scaled_scene(C.load_scene(), users=7),
              C.scaled_scene(C.load_scene(), users=19)]
    cases = [('hover', 21, 0), ('random', 22, 0), ('local', 21, 0), ('random', 23, 1),
             ('hover', 24, 2), ('local', 25, 3), ('random', 26, 4), ('hover', 27, 4)]
    import time
    for kind, scenario, s in cases:
        perf.remove()
        t0 = time.perf_counter()
        ref = run(kind, scenario, seconds, scenes[s])
        t1 = time.perf_counter()
        assert perf.apply()
        new = run(kind, scenario, seconds, scenes[s])
        t2 = time.perf_counter()
        assert len(ref[0]) == len(new[0])
        for i, (a, b) in enumerate(zip(ref[0], new[0])):
            assert a == b, f'report {i} differs ({kind}, scenario {scenario}, scene {s})'
            assert repr(a) == repr(b), f'report {i} repr differs'
        same([dict(r) for r in ref[1]], [dict(r) for r in new[1]], 'readings')
        same(ref[2], new[2])
        for i, (u, v) in enumerate(zip(ref[3], new[3])):
            assert u[0].tobytes() == v[0].tobytes() and u[1].tobytes() == v[1].tobytes(), f'Lite vector {i} differs'
        print(f'OK {kind:6s} scenario {scenario} scene {s}: identical; '
              f'{t1-t0:.2f}s -> {t2-t1:.2f}s ({(t1-t0)/(t2-t1):.2f}x)', flush=True)
    perf.remove()
    # Training adapter: identical physics/rewards/observed fields, diagnostics skipped.
    from dataclasses import replace
    for kind, scenario in (('local', 21), ('random', 22)):
        perf.apply()
        t0 = time.perf_counter()
        ref = run(kind, scenario, seconds, scenes[0])
        t1 = time.perf_counter()
        new = run(kind, scenario, seconds, scenes[0], perf.TrainingAdapter)
        t2 = time.perf_counter()
        for i, (a, b) in enumerate(zip(ref[0], new[0])):
            a = replace(a, uavs=tuple(replace(u, recent_service=(0., 0.)) for u in a.uavs))
            b = replace(b, uavs=tuple(replace(u, recent_service=(0., 0.)) for u in b.uavs))
            assert a == b, f'training adapter report {i} differs'
        same([dict(r) for r in ref[1]], [dict(r) for r in new[1]], 'readings')
        m1 = {k: v for k, v in ref[2].items() if k != 'queue_integrals_physical_grid'}
        m2 = {k: v for k, v in new[2].items() if k != 'queue_integrals_physical_grid'}
        same(m1, m2)
        print(f'OK training adapter {kind} scenario {scenario}: identical except diagnostics; '
              f'{t1-t0:.2f}s -> {t2-t1:.2f}s ({(t1-t0)/(t2-t1):.2f}x)', flush=True)
    perf.remove()


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 300.)
