"""Read-only checks around the design-v3 retrained sweep (torch-free).

  timing       per-stage episode times of the main run -> expected duration of
               the planned points on this computer (an estimate, printed with
               its inputs; the v3 code is faster than these logs under backlog)
  pilot_check  trainability check of the pilot point (PROTOCOL.md section 5),
               judged on M1 training curves only, never on test results
  progress     jobs done per point, measured job-hours, and the remaining time
               extrapolated from the points that have finished
"""
import re
import statistics
import time
from pathlib import Path

import numpy as np

from . import common as C

SAC_STAGES = ('M1', 'M3', 'HoverSAC')
PPO_STAGES = ('M2', 'TFOO', 'TOO', 'NC')


def _episode_times(main_root, seeds):
    """Median wall seconds per training episode of each stage, and per evaluation run."""
    out = {}
    for stage in SAC_STAGES+PPO_STAGES:
        vals = [r['wall_seconds'] for s in seeds for r in C.read_jsonl(C.run_dir(main_root, s, stage)/'train.jsonl')
                if r.get('wall_seconds') is not None]
        out[stage] = statistics.median(vals) if vals else None
    evals = {}
    for s in seeds:
        for p in sorted((C.seed_dir(main_root, s)/'test').glob('*/[0-9][0-9].json')):
            r = C.read_json(p)
            if r.get('wall_seconds') is not None:
                evals.setdefault(p.parent.name, []).append(r['wall_seconds'])
    out['eval'] = {m: statistics.median(v) for m, v in evals.items()}
    return out


def timing(main_root, points, seeds, workers, smoke=False):
    """Expected computer time of the planned points from the main run's own logs."""
    d = C.design(smoke)
    t = _episode_times(main_root, seeds)
    missing = [k for k in SAC_STAGES+PPO_STAGES if t[k] is None]
    if missing:
        raise ValueError(f'no training logs for {missing} under {main_root}')
    b = d['budget']
    train = (b['M1']*t['M1'] + b['M2']*t['M2'] + b['M3']*t['M3'] + b['HoverSAC']*t['HoverSAC'] +
             b['PPO']*(t['TFOO'] + t['TOO'] + t['NC']))
    ev = t['eval']
    mean_eval = float(np.mean(list(ev.values()))) if ev else 15.
    dev_runs = len(d['dev'])*(len(d['upper_candidates'])*(1+len(C.PPO_BASELINES)) + 2*len(d['lower_candidates']))
    test_runs = len(d['test'])*len(C.METHODS)
    per_seed_point = train + (dev_runs+test_runs)*mean_eval
    total = per_seed_point*len(seeds)*len(points)
    lines = ['Episode times of the main run (median wall seconds per 900 s episode):']
    lines += [f'  {k:8s} {t[k]:7.1f} s x {b["PPO"] if k in PPO_STAGES else b[k]} episodes' for k in SAC_STAGES+PPO_STAGES]
    lines += [f'  evaluation runs: ' + ', '.join(f'{m} {v:.1f} s' for m, v in sorted(ev.items()))]
    lines += ['', f'One seed at one point: {per_seed_point/3600:.1f} h of computer time '
              f'({train/3600:.1f} h training, {dev_runs} dev + {test_runs} test runs).',
              f'{len(points)} points x {len(seeds)} seeds: {total/3600:.0f} h of computer time; '
              f'with {workers} parallel workers about {total/3600/workers:.0f} h '
              f'({total/86400/workers:.1f} days) if the workers do not slow each other down.',
              'The v3 simulator patches are faster than these logs when the backlog is large (NC, early M1, '
              'demand close to capacity); after the first points finish, "progress" extrapolates from measured times.']
    return '\n'.join(lines)


def pilot_check(main_root, base, seeds, window=20, ratio=3.0, smoke=False, pilot=None):
    """PROTOCOL.md section 5: for every seed, a pilot point's M1 must (a) improve
    (last `window` episodes better than the first `window`) and (b) end within
    `ratio` x the nominal M1's last-window delay.  pilot: one point name such as
    'u2u_mhz=1.5' (default: every pilot point).  Returns (passed, text) with
    passed True / False / None (M1 not finished for every seed yet)."""
    from .pipeline import point_name
    pilots = [(p, C.point_value(p, v)) for p, v in C.PILOT_POINTS]
    if pilot is not None:
        pilots = [pt for pt in pilots if point_name(*pt) == pilot]
        if not pilots:
            raise ValueError(f'{pilot} is not a pilot point ({[point_name(*pt) for pt in C.PILOT_POINTS]})')
    budget = C.design(smoke)['budget']['M1']
    window = min(window, max(budget//2, 1))
    lines, status = [], []
    for param, value in pilots:
        name = point_name(param, value)
        root = Path(base)/name
        gated = sorted({p for p, g in C.GATED_BY.items() if (g[0], C.point_value(*g)) == (param, value)})
        lines.append(f'Pilot point {name} (gates: {", ".join(gated)}): M1 trainability check '
                     f'(last {window} episodes; threshold {ratio:g} x nominal)')
        for seed in seeds:
            rows = C.read_jsonl(C.run_dir(root, seed, 'M1')/'train.jsonl')
            nominal = C.read_jsonl(C.run_dir(main_root, seed, 'M1')/'train.jsonl')
            if len(rows) < budget:
                lines.append(f'  seed {seed}: M1 has {len(rows)}/{budget} episodes - not finished yet')
                status.append('pending')
                continue
            if len(nominal) < window or (not smoke and len(nominal) < budget):
                raise ValueError(f'nominal M1 log of seed {seed} not found under {main_root}')
            first = float(np.mean([r['truncated_delay_s'] for r in rows[:window]]))
            last = float(np.mean([r['truncated_delay_s'] for r in rows[-window:]]))
            ref = float(np.mean([r['truncated_delay_s'] for r in nominal[-window:]]))
            improve, bounded = last < first, last <= ratio*ref
            status.append('ok' if improve and bounded else 'fail')
            lines.append(f'  seed {seed}: first {first:.3f} s -> last {last:.3f} s (nominal last {ref:.3f} s); '
                         f'improves: {"yes" if improve else "NO"}; within {ratio:g}x nominal: {"yes" if bounded else "NO"}')
    passed = False if 'fail' in status else None if 'pending' in status else True
    lines.append('PASS: the gated points start' if passed else
                 'NOT FINISHED: run the check again later' if passed is None else
                 'FAIL: the gated points wait; stop and discuss (do not change the protocol silently)')
    return passed, '\n'.join(lines)


_START = re.compile(r'^=== start (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ===$', re.M)


def _stamp(text):
    return time.mktime(time.strptime(text.strip(), '%Y-%m-%d %H:%M:%S'))


def job_hours(root):
    """{job marker: hours} for the finished jobs of one output root (last start in the log -> .done stamp)."""
    out = {}
    for done in sorted((Path(root)/'_jobs').glob('*.done')):
        log = done.with_suffix('.log')
        starts = _START.findall(log.read_text(encoding='utf-8', errors='replace')) if log.exists() else []
        try:
            end = _stamp(done.read_text(encoding='utf-8').splitlines()[0])
        except (IndexError, ValueError):
            continue
        if starts:
            out[done.stem] = max(end-_stamp(starts[-1]), 0.)/3600
    return out


def progress(base, points, seeds, workers, smoke=False):
    """Text table: per point jobs done / planned and job-hours; remaining time extrapolated from the
    finished points (mean job-hours per point / workers, i.e. assuming every worker stays busy)."""
    from .pipeline import point_name, retrain_plan
    jobs = [j for j in retrain_plan(base, points, seeds, smoke=smoke, gate=False)]
    lines = [f'{"point":14s} {"jobs done":>10s} {"job-hours":>10s}  state']
    finished_hours, remaining_points, partial_hours = [], 0, 0.
    for param, value in points:
        name = point_name(param, value)
        planned = [j.marker for j in jobs if j.id.startswith(name+'/')]
        hours = job_hours(Path(base)/name)
        done = [m for m in planned if m in hours]
        h = sum(hours[m] for m in done)
        state = 'finished' if len(done) == len(planned) else 'running' if done else 'not started'
        if state == 'finished':
            finished_hours.append(h)
        else:
            remaining_points += 1
            partial_hours += h
        lines.append(f'{name:14s} {len(done):4d}/{len(planned):<5d} {h:10.1f}  {state}')
    lines.append('')
    if finished_hours:
        per_point = statistics.mean(finished_hours)
        left = max(per_point*remaining_points - partial_hours, 0.)
        lines.append(f'finished points: {len(finished_hours)}, mean {per_point:.1f} job-hours per point '
                     f'({min(finished_hours):.1f}-{max(finished_hours):.1f})')
        lines.append(f'remaining: {remaining_points} points, about {left:.0f} job-hours; with {workers} workers about '
                     f'{left/workers:.0f} h ({left/workers/24:.1f} days) if every worker stays busy')
        lines.append('Low-U2U and high-load points run slower than the average; read this as an estimate.')
    else:
        lines.append('no point has finished yet: no extrapolation (see "timing" for the estimate from the main run)')
    return '\n'.join(lines)

