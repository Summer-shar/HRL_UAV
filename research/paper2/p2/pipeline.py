"""Dependency-aware job runner for the whole experiment (resumable).

Each job is a subprocess ``python run.py <args>``; a finished job leaves
``<out>/_jobs/<id>.done``.  Re-running the pipeline skips finished jobs and
resumes interrupted training from its last snapshot.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from . import common as C

RUN = C.PAPER2/'run.py'


class Job:
    def __init__(self, jid, args, deps=(), priority=0, out=None, marker=None):
        self.id, self.args, self.deps, self.priority = jid, list(args), list(deps), priority
        self.out = out                  # output root of this job (None: the pipeline's --out)
        self.marker = marker or jid     # name of its _jobs/<marker>.done inside that root


def plan(seeds, smoke=False, methods=C.METHODS, include_sweeps=True, include_burst=True,
         prefix='', out=None, priority_offset=0):
    """Job graph of one output root.  prefix/out/priority_offset let several roots
    share one scheduler (retrained sweep points); markers stay root-local."""
    d = C.design(smoke)
    flag = ['--smoke'] if smoke else []
    jobs = []

    def add(jid, args, deps=(), priority=0):
        jobs.append(Job(prefix+jid, args+flag, [prefix+x for x in deps], priority+priority_offset, out, jid))
        return jid

    for s in seeds:
        S = f'--seed={s}'
        m1 = add(f's{s}_train_M1', ['train', '--stage=M1', S], priority=100)
        m2 = add(f's{s}_train_M2', ['train', '--stage=M2', S], [m1], priority=90)
        hv = add(f's{s}_train_HoverSAC', ['train', '--stage=HoverSAC', S], [m1], priority=40)
        base = {b: add(f's{s}_train_{b}', ['train', f'--stage={b}', S], priority=60) for b in C.PPO_BASELINES}
        dpu = [add(f's{s}_dev_ProposedUpper_{e}', ['dev', S, '--label=ProposedUpper', f'--candidate={e}'], [m2], 85)
               for e in d['upper_candidates']]
        spu = add(f's{s}_select_upper', ['select', S, '--labels=ProposedUpper'], dpu, 84)
        m3 = add(f's{s}_train_M3', ['train', '--stage=M3', S], [spu], priority=80)
        devs = []
        for c in d['lower_candidates']:
            devs.append(add(f's{s}_dev_ProposedLower_{c}', ['dev', S, '--label=ProposedLower', f'--candidate={c}'],
                            [spu] if c == 0 else [m3], 75))
            devs.append(add(f's{s}_dev_HoverSAC_{c}', ['dev', S, '--label=HoverSAC', f'--candidate={c}'],
                            [m1] if c == 0 else [hv], 30))
        for b, tb in base.items():
            devs += [add(f's{s}_dev_{b}_{e}', ['dev', S, f'--label={b}', f'--candidate={e}'], [tb], 50)
                     for e in d['upper_candidates']]
        sel = add(f's{s}_select_all', ['select', S, '--labels=ProposedLower,HoverSAC,'+','.join(C.PPO_BASELINES)],
                  devs, 70)
        for m in methods:
            add(f's{s}_test_{m}', ['test', S, f'--method={m}'], [sel], 20)
            if include_burst:
                add(f's{s}_burst_{m}', ['burst', S, f'--method={m}'], [sel], 15)
            if include_sweeps:
                for p in d['sweeps']:
                    add(f's{s}_sweep_{m}_{p}', ['sweep', S, f'--method={m}', f'--param={p}'], [sel], 10)
                for p, values in d['saturated'].items():
                    if values:
                        add(f's{s}_sat_{m}_{p}', ['sweep', S, f'--method={m}', f'--param={p}', '--saturated'],
                            [sel], 5)
    finals = [j.marker for j in jobs if any(k in j.marker for k in ('_test_', '_sweep_', '_sat_', '_burst_'))]
    add('report', ['report', f'--seeds={",".join(map(str, seeds))}'], finals, 0)
    return jobs


def point_name(param, value):
    return f'{param}={float(value):g}'


def retrain_plan(base, points, seeds, smoke=False, first=(), main=None, gate=True):
    """All retrained sweep points in one job graph.  Priorities (higher first):
    points in `first`, then the pilot point, then the parameters in
    C.RETRAIN_ORDER, so that on one computer whole curves finish one after the
    other while idle workers still pick up later points.

    Pilot gate (PROTOCOL.md section 5): when the pilot point is planned together
    with other points, a job 'pilot_gate' runs `run.py pilot-check` as soon as the
    pilot's M1 runs are finished, and every job of the other points depends on
    it; if the check fails, only the pilot point goes on.  `main` is the nominal
    output root the check compares with."""
    jobs = []
    first = {(p, float(v)) for p, v in first}
    order = {p: len(C.RETRAIN_ORDER)-i for i, p in enumerate(C.RETRAIN_ORDER)}
    pilot = (C.PILOT_POINT[0], float(C.PILOT_POINT[1]))
    gated = gate and any((p, float(v)) == pilot for p, v in points) and len(points) > 1
    if gated and main is None:
        raise ValueError('the pilot gate needs the nominal output root (--main), or switch it off (--no-pilot-gate)')
    for param, value in points:
        name = point_name(param, value)
        key = (param, float(value))
        boost = (10000 if key in first else 0) + (5000 if key == pilot else 0) + 1000*order.get(param, 0)
        point_jobs = plan(seeds, smoke, include_sweeps=False, include_burst=False, prefix=name+'/',
                          out=Path(base)/name, priority_offset=boost)
        if gated and key != pilot:
            for j in point_jobs:
                j.deps.append('pilot_gate')
        jobs += point_jobs
    if gated:
        pilot_m1 = [f'{point_name(*pilot)}/s{s}_train_M1' for s in seeds]
        args = ['pilot-check', f'--out-base={base}', f'--main={main}', f'--seeds={",".join(map(str, seeds))}']
        jobs.append(Job('pilot_gate', args + (['--smoke'] if smoke else []), pilot_m1, 20000, Path(base), 'pilot_gate'))
    return jobs


def run(jobs, out, workers=4, dry=False):
    out = Path(out)
    jd = out/'_jobs'
    jd.mkdir(parents=True, exist_ok=True)
    root_of = {j.id: Path(j.out) if j.out is not None else out for j in jobs}
    marker = {j.id: root_of[j.id]/'_jobs'/f'{j.marker}.done' for j in jobs}
    log_of = {j.id: root_of[j.id]/'_jobs'/f'{j.marker}.log' for j in jobs}
    for path in marker.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    done = {j.id for j in jobs if marker[j.id].exists()}
    failed, running = set(), {}
    pending = [j for j in jobs if j.id not in done]
    if dry:
        for j in pending:
            print(j.id, ' '.join(j.args), 'deps=', ','.join(j.deps))
        return
    print(f'{len(done)} jobs already done, {len(pending)} to run with {workers} workers; logs in {jd}', flush=True)
    env = dict(os.environ, P2_OUT=str(out))
    while pending or running:
        blocked = {j.id for j in pending if any(dep in failed for dep in j.deps)}
        for j in [j for j in pending if j.id in blocked]:
            failed.add(j.id)
            print(f'SKIP {j.id} (dependency failed)', flush=True)
        pending = [j for j in pending if j.id not in blocked]
        ready = sorted((j for j in pending if all(dep in done for dep in j.deps)), key=lambda j: -j.priority)
        while ready and len(running) < workers:
            j = ready.pop(0)
            pending.remove(j)
            log = log_of[j.id].open('a', encoding='utf-8')
            log.write(f'\n=== start {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n'); log.flush()
            job_env = dict(env, P2_OUT=str(root_of[j.id]))
            proc = subprocess.Popen([sys.executable, str(RUN)]+j.args+[f'--out={root_of[j.id]}'], stdout=log,
                                    stderr=subprocess.STDOUT, cwd=str(C.PAPER2), env=job_env)
            running[j.id] = (proc, log, time.time())
            print(f'START {j.id}', flush=True)
        time.sleep(2)
        for jid, (proc, log, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[jid]
            minutes = (time.time()-t0)/60
            if code == 0:
                marker[jid].write_text(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                done.add(jid)
                print(f'DONE  {jid} ({minutes:.1f} min) [{len(done)}/{len(jobs)}]', flush=True)
            else:
                failed.add(jid)
                print(f'FAIL  {jid} (exit {code}, {minutes:.1f} min) -> see {log_of[jid]}', flush=True)
        if not running and pending and not any(all(dep in done for dep in j.deps) for j in pending):
            for j in pending:
                print(f'SKIP {j.id} (dependency failed)', flush=True)
            break
    if failed:
        print('Failed or skipped jobs: '+', '.join(sorted(failed)), flush=True)
        return 1
    print('All jobs finished.', flush=True)
    return 0
