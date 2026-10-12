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
    def __init__(self, jid, args, deps=(), priority=0):
        self.id, self.args, self.deps, self.priority = jid, list(args), list(deps), priority


def plan(seeds, smoke=False, methods=C.METHODS, include_sweeps=True):
    d = C.design(smoke)
    flag = ['--smoke'] if smoke else []
    jobs = []

    def add(jid, args, deps=(), priority=0):
        jobs.append(Job(jid, args+flag, deps, priority))
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
            if include_sweeps:
                for p in d['sweeps']:
                    add(f's{s}_sweep_{m}_{p}', ['sweep', S, f'--method={m}', f'--param={p}'], [sel], 10)
    finals = [j.id for j in jobs if '_test_' in j.id or '_sweep_' in j.id]
    jobs.append(Job('report', ['report']+flag+[f'--seeds={",".join(map(str, seeds))}'], finals, 0))
    return jobs


def run(jobs, out, workers=4, dry=False):
    out = Path(out)
    jd = out/'_jobs'
    jd.mkdir(parents=True, exist_ok=True)
    done = {j.id for j in jobs if (jd/f'{j.id}.done').exists()}
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
            log = (jd/f'{j.id}.log').open('a', encoding='utf-8')
            log.write(f'\n=== start {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n'); log.flush()
            proc = subprocess.Popen([sys.executable, str(RUN)]+j.args+[f'--out={out}'], stdout=log,
                                    stderr=subprocess.STDOUT, cwd=str(C.PAPER2), env=env)
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
                (jd/f'{jid}.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                done.add(jid)
                print(f'DONE  {jid} ({minutes:.1f} min) [{len(done)}/{len(jobs)}]', flush=True)
            else:
                failed.add(jid)
                print(f'FAIL  {jid} (exit {code}, {minutes:.1f} min) -> see {jd/(jid+".log")}', flush=True)
        if not running and pending and not any(all(dep in done for dep in j.deps) for j in pending):
            for j in pending:
                print(f'SKIP {j.id} (dependency failed)', flush=True)
            break
    if failed:
        print('Failed or skipped jobs: '+', '.join(sorted(failed)), flush=True)
        return 1
    print('All jobs finished.', flush=True)
    return 0
