"""Paper-2 experiment entry point.

  python run.py pipeline [--workers 4] [--seeds 4101,4102,4103] [--smoke]   # everything, resumable
  python run.py bench                                                      # CPU vs GPU speed check
  python run.py train --stage M1|M2|M3|HoverSAC|TFOO|TOO|NC --seed 4101
  python run.py dev --seed 4101 --label ProposedUpper --candidate 400
  python run.py select --seed 4101 --labels ProposedUpper
  python run.py test --seed 4101 --method Proposed
  python run.py sweep --seed 4101 --method Proposed --param cpu_ghz
  python run.py report [--seeds ...]

Outputs go to --out (default: $P2_OUT or research/paper2/outputs).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2 import common as C  # noqa: E402  (also puts the v2.4.0 source on sys.path)


def seeds_arg(text):
    return tuple(int(s) for s in str(text).split(',') if s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('pipeline', 'bench', 'train', 'dev', 'select', 'test', 'sweep', 'report', 'plan'))
    p.add_argument('--out', default=None)
    p.add_argument('--smoke', action='store_true', help='50 s windows and tiny budgets: plumbing check only')
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds', type=seeds_arg)
    p.add_argument('--stage', choices=('M1', 'M2', 'M3', 'HoverSAC', 'TFOO', 'TOO', 'NC'))
    p.add_argument('--label')
    p.add_argument('--labels')
    p.add_argument('--candidate', type=int)
    p.add_argument('--method', choices=tuple(C.METHOD_SPEC))
    p.add_argument('--param', choices=tuple(C.SWEEPS))
    p.add_argument('--workers', type=int, default=int(os.environ.get('P2_WORKERS', '4')))
    p.add_argument('--no-sweeps', action='store_true')
    a = p.parse_args(argv)
    root = C.out_root(a.out)
    seeds = a.seeds or C.design(a.smoke)['seeds']

    if a.command in ('pipeline', 'plan'):
        from p2 import pipeline
        jobs = pipeline.plan(seeds, smoke=a.smoke, include_sweeps=not a.no_sweeps)
        C.write_json(root/'design.json', dict(design=C.design(a.smoke), seeds=list(seeds), smoke=a.smoke,
                                              methods=list(C.METHODS), method_spec=C.METHOD_SPEC,
                                              upper_initial_std=C.UPPER_INITIAL_STD, alpha_m1=C.ALPHA_M1,
                                              alpha_finetune=C.ALPHA_FINETUNE, selection_metric=C.SELECTION_METRIC))
        return pipeline.run(jobs, root, workers=a.workers, dry=a.command == 'plan')
    if a.command == 'bench':
        return bench(root)
    if a.command == 'train':
        from p2 import learn
        init = {}
        if a.stage in ('M2', 'M3', 'HoverSAC'):
            init['m1'] = C.run_dir(root, a.seed, 'M1')/'final.pt'
        if a.stage == 'M3':
            selected = C.read_json(C.seed_dir(root, a.seed)/'selected.json')
            init['upper'] = C.run_dir(root, a.seed, 'M2')/'ckpt'/f'ep{selected["ProposedUpper"]:04d}.pt'
        for path in init.values():
            if not Path(path).exists():
                raise FileNotFoundError(f'{a.stage} needs {path}')
        if a.stage in ('M1', 'M3', 'HoverSAC'):
            learn.train_sac_stage(a.stage, a.seed, root, smoke=a.smoke, init_paths=init)
        else:
            learn.train_ppo_stage(a.stage, a.seed, root, smoke=a.smoke, init_paths=init)
        return 0
    from p2 import evaluate
    if a.command == 'dev':
        print(evaluate.dev_job(root, a.seed, a.label, a.candidate, smoke=a.smoke))
    elif a.command == 'select':
        print(evaluate.select(root, a.seed, a.labels.split(','), smoke=a.smoke))
    elif a.command == 'test':
        evaluate.test_job(root, a.seed, a.method, smoke=a.smoke)
    elif a.command == 'sweep':
        evaluate.sweep_job(root, a.seed, a.method, a.param, smoke=a.smoke)
    elif a.command == 'report':
        from p2 import report, plots
        print('tables:', report.build(root, seeds, smoke=a.smoke))
        print('figures:', plots.build(root, seeds, smoke=a.smoke))
    return 0


def bench(root):
    """Time smoke-sized M1 (SAC learning) and M2 (PPO) on CPU and on GPU."""
    import subprocess
    import shutil
    import torch
    devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
    results = {}
    for device in devices:
        out = Path(root)/'_bench'/device.replace(':', '')
        shutil.rmtree(out, ignore_errors=True)
        env = dict(os.environ, P2_DEVICE=device)
        times = {}
        for stage in ('M1', 'M2'):
            t0 = time.perf_counter()
            subprocess.run([sys.executable, __file__, 'train', f'--stage={stage}', '--seed=4101', '--smoke',
                            f'--out={out}'], check=True, env=env)
            times[stage] = time.perf_counter()-t0
        results[device] = times
        print(f'{device}: M1 smoke {times["M1"]:.1f}s, M2 smoke {times["M2"]:.1f}s', flush=True)
    best = min(results, key=lambda d: sum(results[d].values()))
    print(f'\nFaster device here: {best}.  Use it with:  set P2_DEVICE={best}  (PowerShell: $env:P2_DEVICE="{best}")')
    C.write_json(Path(root)/'_bench'/'bench.json', dict(results=results, faster=best))
    return 0


if __name__ == '__main__':
    sys.exit(main())
