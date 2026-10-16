"""Paper-2 experiment entry point.

  python run.py pipeline [--workers 4] [--seeds 4101,4102,4103] [--smoke]   # everything, resumable
  python run.py bench                                                      # CPU vs GPU speed check
  python run.py train --stage M1|M2|M3|HoverSAC|TFOO|TOO|NC --seed 4101
  python run.py dev --seed 4101 --label ProposedUpper --candidate 400
  python run.py select --seed 4101 --labels ProposedUpper
  python run.py test --seed 4101 --method Proposed
  python run.py sweep --seed 4101 --method Proposed --param cpu_ghz
  python run.py sweep --seed 4101 --method Proposed --param cpu_ghz --saturated   # rho >= 0.95 points
  python run.py burst --seed 4101 --method Proposed     # controlled overload burst (p2/burst.py)
  python run.py report [--seeds ...]
  Design-v3 retrained sweep (PROTOCOL.md): every learned method trained from scratch at
  cpu_ghz 9/11/12/14, load 0.6/0.8/0.9/1.1, users 7/10/16/19 (terminals per region)
  python run.py verify --main D:\\p2_runs_v2 --work D:\\p2_verify   # this code reproduces the main run?
  python run.py timing --main D:\\p2_runs_v2 [--workers 8]           # expected duration on this computer
  python run.py retrain --out-base D:\\p2_sweeps_v3 --points cpu_ghz=9   # pilot point
  python run.py pilot-check --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2
  python run.py retrain --out-base D:\\p2_sweeps_v3                 # all points (resumable)
  python run.py retrain-report --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2
  python run.py retrain-export --out-base D:\\p2_sweeps_v3 --dest D:\\p2_export_A.zip --machine A
                                                           # zip the finished points' records (no weights)
  python run.py revise --from <old out> --out <new out>   # new root, fine-tuning recipe v2,
                                                           # unchanged stages carried over

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
    p.add_argument('command', choices=('pipeline', 'bench', 'train', 'dev', 'select', 'test', 'sweep', 'report', 'plan',
                                        'revise', 'burst', 'retrain', 'retrain-report',
                                        'retrain-export', 'verify', 'timing', 'pilot-check'))
    p.add_argument('--out', default=None)
    p.add_argument('--from', dest='source', default=None, help='revise: the finished output root to derive from')
    p.add_argument('--recipe', default='v2', help='revise: fine-tuning recipe of the new root')
    p.add_argument('--saturated', action='store_true', help='sweep: evaluate the rho >= 0.95 points')
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
    p.add_argument('--out-base', default=None, help='retrain: folder that holds one root per sweep point')
    p.add_argument('--points', default=None, help='retrain: e.g. cpu_ghz=9,users=7 (default: all design-v3 points)')
    p.add_argument('--first', default='', help='retrain: points scheduled before the others')
    p.add_argument('--main', default=None, help='retrain-report: the nominal output root')
    p.add_argument('--dry-run', action='store_true', help='retrain: list the jobs without running them')
    p.add_argument('--dest', default=None, help='retrain-export: zip file to write')
    p.add_argument('--machine', default='A', help='retrain-export: label of this computer')
    p.add_argument('--work', default=None, help='verify: scratch folder (not inside --main)')
    a = p.parse_args(argv)
    if a.command == 'verify':
        if not a.main or not a.work:
            p.error('verify needs --main (the finished nominal run) and --work (a scratch folder)')
        sys.path.insert(0, str(Path(__file__).resolve().parent/'tests'))
        import verify_regression
        return verify_regression.main(['--main', a.main, '--work', a.work] +
                                      (['--seed', str(a.seed)] if a.seed else []))
    if a.command == 'timing':
        from p2 import checks, retrain
        if not a.main:
            p.error('timing needs --main (the finished nominal run)')
        seeds = a.seeds or C.design(a.smoke)['seeds']
        points = retrain.parse_points(a.points) if a.points else retrain.all_points(a.smoke)
        print(checks.timing(Path(a.main), points, seeds, a.workers, smoke=a.smoke))
        return 0
    if a.command == 'pilot-check':
        from p2 import checks
        if not a.main or not a.out_base:
            p.error('pilot-check needs --main and --out-base')
        passed, text = checks.pilot_check(Path(a.main), Path(a.out_base), a.seeds or C.design(a.smoke)['seeds'])
        print(text)
        return 0 if passed else 1
    if a.command in ('retrain', 'retrain-report', 'retrain-export'):
        from p2 import pipeline, report, retrain
        if not a.out_base:
            p.error(f'{a.command} needs --out-base')
        base = Path(a.out_base)
        seeds = a.seeds or C.design(a.smoke)['seeds']
        points = retrain.parse_points(a.points) if a.points else retrain.all_points(a.smoke)
        if a.command == 'retrain-export':
            if not a.dest:
                p.error('retrain-export needs --dest')
            info = retrain.export(base, a.dest, points, seeds, a.machine, smoke=a.smoke)
            print(f"exported {', '.join(info['finished'])} ({info['files']} files)")
            print(f"not finished: {', '.join(info['not_finished']) or 'none'}")
            print(f"{info['dest']}  SHA-256 {info['sha256'].upper()}")
            return 0
        if a.command == 'retrain-report':
            if not a.main:
                p.error('retrain-report needs --main (the nominal output root)')
            print(report.retrain_build(Path(a.main), base, seeds, points, smoke=a.smoke))
            return 0
        for param, value in points:
            retrain.prepare_point(base, param, value, smoke=a.smoke)
        jobs = pipeline.retrain_plan(base, points, seeds, smoke=a.smoke, first=retrain.parse_points(a.first))
        return pipeline.run(jobs, base, workers=a.workers, dry=a.dry_run)
    if a.command == 'revise':
        from p2 import revise
        if not a.source or not a.out:
            p.error('revise needs --from and --out')
        seeds = a.seeds or C.design(a.smoke)['seeds']
        revise.prepare(Path(a.source), Path(a.out), seeds, recipe=a.recipe, smoke=a.smoke)
        return 0
    root = C.out_root(a.out)
    C.set_active_root(root)
    seeds = a.seeds or C.design(a.smoke)['seeds']

    if a.command in ('pipeline', 'plan'):
        from p2 import pipeline
        point = C.scene_override(root) is not None
        jobs = pipeline.plan(seeds, smoke=a.smoke, include_sweeps=not a.no_sweeps and not point,
                             include_burst=not point)
        C.write_json(root/'design.json', dict(design=C.design(a.smoke), seeds=list(seeds), smoke=a.smoke,
                                              methods=list(C.METHODS), method_spec=C.METHOD_SPEC,
                                              upper_initial_std=C.UPPER_INITIAL_STD, alpha_m1=C.ALPHA_M1,
                                              finetune=C.finetune(root), selection_metric=C.SELECTION_METRIC))
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
    elif a.command == 'burst':
        evaluate.burst_job(root, a.seed, a.method, smoke=a.smoke)
    elif a.command == 'sweep':
        evaluate.sweep_job(root, a.seed, a.method, a.param, smoke=a.smoke, saturated=a.saturated)
    elif a.command == 'report':
        from p2 import report, plots
        print('tables:', report.build(root, seeds, smoke=a.smoke))
        if C.scene_override(root) is None:          # sweep-point roots: tables only
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
