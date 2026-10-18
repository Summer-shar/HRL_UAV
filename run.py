"""MH-TPCO experiments (paper 2): single entry point.  See README.md.

  python run.py check-env [--out D:\\p2_check]   # versions, simulator, smoke training on this computer
  python run.py feasibility --main D:\\p2_runs_v2 --out-base D:\\p2_sweeps_v3   # all checks, then the two
                          # pilot points; status in <out-base>\\FEASIBILITY.md (p2/feasibility.py)
  python run.py pipeline [--workers 4] [--seeds 4101,4102,4103] [--smoke]   # main experiment, resumable
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
  cpu_ghz 9/11/12/14, load 0.6/0.8/0.9/1.1, users 7/10/16/19 (terminals per region),
  u2u_mhz 1.5/2/2.5/4
  python run.py verify --main D:\\p2_runs_v2 --work D:\\p2_verify   # this code reproduces the main run?
  python run.py timing --main D:\\p2_runs_v2 [--workers 8]           # expected duration on this computer
  python run.py retrain --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2  # all points (resumable); the
                          # other points wait for their pilot (cpu_ghz=9 or u2u_mhz=1.5) to pass pilot-check
  python run.py pilot-check --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2  # the same check by hand
  python run.py progress --out-base D:\\p2_sweeps_v3 [--workers 8]   # jobs done, measured hours, time left
  python run.py retrain-report --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2
  python run.py retrain-export --out-base D:\\p2_sweeps_v3 --dest D:\\p2_sweeps_v3_records.zip
                                                           # zip the finished points' records (no weights)
  python run.py revise --from <old out> --out <new out>   # new root, fine-tuning recipe v2,
                                                           # unchanged stages carried over

Outputs go to --out (default: $P2_OUT or outputs/ next to run.py).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from p2 import common as C  # noqa: E402


def seeds_arg(text):
    return tuple(int(s) for s in str(text).split(',') if s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('check-env', 'pipeline', 'bench', 'train', 'dev', 'select', 'test', 'sweep',
                                        'report', 'plan', 'revise', 'burst', 'retrain', 'retrain-report',
                                        'retrain-export', 'verify', 'timing', 'pilot-check', 'progress',
                                        'feasibility'))
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
    p.add_argument('--workers', type=int, default=None,
                   help='parallel processes (default: $P2_WORKERS or 4; feasibility: logical CPUs / P2_THREADS, '
                        'at most 8)')
    p.add_argument('--no-sweeps', action='store_true')
    p.add_argument('--out-base', default=None, help='retrain: folder that holds one root per sweep point')
    p.add_argument('--points', default=None, help='retrain: e.g. cpu_ghz=9,users=7 (default: all design-v3 points)')
    p.add_argument('--first', default='', help='retrain: points scheduled before the others')
    p.add_argument('--main', default=None, help='retrain-report: the nominal output root')
    p.add_argument('--dry-run', action='store_true', help='retrain: list the jobs without running them')
    p.add_argument('--dest', default=None, help='retrain-export: zip file to write')
    p.add_argument('--machine', default='local', help='retrain-export: label written into the manifest')
    p.add_argument('--work', default=None, help='verify: scratch folder (not inside --main)')
    p.add_argument('--pilot', default=None, help='pilot-check: one pilot point, e.g. u2u_mhz=1.5 (default: all)')
    p.add_argument('--no-pilot-gate', action='store_true',
                   help='retrain: do not hold the other points until the pilot passes pilot-check')
    a = p.parse_args(argv)
    if a.workers is None:
        auto = max(1, min(8, (os.cpu_count() or 2)//int(os.environ.get('P2_THREADS') or 2)))
        a.workers = int(os.environ.get('P2_WORKERS') or (auto if a.command == 'feasibility' else 4))
    if a.command == 'feasibility':
        from p2 import feasibility
        if not a.main or not a.out_base:
            p.error('feasibility needs --main (the finished main run) and --out-base')
        device = feasibility.device_from_main(a.main)
        if device:
            print(f'P2_DEVICE={device} (from the main run\'s M1 contract)')
        return feasibility.Feasibility(a.main, a.out_base, a.workers, a.seeds or C.design()['seeds']).run()
    if a.command == 'check-env':
        return check_env(Path(a.out) if a.out else C.REPO/'outputs'/'_check_env')
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
    if a.command == 'progress':
        from p2 import checks, retrain
        if not a.out_base:
            p.error('progress needs --out-base')
        points = retrain.parse_points(a.points) if a.points else retrain.all_points(a.smoke)
        print(checks.progress(Path(a.out_base), points, a.seeds or C.design(a.smoke)['seeds'], a.workers,
                              smoke=a.smoke))
        return 0
    if a.command == 'pilot-check':
        from p2 import checks
        if not a.main or not a.out_base:
            p.error('pilot-check needs --main and --out-base')
        passed, text = checks.pilot_check(Path(a.main), Path(a.out_base), a.seeds or C.design(a.smoke)['seeds'],
                                          smoke=a.smoke, pilot=a.pilot)
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
        try:
            jobs = pipeline.retrain_plan(base, points, seeds, smoke=a.smoke, first=retrain.parse_points(a.first),
                                         main=a.main, gate=not a.no_pilot_gate)
        except ValueError as e:
            p.error(str(e))
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
        if not (root/'recipe.json').exists() and not any(root.glob('seed_*')):
            # a new output root uses the fine-tuning recipe of the paper (v2); roots that already hold
            # results without recipe.json are v1 roots of the first run and stay v1 (C.recipe_name)
            C.write_json(root/'recipe.json', dict(finetune='v2', parameters=C.FINETUNE_RECIPES['v2'],
                                                  note='new output root: fine-tuning recipe of the paper'))
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


def check_env(out):
    """Can this computer run the experiments?  Versions, the simulator, and two smoke trainings
    (SAC stage M1 and PPO baseline TFOO with its CVX lower layer) in a scratch folder."""
    import platform
    import shutil
    import subprocess
    from importlib import metadata
    ok = True
    print(f'python {sys.version.split()[0]}  ({sys.executable})\nplatform {platform.platform()}')
    for pkg in ('numpy', 'scipy', 'torch', 'cvxpy', 'clarabel', 'matplotlib'):
        try:
            version = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            version, ok = 'MISSING', False
        print(f'  {pkg:10s} {version}')
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f'torch CUDA available: {cuda}' + (f' ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})'
                                                 if cuda else ''))
    except Exception as e:                              # noqa: BLE001
        print('torch import failed:', e)
        ok = False
    print(f'P2_DEVICE={os.environ.get("P2_DEVICE") or "(not set: cuda:0 if available, else cpu)"}  '
          f'P2_THREADS={os.environ.get("P2_THREADS") or "(not set: 2)"}')
    try:
        import cvxpy
        solvers = cvxpy.installed_solvers()
        print('cvxpy solvers:', ', '.join(solvers))
        if 'CLARABEL' not in solvers:
            print('  CLARABEL missing: the TFOO baseline needs it (pip install clarabel)')
            ok = False
    except Exception as e:                              # noqa: BLE001
        print('cvxpy import failed:', e)
        ok = False
    import numpy as np
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.timing import ClockConfig
    scene = C.load_scene()
    t0 = time.perf_counter()
    adapter = PhysicsAdapter(scene, timing=ClockConfig.from_scene(scene, rollout_seconds=20.))
    adapter.reset(**C.scenario_seeds('test', 21))
    for _ in range(4):
        adapter.start_macro(np.zeros(6))
        for _ in range(10):
            adapter.prepare()
            for _ in range(5):
                adapter.step(np.full((2, 3, 3), 1/3))
    print(f'simulator: 20 s of the nominal scene in {time.perf_counter()-t0:.1f} s, '
          f'{adapter.env.physics.M} terminals')
    if not ok:
        print('\nCHECK FAILED: install the missing packages first (README.md, section 1)')
        return 1
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    for stage in ('M1', 'TFOO'):
        t0 = time.perf_counter()
        log = out/f'{stage}.log'
        with open(log, 'w', encoding='utf-8') as f:
            r = subprocess.run([sys.executable, __file__, 'train', f'--stage={stage}', '--seed=4101', '--smoke',
                                f'--out={out}'], stdout=f, stderr=subprocess.STDOUT)
        print(f'smoke training {stage}: {"ok" if r.returncode == 0 else "FAILED (see "+str(log)+")"} '
              f'in {time.perf_counter()-t0:.0f} s')
        ok = ok and r.returncode == 0
    print('\nALL CHECKS PASSED' if ok else '\nCHECK FAILED')
    return 0 if ok else 1


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
