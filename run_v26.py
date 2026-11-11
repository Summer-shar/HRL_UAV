"""Version 2.6 entry point (new file; run.py and the v2.5.0 code are unchanged).

Whole run, one command (status page <out-base>/V26.md, pushed to GitHub branch paper2-runs):

  python run_v26.py go --main D:\\p2_runs_v2 --out-base D:\\p2_v26 --zeroshot-main D:\\p2_zeroshot_v4

Single steps (used by `go`; --main and --out-base always required):

  python run_v26.py prepare
  python run_v26.py train    --variant ext|burst --stage M3|HoverSAC --seed 4101
  python run_v26.py dev      --variant ext|burst --label ProposedLower|HoverSAC --candidate 125 --seed 4101
  python run_v26.py select   --variant ext|burst --seed 4101
  python run_v26.py burstdev --variant main|ext|burst --seed 4101
  python run_v26.py ablation --seed 4101
  python run_v26.py decide
  python run_v26.py test | burst --variant ext|burst --method Proposed|HoverSAC --seed 4101
  python run_v26.py zeroshot --variant ext|burst --method Proposed|HoverSAC --seed 4101 --point cpu_ghz=9
  python run_v26.py report [--zeroshot-main D:\\p2_zeroshot_v4]
  python run_v26.py plan                                   # list the jobs of phase 1

Design and decision rule: docs/V26.md.  The main output root is only read.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from p2 import common as C  # noqa: E402
from v26 import core as V  # noqa: E402


def seeds_arg(text):
    return tuple(int(s) for s in str(text).split(',') if s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('go', 'prepare', 'train', 'dev', 'select', 'burstdev', 'ablation', 'decide',
                                        'test', 'burst', 'zeroshot', 'report', 'plan'))
    p.add_argument('--main', required=True, help='finished main output root (read only)')
    p.add_argument('--out-base', required=True, help='new folder for everything v2.6 writes')
    p.add_argument('--zeroshot-main', default=None, help='report: the finished zero-shot sweep of the main run')
    p.add_argument('--variant', choices=('main',)+V.VARIANTS)
    p.add_argument('--stage', choices=tuple(V.LABELS.values()))
    p.add_argument('--label', choices=tuple(V.LABELS))
    p.add_argument('--method', choices=tuple(V.METHOD_LABEL))
    p.add_argument('--candidate', type=int)
    p.add_argument('--point')
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds', type=seeds_arg)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--smoke', action='store_true', help='tiny budgets and 50 s windows: plumbing check only')
    p.add_argument('--no-publish', action='store_true', help='go: do not push the status to GitHub')
    a = p.parse_args(argv)
    main_root, base = V.check_roots(a.main, a.out_base)
    seeds = a.seeds or V._ORIGINAL_DESIGN(a.smoke)['seeds']
    if a.workers is None:
        a.workers = int(os.environ.get('P2_WORKERS') or
                        max(1, min(8, (os.cpu_count() or 2)//int(os.environ.get('P2_THREADS') or 2))))
    if not os.environ.get('P2_DEVICE'):
        from p2 import feasibility
        feasibility.device_from_main(main_root)          # same device as the main run (sets P2_DEVICE)

    def need(*names):
        missing = [n for n in names if getattr(a, n) is None]
        if missing:
            p.error(f'{a.command} needs ' + ', '.join('--'+n for n in missing))

    if a.command == 'go':
        from v26.stage import Go
        return Go(main_root, base, a.workers, seeds, smoke=a.smoke, publish=not a.no_publish,
                  zeroshot_main=a.zeroshot_main).run()
    if a.command == 'plan':
        for j in V.plan_phase1(seeds, a.smoke):
            print(j.priority, j.id, ' '.join(j.args), 'deps=', ','.join(j.deps))
        return 0
    if a.command == 'prepare':
        print('files copied:', V.prepare(main_root, base, seeds, a.smoke))
    elif a.command == 'train':
        need('variant', 'stage', 'seed')
        V.train_job(base, main_root, a.variant, a.stage, a.seed, a.smoke)
    elif a.command == 'dev':
        need('variant', 'label', 'candidate', 'seed')
        print(V.dev_job(base, a.variant, a.label, a.candidate, a.seed, a.smoke))
    elif a.command == 'select':
        need('variant', 'seed')
        print(V.select_job(base, a.variant, a.seed, a.smoke))
    elif a.command == 'burstdev':
        need('variant', 'seed')
        V.burstdev_job(base, main_root, a.variant, a.seed, a.smoke)
    elif a.command == 'ablation':
        need('seed')
        V.ablation_job(base, main_root, a.seed, a.smoke)
    elif a.command == 'decide':
        print(V.decision_text(V.decide(base, seeds, a.smoke)))
    elif a.command in ('test', 'burst'):
        need('variant', 'method', 'seed')
        (V.test_job if a.command == 'test' else V.burst_job)(base, main_root, a.variant, a.seed, a.method, a.smoke)
    elif a.command == 'zeroshot':
        need('variant', 'method', 'seed', 'point')
        V.zeroshot_job(base, main_root, a.variant, a.seed, a.method, a.point, a.smoke)
    elif a.command == 'report':
        from v26 import report
        print(report.build(base, main_root, seeds, a.smoke, a.zeroshot_main))
    return 0


if __name__ == '__main__':
    sys.exit(main())
