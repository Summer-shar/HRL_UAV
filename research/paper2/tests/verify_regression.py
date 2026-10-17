"""Regression check against the finished main run (needs torch, cvxpy and the main run's files).

Re-runs, with the code of THIS checkout, the first episodes of every training
stage of one seed (M1, M3, HoverSAC: 3 episodes; M2, TFOO, TOO, NC: 5 episodes,
i.e. up to and including the first PPO update) and the test runs of every
method on two test scenarios, and compares every logged value with the main
run's own records (everything except wall_seconds).  The main run is only read.

    python tests/verify_regression.py --main D:\\p2_runs_v2 --work D:\\p2_verify

Exit code 0 = every record identical.  If something differs, run the same
command from a checkout of the code that produced the main run (commit d617f05,
or the commit recorded with it) with another --work folder: if that checkout
differs from the records in the same way, the difference comes from this
computer's packages, not from the code change; the two output files
(verify_<code>.json) can then be compared with each other.
The script only uses functions that already existed in d617f05.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

PAPER2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER2))
from p2 import common as C  # noqa: E402

SHORT = dict(M1=3, M3=3, HoverSAC=3, PPO=5)


def code_digest():
    files = sorted([*PAPER2.glob('p2/*.py'), *C.SRC.glob('hrl_v2/*.py'), *C.SRC.glob('hrl_uav/*.py'),
                    *C.SRC.glob('hrl_v2/configs/*.json')])
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


# Top-level record fields that later pipeline versions added on purpose.  The
# main run's M1 / M2 / TFOO / TOO / NC logs come from the v1 run (commit
# d458a73, carried into the v2 root by run.py revise) and lack them; a field
# that exists only in the new record is listed, never treated as a difference.
ADDED_FIELDS = {'actor_lr': 'SAC rows, commit cfdcf3f (fine-tuning recipe v2)',
                'users': 'evaluation rows of design-v3 users points (never at 13 per region)'}


def compare(new, old, skip=('wall_seconds',), top=True):
    """First difference between two JSON-like records, or None.  At the top level a
    field present only in the new record is tolerated when it is in ADDED_FIELDS."""
    if isinstance(new, dict) and isinstance(old, dict):
        keys = (set(new) | set(old)) - set(skip)
        for k in sorted(keys, key=str):
            if k not in old and k in new and top and k in ADDED_FIELDS:
                continue
            if k not in new or k not in old:
                return f'{k}: only in {"new" if k in new else "main run"}'
            d = compare(new[k], old[k], (), top=False)
            if d:
                return f'{k}.{d}'
        return None
    if isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)):
        if len(new) != len(old):
            return f'length {len(new)} != {len(old)}'
        for i, (a, b) in enumerate(zip(new, old)):
            d = compare(a, b, (), top=False)
            if d:
                return f'[{i}].{d}'
        return None
    return None if new == old else f'{new!r} != {old!r}'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--main', required=True, help='finished nominal output root (read only)')
    p.add_argument('--work', required=True, help='empty or reusable scratch folder for this check')
    p.add_argument('--seed', type=int, default=4101)
    p.add_argument('--scenarios', default='21,22')
    p.add_argument('--skip-train', action='store_true')
    a = p.parse_args(argv)
    main_root, work = Path(a.main).resolve(), Path(a.work).resolve()
    if work == main_root or main_root in work.parents or work in main_root.parents:
        raise SystemExit('--work must be a separate folder, neither inside nor around --main')
    for need in ('selected.json', 'train/M1/train.jsonl', 'train/M1/final.pt'):
        if not (C.seed_dir(main_root, a.seed)/need).exists():
            raise SystemExit(f'{C.seed_dir(main_root, a.seed)/need} not found: is --main the finished main run?')
    label = code_digest()
    root = work/f'train_{label}'
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if (main_root/'recipe.json').exists():
        shutil.copy2(main_root/'recipe.json', root/'recipe.json')
    from p2 import learn, evaluate, perf
    results = dict(code=label, environment=dict(python=sys.version.split()[0], platform=platform.platform()),
                   main=str(main_root), seed=a.seed, train={}, test={}, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    try:
        import numpy, torch
        results['environment'].update(numpy=numpy.__version__, torch=torch.__version__)
    except ImportError:
        pass
    seed, msd = a.seed, C.seed_dir(main_root, a.seed)
    if not a.skip_train:
        for k, v in SHORT.items():
            C.BUDGET[k] = v
        selected = C.read_json(msd/'selected.json')
        m1 = msd/'train'/'M1'/'final.pt'
        stages = [('M1', 'sac', {}), ('M2', 'ppo', dict(m1=m1)),
                  ('M3', 'sac', dict(m1=m1, upper=msd/'train'/'M2'/'ckpt'/f'ep{selected["ProposedUpper"]:04d}.pt')),
                  ('HoverSAC', 'sac', dict(m1=m1)), ('TFOO', 'ppo', {}), ('TOO', 'ppo', {}), ('NC', 'ppo', {})]
        for name, kind, init in stages:
            t0 = time.perf_counter()
            (learn.train_sac_stage if kind == 'sac' else learn.train_ppo_stage)(name, seed, root, init_paths=init)
            new = C.read_jsonl(C.run_dir(root, seed, name)/'train.jsonl')
            old = C.read_jsonl(msd/'train'/name/'train.jsonl')[:len(new)]
            diffs = [f'episode {r.get("episode")}: {d}' for r, o in zip(new, old) for d in [compare(r, o)] if d]
            if len(old) < len(new):
                diffs.append(f'main run has only {len(old)} logged episodes')
            results['train'][name] = dict(episodes=len(new), identical=not diffs, first_difference=diffs[:3],
                                          seconds=round(time.perf_counter()-t0, 1))
            print(f'train {name:8s} {len(new)} episodes: {"identical" if not diffs else "DIFFERENT: "+diffs[0]}', flush=True)
    scenarios = [int(s) for s in a.scenarios.split(',') if s]
    scene = C.load_scene()
    for method in C.METHODS:
        m_, learner_path, upper_path = evaluate.method_paths(main_root, seed, method)
        make = evaluate.factory(m_, seed, learner_path, upper_path)
        diffs = []
        for sc in scenarios:
            out = work/f'test_{label}'/method/f'{sc:02d}.json'
            if out.exists():
                out.unlink()
            new = evaluate.evaluate_one(make, scene, 'test', sc, out,
                                        identity=dict(seed=seed, method=method,
                                                      learner=str(learner_path) if learner_path else None,
                                                      upper=str(upper_path) if upper_path else None))
            old_path = msd/'test'/method/f'{sc:02d}.json'
            if not old_path.exists():
                diffs.append(f'scenario {sc}: no main-run record')
                continue
            d = compare(C.read_json(out), C.read_json(old_path), skip=('wall_seconds', 'learner', 'upper'))
            if d:
                diffs.append(f'scenario {sc}: {d}')
        results['test'][method] = dict(scenarios=scenarios, identical=not diffs, first_difference=diffs[:3])
        print(f'test  {method:8s} scenarios {scenarios}: {"identical" if not diffs else "DIFFERENT: "+diffs[0]}', flush=True)
    results['perf_patches_active'] = perf.apply()
    ok = all(v['identical'] for v in results['train'].values()) and all(v['identical'] for v in results['test'].values())
    results['all_identical'] = ok
    out = work/f'verify_{label}.json'
    C.write_json(out, results)
    print(('ALL IDENTICAL' if ok else 'DIFFERENCES FOUND') + f' -> {out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
