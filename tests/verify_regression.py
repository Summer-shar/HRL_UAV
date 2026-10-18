"""Regression check against the finished main run (needs torch, cvxpy and the main run's files).

Re-runs, with the code of THIS checkout, the first episodes of every training
stage of one seed (M1, M3, HoverSAC: 3 episodes; M2, TFOO, TOO, NC: 5 episodes,
i.e. up to and including the first PPO update) and the test runs of every
method on two test scenarios, and compares every logged value with the main
run's own records (everything except wall_seconds; every field is compared and
every value difference counted, not only the first).  Fields that the pipeline
added after the main run was recorded (ADDED_FIELDS) are listed, not compared.
The main run is only read.

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from p2 import common as C  # noqa: E402

SHORT = dict(M1=3, M3=3, HoverSAC=3, PPO=5)


def code_digest():
    files = sorted([*C.SRC.glob('p2/*.py'), *C.SRC.glob('hrl_v2/*.py'), *C.SRC.glob('hrl_uav/*.py'),
                    *C.SRC.glob('hrl_v2/configs/*.json')])
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


# Top-level record fields that later pipeline versions added on purpose.  The
# main run carries records of older pipeline versions (run.py revise): its
# M1 / M2 / TFOO / TOO / NC training logs come from the v1 run (commit d458a73)
# and its TFOO / TOO / NC test records from the first run before d458a73.  A
# field that exists only in the new record and is listed here is reported, never
# treated as a difference.  The list was checked against copies of every
# main-run record (paper repository results_v2/raw, results_real/raw): these are
# the only fields the new code writes that some main-run records lack.
ADDED_FIELDS = {'actor_lr': 'SAC training rows, added in commit cfdcf3f (fine-tuning recipe v2)',
                'u2u_mhz': 'evaluation records, added in commit d458a73 (U2U sweep); the main run\'s '
                           'TFOO / TOO / NC test records were written before it',
                'users': 'evaluation records of design-v3 users points (never at 13 per region)'}


def differences(new, old, skip=(), path='', top=True):
    """Every difference between two JSON-like records, as (kind, path, detail).
    kind 'added': a top-level field in ADDED_FIELDS present only in the new record;
    kind 'value': anything else (a different value, or a field missing on one side)."""
    if isinstance(new, dict) and isinstance(old, dict):
        out = []
        for k in sorted((set(new) | set(old)) - set(skip), key=str):
            p = f'{path}.{k}' if path else str(k)
            if k not in old:
                out.append(('added' if top and k in ADDED_FIELDS else 'value', p, 'only in new'))
            elif k not in new:
                out.append(('value', p, 'only in main run'))
            else:
                out += differences(new[k], old[k], (), p, False)
        return out
    if isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)):
        if len(new) != len(old):
            return [('value', path, f'length {len(new)} != {len(old)}')]
        out = []
        for i, (a, b) in enumerate(zip(new, old)):
            out += differences(a, b, (), f'{path}[{i}]', False)
        return out
    return [] if new == old else [('value', path, f'{new!r} != {old!r}')]


def compare(new, old, skip=('wall_seconds',), top=True):
    """First value difference as text, or None (fields in ADDED_FIELDS tolerated)."""
    d = [x for x in differences(new, old, skip, top=top) if x[0] == 'value']
    return f'{d[0][1]}: {d[0][2]}' if d else None


def summarize(pairs):
    """pairs: [(label, new, old, skip)] -> (value differences as text, sorted added fields)."""
    values, added = [], set()
    for label, new, old, skip in pairs:
        for kind, path, detail in differences(new, old, skip):
            if kind == 'added':
                added.add(path)
            else:
                values.append(f'{label}: {path}: {detail}')
    return values, sorted(added)


def verdict(values, added):
    if values:
        return f'DIFFERENT ({len(values)} values): ' + values[0]
    return 'identical' + (f' (fields added after the main run, not compared: {", ".join(added)})' if added else '')


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
            diffs, added = summarize([(f'episode {r.get("episode")}', r, o, ('wall_seconds',))
                                      for r, o in zip(new, old)])
            if len(old) < len(new):
                diffs.append(f'main run has only {len(old)} logged episodes')
            results['train'][name] = dict(episodes=len(new), identical=not diffs, value_differences=len(diffs),
                                          first_differences=diffs[:10], added_fields=added,
                                          seconds=round(time.perf_counter()-t0, 1))
            print(f'train {name:8s} {len(new)} episodes: {verdict(diffs, added)}', flush=True)
    scenarios = [int(s) for s in a.scenarios.split(',') if s]
    scene = C.load_scene()
    for method in C.METHODS:
        m_, learner_path, upper_path = evaluate.method_paths(main_root, seed, method)
        make = evaluate.factory(m_, seed, learner_path, upper_path)
        diffs, pairs = [], []
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
            pairs.append((f'scenario {sc}', C.read_json(out), C.read_json(old_path), ('wall_seconds', 'learner', 'upper')))
        values, added = summarize(pairs)
        diffs += values
        results['test'][method] = dict(scenarios=scenarios, identical=not diffs, value_differences=len(diffs),
                                       first_differences=diffs[:10], added_fields=added)
        print(f'test  {method:8s} scenarios {scenarios}: {verdict(diffs, added)}', flush=True)
    results['perf_patches_active'] = perf.apply()
    ok = all(v['identical'] for v in results['train'].values()) and all(v['identical'] for v in results['test'].values())
    results['all_identical'] = ok
    added = sorted({f for part in ('train', 'test') for v in results[part].values() for f in v['added_fields']})
    results['added_fields'] = {f: ADDED_FIELDS[f] for f in added}
    out = work/f'verify_{label}.json'
    C.write_json(out, results)
    for f in added:
        print(f'note: {f} exists only in the new records ({ADDED_FIELDS[f]})')
    print(('ALL IDENTICAL' if ok else 'DIFFERENCES FOUND') + f' -> {out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
