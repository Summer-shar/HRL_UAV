"""Bitwise consistency checks of the zero-shot evaluation (design v4); run.py zeroshot-check.

nominal     The zero-shot path (evaluate.zeroshot_test_job) in a check root whose
            scene is the nominal one, for every method, seed and test scenario,
            compared with the main run's own test records.  This ties the
            zero-shot evaluation to the numbers of the main table.
regression  Points already evaluated under <out-base> (one per parameter of the
            first v4 run, REGRESSION_POINTS) recomputed with this code for one
            seed and REGRESSION_METHODS, compared with the records there.  This
            justifies adding new points to an output folder whose records were
            written by an earlier commit.  Points that do not exist yet are skipped.

Every field is compared except wall_seconds, the checkpoint paths and trained_at
(a label only the zero-shot records carry); fields that the pipeline added after
the main run (tests/verify_regression.ADDED_FIELDS) are listed, not compared.
New records go to <out-base>/_zeroshot/consistency/; existing records are only read.
The result is written to <out-base>/_zeroshot/consistency.json.
"""
import sys
import time
from pathlib import Path

from . import common as C

NOMINAL_SCENE = dict(param='cpu_ghz', value=C.NOMINAL['cpu_ghz'])   # scaled_scene leaves the scene unchanged
REGRESSION_POINTS = (('cpu_ghz', 11.0), ('load', 0.8), ('users', 10), ('u2u_mhz', 2.5))
REGRESSION_METHODS = ('Proposed', 'TFOO')
SKIP = ('wall_seconds', 'learner', 'upper', 'trained_at')


def work_dir(base):
    return Path(base)/'_zeroshot'/'consistency'


def _verify_module():
    tests = str(C.REPO/'tests')
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import verify_regression
    return verify_regression


def prepare(base, main, seeds, smoke=False, methods=C.METHODS, regression_methods=REGRESSION_METHODS,
            regression_points=REGRESSION_POINTS):
    """Check roots and their jobs: [(kind, root, reference_root, seeds, methods)], [Job]."""
    from .pipeline import Job, point_name
    work = work_dir(base)
    checks = []
    nominal = work/'nominal'
    nominal.mkdir(parents=True, exist_ok=True)
    C.write_json(nominal/'scene.json', dict(NOMINAL_SCENE, design='v4', main=str(Path(main).resolve()),
                                            purpose='consistency check: the zero-shot path at the nominal scene'))
    checks.append(('nominal', nominal, Path(main), tuple(seeds), tuple(methods)))
    for param, value in regression_points:
        name = point_name(param, value)
        existing = Path(base)/name
        if not (existing/'scene.json').exists() or not any((C.seed_dir(existing, seeds[0])/'test').glob('*/*.json')):
            continue
        root = work/'regression'/name
        root.mkdir(parents=True, exist_ok=True)
        scene = dict(C.read_json(existing/'scene.json'), purpose=f'consistency check: recompute {existing}')
        C.write_json(root/'scene.json', scene)
        checks.append(('regression', root, existing, (seeds[0],), tuple(regression_methods)))
    jobs = []
    for kind, root, _, check_seeds, check_methods in checks:
        for seed in check_seeds:
            for method in check_methods:
                marker = f's{seed}_check_{method}'
                args = ['zeroshot-test', f'--seed={seed}', f'--method={method}', f'--main={Path(main).resolve()}'] + \
                    (['--smoke'] if smoke else [])
                slow = 2000 if method in ('NC', 'TFOO') else 1000
                jobs.append(Job(f'{kind}/{root.name}/{marker}', args, [], slow, root, marker))
    return checks, jobs


def compare(checks, smoke=False):
    """Every record of the check roots against its reference record."""
    V = _verify_module()
    d = C.design(smoke)
    out = dict(checks=[], value_differences=0, missing=0, compared=0, added_fields=[])
    added = set()
    for kind, root, ref, check_seeds, check_methods in checks:
        entry = dict(kind=kind, root=str(root), reference=str(ref), seeds=list(check_seeds),
                     methods=list(check_methods), compared=0, missing=[], differences=[])
        for seed in check_seeds:
            for method in check_methods:
                for sc in d['test']:
                    new_path = C.seed_dir(root, seed)/'test'/method/f'{sc:02d}.json'
                    old_path = C.seed_dir(ref, seed)/'test'/method/f'{sc:02d}.json'
                    if not new_path.exists() or not old_path.exists():
                        entry['missing'].append(f'seed {seed} {method} scenario {sc}: '
                                                f'{"new" if not new_path.exists() else "reference"} record missing')
                        continue
                    for k, path, detail in V.differences(C.read_json(new_path), C.read_json(old_path), SKIP):
                        if k == 'added':
                            added.add(path)
                        else:
                            entry['differences'].append(f'seed {seed} {method} scenario {sc}: {path}: {detail}')
                    entry['compared'] += 1
        entry['identical'] = not entry['differences'] and not entry['missing']
        out['checks'].append(entry)
        out['compared'] += entry['compared']
        out['value_differences'] += len(entry['differences'])
        out['missing'] += len(entry['missing'])
    out['added_fields'] = {f: V.ADDED_FIELDS[f] for f in sorted(added)}
    out['all_identical'] = bool(out['checks']) and all(c['identical'] for c in out['checks'])
    return out


def run(base, main, seeds, workers, smoke=False, **kw):
    """Run the check jobs (resumable) and compare; records left by other code are discarded first."""
    import shutil
    from . import pipeline
    from .retrain import code_fingerprint
    work, fp = work_dir(base), code_fingerprint()[0]
    stamp = work/'code.json'
    if work.exists() and (not stamp.exists() or C.read_json(stamp).get('code_sha256') != fp):
        shutil.rmtree(work)                     # evaluate_one never recomputes an existing record
    work.mkdir(parents=True, exist_ok=True)
    C.write_json(stamp, dict(code_sha256=fp))
    checks, jobs = prepare(base, main, seeds, smoke, **kw)
    print(f'{len(checks)} check roots: ' + ', '.join(f'{k} {Path(r).name}' for k, r, *_ in checks), flush=True)
    code = pipeline.run(jobs, work_dir(base), workers=workers)
    result = compare(checks, smoke)
    result.update(code_sha256=code_fingerprint()[0], jobs_exit=code, finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                  skipped_regression_points=[f'{p}={v:g}' for p, v in kw.get('regression_points', REGRESSION_POINTS)
                                             if not any(Path(r).name == f'{p}={v:g}' for k, r, *_ in checks)])
    C.write_json(Path(base)/'_zeroshot'/'consistency.json', result)
    for c in result['checks']:
        status = 'IDENTICAL' if c['identical'] else \
            f'DIFFERENT ({len(c["differences"])} values, {len(c["missing"])} missing)'
        print(f'{c["kind"]:10s} {Path(c["root"]).name:14s} {c["compared"]:4d} records vs {c["reference"]}: {status}',
              flush=True)
        for line in (c['differences'] + c['missing'])[:10]:
            print('   ', line, flush=True)
    for f, why in result['added_fields'].items():
        print(f'note: {f} exists only in the new records ({why})', flush=True)
    if result['skipped_regression_points']:
        print('regression points not present (skipped): ' + ', '.join(result['skipped_regression_points']), flush=True)
    print(('ALL IDENTICAL' if result['all_identical'] else 'DIFFERENCES FOUND') +
          f' ({result["compared"]} records) -> {Path(base)/"_zeroshot"/"consistency.json"}', flush=True)
    return 0 if result['all_identical'] and code == 0 else 1
