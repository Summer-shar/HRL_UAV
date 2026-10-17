"""Retrained sweep points: every learned method trained again, from scratch, at each value.

Design v3 (PROTOCOL.md): cpu_ghz 9/11/12/14, load 0.6/0.8/0.9/1.1, users
(terminals per region) 7/10/16/19 and u2u_mhz 1.5/2/2.5/4; the nominal value of
each parameter is the main run.  Each point is its own output root  <base>/<param>=<value>/  holding
  scene.json   {"param", "value", "design": "v3", "prune_snapshots": true}
               -> common.load_scene(root)
  recipe.json  fine-tuning recipe v2 (as the main results)
and the usual seed_*/ tree.  Stages, budgets, hyper-parameters, seeds,
training scenarios, initial weights, exploration streams and the dev
selection rule are those of the main run; only the scene parameter differs.
Design-v3 points are evaluated on the full test split (scenarios 21-40, as the
main table).  No nested sweeps or burst runs.  Resume snapshots (state.pt,
which carry the replay buffer) are deleted when a stage finishes, to save disk.

    python run.py retrain --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2   # everything, pilots gate
    python run.py pilot-check --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2
    python run.py retrain-report --out-base D:\\p2_sweeps_v3 --main D:\\p2_runs_v2

Finished points can be zipped with retrain-export (records only: no weights,
trajectories or logs), e.g. as a backup or for the paper repository.
"""
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

from . import common as C


def parse_points(text):
    """'cpu_ghz=9,users=7' -> [('cpu_ghz', 9.0), ('users', 7)]."""
    points = []
    for item in filter(None, (t.strip() for t in str(text).split(','))):
        param, _, value = item.partition('=')
        if param not in C.RETRAIN_SWEEPS and param not in C.SWEEPS:
            raise ValueError(f'unknown sweep parameter {param!r}')
        points.append((param, C.point_value(param, value)))
    return points


def all_points(smoke=False):
    """Every non-nominal value of the design-v3 retrained sweep: the pilot points
    first, then the parameters in C.RETRAIN_ORDER."""
    d = C.design(smoke)
    points = [(p, C.point_value(p, v)) for p in C.RETRAIN_ORDER for v in d['retrain_sweeps'][p]
              if C.point_value(p, v) != C.point_value(p, C.NOMINAL[p])]
    pilots = [(p, C.point_value(p, v)) for p, v in C.PILOT_POINTS]
    pilots = [pt for pt in pilots if pt in points]
    return pilots + [pt for pt in points if pt not in pilots]


def prepare_point(base, param, value, smoke=False, recipe='v2'):
    from .pipeline import point_name
    value = C.point_value(param, value)
    if value == C.point_value(param, C.NOMINAL[param]):
        raise ValueError('the nominal point is the main run; it is not retrained here')
    root = Path(base)/point_name(param, value)
    root.mkdir(parents=True, exist_ok=True)
    scene = dict(param=param, value=value, design='v3', prune_snapshots=True)
    path = root/'scene.json'
    if path.exists():
        old = C.read_json(path)
        if (old.get('param'), C.point_value(param, old.get('value')), old.get('design')) != (param, value, 'v3'):
            raise ValueError(f'{root} already belongs to {old} (design v3 needs its own --out-base)')
    else:
        C.write_json(path, dict(scene, created=time.strftime('%Y-%m-%d %H:%M:%S')))
    marker = root/'recipe.json'
    if marker.exists():
        if C.read_json(marker).get('finetune') != recipe:
            raise ValueError(f'{root} uses another fine-tuning recipe')
    else:
        C.write_json(marker, dict(finetune=recipe, parameters=C.FINETUNE_RECIPES[recipe],
                                  note='retrained sweep point; same recipe as the main results'))
    scene_json = json.dumps(C.load_scene(root), sort_keys=True, separators=(',', ':'))
    C.write_json(root/'design.json', dict(design=C.design(smoke), point=scene, smoke=smoke,
                                          methods=list(C.METHODS), finetune=C.finetune(root),
                                          test_scenarios=list(C.test_scenarios(root, C.design(smoke))),
                                          demand_over_capacity=C.utilization(param, value),
                                          scene_sha256=hashlib.sha256(scene_json.encode()).hexdigest(),
                                          code_sha256=code_fingerprint()[0]))
    return root


def point_finished(root, seeds, smoke=False, methods=C.METHODS):
    """Every method has a test record on every test scenario of the point for every seed."""
    d = C.design(smoke)
    scenarios = C.test_scenarios(root, d)
    return all((C.seed_dir(root, s)/'test'/m/f'{sc:02d}.json').exists()
               for s in seeds for m in methods for sc in scenarios)


EXPORT_PATTERNS = ('scene.json', 'recipe.json', 'design.json', 'results/*.md', 'results/*.csv', 'results/*.json',
                   'seed_*/selected.json', 'seed_*/selection_report.json', 'seed_*/test/*/[0-9][0-9].json',
                   'seed_*/train/*/train.jsonl', 'seed_*/train/*/done.json', 'seed_*/dev/*/*/summary.json',
                   '_jobs/*.done')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_fingerprint():
    """SHA-256 over the pipeline, simulator and RL code and configs (identifies the code without git)."""
    files = sorted([*C.SRC.glob('p2/*.py'), C.REPO/'run.py', *C.SRC.glob('hrl_v2/*.py'),
                    *C.SRC.glob('hrl_v2/configs/*.json'), *C.SRC.glob('hrl_uav/*.py')])
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(C.REPO)).replace('\\', '/').encode())
        h.update(f.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest(), len(files)


def environment():
    from importlib import metadata
    versions = {}
    for pkg in ('numpy', 'scipy', 'torch', 'cvxpy', 'clarabel', 'matplotlib'):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    fp, n = code_fingerprint()
    return dict(platform=platform.platform(), python=sys.version.split()[0], processor=platform.processor(),
                logical_cpus=os.cpu_count(), packages=versions, P2_DEVICE=os.environ.get('P2_DEVICE'),
                P2_THREADS=os.environ.get('P2_THREADS'), code_sha256=fp, code_files=n)


def export(base, dest, points, seeds, machine, smoke=False, require_report=True):
    """Zip the records of the finished points (no weights, trajectories or logs) plus a manifest.
    A point counts as finished when its report job is done (require_report) and every test record exists."""
    from .pipeline import point_name
    base, dest = Path(base), Path(dest)
    finished, missing, files = [], [], []
    for param, value in points:
        name = point_name(param, value)
        root = base/name
        done = root.exists() and point_finished(root, seeds, smoke)
        if not done or (require_report and not (root/'_jobs'/'report.done').exists()):
            missing.append(name)
            continue
        finished.append(name)
        for pattern in EXPORT_PATTERNS:
            files += [p for p in sorted(root.glob(pattern)) if p.is_file()]
    if not finished:
        raise ValueError(f'no finished point under {base} among {[point_name(*p) for p in points]}')
    manifest = dict(machine=machine, created=time.strftime('%Y-%m-%d %H:%M:%S'), out_base=str(base),
                    finished=finished, not_finished=missing, seeds=list(seeds), smoke=smoke,
                    environment=environment(),
                    files={str(p.relative_to(base)).replace('\\', '/'): _sha256(p) for p in files})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel, p in zip(manifest['files'], files):
            z.write(p, rel)
        z.writestr(f'manifest_{machine}.json', json.dumps(manifest, indent=1, ensure_ascii=False))
    os.replace(tmp, dest)
    return dict(dest=str(dest), sha256=_sha256(dest), finished=finished, not_finished=missing,
                files=len(files))
