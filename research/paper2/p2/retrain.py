"""Retrained sweep points: every learned method trained again at each parameter value.

Each point is its own output root  <base>/<param>=<value>/  holding
  scene.json   {"param", "value", "prune_snapshots": true}  -> common.load_scene(root)
  recipe.json  fine-tuning recipe v2 (as the main results)
and the usual seed_*/ tree.  Training scenarios, seeds, initial weights and
exploration streams are identical to the nominal run; only the scene
parameter differs.  Points are evaluated on the sweep scenarios (21-28), the
same scenarios as the non-retrained sweep, so the two can be compared point by
point.  No nested sweeps or burst runs.  Resume snapshots (state.pt, which
carry the replay buffer) are deleted when a stage finishes, to save disk.

    python run.py retrain --out-base D:\\p2_retrain --first u2u_mhz=1.5,u2u_mhz=2
    python run.py retrain-report --out-base D:\\p2_retrain --main D:\\p2_runs_v2
    python run.py retrain-export --out-base D:\\p2_retrain --dest D:\\p2_export_A.zip --machine A

Points may be split between computers (--points); each computer exports its
finished points with retrain-export (records only: no weights, trajectories or
logs) and the exports are merged into one folder for retrain-report.
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
    """'u2u_mhz=1.5,cpu_ghz=9' -> [('u2u_mhz', 1.5), ('cpu_ghz', 9.0)]."""
    points = []
    for item in filter(None, (t.strip() for t in str(text).split(','))):
        param, _, value = item.partition('=')
        if param not in C.SWEEPS:
            raise ValueError(f'unknown sweep parameter {param!r}')
        points.append((param, float(value)))
    return points


def all_points(smoke=False):
    """Every stable sweep value except the nominal one."""
    d = C.design(smoke)
    return [(p, float(v)) for p, values in d['sweeps'].items() for v in values if float(v) != C.NOMINAL[p]]


def prepare_point(base, param, value, smoke=False, recipe='v2'):
    from .pipeline import point_name
    if float(value) == C.NOMINAL[param]:
        raise ValueError('the nominal point is the main run; it is not retrained here')
    root = Path(base)/point_name(param, value)
    root.mkdir(parents=True, exist_ok=True)
    scene = dict(param=param, value=float(value), prune_snapshots=True)
    path = root/'scene.json'
    if path.exists():
        old = C.read_json(path)
        if (old.get('param'), float(old.get('value'))) != (param, float(value)):
            raise ValueError(f'{root} already belongs to {old}')
    else:
        C.write_json(path, dict(scene, created=time.strftime('%Y-%m-%d %H:%M:%S')))
    marker = root/'recipe.json'
    if marker.exists():
        if C.read_json(marker).get('finetune') != recipe:
            raise ValueError(f'{root} uses another fine-tuning recipe')
    else:
        C.write_json(marker, dict(finetune=recipe, parameters=C.FINETUNE_RECIPES[recipe],
                                  note='retrained sweep point; same recipe as the main results'))
    C.write_json(root/'design.json', dict(design=C.design(smoke), point=scene, smoke=smoke,
                                          methods=list(C.METHODS), finetune=C.finetune(root),
                                          test_scenarios=list(C.test_scenarios(root, C.design(smoke)))))
    return root


def point_finished(root, seeds, smoke=False, methods=C.METHODS):
    """Every method has a test record on every sweep scenario for every seed."""
    d = C.design(smoke)
    return all((C.seed_dir(root, s)/'test'/m/f'{sc:02d}.json').exists()
               for s in seeds for m in methods for sc in d['sweep_scenarios'])


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
    """SHA-256 over the paper-2 code and the v2.4.0 source/configs (identifies the code without git)."""
    files = sorted([*C.PAPER2.glob('p2/*.py'), C.PAPER2/'run.py', *C.SRC.glob('hrl_v2/*.py'),
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
