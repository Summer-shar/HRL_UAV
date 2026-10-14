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
"""
import time
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
