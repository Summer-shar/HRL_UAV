"""Derive a new output root that differs from a finished one only in the
fine-tuning recipe of M3 and HoverSAC (see common.FINETUNE_RECIPES).

Everything that cannot depend on that recipe is carried over unchanged (hard
links where the file system allows, copies otherwise):

* train/M1, train/M2, train/TFOO|TOO|NC          checkpoints, logs, contracts
* dev/ProposedUpper, dev/TFOO|TOO|NC              validation summaries
* dev/ProposedLower/0, dev/HoverSAC/0             candidate 0 = the M1 SAC, no fine-tuning
* test/, sweep/ and burst/ records of TFOO, TOO, NC, CA   methods without a fine-tuned SAC
* the ProposedUpper and PPO-baseline entries of selected.json / selection_report.json

The jobs whose outputs are complete in the new root are marked done, so
``run.py pipeline --out <new root>`` runs only M3, the HoverSAC fine-tuning,
their dev candidates, the selection, the test / sweep evaluation of Proposed
and HoverSAC, any record that was missing in the old root, and the report.
The old root is only read.
"""
import os
import shutil
import time
from pathlib import Path

from . import common as C

CARRY_TRAIN = ('M1', 'M2') + tuple(C.PPO_BASELINES)
CARRY_DEV_LABELS = ('ProposedUpper',) + tuple(C.PPO_BASELINES)
CARRY_DEV_ZERO = ('ProposedLower', 'HoverSAC')          # only candidate 0 (the M1 SAC)
FIXED_METHODS = tuple(m for m in C.METHODS if m not in ('Proposed', 'HoverSAC'))
FINETUNED = ('Proposed', 'HoverSAC')


def _place(src, dst):
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return 1


def _carry(src, dst):
    if not src.exists():
        return 0
    if src.is_file():
        return _place(src, dst)
    return sum(_place(f, dst/f.relative_to(src)) for f in src.rglob('*') if f.is_file())


def _args(job):
    out = {'command': job.args[0]}
    for a in job.args[1:]:
        if a.startswith('--'):
            k, _, v = a[2:].partition('=')
            out[k] = v if v else True
    return out


def _eval_files(root, seed, method, kind, d, param=None):
    sd = C.seed_dir(root, seed)
    if kind == 'test':
        return [sd/'test'/method/f'{s:02d}.json' for s in d['test']]
    values = d['saturated' if kind == 'sat' else 'sweeps'][param]
    return [sd/'sweep'/f'{param}={v:g}'/method/f'{s:02d}.json'
            for v in values if float(v) != C.NOMINAL[param] for s in d['sweep_scenarios']]


def carried_done(job, root, d):
    """True when this job's outputs exist in the new root and cannot depend on the recipe."""
    a = _args(job)
    cmd = a['command']
    if cmd == 'report' or 'seed' not in a:
        return False
    seed = int(a['seed'])
    sd = C.seed_dir(root, seed)
    if cmd == 'train':
        return a['stage'] in CARRY_TRAIN and (C.run_dir(root, seed, a['stage'])/'final.pt').exists()
    if cmd == 'dev':
        label, cand = a['label'], int(a['candidate'])
        if label in CARRY_DEV_LABELS or (label in CARRY_DEV_ZERO and cand == 0):
            return (sd/'dev'/label/str(cand)/'summary.json').exists()
        return False
    if cmd == 'select':
        labels = a['labels'].split(',')
        if set(labels) <= set(CARRY_DEV_LABELS):
            sel = sd/'selected.json'
            return sel.exists() and all(lb in C.read_json(sel) for lb in labels)
        return False
    if cmd == 'burst':
        files = [sd/'burst'/a['method']/f'{s:02d}.json' for s in d['sweep_scenarios']]
        return a['method'] in FIXED_METHODS and all(f.exists() for f in files)
    if cmd in ('test', 'sweep'):
        if a['method'] not in FIXED_METHODS:
            return False
        kind = 'test' if cmd == 'test' else ('sat' if a.get('saturated') else 'sweep')
        files = _eval_files(root, seed, a['method'], kind, d, a.get('param'))
        return bool(files) and all(f.exists() for f in files)
    return False


def prepare(src, dst, seeds, recipe='v2', smoke=False):
    from . import pipeline
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if src == dst:
        raise ValueError('--from and --out must differ; the old root is kept as it is')
    if recipe not in C.FINETUNE_RECIPES:
        raise ValueError(f'unknown recipe {recipe!r}; known: {", ".join(C.FINETUNE_RECIPES)}')
    old = C.recipe_name(src)
    if old == recipe:
        raise ValueError(f'{src} already uses recipe {recipe}')
    marker = dst/'recipe.json'
    if marker.exists():
        m = C.read_json(marker)
        if m.get('finetune') != recipe or Path(m.get('derived_from', '')).resolve() != src:
            raise ValueError(f'{dst} was prepared for recipe {m.get("finetune")} from {m.get("derived_from")}')
    d = C.design(smoke)
    for seed in seeds:
        sd = C.seed_dir(src, seed)
        sel = sd/'selected.json'
        if not sel.exists() or not all(lb in C.read_json(sel) for lb in CARRY_DEV_LABELS):
            raise FileNotFoundError(f'{sel} lacks the ProposedUpper / baseline selections; finish the old run first')
        for stage in CARRY_TRAIN:
            if not (C.run_dir(src, seed, stage)/'final.pt').exists():
                raise FileNotFoundError(f'{C.run_dir(src, seed, stage)} has no final.pt')

    dst.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        C.write_json(marker, dict(finetune=recipe, parameters=C.FINETUNE_RECIPES[recipe], derived_from=str(src),
                                  derived_from_recipe=old, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                                  carried=dict(train=list(CARRY_TRAIN), dev=list(CARRY_DEV_LABELS),
                                               dev_candidate0=list(CARRY_DEV_ZERO), evaluation=list(FIXED_METHODS)),
                                  rerun=list(FINETUNED)))
    placed = 0
    for seed in seeds:
        s_src, s_dst = C.seed_dir(src, seed), C.seed_dir(dst, seed)
        for stage in CARRY_TRAIN:
            placed += _carry(C.run_dir(src, seed, stage), C.run_dir(dst, seed, stage))
        for label in CARRY_DEV_LABELS:
            placed += _carry(s_src/'dev'/label, s_dst/'dev'/label)
        for label in CARRY_DEV_ZERO:
            placed += _carry(s_src/'dev'/label/'0', s_dst/'dev'/label/'0')
        for m in FIXED_METHODS:
            placed += _carry(s_src/'test'/m, s_dst/'test'/m)
            placed += _carry(s_src/'burst'/m, s_dst/'burst'/m)
            for point in sorted(s_src.glob('sweep/*')):
                placed += _carry(point/m, s_dst/'sweep'/point.name/m)
        for name in ('selected.json', 'selection_report.json'):
            path = s_dst/name
            if not path.exists() and (s_src/name).exists():
                old_sel = C.read_json(s_src/name)
                C.write_json(path, {k: v for k, v in old_sel.items() if k in CARRY_DEV_LABELS})
    jobs = pipeline.plan(seeds, smoke=smoke)
    jd = dst/'_jobs'
    jd.mkdir(parents=True, exist_ok=True)
    done = [j.id for j in jobs if carried_done(j, dst, d)]
    for jid in done:
        path = jd/f'{jid}.done'
        if not path.exists():
            path.write_text(f'carried from {src} ({time.strftime("%Y-%m-%d %H:%M:%S")})\n')
    todo = [j.id for j in jobs if j.id not in done]
    C.write_json(dst/'revise_manifest.json', dict(source=str(src), recipe=recipe, seeds=list(seeds),
                                                 files_placed=placed, carried_jobs=done, jobs_to_run=todo))
    print(f'Prepared {dst} (recipe {recipe}, from {src}, recipe {old}): {placed} files placed, '
          f'{len(done)} jobs carried, {len(todo)} to run.')
    print(f'Next:  python run.py pipeline --out "{dst}"' + (' --smoke' if smoke else '')
          + f' --seeds {",".join(map(str, seeds))}')
    return done, todo
