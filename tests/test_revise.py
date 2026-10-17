"""Torch-free test of ``run.py revise``: a v2 root derived from a finished v1 root.

Builds a fake finished smoke root (dummy checkpoints and records), derives a
new root and checks what is carried, what is re-run, and that the old root is
never written.  Usage:  python tests/test_revise.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p2 import common as C  # noqa: E402
from p2 import pipeline, revise  # noqa: E402


def touch(path, text='x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fake_finished_root(root, seed, d):
    for stage in ('M1', 'M2', 'M3', 'HoverSAC') + tuple(C.PPO_BASELINES):
        rd = C.run_dir(root, seed, stage)
        touch(rd/'final.pt', f'{stage} weights')
        touch(rd/'train.jsonl', json.dumps(dict(episode=1, truncated_delay_s=1.0))+'\n')
        for c in d['upper_candidates'] + d['lower_candidates']:
            touch(rd/'ckpt'/f'ep{c:04d}.pt')
    sd = C.seed_dir(root, seed)
    labels = {'ProposedUpper': d['upper_candidates'], 'ProposedLower': d['lower_candidates'],
              'HoverSAC': d['lower_candidates'], **{b: d['upper_candidates'] for b in C.PPO_BASELINES}}
    for label, cands in labels.items():
        for c in cands:
            C.write_json(sd/'dev'/label/str(c)/'summary.json', dict(label=label, candidate=c, cost_per_s=1.))
    C.write_json(sd/'selected.json', {label: cands[0] for label, cands in labels.items()})
    C.write_json(sd/'selection_report.json', {label: dict(selected=cands[0]) for label, cands in labels.items()})
    for m in C.METHODS:
        for s in d['test']:
            C.write_json(sd/'test'/m/f'{s:02d}.json', dict(method=m, scenario=s))
            touch(sd/'test'/m/'traj'/f'{s:02d}.npz')
        for p, values in d['sweeps'].items():
            for v in values:
                if float(v) != C.NOMINAL[p]:
                    for s in d['sweep_scenarios']:
                        C.write_json(sd/'sweep'/f'{p}={v:g}'/m/f'{s:02d}.json', dict(method=m, scenario=s))


def snapshot(root):
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p2revise_'))
    old, new = tmp/'v1', tmp/'v2'
    d = C.design(smoke=True)
    seed = d['seeds'][0]
    fake_finished_root(old, seed, d)
    before = snapshot(old)
    assert C.recipe_name(old) == 'v1'

    done, todo = revise.prepare(old, new, [seed], smoke=True)
    assert snapshot(old) == before, 'the old root must not be written'
    assert C.recipe_name(new) == 'v2' and C.finetune(new)['alpha'] == C.FINETUNE_RECIPES['v2']['alpha']
    assert C.finetune(new, smoke=True, episodes=3)['actor_frozen_episodes'] == 2

    sd = C.seed_dir(new, seed)
    # carried
    for stage in ('M1', 'M2') + tuple(C.PPO_BASELINES):
        assert (C.run_dir(new, seed, stage)/'final.pt').exists(), stage
    assert (sd/'dev'/'ProposedLower'/'0'/'summary.json').exists()
    assert (sd/'dev'/'HoverSAC'/'0'/'summary.json').exists()
    assert (sd/'test'/'TFOO'/'traj'/f'{d["test"][0]:02d}.npz').exists()
    # not carried: anything that depends on the fine-tuning recipe
    for stage in ('M3', 'HoverSAC'):
        assert not C.run_dir(new, seed, stage).exists(), stage
    for c in d['lower_candidates'][1:]:
        assert not (sd/'dev'/'ProposedLower'/str(c)).exists()
        assert not (sd/'dev'/'HoverSAC'/str(c)).exists()
    for m in ('Proposed', 'HoverSAC'):
        assert not (sd/'test'/m).exists() and not list(sd.glob(f'sweep/*/{m}')), m
    sel = C.read_json(sd/'selected.json')
    assert set(sel) == {'ProposedUpper', *C.PPO_BASELINES}, sel
    # hard links where possible (same file system here)
    a, b = C.run_dir(old, seed, 'M1')/'final.pt', C.run_dir(new, seed, 'M1')/'final.pt'
    assert os.stat(a).st_ino == os.stat(b).st_ino

    expected = {f's{seed}_train_M3', f's{seed}_train_HoverSAC', f's{seed}_select_all', 'report',
                f's{seed}_test_Proposed', f's{seed}_test_HoverSAC'}
    expected |= {f's{seed}_dev_{lb}_{c}' for lb in ('ProposedLower', 'HoverSAC') for c in d['lower_candidates'][1:]}
    expected |= {f's{seed}_sweep_{m}_{p}' for m in ('Proposed', 'HoverSAC') for p in d['sweeps']}
    expected |= {f's{seed}_burst_{m}' for m in C.METHODS}      # no burst records in the fake v1 root
    assert set(todo) == expected, sorted(set(todo) ^ expected)
    ids = {j.id for j in pipeline.plan([seed], smoke=True)}
    assert set(done) | set(todo) == ids
    assert all((new/'_jobs'/f'{j}.done').exists() for j in done)

    # idempotent; a different recipe or source is refused
    assert revise.prepare(old, new, [seed], smoke=True)[1] == todo
    for bad in (dict(src=old, dst=new, recipe='v1'), dict(src=new, dst=tmp/'v3', recipe='v2'),
                dict(src=old, dst=old, recipe='v2')):
        try:
            revise.prepare(bad['src'], bad['dst'], [seed], recipe=bad['recipe'], smoke=True)
        except ValueError as e:
            print('refused as expected:', e)
        else:
            raise AssertionError(f'should refuse {bad}')
    other = tmp/'v4'
    C.write_json(other/'recipe.json', dict(finetune='v2', derived_from=str(tmp/'elsewhere')))
    try:
        revise.prepare(old, other, [seed], smoke=True)
    except ValueError as e:
        print('refused as expected:', e)
    else:
        raise AssertionError('a root prepared from another source must be refused')
    # the full design plans saturated points for every method
    full = pipeline.plan(C.design(False)['seeds'])
    sat = [j for j in full if '_sat_' in j.id]
    assert len(sat) == len(C.design(False)['seeds'])*len(C.METHODS)*2 and all('--saturated' in j.args for j in sat)
    print(f'OK: {len(done)} carried, {len(todo)} to run; outputs in {tmp}')


if __name__ == '__main__':
    main()
