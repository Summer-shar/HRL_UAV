"""Torch-free test of the record comparison used by run.py verify (tests/verify_regression.py).

Every field is compared, not only up to the first difference: a value that differs
in a field sorted after a later-added field (u2u_mhz) must still be found.
Later-added top-level fields are listed, never counted as differences; a field
missing in the new record, a nested extra field or a changed value is a difference.
Usage:  python tests/test_verify_compare.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_regression as V  # noqa: E402


def main():
    old = dict(cost_per_s=6.45, energy_w={'cpu': 120.0, 'u2u': 3.2}, method='TFOO', solver_calls=1800,
               truncated_delay_s=0.684, unfinished_rate=0.0007, upload_stage_s=0.29, wall_seconds=15.0,
               learner='D:/old/final.pt')
    skip = ('wall_seconds', 'learner', 'upper')
    new = dict(copy.deepcopy(old), u2u_mhz=3.0, wall_seconds=9.0, learner='D:/new/final.pt')
    values, added = V.summarize([('scenario 21', new, old, skip)])
    assert values == [] and added == ['u2u_mhz'], (values, added)
    assert V.verdict(values, added).startswith('identical (fields added after the main run')
    assert V.compare(new, old, skip) is None

    bad = copy.deepcopy(new)
    bad['unfinished_rate'] = 0.0008                       # sorts after u2u_mhz: the case the old check missed
    bad['energy_w']['u2u'] = 3.3
    values, added = V.summarize([('scenario 21', bad, old, skip)])
    assert len(values) == 2 and any('unfinished_rate' in v for v in values) and any('energy_w.u2u' in v for v in values)
    assert V.verdict(values, added).startswith('DIFFERENT (2 values)')

    gone = copy.deepcopy(new)
    del gone['solver_calls']                              # missing in the new record
    nested = copy.deepcopy(new)
    nested['energy_w']['u2u_mhz'] = 3.0                   # tolerated only at the top level
    unknown = dict(copy.deepcopy(new), new_metric=1.0)    # not in ADDED_FIELDS
    for case, field in ((gone, 'solver_calls'), (nested, 'energy_w.u2u_mhz'), (unknown, 'new_metric')):
        values, _ = V.summarize([('scenario 21', case, old, skip)])
        assert len(values) == 1 and field in values[0], values

    rows_old = [dict(episode=i, delay=float(i)) for i in range(1, 4)]
    rows_new = [dict(r, actor_lr=3e-5) for r in rows_old]
    values, added = V.summarize([(f'episode {n["episode"]}', n, o, ('wall_seconds',)) for n, o in zip(rows_new, rows_old)])
    assert values == [] and added == ['actor_lr']
    rows_new[2]['delay'] = 3.5
    values, _ = V.summarize([(f'episode {n["episode"]}', n, o, ('wall_seconds',)) for n, o in zip(rows_new, rows_old)])
    assert values == ['episode 3: delay: 3.5 != 3.0'], values
    print('OK')


if __name__ == '__main__':
    main()
