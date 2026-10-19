"""Torch-free test of the read-only checks of design v3 (p2/checks.py).

pilot_check: per pilot point, pass / fail / not finished, judged on fabricated M1
training logs against a fabricated nominal M1 log; the U2U pilot is judged on its
own root only.  timing: runs on fabricated logs and reports every planned point.
Usage:  python tests/test_checks.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p2 import common as C  # noqa: E402
from p2 import checks  # noqa: E402
from p2.pipeline import point_name  # noqa: E402


def write_log(root, seed, stage, delays, wall=10.):
    path = C.run_dir(root, seed, stage)/'train.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for i, d in enumerate(delays):
        C.append_jsonl(path, dict(episode=i+1, truncated_delay_s=float(d), cost_per_s=float(d)*9, wall_seconds=wall))


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p2checks_'))
    main_root, base = tmp/'main', tmp/'sweeps'
    seeds = list(C.design()['seeds'])
    n = C.BUDGET['M1']
    cpu, u2u = (point_name(p, C.point_value(p, v)) for p, v in C.PILOT_POINTS)
    for s in seeds:
        write_log(main_root, s, 'M1', [2.0 - 1.4*i/n for i in range(n)])           # nominal ends near 0.6 s
    good = [3.0 - 2.0*i/n for i in range(n)]                                        # improves, ends near 1.0 s
    for s in seeds:
        write_log(base/cpu, s, 'M1', good)
    # U2U pilot not started: the CPU pilot alone passes, the U2U pilot is pending
    ok, text = checks.pilot_check(main_root, base, seeds, pilot=cpu)
    assert ok is True and 'PASS' in text, text
    ok, text = checks.pilot_check(main_root, base, seeds, pilot=u2u)
    assert ok is None and 'NOT FINISHED' in text, text
    ok, _ = checks.pilot_check(main_root, base, seeds)                                # both pilots
    assert ok is None
    # U2U pilot: one seed diverges (ends above 3 x nominal) -> that gate fails, the CPU gate still passes
    for s in seeds:
        write_log(base/u2u, s, 'M1', good if s != seeds[1] else [3.0 + 0.5*i/n for i in range(n)])
    ok, text = checks.pilot_check(main_root, base, seeds, pilot=u2u)
    assert ok is False and 'FAIL' in text and f'seed {seeds[1]}' in text, text
    assert checks.pilot_check(main_root, base, seeds, pilot=cpu)[0] is True
    assert checks.pilot_check(main_root, base, seeds)[0] is False
    # no learning progress (flat curve, inside the 3x bound) also fails
    write_log(base/u2u, seeds[1], 'M1', [1.0]*n)
    ok, text = checks.pilot_check(main_root, base, seeds, pilot=u2u)
    assert ok is False and 'improves: NO' in text, text
    write_log(base/u2u, seeds[1], 'M1', good)
    assert checks.pilot_check(main_root, base, seeds, pilot=u2u)[0] is True
    try:
        checks.pilot_check(main_root, base, seeds, pilot='load=0.8')
    except ValueError as e:
        print('refused as expected:', e)
    else:
        raise AssertionError('a non-pilot point must be refused')
    # the gate needs the nominal log
    try:
        checks.pilot_check(tmp/'empty', base, seeds, pilot=cpu)
    except ValueError as e:
        print('refused as expected:', e)
    else:
        raise AssertionError('missing nominal M1 log must be refused')
    print(checks.pilot_check(main_root, base, seeds)[1])

    for s in seeds:
        for stage in checks.SAC_STAGES + checks.PPO_STAGES:
            if stage != 'M1':
                write_log(main_root, s, stage, [1.0]*5, wall=20.)
        test = C.seed_dir(main_root, s)/'test'/'CA'
        test.mkdir(parents=True, exist_ok=True)
        C.write_json(test/'21.json', dict(wall_seconds=15.))
    from p2 import retrain
    text = checks.timing(main_root, retrain.all_points(), seeds, 8)
    assert '16 points x 3 seeds' in text, text
    print(text)
    # progress: one point finished (every planned job has a log and a .done marker), one half done
    from p2.pipeline import retrain_plan
    pts = [('cpu_ghz', 9.0), ('u2u_mhz', 1.5), ('load', 0.8)]
    jobs = retrain_plan(base, pts, seeds, gate=False)
    for k, (param, value) in enumerate(pts[:2]):
        name = point_name(param, value)
        mine = [j for j in jobs if j.id.startswith(name+'/')]
        for i, j in enumerate(mine if k == 0 else mine[:len(mine)//2]):
            jd = base/name/'_jobs'
            jd.mkdir(parents=True, exist_ok=True)
            (jd/f'{j.marker}.log').write_text('=== start 2026-10-18 08:00:00 ===\nfailed\n'
                                             '=== start 2026-10-18 09:00:00 ===\nok\n', encoding='utf-8')
            (jd/f'{j.marker}.done').write_text('2026-10-18 09:30:00\n', encoding='utf-8')
    text = checks.progress(base, pts, seeds, 8)
    print(text)
    n = len([j for j in jobs if j.id.startswith('cpu_ghz=9/')])
    assert f'{n:4d}/{n:<5d} {n*0.5:10.1f} {n*0.5:9.1f}  finished' in text, text
    assert 'remaining: 1 running and 1 waiting points, about 93 job-hours' in text, text   # 31 + 62
    assert abs(checks.job_hours(base/'cpu_ghz=9')[jobs[0].marker] - 0.5) < 1e-9
    # a running point slower than the finished one (1.5 h per job): extrapolated from its own share,
    # never "0 job-hours left" (the estimate the first version printed)
    for j in [j for j in jobs if j.id.startswith('u2u_mhz=1.5/')][:n//2]:
        (base/'u2u_mhz=1.5'/'_jobs'/f'{j.marker}.done').write_text('2026-10-18 10:30:00\n', encoding='utf-8')
    text = checks.progress(base, pts, seeds, 8)
    print(text)
    assert f'{n//2*1.5:10.1f} {n*1.5:9.1f}  running' in text, text
    assert f'about {n//2*1.5 + (n*0.5+n*1.5)/2:.0f} job-hours' in text, text          # 93 left + 124 waiting
    print('OK; outputs in', tmp)


if __name__ == '__main__':
    main()
