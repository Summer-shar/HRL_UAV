"""Torch-free test of the feasibility command's bookkeeping (p2/feasibility.py) with a stand-in run.py.

Checks: all steps pass -> exit 0 and FEASIBILITY.md says so; a re-run skips steps 1-4
and resumes the pilots; a failing step stops the command, names the step and its log,
and leaves later steps untouched; a failed pilot check stops with the protocol message;
another code fingerprint repeats the checks.
Usage:  python tests/test_feasibility.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p2 import feasibility as F  # noqa: E402

FAKE = '''import os, sys
cmd = sys.argv[1] if len(sys.argv) > 1 else ''
print('fake', ' '.join(sys.argv[1:]))
if cmd == 'verify':
    print('train M1       3 episodes: identical')
    print('test  TFOO     scenarios [21, 22]: identical (fields added after the main run, not compared: u2u_mhz)')
    print('ALL IDENTICAL' if os.environ.get('FAKE_VERIFY', '0') == '0' else 'DIFFERENCES FOUND')
    sys.exit(int(os.environ.get('FAKE_VERIFY', '0')))
if cmd == 'pilot-check':
    print('PASS' if os.environ.get('FAKE_PILOT', '0') == '0' else 'FAIL')
    sys.exit(int(os.environ.get('FAKE_PILOT', '0')))
'''


def run(base, main, fake):
    f = F.Feasibility(main, base, 2, (4101, 4102, 4103))
    f.run_py = str(fake)
    return f.run(), (Path(base)/'FEASIBILITY.md').read_text(encoding='utf-8'), \
        json.loads((Path(base)/'_feasibility'/'state.json').read_text())


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p2feas_'))
    fake = tmp/'fake_run.py'
    fake.write_text(FAKE)
    F.TESTS = ([str(fake), 'test'],)
    main_root = tmp/'main'
    main_root.mkdir()
    code, md, state = run(tmp/'ok', main_root, fake)
    assert code == 0 and '可行性阶段通过' in md and all(s['status'] == 'passed' for s in state['steps'].values()), md
    assert 'ALL IDENTICAL' in md and 'u2u_mhz' in md
    code, md, state = run(tmp/'ok', main_root, fake)                    # re-run: checks skipped
    st = state['steps']
    assert code == 0 and all(st[k]['status'] == 'skipped' for k in ('check-env', 'tests', 'verify', 'smoke'))
    assert st['pilots']['status'] == 'passed' and '已通过（沿用上次）' in md

    os.environ['FAKE_VERIFY'] = '1'
    code, md, state = run(tmp/'bad', main_root, fake)
    st = state['steps']
    assert code == 1 and '已停止：第 3 步' in md and 'verify.log' in md, md
    assert st['verify']['status'] == 'failed' and st['smoke']['status'] == 'pending' and st['pilots']['status'] == 'pending'
    assert 'DIFFERENCES FOUND' in md
    os.environ['FAKE_VERIFY'] = '0'
    code, md, state = run(tmp/'bad', main_root, fake)                   # fixed: resumes at verify
    assert code == 0 and state['steps']['tests']['status'] == 'skipped' and state['steps']['verify']['status'] == 'passed'

    os.environ['FAKE_PILOT'] = '1'
    code, md, state = run(tmp/'pilot', main_root, fake)
    assert code == 1 and '试点检查未通过' in md and state['steps']['summary']['status'] == 'failed', md
    os.environ['FAKE_PILOT'] = '0'

    s = json.loads((tmp/'ok'/'_feasibility'/'state.json').read_text())
    s['code'] = 'other'
    (tmp/'ok'/'_feasibility'/'state.json').write_text(json.dumps(s))
    code, md, state = run(tmp/'ok', main_root, fake)                    # other code: checks repeated
    assert code == 0 and state['steps']['check-env']['status'] == 'passed'
    print(md[:1500])
    print('OK; outputs in', tmp)


if __name__ == '__main__':
    main()
