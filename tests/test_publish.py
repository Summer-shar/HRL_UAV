"""Publishing run status to a git branch (p2/publish.py), against a local bare repository.

Checks: first push creates the branch when it does not exist; the status page, step
logs (last lines only), result tables and the last lines of unfinished job logs are
pushed, finished job logs and weights are not; a second push records only changes;
a commit pushed by someone else meanwhile (instructions) is kept; an unreachable remote
is reported, never raised.  Needs git on PATH.
Usage:  python tests/test_publish.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p2.publish import Publisher  # noqa: E402


def git(*args, cwd):
    return subprocess.run(['git', *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p2pub_'))
    remote = tmp/'remote.git'
    git('init', '-q', '--bare', str(remote), cwd=tmp)
    base = tmp/'p2_sweeps_v3'
    (base/'_feasibility'/'verify').mkdir(parents=True)
    (base/'FEASIBILITY.md').write_text('# status\nrunning\n', encoding='utf-8')
    (base/'_feasibility'/'state.json').write_text('{}', encoding='utf-8')
    (base/'_feasibility'/'pilots.log').write_text('\n'.join(f'line {i}' for i in range(1000)), encoding='utf-8')
    (base/'_feasibility'/'verify'/'verify_abc.json').write_text('{"all_identical": true}', encoding='utf-8')
    (base/'results').mkdir()
    (base/'results'/'retrain_sweep_cpu_ghz.csv').write_text('a,b\n1,2\n', encoding='utf-8')
    jobs = base/'cpu_ghz=9'/'_jobs'
    jobs.mkdir(parents=True)
    (jobs/'s4101_train_M1.log').write_text('Traceback ...\nValueError: boom\n', encoding='utf-8')
    (jobs/'s4101_train_TFOO.log').write_text('ok\n', encoding='utf-8')
    (jobs/'s4101_train_TFOO.done').write_text('2026-10-18 10:00:00\n', encoding='utf-8')
    (base/'cpu_ghz=9'/'seed_4101').mkdir()
    (base/'cpu_ghz=9'/'seed_4101'/'final.pt').write_bytes(b'weights')

    pub = Publisher(base, remote=str(remote))
    assert pub.publish('first') is True, pub.last_error
    check = tmp/'check'
    git('clone', '-q', '-b', 'paper2-runs', str(remote), str(check), cwd=tmp)
    files = sorted(p.relative_to(check/'p2_sweeps_v3').as_posix() for p in (check/'p2_sweeps_v3').rglob('*') if p.is_file())
    print(files)
    assert 'FEASIBILITY.md' in files and 'results/retrain_sweep_cpu_ghz.csv' in files
    assert '_feasibility/verify/verify_abc.json' in files and 'PUBLISHED.txt' in files
    assert 'unfinished_job_logs/cpu_ghz=9__s4101_train_M1.log' in files
    assert not any('TFOO' in f or f.endswith('.pt') for f in files)
    log = (check/'p2_sweeps_v3'/'_feasibility'/'pilots.log').read_text()
    assert log.startswith('... (700 earlier lines') and log.rstrip().endswith('line 999')
    assert 'GitHub 推送：成功' in pub.status_line()

    # someone else (the cloud side) adds instructions meanwhile; the next push keeps them
    (check/'NEXT.md').write_text('next steps\n', encoding='utf-8')
    git('add', 'NEXT.md', cwd=check)
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'instructions', cwd=check)
    git('push', '-q', 'origin', 'paper2-runs', cwd=check)
    (base/'FEASIBILITY.md').write_text('# status\npassed\n', encoding='utf-8')
    n0 = int(git('rev-list', '--count', 'paper2-runs', cwd=remote))
    assert pub.publish('second') is True, pub.last_error
    git('pull', '-q', cwd=check)
    assert (check/'NEXT.md').exists() and 'passed' in (check/'p2_sweeps_v3'/'FEASIBILITY.md').read_text()
    assert int(git('rev-list', '--count', 'paper2-runs', cwd=remote)) == n0+1
    assert pub.publish('interval', force=False) is None             # throttled

    bad = Publisher(tmp/'other', remote=str(tmp/'missing.git'))
    (tmp/'other').mkdir()
    (tmp/'other'/'FEASIBILITY.md').write_text('x', encoding='utf-8')
    assert bad.publish('x') is False and bad.last_error and '失败' in bad.status_line()
    print('OK; outputs in', tmp)


if __name__ == '__main__':
    main()
