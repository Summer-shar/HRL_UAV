"""Run requests from GitHub without anyone at the computer.

Polls the branch the code was cloned from (default paper2-release) every few
minutes for the file RUN_REQUEST.json, e.g.

    {"id": "v4-1", "run": "zeroshot", "note": "zero-shot sweep, design v4"}

When the id is new and nothing started by this watcher is still running, it
pulls the code (fast-forward only) and starts ONE of the fixed commands below;
the request only names the command, every argument comes from this command
line, so nothing else can be run.  A failed run waits for a request with a new
id (after a fix); the experiment commands themselves resume where they stopped.

    python tools/watch.py --main D:\\p2_runs_v2 --zeroshot-out D:\\p2_zeroshot_v4 --sweep-out D:\\p2_sweeps_v3

Allowed commands:
    zeroshot     run.py zeroshot    --main <main> --out-base <zeroshot-out>
    feasibility  run.py feasibility --main <main> --out-base <sweep-out>
    sweep        run.py sweep       --main <main> --out-base <sweep-out>
    stop         stop watching
State and logs: <state-dir> (default: <zeroshot-out>/../p2_watch).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GIT_ENV = dict(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def git(args, repo, timeout=300):
    try:
        r = subprocess.run(['git', *args], cwd=str(repo), capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace', env=dict(os.environ, **GIT_ENV))
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def commands(a):
    run = str(REPO/'run.py')
    return {'zeroshot': [sys.executable, run, 'zeroshot', '--main', a.main, '--out-base', a.zeroshot_out],
            'feasibility': [sys.executable, run, 'feasibility', '--main', a.main, '--out-base', a.sweep_out],
            'sweep': [sys.executable, run, 'sweep', '--main', a.main, '--out-base', a.sweep_out]}


def read_request(repo, branch):
    code, out = git(['fetch', '-q', 'origin', branch], repo)
    if code != 0:
        return None, f'fetch failed: {out[-300:]}'
    code, out = git(['show', f'origin/{branch}:RUN_REQUEST.json'], repo)
    if code != 0:
        return None, 'no RUN_REQUEST.json on the branch yet'
    try:
        request = json.loads(out)
    except ValueError:
        return None, 'RUN_REQUEST.json is not valid JSON'
    if not isinstance(request, dict) or not request.get('id') or request.get('run') not in ('zeroshot', 'feasibility',
                                                                                         'sweep', 'stop'):
        return None, f'ignored request: {request!r}'
    return request, ''


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--main', required=True)
    p.add_argument('--zeroshot-out', required=True)
    p.add_argument('--sweep-out', required=True)
    p.add_argument('--branch', default='paper2-release')
    p.add_argument('--repo', default=str(REPO))
    p.add_argument('--state-dir', default=None)
    p.add_argument('--interval', type=float, default=180., help='seconds between polls')
    p.add_argument('--hours', type=float, default=96., help='stop watching after this many hours')
    p.add_argument('--once', action='store_true', help='one poll (tests)')
    a = p.parse_args(argv)
    repo = Path(a.repo)
    state_dir = Path(a.state_dir) if a.state_dir else Path(a.zeroshot_out).resolve().parent/'p2_watch'
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir/'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else dict(done=[])
    child, child_id, log = None, None, None
    deadline = time.time() + a.hours*3600
    print(f'[{now()}] watching origin/{a.branch}:RUN_REQUEST.json every {a.interval:.0f} s; state {state_path}',
          flush=True)
    while time.time() < deadline:
        if child is not None and child.poll() is not None:
            print(f'[{now()}] run {child_id} ended with exit code {child.returncode}', flush=True)
            state.setdefault('finished', {})[child_id] = dict(exit=child.returncode, ended=now())
            state_path.write_text(json.dumps(state, indent=1), encoding='utf-8')
            log.close()
            child, child_id, log = None, None, None
        if child is None:
            request, why = read_request(repo, a.branch)
            if request is None:
                print(f'[{now()}] waiting: {why}', flush=True)
            elif request['id'] in state['done']:
                print(f'[{now()}] waiting: request {request["id"]} already started earlier', flush=True)
            elif request['run'] == 'stop':
                state['done'].append(request['id'])
                state_path.write_text(json.dumps(state, indent=1), encoding='utf-8')
                print(f'[{now()}] stop requested ({request["id"]}); exiting', flush=True)
                return 0
            else:
                code, out = git(['pull', '--ff-only', 'origin', a.branch], repo)
                print(f'[{now()}] pull: {"ok" if code == 0 else "FAILED"} {out[-200:]}', flush=True)
                if code == 0:
                    argv = commands(a)[request['run']]
                    log = open(state_dir/f'run_{request["id"]}.log', 'a', encoding='utf-8')
                    log.write(f'=== {now()} {" ".join(argv)}\n')
                    log.flush()
                    child = subprocess.Popen(argv, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT)
                    child_id = request['id']
                    state['done'].append(child_id)
                    state.setdefault('started', {})[child_id] = dict(run=request['run'], at=now(), pid=child.pid)
                    state_path.write_text(json.dumps(state, indent=1), encoding='utf-8')
                    print(f'[{now()}] started {request["run"]} for request {child_id} (pid {child.pid}); '
                          f'log {state_dir/("run_" + child_id + ".log")}', flush=True)
        if a.once:
            if child is not None:
                child.wait()
                log.close()
            return 0
        time.sleep(a.interval)
    print(f'[{now()}] watch time over; exiting', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
