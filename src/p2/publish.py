"""Push the small status files of a run to a GitHub branch, so the run can be followed remotely.

The branch (default ``paper2-runs`` of the repository this code was cloned from) holds
one folder per output directory, e.g. ``p2_sweeps_v3/``, with the status page
(FEASIBILITY.md / SWEEP.md), the step logs (last lines only), the verify record,
the result tables of retrain-report, the pilot-gate logs and the last lines of
every job log that has not finished (failed or still running).  No weights,
trajectories or full logs.

A separate clone of that branch lives in ``<out-base>/_publish/repo``; the code
checkout is never touched.  Publishing never raises: when git or the network
fails, the error is recorded in ``last_error`` and the experiment continues.

    python run.py publish --out-base D:\\p2_sweeps_v3          # push the current state once
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import common as C

BRANCH = 'paper2-runs'
MAX_JOB_LOGS = 40


# never wait for a credential prompt: without stored credentials git fails at once
GIT_ENV = dict(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never', GIT_ASKPASS='', SSH_ASKPASS='')


def _git(args, cwd, timeout=180):
    try:
        r = subprocess.run(['git', *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace', env=dict(os.environ, **GIT_ENV))
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def _tail_text(path, lines):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    if lines:
        parts = text.splitlines()
        if len(parts) > lines:
            text = f'... ({len(parts)-lines} earlier lines not published)\n' + '\n'.join(parts[-lines:]) + '\n'
    return text


class Publisher:
    def __init__(self, base, name=None, branch=BRANCH, remote=None, interval=1800):
        self.base = Path(base).resolve()
        self.name = name or self.base.name
        self.branch, self.interval = branch, interval
        self.repo = self.base/'_publish'/'repo'
        if remote is None:
            code, out = _git(['remote', 'get-url', 'origin'], C.REPO)
            remote = out.splitlines()[-1].strip() if code == 0 and out else None
        self.remote = remote
        self.last, self.last_error, self.last_ok = 0., None, None

    # ---- files --------------------------------------------------------------------------------
    def files(self):
        """[(source, path inside the branch folder, tail lines or None)]"""
        b, out = self.base, []
        for name in ('FEASIBILITY.md', 'SWEEP.md'):
            if (b/name).exists():
                out.append((b/name, name, None))
        for stage in ('_feasibility', '_sweep'):
            d = b/stage
            if not d.exists():
                continue
            for p in sorted(d.glob('*.json')):
                out.append((p, f'{stage}/{p.name}', None))
            for p in sorted(d.glob('*.log')):
                out.append((p, f'{stage}/{p.name}', 300))
            for p in sorted((d/'verify').glob('verify_*.json')):
                out.append((p, f'{stage}/verify/{p.name}', None))
            for p in sorted((d/'check_env').glob('*.log')):
                out.append((p, f'{stage}/check_env/{p.name}', 120))
            for p in sorted((d/'smoke'/'results').glob('*.md')):
                out.append((p, f'{stage}/smoke_results/{p.name}', None))
        for p in sorted((b/'results').glob('*')):
            if p.is_file() and p.suffix in ('.md', '.csv', '.json'):
                out.append((p, f'results/{p.name}', None))
        for p in sorted((b/'_jobs').glob('*.log')):
            out.append((p, f'_jobs/{p.name}', 200))
        pending = []
        for log in b.glob('*/_jobs/*.log'):
            if not log.with_suffix('.done').exists():
                pending.append(log)
        for d in (b/'_feasibility'/'smoke').glob('*/_jobs/*.log') if (b/'_feasibility'/'smoke').exists() else []:
            if not d.with_suffix('.done').exists():
                pending.append(d)
        pending.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for log in pending[:MAX_JOB_LOGS]:
            rel = log.relative_to(b).as_posix()
            out.append((log, f'unfinished_job_logs/{rel.replace("/_jobs/", "__")}', 80))
        return out

    # ---- git ----------------------------------------------------------------------------------
    def _ensure_repo(self):
        if (self.repo/'.git').exists():
            return True
        if not self.remote:
            self.last_error = 'no git remote: the code folder is not a git clone (git clone the repository instead)'
            return False
        self.repo.parent.mkdir(parents=True, exist_ok=True)
        code, out = _git(['clone', '--single-branch', '--branch', self.branch, '--depth', '1', self.remote,
                          str(self.repo)], self.repo.parent, timeout=600)
        if code != 0:
            shutil.rmtree(self.repo, ignore_errors=True)
            self.repo.mkdir(parents=True)
            for args in (['init', '-q'], ['remote', 'add', 'origin', self.remote], ['checkout', '-q', '-b', self.branch]):
                code, out2 = _git(args, self.repo)
                if code != 0:
                    self.last_error = f'git {" ".join(args)}: {out2} (clone said: {out})'
                    return False
        for key, value in (('user.name', 'p2-runner'), ('user.email', 'p2-runner@localhost')):
            code, current = _git(['config', key], self.repo)
            if code != 0 or not current:
                _git(['config', key, value], self.repo)
        return True

    def publish(self, message, force=True):
        """Copy the files and push; returns True / False.  force=False: at most once per interval."""
        if not force and time.time()-self.last < self.interval:
            return None
        if self.last_ok is False and time.time()-self.last < 600:
            return None                                         # after a failure, retry at most every 10 minutes
        self.last = time.time()
        try:
            if not self._ensure_repo():
                self.last_ok = False
                return False
            _git(['pull', '-q', '--rebase', 'origin', self.branch], self.repo, timeout=300)
            dest = self.repo/self.name
            if dest.exists():
                shutil.rmtree(dest)
            for src, rel, lines in self.files():
                try:
                    text = _tail_text(src, lines)
                except OSError:
                    continue
                target = dest/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, 'w', encoding='utf-8', newline='\n') as f:    # Path.write_text(newline=) needs 3.10
                    f.write(text)
            (dest/'PUBLISHED.txt').write_text(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {message}\n', encoding='utf-8')
            _git(['add', '-A', self.name], self.repo)
            code, out = _git(['commit', '-q', '-m', f'{self.name}: {message}'], self.repo)
            if code != 0 and 'nothing to commit' not in out:
                self.last_error, self.last_ok = f'git commit: {out}', False
                return False
            for attempt in range(2):
                code, out = _git(['push', '-q', 'origin', f'HEAD:{self.branch}'], self.repo, timeout=600)
                if code == 0:
                    self.last_error, self.last_ok = None, True
                    return True
                _git(['pull', '-q', '--rebase', 'origin', self.branch], self.repo, timeout=300)
            self.last_error, self.last_ok = f'git push: {out}', False
            return False
        except Exception as e:                                  # noqa: BLE001 (never stop the experiment)
            self.last_error, self.last_ok = f'{type(e).__name__}: {e}', False
            return False

    def status_line(self):
        if self.last_ok is None:
            return f'GitHub 推送：尚未推送（分支 {self.branch}，文件夹 {self.name}/）'
        if self.last_ok:
            return f'GitHub 推送：成功（分支 {self.branch}，文件夹 {self.name}/）'
        return f'**GitHub 推送：失败（实验不受影响，继续运行）：{self.last_error}**'
