"""One command for the feasibility stage (README section 4, step 1): every check, then the two pilot points.

    python run.py feasibility --main D:\\p2_runs_v2 --out-base D:\\p2_sweeps_v3 [--workers 8]

Steps, each logged to <out-base>/_feasibility/<step>.log:
  1 check-env      versions, simulator, smoke trainings
  2 tests          the self-tests of tests/
  3 verify         this code against the main run's records
  4 smoke          tiny retrained sweep at users=7 and u2u_mhz=1.5 (plumbing with real networks)
  5 pilots         cpu_ghz=9 and u2u_mhz=1.5, 3 seeds, full pipeline (about half a day to a day)
  6 summary        pilot-check, progress (time per point, remaining time), retrain-report
<out-base>/FEASIBILITY.md always shows the current state: which step runs, when it was last updated
(every 10 minutes during step 5), and, if the command stopped, which step failed, why, and the log.
A failing step stops the command.  Re-running the same command skips steps 1-4 that passed with the
same code and main run, and resumes step 5 where it stopped.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import common as C

PILOTS = (('cpu_ghz', 9.0), ('u2u_mhz', 1.5))
SMOKE = 'users=7,u2u_mhz=1.5'
TESTS = (['test_users.py'], ['test_retrain.py'], ['test_checks.py'], ['test_eval_report.py'],
         ['test_verify_compare.py'], ['test_perf_equivalence.py', '150'], ['test_identity.py'])
TITLES = {'check-env': '环境检查', 'tests': '自检脚本', 'verify': '与主实验记录比较', 'smoke': '扫参管道冒烟',
          'pilots': '两个试点（3 个种子，完整流程）', 'summary': '试点检查、耗时、汇总'}
STATUS_TEXT = {'pending': '未开始', 'running': '运行中', 'passed': '通过', 'failed': '失败', 'skipped': '已通过（沿用上次）'}


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def tail(path, n=25):
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return ''
    return '\n'.join(lines[-n:])


class Feasibility:
    def __init__(self, main, base, workers, seeds):
        self.main, self.base, self.workers, self.seeds = Path(main).resolve(), Path(base).resolve(), workers, seeds
        self.dir = self.base/'_feasibility'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir/'state.json'
        from .retrain import code_fingerprint
        self.code = code_fingerprint()[0]
        old = C.read_json(self.state_path) if self.state_path.exists() else {}
        same = old.get('code') == self.code and old.get('main') == str(self.main)
        self.state = dict(code=self.code, main=str(self.main), started=old.get('started') if same else now(),
                          steps={k: (old.get('steps', {}).get(k) if same else None) or dict(status='pending')
                                 for k in TITLES})
        self.note, self.progress_text = '', ''
        self.run_py = str(C.REPO/'run.py')

    # ---- bookkeeping -------------------------------------------------------------------------
    def save(self, headline):
        C.write_json(self.state_path, self.state)
        lines = [f'# 可行性阶段状态', '', f'**{headline}**', '', f'最后更新：{now()}（运行中每 10 分钟更新一次；'
                 f'如果这个时间停在很久以前、而命令窗口已经没有输出，说明程序已经退出）', '',
                 f'- 主实验目录：`{self.main}`', f'- 输出目录：`{self.base}`',
                 f'- 代码指纹：`{self.code[:12]}`；P2_DEVICE={os.environ.get("P2_DEVICE") or "未设置"}；'
                 f'P2_THREADS={os.environ.get("P2_THREADS") or "未设置（2）"}；进程数 {self.workers}', '',
                 '| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |', '|---|---|---|---|---|---|']
        for i, (k, title) in enumerate(TITLES.items(), 1):
            s = self.state['steps'][k]
            sec = s.get('seconds')
            took = '' if sec is None else (f'{sec/60:.0f} 分钟' if sec < 5400 else f'{sec/3600:.1f} 小时')
            lines.append(f'| {i} | {title} | {STATUS_TEXT[s["status"]]} | {s.get("started", "")} | '
                         f'{s.get("ended", "")} | {took} |')
        if self.note:
            lines += ['', self.note]
        if self.progress_text:
            lines += ['', '## 进度（run.py progress）', '', '```', self.progress_text, '```']
        for k in TITLES:
            s = self.state['steps'][k]
            if s.get('excerpt'):
                lines += ['', f'## {TITLES[k]}：输出摘录', '', '```', s['excerpt'], '```']
        (self.base/'FEASIBILITY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def fresh_log(self, key):
        """Start <dir>/<key>.log afresh; the previous attempt is kept as <key>.prev.log."""
        log = self.dir/f'{key}.log'
        if log.exists():
            os.replace(log, self.dir/f'{key}.prev.log')

    def run_commands(self, key, commands, poll=None):
        """Run argv lists one after another into <dir>/<key>.log; False at the first non-zero exit."""
        log = self.dir/f'{key}.log'
        for argv in commands:
            with open(log, 'a', encoding='utf-8') as f:
                f.write(f'\n=== {now()}  {" ".join(argv)}\n')
                f.flush()
                proc = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, cwd=str(C.REPO))
                last = time.time()
                while True:
                    try:
                        proc.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        if poll and time.time()-last >= 600:
                            poll()
                            last = time.time()
                if proc.returncode != 0:
                    f.write(f'=== exit code {proc.returncode}\n')
                    return False, argv
        return True, None

    def step(self, key, commands, excerpt=None, poll=None, skippable=True):
        s = self.state['steps'][key]
        if skippable and s['status'] in ('passed', 'skipped'):
            s['status'] = 'skipped'
            print(f'[{now()}] {TITLES[key]}: passed before with the same code, skipped', flush=True)
            return True
        self.state['steps'][key] = s = dict(status='running', started=now())
        self.save(f'运行中：{TITLES[key]}')
        print(f'[{now()}] {TITLES[key]} ...  (log: {self.dir/(key+".log")})', flush=True)
        self.fresh_log(key)
        t0 = time.time()
        ok, failed = self.run_commands(key, commands, poll)
        s.update(status='passed' if ok else 'failed', ended=now(), seconds=round(time.time()-t0))
        text = tail(self.dir/f'{key}.log', 400)
        s['excerpt'] = excerpt(text) if excerpt else '\n'.join(text.splitlines()[-12:])
        print(f'[{now()}] {TITLES[key]}: {"passed" if ok else "FAILED"}', flush=True)
        if not ok:
            self.note = (f'## 停止原因\n\n第 {list(TITLES).index(key)+1} 步“{TITLES[key]}”失败，失败的命令：\n\n'
                         f'```\n{" ".join(failed)}\n```\n\n完整日志：`{self.dir/(key+".log")}`。'
                         f'修正原因后原样重新运行同一条命令，已通过的步骤会跳过。')
            self.save(f'已停止：第 {list(TITLES).index(key)+1} 步“{TITLES[key]}”失败')
            print('\n'.join(text.splitlines()[-15:]), flush=True)
            print(f'STOPPED at step {list(TITLES).index(key)+1} ({key}); status: {self.base/"FEASIBILITY.md"}', flush=True)
        return ok

    def poll_progress(self):
        from .checks import progress
        try:
            self.progress_text = progress(self.base, list(PILOTS), self.seeds, self.workers)
        except Exception as e:                                  # noqa: BLE001 (status only)
            self.progress_text = f'(progress unavailable: {e})'
        self.save('运行中：两个试点（3 个种子，完整流程）')

    # ---- the stage ---------------------------------------------------------------------------
    def run(self):
        py, run = sys.executable, self.run_py
        seeds = ','.join(map(str, self.seeds))
        tests = C.REPO/'tests'
        free = shutil.disk_usage(self.base).free/2**30
        self.note = f'输出目录所在磁盘剩余 {free:.0f} GB。' if free >= 30 else \
            f'**注意：输出目录所在磁盘只剩 {free:.0f} GB，全部 16 个点可能不够。**'
        steps = [
            ('check-env', [[py, run, 'check-env', f'--out={self.dir/"check_env"}']], None),
            ('tests', [[py, str(tests/t[0]), *t[1:]] for t in TESTS], None),
            ('verify', [[py, run, 'verify', f'--main={self.main}', f'--work={self.dir/"verify"}']],
             lambda t: '\n'.join(x for x in t.splitlines() if x.startswith(('train ', 'test ', 'note:', 'ALL', 'DIFF')))),
            ('smoke', [[py, run, 'retrain', f'--out-base={self.dir/"smoke"}', f'--points={SMOKE}', '--smoke',
                        '--seeds=4101', f'--workers={self.workers}'],
                       [py, run, 'retrain-report', f'--out-base={self.dir/"smoke"}', f'--main={self.main}',
                        f'--points={SMOKE}', '--smoke', '--seeds=4101']], None),
        ]
        for key, commands, excerpt in steps:
            if not self.step(key, commands, excerpt):
                return 1
        points = ','.join(f'{p}={v:g}' for p, v in PILOTS)
        self.poll_progress()
        if not self.step('pilots', [[py, run, 'retrain', f'--out-base={self.base}', f'--main={self.main}',
                                     f'--points={points}', f'--seeds={seeds}', f'--workers={self.workers}']],
                         poll=self.poll_progress, skippable=False):
            return 1
        self.poll_progress()
        self.state['steps']['summary'] = dict(status='running', started=now())
        self.save('运行中：试点检查、耗时、汇总')
        self.fresh_log('summary')
        t0 = time.time()
        ok_check, _ = self.run_commands('summary', [[py, run, 'pilot-check', f'--out-base={self.base}',
                                                     f'--main={self.main}', f'--seeds={seeds}']])
        self.run_commands('summary', [[py, run, 'progress', f'--out-base={self.base}', f'--seeds={seeds}',
                                       f'--workers={self.workers}'],
                                      [py, run, 'retrain-report', f'--out-base={self.base}', f'--main={self.main}',
                                       f'--points={points}', f'--seeds={seeds}']])
        s = self.state['steps']['summary']
        s.update(status='passed' if ok_check else 'failed', ended=now(), seconds=round(time.time()-t0),
                 excerpt=tail(self.dir/'summary.log', 60))
        self.progress_text = ''
        report = self.base/'results'/'retrain_summary.md'
        if ok_check:
            self.note = ('## 结论：可行性阶段通过\n\n两个试点的 M1 都通过可训练性检查（判据见 docs/PROTOCOL.md 第 5 节）。'
                         f'汇总表：`{report}`。下面“试点检查、耗时、汇总”的摘录里有按实测耗时外推的剩余时间'
                         '（两个试点最慢，外推偏长）。确认后，用同一个输出目录运行其余点：\n\n'
                         f'```\npython run.py retrain --out-base {self.base} --main {self.main} --workers {self.workers}\n```')
            self.save('可行性阶段通过')
        else:
            self.note = ('## 停止原因\n\n试点检查未通过（判据见 docs/PROTOCOL.md 第 5 节），按方案先讨论，不继续其余点。'
                         f'日志：`{self.dir/"summary.log"}`，汇总表：`{report}`。')
            self.save('已停止：试点检查未通过')
        print((self.base/'FEASIBILITY.md').read_text(encoding='utf-8'), flush=True)
        return 0 if ok_check else 1


def device_from_main(main):
    """P2_DEVICE of the main run (its M1 contract), used when the variable is not set."""
    path = Path(main)/'seed_4101'/'train'/'M1'/'contract.json'
    if os.environ.get('P2_DEVICE') or not path.exists():
        return None
    try:
        device = json.loads(path.read_text(encoding='utf-8')).get('device')
    except (OSError, ValueError):
        return None
    if device:
        os.environ['P2_DEVICE'] = device
    return device
