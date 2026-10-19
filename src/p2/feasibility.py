"""The two run stages of the retrained sweep, each one command with a status page pushed to GitHub.

    python run.py feasibility --main D:\\p2_runs_v2 --out-base D:\\p2_sweeps_v3     # stage 1
    python run.py sweep       --main D:\\p2_runs_v2 --out-base D:\\p2_sweeps_v3     # stage 2 (after stage 1)

Stage 1, feasibility (status page <out-base>/FEASIBILITY.md):
  1 check-env      versions, simulator, smoke trainings
  2 tests          the self-tests of tests/
  3 verify         this code against the main run's records
  4 smoke          tiny retrained sweep at users=7 and u2u_mhz=1.5 (plumbing with real networks)
  5 pilots         cpu_ghz=9 and u2u_mhz=1.5, 3 seeds, full pipeline
  6 summary        pilot-check, progress (time per point, remaining time), retrain-report
Stage 2, sweep (status page <out-base>/SWEEP.md): every point (the pilot gates are part
of the job graph), with retrain-report and progress every 30 minutes, then the final
report and a records-only zip.  It refuses to start unless stage 1 passed.

Every step is logged to <out-base>/_feasibility|_sweep/<step>.log.  The status page shows
which step runs and when it was last updated (every 10 minutes during long steps); when
a step fails, the command stops and the page names the step, the command and the log.
Re-running the same command skips the checks that passed with the same code and resumes
the training where it stopped.  Status page, logs (last lines) and result tables are
pushed to the GitHub branch paper2-runs (p2/publish.py) at every step and every 30
minutes; a failed push never stops the experiment.
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
         ['test_verify_compare.py'], ['test_feasibility.py'], ['test_publish.py'], ['test_perf_equivalence.py', '150'],
         ['test_identity.py'])
STAGES = {
    'feasibility': dict(page='FEASIBILITY.md', dir='_feasibility', title='可行性阶段',
                        steps={'check-env': '环境检查', 'tests': '自检脚本', 'verify': '与主实验记录比较',
                               'smoke': '扫参管道冒烟', 'pilots': '两个试点（3 个种子，完整流程）',
                               'summary': '试点检查、耗时、汇总'}),
    'sweep': dict(page='SWEEP.md', dir='_sweep', title='全部扫参点',
                  steps={'sweep': '全部 16 个点（含两个门控）', 'summary': '最终汇总与导出'}),
}
STATUS_TEXT = {'pending': '未开始', 'running': '运行中', 'passed': '通过', 'failed': '失败',
               'skipped': '已通过（沿用上次）'}


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def tail(path, n=25):
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return ''
    return '\n'.join(lines[-n:])


class Stage:
    kind = None

    def __init__(self, main, base, workers, seeds, publish=True):
        spec = STAGES[self.kind]
        self.title, self.steps, self.page = spec['title'], spec['steps'], spec['page']
        self.main, self.base, self.workers, self.seeds = Path(main).resolve(), Path(base).resolve(), workers, seeds
        self.dir = self.base/spec['dir']
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir/'state.json'
        from .retrain import code_fingerprint
        self.code = code_fingerprint()[0]
        old = C.read_json(self.state_path) if self.state_path.exists() else {}
        same = old.get('code') == self.code and old.get('main') == str(self.main)
        self.state = dict(code=self.code, main=str(self.main), started=old.get('started') if same else now(),
                          steps={k: (old.get('steps', {}).get(k) if same else None) or dict(status='pending')
                                 for k in self.steps})
        self.note, self.progress_text = '', ''
        self.run_py = str(C.REPO/'run.py')
        if publish:
            from .publish import Publisher
            self.publisher = Publisher(self.base)
        else:
            self.publisher = None

    def number(self, key):
        return list(self.steps).index(key)+1

    # ---- status page -------------------------------------------------------------------------
    def render(self, headline):
        lines = [f'# {self.title}：状态', '', f'**{headline}**', '',
                 f'最后更新：{now()}（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）', '',
                 f'- 主实验目录：`{self.main}`', f'- 输出目录：`{self.base}`',
                 f'- 代码指纹：`{self.code[:12]}`；P2_DEVICE={os.environ.get("P2_DEVICE") or "未设置"}；'
                 f'P2_THREADS={os.environ.get("P2_THREADS") or "未设置（2）"}；进程数 {self.workers}']
        if self.publisher:
            lines.append(f'- {self.publisher.status_line()}')
        lines += ['', '| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |', '|---|---|---|---|---|---|']
        for k, title in self.steps.items():
            s = self.state['steps'][k]
            sec = s.get('seconds')
            took = '' if sec is None else (f'{sec/60:.0f} 分钟' if sec < 5400 else f'{sec/3600:.1f} 小时')
            lines.append(f'| {self.number(k)} | {title} | {STATUS_TEXT[s["status"]]} | {s.get("started", "")} | '
                         f'{s.get("ended", "")} | {took} |')
        if self.note:
            lines += ['', self.note]
        if self.progress_text:
            lines += ['', '## 进度（run.py progress）', '', '```', self.progress_text, '```']
        for k in self.steps:
            s = self.state['steps'][k]
            if s.get('excerpt'):
                lines += ['', f'## {self.steps[k]}：输出摘录', '', '```', s['excerpt'], '```']
        return '\n'.join(lines)+'\n'

    def save(self, headline, event='update', force=True):
        """Write state and status page; push to GitHub (force=False: at most every 30 minutes)."""
        C.write_json(self.state_path, self.state)
        (self.base/self.page).write_text(self.render(headline), encoding='utf-8')
        if self.publisher:
            if self.publisher.publish(f'{self.kind}: {event}', force=force) is not None:
                (self.base/self.page).write_text(self.render(headline), encoding='utf-8')   # push result shown

    # ---- running -----------------------------------------------------------------------------
    def fresh_log(self, key):
        """Start <dir>/<key>.log afresh; the previous attempt is kept as <key>.prev.log."""
        log = self.dir/f'{key}.log'
        if log.exists():
            os.replace(log, self.dir/f'{key}.prev.log')

    def run_commands(self, key, commands, poll=None):
        """Run argv lists one after another into <dir>/<key>.log; (False, argv) at the first non-zero exit."""
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
            print(f'[{now()}] {self.steps[key]}: passed before with the same code, skipped', flush=True)
            return True
        self.state['steps'][key] = s = dict(status='running', started=now())
        self.save(f'运行中：{self.steps[key]}', f'step {self.number(key)} {key} started')
        print(f'[{now()}] {self.steps[key]} ...  (log: {self.dir/(key+".log")})', flush=True)
        self.fresh_log(key)
        t0 = time.time()
        ok, failed = self.run_commands(key, commands, poll)
        s.update(status='passed' if ok else 'failed', ended=now(), seconds=round(time.time()-t0))
        text = tail(self.dir/f'{key}.log', 400)
        s['excerpt'] = excerpt(text) if excerpt else '\n'.join(text.splitlines()[-12:])
        print(f'[{now()}] {self.steps[key]}: {"passed" if ok else "FAILED"}', flush=True)
        if ok:
            self.save(f'运行中：{self.steps[key]}完成', f'step {self.number(key)} {key} passed')
        else:
            self.note = (f'## 停止原因\n\n第 {self.number(key)} 步“{self.steps[key]}”失败，失败的命令：\n\n'
                         f'```\n{" ".join(failed)}\n```\n\n完整日志：`{self.dir/(key+".log")}`。'
                         f'修正原因后原样重新运行同一条命令，已通过的步骤会跳过。')
            self.save(f'已停止：第 {self.number(key)} 步“{self.steps[key]}”失败',
                      f'STOPPED at step {self.number(key)} {key}')
            print('\n'.join(text.splitlines()[-15:]), flush=True)
            print(f'STOPPED at step {self.number(key)} ({key}); status: {self.base/self.page}', flush=True)
        return ok

    def poll_progress(self, points, headline, report=False):
        from .checks import progress
        try:
            self.progress_text = progress(self.base, points, self.seeds, self.workers)
        except Exception as e:                                  # noqa: BLE001 (status only)
            self.progress_text = f'(progress unavailable: {e})'
        if report and self.publisher and time.time()-self.publisher.last >= self.publisher.interval:
            self.run_commands('report', [[sys.executable, self.run_py, 'retrain-report', f'--out-base={self.base}',
                                          f'--main={self.main}', f'--seeds={",".join(map(str, self.seeds))}']])
        self.save(headline, 'progress', force=False)

    def finish(self, ok, headline, note, event):
        self.progress_text = ''
        self.note = note
        self.save(headline, event)
        print((self.base/self.page).read_text(encoding='utf-8'), flush=True)
        return 0 if ok else 1

    def disk_note(self, need_gb):
        free = shutil.disk_usage(self.base).free/2**30
        return f'输出目录所在磁盘剩余 {free:.0f} GB。' if free >= need_gb else \
            f'**注意：输出目录所在磁盘只剩 {free:.0f} GB，可能不够。**'


class Feasibility(Stage):
    kind = 'feasibility'

    def run(self):
        py, run = sys.executable, self.run_py
        seeds = ','.join(map(str, self.seeds))
        tests = C.REPO/'tests'
        self.note = self.disk_note(30)
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
        running = '运行中：两个试点（3 个种子，完整流程）'
        self.poll_progress(list(PILOTS), running)
        if not self.step('pilots', [[py, run, 'retrain', f'--out-base={self.base}', f'--main={self.main}',
                                     f'--points={points}', f'--seeds={seeds}', f'--workers={self.workers}']],
                         poll=lambda: self.poll_progress(list(PILOTS), running), skippable=False):
            return 1
        self.state['steps']['summary'] = dict(status='running', started=now())
        self.save('运行中：试点检查、耗时、汇总', 'step 6 summary started')
        self.fresh_log('summary')
        t0 = time.time()
        ok_check, _ = self.run_commands('summary', [[py, run, 'pilot-check', f'--out-base={self.base}',
                                                     f'--main={self.main}', f'--seeds={seeds}']])
        self.run_commands('summary', [[py, run, 'progress', f'--out-base={self.base}', f'--seeds={seeds}',
                                       f'--workers={self.workers}'],
                                      [py, run, 'retrain-report', f'--out-base={self.base}', f'--main={self.main}',
                                       f'--points={points}', f'--seeds={seeds}']])
        self.state['steps']['summary'].update(status='passed' if ok_check else 'failed', ended=now(),
                                              seconds=round(time.time()-t0), excerpt=tail(self.dir/'summary.log', 60))
        report = self.base/'results'/'retrain_summary.md'
        if ok_check:
            return self.finish(True, '可行性阶段通过', (
                '## 结论：可行性阶段通过\n\n两个试点的 M1 都通过可训练性检查（判据见 docs/PROTOCOL.md 第 5 节）。'
                f'汇总表：`{report}`。下面“试点检查、耗时、汇总”的摘录里有按实测耗时外推的剩余时间'
                '（两个试点最慢，外推偏长）。确认后运行第二阶段：\n\n'
                f'```\npython run.py sweep --main {self.main} --out-base {self.base}\n```'), 'PASSED')
        return self.finish(False, '已停止：试点检查未通过', (
            '## 停止原因\n\n试点检查未通过（判据见 docs/PROTOCOL.md 第 5 节），按方案先讨论，不继续其余点。'
            f'日志：`{self.dir/"summary.log"}`，汇总表：`{report}`。'), 'STOPPED: pilot check failed')


class Sweep(Stage):
    kind = 'sweep'

    def run(self):
        feas = self.base/'_feasibility'/'state.json'
        steps = C.read_json(feas)['steps'] if feas.exists() else {}
        if steps.get('pilots', {}).get('status') != 'passed' or \
                steps.get('summary', {}).get('status') not in ('passed', 'failed'):
            print(f'Stage 1 (feasibility) has not finished under {self.base}: run "python run.py feasibility" first '
                  f'(docs/PROTOCOL.md section 4).', flush=True)
            return 2
        # a pilot that failed its check holds only the points it gates (pilot_gate_* in the job graph)
        from .retrain import all_points, point_finished
        py, run = sys.executable, self.run_py
        seeds = ','.join(map(str, self.seeds))
        points = all_points()
        self.note = self.disk_note(60)
        running = '运行中：全部 16 个点'
        self.poll_progress(points, running)
        self.step('sweep', [[py, run, 'retrain', f'--out-base={self.base}', f'--main={self.main}',
                             f'--seeds={seeds}', f'--workers={self.workers}']],
                  poll=lambda: self.poll_progress(points, running, report=True), skippable=False)
        swept_ok = self.state['steps']['sweep']['status'] == 'passed'
        self.state['steps']['summary'] = dict(status='running', started=now())
        self.save('运行中：最终汇总与导出', 'step 2 summary started')
        self.fresh_log('summary')
        t0 = time.time()
        ok, _ = self.run_commands('summary', [
            [py, run, 'progress', f'--out-base={self.base}', f'--seeds={seeds}', f'--workers={self.workers}'],
            [py, run, 'retrain-report', f'--out-base={self.base}', f'--main={self.main}', f'--seeds={seeds}'],
            [py, run, 'retrain-export', f'--out-base={self.base}', f'--dest={self.base/"records.zip"}',
             f'--seeds={seeds}']])
        finished = [f'{p}={v:g}' for p, v in points if point_finished(self.base/f'{p}={v:g}', self.seeds)]
        held = held_points(self.base, points)
        missing = [f'{p}={v:g}' for p, v in points if f'{p}={v:g}' not in finished and f'{p}={v:g}' not in held]
        self.state['steps']['summary'].update(status='passed' if ok and not missing else 'failed', ended=now(),
                                              seconds=round(time.time()-t0), excerpt=tail(self.dir/'summary.log', 40))
        report = self.base/'results'/'retrain_summary.md'
        held_text = (f'\n\n因试点检查未通过而挂起的点（按 docs/PROTOCOL.md 第 5 节先讨论）：{", ".join(held)}。'
                     if held else '')
        if not missing and ok:
            return self.finish(True, '扫参完成' + ('（部分点因门控挂起）' if held else ''), (
                f'## 结论：{len(finished)} 个点完成\n\n汇总表：`{report}`，各参数表 `results/retrain_sweep_<参数>.csv`，'
                f'只含记录的压缩包：`{self.base/"records.zip"}`。' + held_text), 'FINISHED')
        if not missing:
            return self.finish(False, '已停止：最终汇总失败', (
                f'## 停止原因\n\n全部点已完成，但汇总或导出失败，见 `{self.dir/"summary.log"}`。'
                '修正后原样重新运行同一条命令。'), 'STOPPED: summary failed')
        return self.finish(False, f'已停止：{len(missing)} 个点未完成', (
            f'## 停止原因\n\n未完成的点：{", ".join(missing) or "无"}。可能原因：任务失败（见 `{self.dir/"sweep.log"}` 末尾的 '
            '“Failed or skipped jobs”），或门控未通过（`_jobs/pilot_gate_*.log`）。修正后原样重新运行同一条命令即可续跑。'),
            f'STOPPED: {len(missing)} points unfinished')


def held_points(base, points):
    """Points whose pilot gate has not passed (no pilot_gate_<param>.done in <base>/_jobs)."""
    from .pipeline import point_name
    held = []
    for param, value in points:
        gate_param, gate_value = C.GATED_BY[param]
        if (param, C.point_value(param, value)) == (gate_param, C.point_value(gate_param, gate_value)):
            continue                                            # the pilot itself is never held
        if not (Path(base)/'_jobs'/f'pilot_gate_{gate_param}.done').exists():
            held.append(point_name(param, value))
    return held


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
