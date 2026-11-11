"""One command for the whole v2.6 run, with a status page pushed to GitHub.

    python run_v26.py go --main D:\\p2_runs_v2 --out-base D:\\p2_v26 --zeroshot-main D:\\p2_zeroshot_v4

Steps (status page <out-base>/V26.md; re-running the same command resumes):
  1 smoke     tiny main run + tiny v2.6 run in <out-base>/_smoke (plumbing with real networks)
  2 prepare   copy the window-100 snapshots and dev records from the main root (read only)
  3 phase1    continue M3 / HoverSAC to 300 windows (variants ext and burst), dev evaluation of
              the new candidates, selection, burst experiment on dev, no-stage-3 ablation
  4 decide    choose ext / burst / main from dev data only (DECISION.md)
  5 phase2    test, burst experiment and zero-shot sweep of the adopted variant
  6 report    results/v26_summary.md and the consolidated records
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from p2 import common as C
from p2.publish import Publisher, _tail_text  # noqa: F401  (unchanged publisher; files() extended below)
from . import core as V
from . import report as R

STEPS = dict(smoke='管道冒烟（极小训练量，1 个种子）', prepare='从主实验目录复制第 100 个窗口的快照（只读）',
             phase1='继续微调到 300 个窗口（ext、burst 两个版本）+ 验证集评估与选择 + 无第三阶段消融',
             decide='按验证集数据选择版本', phase2='所选版本的测试、突发实验与零样本扫参', report='汇总')
STATUS = dict(pending='未开始', running='运行中', passed='通过', failed='失败', skipped='已通过（沿用上次）')


def code_hash():
    files = sorted((C.SRC/'v26').glob('*.py'))+[V.RUN]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


class V26Publisher(Publisher):
    def files(self):
        b, out = self.base, []
        for name in (V.STATUS_PAGE, 'DECISION.md'):
            if (b/name).exists():
                out.append((b/name, name, None))
        if (b/'_v26'/'state.json').exists():
            out.append((b/'_v26'/'state.json', '_v26/state.json', None))
        for p in sorted((b/'_v26').glob('*.log')):
            out.append((p, f'_v26/{p.name}', 300))
        for p in sorted((b/'results').glob('*')):
            if p.is_file() and p.suffix in ('.md', '.csv', '.json'):
                out.append((p, f'results/{p.name}', None))
        pending = [log for log in (b/'_jobs').glob('*.log') if not log.with_suffix('.done').exists()]
        pending += [log for log in (b/'_smoke').glob('**/_jobs/*.log') if not log.with_suffix('.done').exists()]
        pending.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for log in pending[:40]:
            out.append((log, 'unfinished_job_logs/'+log.relative_to(b).as_posix().replace('/', '__'), 80))
        return out


class Go:
    def __init__(self, main, base, workers, seeds, smoke=False, publish=True, zeroshot_main=None):
        self.main, self.base = V.check_roots(main, base)
        self.workers, self.seeds, self.smoke = workers, tuple(seeds), smoke
        self.zeroshot_main = str(Path(zeroshot_main).resolve()) if zeroshot_main else None
        self.dir = self.base/'_v26'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.code = code_hash()
        self.state_path = self.dir/'state.json'
        old = C.read_json(self.state_path) if self.state_path.exists() else {}
        same = old.get('code') == self.code and old.get('main') == str(self.main)
        self.state = dict(code=self.code, main=str(self.main), started=old.get('started') if same else V.now(),
                          steps={k: (old.get('steps', {}).get(k) if same else None) or dict(status='pending')
                                 for k in STEPS})
        self.progress, self.note = None, ''
        self.publisher = V26Publisher(self.base) if publish and not smoke else None

    # ---- status ---------------------------------------------------------------------------
    def render(self, headline):
        lines = ['# v2.6（继续微调 + 训练中加入区域突发）：状态', '', f'**{headline}**', '',
                 f'最后更新：{V.now()}（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）', '',
                 f'- 主实验目录（只读）：`{self.main}`', f'- 输出目录：`{self.base}`',
                 f'- 代码：`{self.code[:12]}`；P2_DEVICE={os.environ.get("P2_DEVICE") or "未设置"}；进程数 {self.workers}；'
                 f'种子 {", ".join(map(str, self.seeds))}']
        if self.publisher:
            lines.append(f'- {self.publisher.status_line()}')
        lines += ['', '| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |', '|---|---|---|---|---|---|']
        for i, (k, title) in enumerate(STEPS.items(), 1):
            s = self.state['steps'][k]
            sec = s.get('seconds')
            took = '' if sec is None else (f'{sec/60:.0f} 分钟' if sec < 5400 else f'{sec/3600:.1f} 小时')
            lines.append(f'| {i} | {title} | {STATUS[s["status"]]} | {s.get("started", "")} | {s.get("ended", "")} | {took} |')
        if self.progress:
            p = self.progress
            lines += ['', f'任务进度：{p["done"]}/{p["total"]} 完成；运行中 {len(p["running"])} 个'
                      + (f'：{", ".join(p["running"][:12])}' if p['running'] else '')
                      + (f'；**失败 {len(p["failed"])} 个：{", ".join(p["failed"][:12])}**' if p['failed'] else '')]
        if self.note:
            lines += ['', self.note]
        if (self.base/'DECISION.md').exists():
            lines += ['', '---', '', (self.base/'DECISION.md').read_text(encoding='utf-8')]
        return '\n'.join(lines)+'\n'

    def save(self, headline, event, force=True):
        C.write_json(self.state_path, self.state)
        (self.base/V.STATUS_PAGE).write_text(self.render(headline), encoding='utf-8')
        if self.publisher and self.publisher.publish(f'v26: {event}', force=force) is not None:
            (self.base/V.STATUS_PAGE).write_text(self.render(headline), encoding='utf-8')

    def step(self, key, action, skippable=True):
        s = self.state['steps'][key]
        if skippable and s['status'] in ('passed', 'skipped'):
            s['status'] = 'skipped'
            print(f'[{V.now()}] {STEPS[key]}: passed before with the same code, skipped', flush=True)
            return True
        self.state['steps'][key] = s = dict(status='running', started=V.now())
        self.progress = None
        self.save(f'运行中：{STEPS[key]}', f'{key} started')
        print(f'[{V.now()}] {STEPS[key]} ...', flush=True)
        t0 = time.time()
        try:
            problem = action()
        except Exception as e:                                  # noqa: BLE001 (reported on the status page)
            problem = f'{type(e).__name__}: {e}'
        s.update(status='failed' if problem else 'passed', ended=V.now(), seconds=round(time.time()-t0))
        if problem:
            self.note = (f'## 停止原因\n\n步骤“{STEPS[key]}”失败：\n\n```\n{problem}\n```\n\n'
                         f'任务日志在 `{self.base/"_jobs"}`，步骤日志在 `{self.dir}`。修正原因后原样重新运行同一条命令，'
                         f'已完成的任务会跳过，训练从最近的快照继续。')
            self.save(f'已停止：步骤“{STEPS[key]}”失败', f'STOPPED at {key}')
            print(f'[{V.now()}] STOPPED at {key}: {problem}', flush=True)
            return False
        self.save(f'运行中：{STEPS[key]}完成', f'{key} passed')
        print(f'[{V.now()}] {STEPS[key]}: passed', flush=True)
        return True

    def jobs(self, jobs, headline):
        def poll(progress):
            self.progress = progress
            self.save(headline, 'progress', force=False)
        failed = V.run_jobs(jobs, self.base, self.main, workers=self.workers, smoke=self.smoke, poll=poll)
        return ('failed or skipped jobs: '+', '.join(failed)) if failed else None

    # ---- steps ----------------------------------------------------------------------------
    def smoke_step(self):
        import shutil
        py, smoke = sys.executable, self.base/'_smoke'
        shutil.rmtree(smoke, ignore_errors=True)                 # smoke data only; always from scratch
        seed = str(self.seeds[0])
        commands = [[py, str(C.REPO/'run.py'), 'pipeline', '--smoke', '--no-sweeps', f'--seeds={seed}',
                     f'--workers={self.workers}', f'--out={smoke/"main"}'],
                    [py, str(V.RUN), 'go', '--smoke', f'--main={smoke/"main"}', f'--out-base={smoke/"v26"}',
                     f'--seeds={seed}', f'--workers={self.workers}', '--no-publish']]
        log = self.dir/'smoke.log'
        for argv in commands:
            with open(log, 'a', encoding='utf-8') as f:
                f.write(f'\n=== {V.now()}  {" ".join(argv)}\n')
                f.flush()
                code = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, cwd=str(C.REPO))
            if code != 0:
                return f'exit code {code}: {" ".join(argv)}\n(log: {log})'
        return None

    def run(self):
        if not self.smoke:
            if not self.step('smoke', self.smoke_step):
                return 1
        else:
            self.state['steps']['smoke'] = dict(status='skipped')
        if not self.step('prepare', lambda: V.prepare(self.main, self.base, self.seeds, self.smoke) and None):
            return 1
        if not self.step('phase1', lambda: self.jobs(V.plan_phase1(self.seeds, self.smoke), '运行中：继续微调与验证集评估'),
                         skippable=False):
            return 1
        force = 'ext' if self.smoke else None                    # smoke: always exercise the evaluation path
        if not self.step('decide', lambda: V.decide(self.base, self.seeds, self.smoke, force=force) and None,
                         skippable=False):
            return 1
        decision = C.read_json(self.base/'results'/'v26_decision.json')
        if not self.step('phase2', lambda: self.jobs(V.plan_phase2(self.base, decision, self.seeds, self.smoke,
                                                                 force=self.smoke),
                                                     f'运行中：{decision["adopted"]} 的测试与扫参'), skippable=False):
            return 1
        if not self.step('report', lambda: R.build(self.base, self.main, self.seeds, self.smoke,
                                                   self.zeroshot_main) and None, skippable=False):
            return 1
        adopted = decision['adopted']
        self.progress = None
        self.note = ('## 完成\n\n' + ('按验证集数据，没有比现有模型更好的候选，现有结果保持不变。' if adopted == 'main'
                                     else f'采用版本 `{adopted}`。') +
                     f'\n\n汇总：`{self.base/"results"/"v26_summary.md"}`。')
        self.save(f'完成：采用 {adopted}', 'FINISHED')
        print(f'[{V.now()}] FINISHED: adopted {adopted}; summary {self.base/"results"/"v26_summary.md"}', flush=True)
        return 0
