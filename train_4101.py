"""PyCharm: right-click this file -> Run. Fresh seed 4101; live console output.

Use the project's .venv/Scripts/python.exe interpreter. No command-line
arguments are needed. Preparation is reused when its identity matches.
Each invocation creates a new run; it never resumes an old checkpoint.
"""
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / 'src/hrl_v2/configs/v2_2_1_lite.json'
WORKSPACE = PROJECT / 'outputs/lite_v2_2_1'
SEED = 4101


class ConsoleLog:
    """Flush to both PyCharm's Run console and a durable text log."""
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextlib.contextmanager
def single_run_lock(path):
    """OS-held lock is released even when PyCharm's Stop kills the process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('This launcher is already running. Do not start another copy.') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run():
    sys.path.insert(0, str(PROJECT / 'src'))
    from hrl_v2.runtime import configure
    from hrl_v2.checkpoint import identity, runtime_record, runtime_signature, atomic_json
    from hrl_v2.evaluate import gate_manifest
    from hrl_v2.run import train
    from hrl_v2.baseline_reference import freeze

    configure()  # Fail early if the selected interpreter has no supported GPU.
    binding = identity(WORKSPACE, CONFIG)
    validation_path = WORKSPACE / 'validation/validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    valid = (validation.get('status') == 'passed' and validation.get('identity') == binding
             and runtime_signature(validation.get('runtime', {})) == runtime_signature(runtime_record()))
    print('[1/3] Checking implementation validation...', flush=True)
    if not valid:
        env = dict(os.environ, PYTHONPATH=str(PROJECT / 'src'))
        subprocess.run([sys.executable, '-u', '-m', 'hrl_v2.validate', '--config', str(CONFIG),
                        '--output', str(validation_path.parent), '--skip-benchmark'],
                       cwd=str(PROJECT), env=env, check=True)
    else:
        print('Matching validation passed; reuse it.', flush=True)

    print('[2/3] Preparing frozen dev baselines (20 scenes, G1 >= 12 wins).', flush=True)
    baseline_dir = WORKSPACE / 'baselines'
    reference_path = baseline_dir / 'summary.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8')) if reference_path.exists() else {}
    ready = (reference.get('identity') == binding and reference.get('master') == SEED
             and reference.get('manifest') == gate_manifest(20) and reference.get('required_wins') == 12)
    if not ready:
        print('Existing matching scene results are reused. New scene progress appears below.', flush=True)
        freeze(CONFIG, baseline_dir, 20, 12, master=SEED)
    else:
        print('Frozen 20-scene baseline reference ready; reuse it.', flush=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = WORKSPACE / f'seed_4101_{stamp}'
    print('[3/3] Starting fresh seed 4101 on cuda:0.', flush=True)
    print('Schedule: SAC 1-300; G1 gate; PPO 301-400; alternating blocks 401-600.', flush=True)
    print(f'Run directory: {output}', flush=True)
    print('Each completed episode prints reward, delay, time and update counts.', flush=True)
    atomic_json(WORKSPACE / 'latest_manual_launch.json', dict(pid=os.getpid(),output=str(output),
        config=str(CONFIG),seed=SEED,started=datetime.now().isoformat(),resume=False))
    started = time.perf_counter()
    result = train(WORKSPACE, CONFIG, SEED, output, baseline_reference=reference_path)
    print(f'Run stopped with status={result["status"]}; wall hours={(time.perf_counter()-started)/3600:.3f}', flush=True)
    print(f'Checkpoint pointer: {output / "latest.json"}', flush=True)


def main():
    logs = WORKSPACE / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / ('pycharm_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
    with single_run_lock(WORKSPACE / 'manual_launch.lock'), path.open('w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(ConsoleLog(sys.stdout, log)), contextlib.redirect_stderr(ConsoleLog(sys.stderr, log)):
            print(f'Live console log: {path}', flush=True)
            try:
                run()
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
