"""Bounded V4 functional acceptance; never launches research training/evaluation."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import traceback
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_tests(names):
    suite = unittest.TestSuite()
    for name in names:
        path = SOURCE_ROOT / 'tests' / name
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    return result, buffer.getvalue()


def validate(project_root, protocol_path, output):
    from hrl_uav.industrial_hotspot_experiment import load_protocol, source_hashes
    from hrl_uav.industrial_experiment import digest
    from hrl_uav.industrial_env import IndustrialEnv

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol, scene = load_protocol(protocol_path, project_root)
    ledger_path = output.parent / 'functional_budget.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
        if ledger.get('schema') != 'industrial_hotspot_functional_budget_v1':
            raise ValueError('Invalid shared functional budget ledger')
    else:
        ledger = dict(schema='industrial_hotspot_functional_budget_v1',
                      physical_steps=0, backward_calls=0, attempts=[])
    limits = dict(physical_steps=5000, backward_calls=300)
    before = {name: ledger[name] for name in limits}
    attempt = dict(output=str(output.resolve()), status='running', before=before)
    ledger['attempts'].append(attempt)
    atomic_json(ledger_path, ledger)
    started = time.monotonic()
    report = dict(schema='industrial_hotspot_acceptance_v1', passed=False,
                  source_sha256=source_hashes(project_root),
                  protocol_sha256=digest(protocol), scene_sha256=digest(scene),
                  service_steps=0, backward_calls=0, test_count=0,
                  physical_step_cap_all_attempts=5000,
                  backward_cap_all_attempts=300,
                  research_training_started=False,
                  performance_diagnostic_executed=False,
                  checkpoints_are_functional_fixtures_only=True)
    atomic_json(output / 'preregister.json', report)

    def account(name):
        if time.monotonic() - started > 600:
            raise TimeoutError('Functional acceptance exceeded 600 s wall limit')
        if ledger[name] >= limits[name]:
            raise RuntimeError('Cumulative functional budget exhausted: ' + name)
        # Write before invocation: even an interrupted or failed call is charged.
        ledger[name] += 1
        atomic_json(ledger_path, ledger)

    original_step = IndustrialEnv.micro_step
    original_backward = None

    def counted_step(env, *args, **kwargs):
        if env.time_s + env.dt > 20 + 1e-8:
            raise RuntimeError('A functional fixture exceeded 20 s of service')
        account('physical_steps')
        return original_step(env, *args, **kwargs)

    IndustrialEnv.micro_step = counted_step
    try:
        result, log = run_tests(['test_industrial_hotspot_scene.py',
                                 'test_industrial_hotspot_gate.py'])
        (output / 'service_gate_tests.txt').write_text(log, encoding='utf-8')
        report['test_count'] += result.testsRun
        if not result.wasSuccessful():
            raise AssertionError('Scene or gate tests failed; see service_gate_tests.txt')
        if 'torch' in sys.modules:
            raise AssertionError('Scene and pure gate tests imported torch')
        report['service_and_pure_gate_passed_without_torch'] = True

        from hrl_uav.industrial_experiment import runtime_setup
        torch = runtime_setup(protocol, 71)
        original_backward = torch.autograd.backward

        def counted_backward(*args, **kwargs):
            account('backward_calls')
            return original_backward(*args, **kwargs)

        torch.autograd.backward = counted_backward
        result, log = run_tests(['test_industrial_hotspot_training.py'])
        (output / 'learning_restore_tests.txt').write_text(log, encoding='utf-8')
        report['test_count'] += result.testsRun
        if not result.wasSuccessful():
            raise AssertionError('Training/restore tests failed; see learning_restore_tests.txt')
        if source_hashes(project_root) != report['source_sha256']:
            raise RuntimeError('Runtime sources changed during acceptance')
        report['passed'] = True
        report['status'] = 'complete'
        report['result_kind'] = 'functional_acceptance_passed'
        attempt['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        attempt['status'] = 'failed'
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        IndustrialEnv.micro_step = original_step
        if original_backward is not None:
            import torch
            torch.autograd.backward = original_backward
        report['service_steps'] = ledger['physical_steps'] - before['physical_steps']
        report['backward_calls'] = ledger['backward_calls'] - before['backward_calls']
        report['cumulative_all_attempts'] = {name: ledger[name] for name in limits}
        report['wall_seconds'] = time.monotonic() - started
        attempt['after'] = report['cumulative_all_attempts']
        atomic_json(ledger_path, ledger)
        atomic_json(output / 'acceptance.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--protocol', type=Path, default=Path('configs/industrial_hotspot_v4.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.project_root, args.protocol, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
