"""Collect an explicit release inventory without running simulation or learning."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUN = 'industrial_hotspot_records/load24725_fresh_sac_lr1e5_20260924'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    acceptance = 'industrial_device_sac_pretrain/validation/attempt02/acceptance.json'
    sealed = read(acceptance)['identity']['source_sha256']
    for name, expected in sealed.items():
        assert sha(name) == expected, name
    contract = read(RUN + '/joint400/contract.json')
    for name, expected in contract['runner_sha256'].items():
        path = 'industrial_device_joint_fresh/' + name
        assert sha(path) == expected, path
        sealed[path] = expected

    paths = set(sealed)
    for package in ('industrial_device_pilot', 'industrial_device_sac_pretrain',
                    'industrial_device_joint_fresh'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / package).glob('*.py'))
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / package).glob('*.md'))
    paths.update(('tests/test_industrial_hotspot_scene.py',
                  'tests/test_industrial_hotspot_gate.py',
                  'tests/test_industrial_hotspot_training.py'))
    scanned = set()
    while True:
        pending = [p for p in paths - scanned if p.endswith('.py')]
        if not pending:
            break
        for name in pending:
            tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
            scanned.add(name)
            parts = list(Path(name).with_suffix('').parts)
            package = parts[:-1]
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = (package[:len(package) - node.level + 1] if node.level else [])
                    base += (node.module or '').split('.') if node.module else []
                    modules.append('.'.join(base))
                    modules.extend('.'.join(base + [a.name]) for a in node.names)
            for module in modules:
                stem = module.replace('.', '/')
                for candidate in (stem + '.py', stem + '/__init__.py'):
                    if (ROOT / candidate).is_file():
                        paths.add(candidate)
                parent = Path(stem).parent
                while str(parent) != '.':
                    init = (parent / '__init__.py').as_posix()
                    if (ROOT / init).is_file():
                        paths.add(init)
                    parent = parent.parent

    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'configs').glob('industrial*.json'))
    paths.update((acceptance,
        'industrial_hotspot_records/load24725_seed1_20260923/training/protocol.json',
        'industrial_hotspot_records/load24725_seed1_20260923/training/scene.json',
        'industrial_hotspot_records/load24725_seed1_20260923/training/evaluations/pretrain-0200/report.json',
        'industrial_hotspot_records/load24725_device_lr_continue_20260924/phase75/protocol.json'))
    for stage in ('training', 'joint25_run02', 'joint50', 'joint400'):
        for name in ('protocol.json', 'scene.json', 'manifest.json', 'progress.json',
                     'selection.json', 'contract.json', 'preflight.json',
                     'startup_verified.json', 'final_verified.json', 'report.md'):
            path = RUN + '/' + stage + '/' + name
            if (ROOT / path).is_file():
                paths.add(path)
        paths.update(p.relative_to(ROOT).as_posix()
                     for p in (ROOT / RUN / stage / 'evaluations').glob('*/report.json'))

    for stage in ('training', 'joint400'):
        for node in read(RUN + '/' + stage + '/selection.json')['evaluations']:
            path = Path(node['report_path']).relative_to(ROOT).as_posix()
            assert path in paths and sha(path) == node['report_sha256'], path
    paths.update(('HANDOVER_20260924.md', 'HANDOVER_20260925.md', 'PROJECT_FACTS.md',
                  'README.md', 'handover_release_20260925/prepare_release.py',
                  'handover_release_20260925/export_episodes.py', '.gitattributes'))
    paths.update(p.relative_to(ROOT).as_posix() for p in (OUT / 'episodes').iterdir() if p.is_file())

    # Preserve bytes used by source/report integrity checks on all platforms.
    attribute_paths = sorted(paths | {
        'handover_release_20260925/manifest.json', 'handover_release_20260925/paths.txt'})
    (ROOT / '.gitattributes').write_text(
        '# Preserve experiment source and evidence SHA256 across checkout platforms.\n' +
        ''.join('/' + p + ' -text\n' for p in attribute_paths), encoding='utf-8')
    missing = [p for p in paths if not (ROOT / p).is_file()]
    assert not missing, missing
    assert not any(p.endswith(('.pt', '.jsonl')) for p in paths)
    inventory = {p: dict(sha256=sha(p), bytes=(ROOT / p).stat().st_size) for p in sorted(paths)}
    manifest_path = 'handover_release_20260925/manifest.json'
    (ROOT / manifest_path).write_text(json.dumps(dict(
        scope='Selected current sources, configuration, frozen reports, all 700 episode results and lossless compressed episode logs; no weights',
        simulation_steps=0, optimizer_steps=0, sealed_source_files=len(sealed),
        files=inventory), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'paths.txt').write_text('\n'.join(sorted(paths | {manifest_path,
        'handover_release_20260925/paths.txt'})) + '\n', encoding='utf-8')
    print(json.dumps(dict(files=len(paths) + 2, python_files=len(scanned),
                         sealed_files=len(sealed), total_bytes=sum(v['bytes'] for v in inventory.values()),
                         largest=sorted(inventory, key=lambda p: inventory[p]['bytes'], reverse=True)[:3])))


if __name__ == '__main__':
    main()
