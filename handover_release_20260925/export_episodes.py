"""Export all recorded training episodes and lossless logs; no model execution."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'episodes'
RUN = ROOT / 'industrial_hotspot_records/load24725_fresh_sac_lr1e5_20260924'
PARTS = [('training', 1, 300), ('joint25_run02', 1, 25),
         ('joint50', 26, 50), ('joint400', 51, 400)]


def flatten(value, prefix='', out=None):
    out = {} if out is None else out
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, (prefix + '.' if prefix else '') + str(key), out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten(item, prefix + '.' + str(index), out)
    else:
        out[prefix] = value
    return out


def total(value):
    return sum(total(v) for v in value) if isinstance(value, list) else float(value)


def main():
    OUT.mkdir(exist_ok=True)
    rows, sources = [], []
    for stage, first, last in PARTS:
        path = RUN / stage / 'episodes.jsonl'
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        assert [r['episode'] for r in records] == list(range(first, last + 1)), stage
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        assert gzip.decompress(compressed) == raw
        archive = OUT / (stage + '.episodes.jsonl.gz')
        archive.write_bytes(compressed)
        sources.append(dict(source=path.relative_to(ROOT).as_posix(), first=first, last=last,
                            episodes=len(records), bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest(),
                            archive=archive.relative_to(ROOT).as_posix()))
        for record in records:
            trajectory = record['trajectory']
            assert len(trajectory) == 180 and record['metrics']['physical_steps'] == 9000
            assert record['seconds'] == 900 and record['metrics']['seconds'] == 900
            assert record['truncated'] and not record['terminated']
            assert math.isclose(record['physical_return'], -record['metrics']['cost_sum'], abs_tol=1e-6)
            assert all(math.isfinite(record['metrics'][k]) for k in ('weighted_delay_ms', 'cost_sum'))
            row = flatten({k: v for k, v in record.items() if k != 'trajectory'})
            unfinished = [p['arrived'] - p['completed'] for p in record['metrics']['priority']]
            assert sum(unfinished) == record['metrics']['unfinished_parents']
            for index, value in enumerate(unfinished):
                row['derived.priority.' + str(index) + '.unfinished'] = value
            row['source_run'] = stage
            row['source_log'] = path.relative_to(ROOT).as_posix()
            row['trajectory_samples'] = len(trajectory)
            for field in ('unfinished_counts', 'upload_mbits', 'ready_gcycles', 'transfer_mbits',
                          'transfer_future_gcycles', 'cpu_gcycles'):
                values = [total(point[field]) for point in trajectory]
                row['sample5s_mean.' + field] = statistics.mean(values)
                row['sample5s_max.' + field] = max(values)
            rows.append(row)
    assert len(rows) == 700
    assert [r['episode'] for r in rows if r['stage'] == 'pretrain'] == list(range(1, 301))
    assert [r['episode'] for r in rows if r['stage'] == 'joint'] == list(range(1, 401))

    leading = ['stage', 'episode', 'source_run', 'seconds', 'physical_return', 'discounted_return',
               'metrics.mean_cost_per_s', 'metrics.weighted_delay_ms',
               'metrics.priority.0.mean_delay_ms', 'metrics.priority.1.mean_delay_ms',
               'derived.priority.0.unfinished', 'derived.priority.1.unfinished']
    fields = leading + sorted(set().union(*(set(r) for r in rows)) - set(leading))
    for name, selected in [('all_episodes', rows),
                           ('pretrain_001_300', [r for r in rows if r['stage'] == 'pretrain']),
                           ('joint_001_400', [r for r in rows if r['stage'] == 'joint'])]:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
        (OUT / (name + '.csv')).write_bytes(stream.getvalue().encode('utf-8-sig'))
    blocks = []
    for stage in ('pretrain', 'joint'):
        group = [r for r in rows if r['stage'] == stage]
        for start in range(0, len(group), 50):
            block = group[start:start + 50]
            blocks.append(dict(stage=stage, first=block[0]['episode'], last=block[-1]['episode'],
                mean_episode_delay_ms=statistics.mean(r['metrics.weighted_delay_ms'] for r in block),
                mean_cost_per_s=statistics.mean(r['metrics.mean_cost_per_s'] for r in block),
                mean_episode_low_unfinished=statistics.mean(r['derived.priority.1.unfinished'] for r in block)))
    audit = dict(passed=True, episodes=700, pretrain=300, joint=400,
                 physical_steps=6300000, trajectory_samples=126000,
                 simulation_steps_added=0, optimizer_steps_added=0, sources=sources,
                 note='Training episodes vary in scenarios and include exploration/updates. Block delays are arithmetic means of per-episode weighted completed-parent delay, not pooled frozen evaluation.',
                 blocks=blocks)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, stage in enumerate(('pretrain', 'joint')):
        group = [r for r in rows if r['stage'] == stage]
        x = [r['episode'] for r in group]
        for index, (key, label) in enumerate((('metrics.weighted_delay_ms', '加权已完成时延（ms）'),
                                              ('physical_return', '每轮未折扣物理回报'))):
            y = [r[key] for r in group]
            ax = axes[index, col]
            ax.plot(x, y, color='0.70', linewidth=0.7, label='每轮原始值')
            smooth = [statistics.mean(y[max(0, i-19):i+1]) for i in range(len(y))]
            ax.plot(x, smooth, color='black', linewidth=1.3, label='最近20轮均值')
            ax.set_ylabel(label)
            ax.set_xlabel('预训练轮次' if stage == 'pretrain' else '联合训练轮次')
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        axes[0, col].set_title('SAC悬停预训练1—300轮' if stage == 'pretrain' else '联合训练1—400轮（从预训练250轮接入）')
    fig.suptitle('逐轮训练日志：场景逐轮变化，含探索和学习；不等同于冻结开发评价', fontsize=12)
    fig.savefig(OUT / 'training_curves.svg')
    fig.savefig(OUT / 'training_curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps(audit, ensure_ascii=True))


if __name__ == '__main__':
    main()
