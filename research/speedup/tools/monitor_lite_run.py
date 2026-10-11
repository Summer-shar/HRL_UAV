"""Read committed episode logs; report milestones without touching a learner."""
import argparse
import json
from pathlib import Path
import statistics
from hrl_v2.checkpoint import atomic_json
from hrl_v2.schedule import phase


def inspect(output):
    output=Path(output)
    latest=json.loads((output/'latest.json').read_text())
    checkpoint=Path(latest['path']).name
    completed=int(checkpoint.split('-')[1])
    rows={}
    path=output/'train_metrics.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue  # writer may be finishing its last line
            if row['global_episode']<=completed: rows[row['global_episode']]=row
    ordered=[rows[k] for k in sorted(rows)]
    groups={layer:[r for r in ordered if r['layer']==layer] for layer in ('lower','upper')}
    timing={layer:dict(episodes=len(rs),mean_work_seconds=statistics.mean(r['work_seconds'] for r in rs) if rs else None,
                       mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rs if r['wall_seconds'] is not None) if any(r['wall_seconds'] is not None for r in rs) else None)
            for layer,rs in groups.items()}
    smooth=[dict(episode=r['global_episode'],truncated_delay_10_episode_mean_s=
                statistics.mean(x['truncated_delay_s'] for x in ordered[max(0,i-9):i+1])) for i,r in enumerate(ordered)]
    # Distinguish cleared replay, critic-only warmup and full lower updates.
    def category(row):
        if row['layer']=='upper':return 'upper'
        if row['sac_delta']['events']==0:return 'lower_empty_replay'
        return 'lower_actor' if row['sac_delta']['actor_updates'] else 'lower_critic_only'
    buckets={}
    for row in ordered:
        if row['checkpoint_seconds'] is not None:buckets.setdefault(category(row),[]).append(row['work_seconds']+row['checkpoint_seconds'])
    forecasts={k:statistics.mean(v) for k,v in buckets.items()}
    known=0.;unknown={};previous=None
    for number in range(1,601):
        p=phase(number);block=(p.stage,p.cycle,p.layer)
        key=('upper' if p.layer=='upper' else 'lower_empty_replay' if block!=previous
             else 'lower_actor' if p.actor_enabled else 'lower_critic_only')
        previous=block
        if key in forecasts:known+=forecasts[key]
        else:unknown[key]=unknown.get(key,0)+1
    result=dict(completed_episode=completed,recorded_episodes=len(ordered),timing=timing,
        first_10=[{k:r[k] for k in ('global_episode','layer','work_seconds','wall_seconds','sac_delta','resources')} for r in ordered[:10]],
        recent=ordered[-10:],delay_trend=smooth,latest_checkpoint=latest,
        budget=dict(calibrated_category_means_s=forecasts,known_component_hours=known/3600,
                    unmeasured_episode_categories=unknown,
                    development_evaluation='not extrapolated from short windows; add measured panel durations when available',
                    note='Warmup, replay resets and checkpoint IO included; upper mean includes measured PPO batches only.'),
        gate_files=[str(p) for p in sorted((output/'gates').glob('*.json'))],
        failure=json.loads((output/'technical_failure.json').read_text()) if (output/'technical_failure.json').exists() else None)
    atomic_json(output/'monitor_snapshot.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(inspect(args.output),ensure_ascii=False,indent=2))
