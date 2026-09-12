"""Offline arithmetic only: the fixed local counterfactual matrix, no simulator import."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    if not rows:
        path.write_text('',encoding='utf-8')
        return
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),
        **{f'p{p}':float(np.percentile(a,p)) for p in (10,25,75,90)},
        min=float(a.min()),max=float(a.max()))


def ranking(rows,ctol,rtol):
    c=np.asarray([r['C60'] for r in rows])
    reward=np.asarray([r['ppo_return60'] for r in rows])
    physical=np.flatnonzero(c<=c.min()+ctol)
    training=np.flatnonzero(reward>=reward.max()-rtol)
    zero=next(i for i,r in enumerate(rows) if r['action']=='zero')
    best_nonzero=min((i for i in range(len(rows)) if i!=zero),key=lambda i:c[i])
    conflicts=[i for i in range(len(rows)) if c[i]<c[zero]-ctol and reward[i]<reward[zero]-rtol]
    discordant=concordant=ties=0
    for i in range(len(rows)):
        for j in range(i):
            dc,dr=c[i]-c[j],reward[i]-reward[j]
            if abs(dc)<=ctol or abs(dr)<=rtol:
                ties+=1
            elif dc*dr>0:
                discordant+=1
            else:
                concordant+=1
    return dict(best_nonzero_index=int(best_nonzero),physical_indices=physical.tolist(),
        training_indices=training.tolist(),best_sets_overlap=bool(set(physical)&set(training)),
        conflict_indices=conflicts,pair_concordant=concordant,pair_discordant=discordant,pair_ties=ties,
        physical_training_exact_argbest_equal=bool(np.argmin(c)==np.argmax(reward)))


def analyze(source):
    source=Path(source).resolve()
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETE' or manifest['completed_branches']!=520:
        raise ValueError('Only the complete fixed matrix can be analyzed')
    out=source/'analysis'
    out.mkdir(exist_ok=False)
    states=json.loads((source/'states.json').read_text(encoding='utf-8'))
    checks=json.loads((source/'checks.json').read_text(encoding='utf-8'))
    with (source/'per_branch.csv').open(encoding='utf-8-sig',newline='') as f:
        raw=list(csv.DictReader(f))
    text_keys={'action','initial_state_sha256'}
    rows=[{k:v if k in text_keys else float(v) for k,v in r.items()} for r in raw]
    if len(rows)!=520 or len(states)!=40 or len(checks)!=520:
        raise ValueError('Incomplete result records')
    for check in checks:
        for key in ('initial_state_equal','CRN_equal','original_unchanged','environment_rng_equal','global_rng_unchanged'):
            if not check[key]:
                raise ValueError(f'Failed pairing: {check}')
        if check['action']=='zero' and not check['full_zero_terminal_equal']:
            raise ValueError('Zero continuation mismatch')
    ctol=protocol['numerical_tolerances']['C60_absolute']
    rtol=protocol['numerical_tolerances']['return_absolute']
    summaries,conflicts,action_features=[],[],[]
    scene_ids=protocol['scenes']
    auxiliary=('C60_holding','C60_energy','full_energy_j','energy_compute_j','energy_upload_j',
        'energy_u2u_j','energy_propulsion_j','queue_total_mbit','queue_upload_mbit','queue_compute_mbit',
        'queue_transfer_mbit','unfinished','unfinished_high_count','unfinished_low_count',
        'completed_weighted_delay_ms','completed_high_delay_ms','completed_low_delay_ms',
        'mean_queue_upload_mbit','mean_queue_compute_mbit','mean_queue_transfer_mbit')
    for state in states:
        selected=[r for r in rows if r['seed']==state['seed'] and r['start_s']==state['start_s']]
        if len(selected)!=13 or {r['action'] for r in selected}!={a['name'] for a in protocol['actions']}:
            raise ValueError('Wrong action set')
        selected.sort(key=lambda r:next(i for i,a in enumerate(protocol['actions']) if a['name']==r['action']))
        zero=selected[0]
        ranked=ranking(selected,ctol,rtol)
        best=selected[ranked['best_nonzero_index']]
        index=(int(best['dimension'])-1)//2
        roles=np.empty(3,dtype=object)
        roles[np.argsort(-np.asarray(state['current_load']))]=['heavy','medium','light']
        offsets=np.asarray(state['centroid_offset'])
        dc=best['C60']-zero['C60']
        summary=dict(seed=state['seed'],start_s=state['start_s'],zero_C60=zero['C60'],
            best_nonzero_delta=dc,best_relative_percent=100*dc/zero['C60'],
            has_nonzero_improvement=dc < -ctol,zero_is_best=dc>=-ctol,
            zero_strictly_best=dc>ctol,zero_tied_best=abs(dc)<=ctol,
            best_action=best['action'],best_source_uav=index,best_source_role=roles[index],
            physical_best_actions=';'.join(selected[i]['action'] for i in ranked['physical_indices']),
            training_best_actions=';'.join(selected[i]['action'] for i in ranked['training_indices']),
            physical_training_best_agreement=ranked['best_sets_overlap'],
            exact_argbest_agreement=ranked['physical_training_exact_argbest_equal'],
            improvement_rank_conflicts=len(ranked['conflict_indices']),
            pair_concordant=ranked['pair_concordant'],pair_discordant=ranked['pair_discordant'],
            pair_ties=ranked['pair_ties'],best_delta_ppo_return=best['ppo_return60']-zero['ppo_return60'],
            state_compute_queue=float(np.sum(state['compute_queue'])),
            state_transfer_queue=float(np.sum(state['transfer_queue'])),
            state_upload_queue=float(np.sum(state['upload_queue'])),
            state_centroid_offset=float(np.linalg.norm(offsets,axis=1).mean()),
            state_current_load=float(np.sum(state['current_load'])),phase_age=state['phase_age'],
            **{f'best_delta_{key}':best[key]-zero[key] for key in auxiliary})
        summaries.append(summary)
        for i in ranked['conflict_indices']:
            row=selected[i]
            conflicts.append(dict(seed=state['seed'],start_s=state['start_s'],action=row['action'],
                delta_C60=row['C60']-zero['C60'],delta_return=row['ppo_return60']-zero['ppo_return60'],
                delta_holding=row['C60_holding']-zero['C60_holding'],
                delta_energy=row['C60_energy']-zero['C60_energy']))
        for row in selected[1:]:
            i=(int(row['dimension'])-1)//2
            action_features.append(dict(seed=state['seed'],start_s=state['start_s'],action=row['action'],
                source_uav=i,axis=(int(row['dimension'])-1)%2,sign=int(row['sign']),role=roles[i],
                delta_C60=row['delta_C60'],delta_return=row['delta_ppo_return60'],
                improved=row['delta_C60'] < -ctol,source_load=state['current_load'][i],
                compute_queue=state['compute_queue'][i],transfer_queue=sum(state['transfer_queue'][i]),
                upload_queue=state['upload_queue'][i],centroid_offset=float(np.linalg.norm(offsets[i]))))
    grouped=[]
    for field in ('role','source_uav','action','source_load'):
        for value in sorted({r[field] for r in action_features},key=str):
            sample=[r for r in action_features if r[field]==value]
            grouped.append(dict(group=field,value=value,n=len(sample),
                improved_fraction=float(np.mean([r['improved'] for r in sample])),
                mean_delta_C60=float(np.mean([r['delta_C60'] for r in sample])),
                median_delta_C60=float(np.median([r['delta_C60'] for r in sample]))))
    # Descriptive bins fixed at empirical quartiles, not used to select new simulations.
    for field in ('compute_queue','transfer_queue','upload_queue','centroid_offset'):
        boundaries=np.unique(np.quantile([r[field] for r in action_features],[0,.25,.5,.75,1]))
        for i in range(len(boundaries)-1):
            lo,hi=boundaries[i:i+2]
            sample=[r for r in action_features if r[field]>=lo and
                    (r[field]<hi or i==len(boundaries)-2 and r[field]<=hi)]
            if sample:
                grouped.append(dict(group=field,value=f'[{lo:.9g},{hi:.9g}]',n=len(sample),
                    improved_fraction=float(np.mean([r['improved'] for r in sample])),
                    mean_delta_C60=float(np.mean([r['delta_C60'] for r in sample])),
                    median_delta_C60=float(np.median([r['delta_C60'] for r in sample]))))
    scene_summary=[]
    for seed in scene_ids:
        sample=[s for s in summaries if s['seed']==seed]
        scene_summary.append(dict(seed=seed,states=len(sample),
            improved_states=sum(s['has_nonzero_improvement'] for s in sample),
            mean_best_delta_C60=float(np.mean([s['best_nonzero_delta'] for s in sample])),
            mean_best_relative_percent=float(np.mean([s['best_relative_percent'] for s in sample])),
            ranking_agreement=sum(s['physical_training_best_agreement'] for s in sample),
            conflict_states=sum(s['improvement_rank_conflicts']>0 for s in sample)))
    improved=[s for s in summaries if s['has_nonzero_improvement']]
    direction_counts={a['name']:sum(s['best_action']==a['name'] for s in summaries) for a in protocol['actions'][1:]}
    counts=dict(states=40,branches=520,improved_states=len(improved),
        improved_fraction=len(improved)/40,zero_best_states=sum(s['zero_is_best'] for s in summaries),
        zero_strict_best_states=sum(s['zero_strictly_best'] for s in summaries),
        zero_tied_best_states=sum(s['zero_tied_best'] for s in summaries),
        best_agreement_states=sum(s['physical_training_best_agreement'] for s in summaries),
        conflict_states=sum(s['improvement_rank_conflicts']>0 for s in summaries),conflict_actions=len(conflicts),
        pair_concordant=sum(s['pair_concordant'] for s in summaries),
        pair_discordant=sum(s['pair_discordant'] for s in summaries),pair_ties=sum(s['pair_ties'] for s in summaries))
    report=dict(counts=counts,best_nonzero_delta=distribution([s['best_nonzero_delta'] for s in summaries]),
        relative_percent=distribution([s['best_relative_percent'] for s in summaries]),
        zero_C60=distribution([s['zero_C60'] for s in summaries]),
        mean_oracle_gain_with_zero=float(np.mean([min(0,s['best_nonzero_delta']) for s in summaries])),
        best_directions=direction_counts,per_scene=scene_summary,
        best_candidate_component_means={k:float(np.mean([s['best_delta_'+k] for s in summaries])) for k in auxiliary},
        improvement_only_component_means={k:float(np.mean([s['best_delta_'+k] for s in improved])) for k in auxiliary} if improved else {},
        projection_macro_count=sum(r['projection_macro_count'] for r in rows),
        B_projection_count=sum(r['B_projection_count'] for r in rows),
        source_unchanged=manifest['source_unchanged'],sac_frozen_checks=manifest['sac_frozen_checks'],
        scope='40 repeated-development states from one frozen SAC; minima are hindsight per-state selection, not an implementable learned policy. No confidence/generalization/causal-mechanism claims.',
        objective_scope=protocol['tail'],
        input_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                      ('manifest.json','protocol.json','states.json','checks.json','per_branch.csv')})
    write_csv(out/'state_summary.csv',summaries)
    write_csv(out/'action_state_features.csv',action_features)
    write_csv(out/'feature_groups.csv',grouped)
    write_csv(out/'ranking_conflicts.csv',conflicts)
    write_csv(out/'scene_summary.csv',scene_summary)
    write_json(out/'summary.json',report)
    lines=['# B-centered local counterfactual value landscape','',
        '## Observed results',
        f'- Improved states: {len(improved)}/40; zero best (including ties): {counts["zero_best_states"]}/40.',
        f'- Best-nonzero Delta C60: {json.dumps(report["best_nonzero_delta"])}',
        f'- Relative percentage: {json.dumps(report["relative_percent"])}',
        f'- Physical/discounted-prefix best-action agreement: {counts["best_agreement_states"]}/40.',
        f'- Physical improvement but worse discounted prefix: {counts["conflict_actions"]} actions in {counts["conflict_states"]} states.',
        f'- Pair rankings: {counts["pair_concordant"]} concordant / {counts["pair_discordant"]} discordant / {counts["pair_ties"]} tied.',
        '', '## Per-scene results','',
        '| Scene | Improved states | Mean best Delta C60 | Mean relative % | Best agreement | Conflict states |',
        '|---|---:|---:|---:|---:|---:|']
    lines += [f'| {s["seed"]} | {s["improved_states"]}/8 | {s["mean_best_delta_C60"]:.9f} | {s["mean_best_relative_percent"]:.6f} | {s["ranking_agreement"]}/8 | {s["conflict_states"]} |' for s in scene_summary]
    lines += ['', '## Descriptive associations',
        f'Best nonzero directions: {json.dumps(direction_counts)}',
        'Feature bins and role associations: feature_groups.csv; all actions: action_state_features.csv.',
        'Source roles are assigned from current load at each branch time, never permanently by UAV ID.',
        '', '## Objective definitions and checks',
        protocol['C60'],protocol['ppo_return60'],protocol['tail'],
        'The six-macro discounted prefix exactly equals native rollout reward accumulation; no value-network tail is evaluated.',
        'All 520 clones restore the full physical/global RNG snapshot. All 40 zero terminal states equal the uninterrupted B continuation.',
        'Task-birth, physical-step user motion, load and centroid digests match; environment RNG matches each zero branch.',
        'Queue averages use 0.5-second frame-end samples; terminal counts are at branch end. Completion delays include tasks born before branching but completed inside this branch.',
        'Full energy is distinct from the original incremental energy objective. All four full-energy components are retained.',
        '', '## Interpretation limits',report['scope'],
        'A finite directional grid and a 60-second continuation do not certify any global or continuous local optimum.',
        'A best direction chosen after observing the future is not evidence that an observable-state policy can predict that direction.',
        'Ranking differences establish a finite-horizon weighting difference, not a proven PPO-training failure mechanism.',
        'No further evaluation or training is authorized or performed by this analyzer.']
    (out/'result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='results/b_centered_local_counterfactual_sac101')
    analyze(parser.parse_args().source)
