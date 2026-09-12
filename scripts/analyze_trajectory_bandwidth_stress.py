"""Offline paired mechanism analysis of a completed fixed3/6MHz matrix."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        for key,value in list(row.items()):
            if key.endswith('sha256'):
                continue
            try:
                row[key]=float(value)
            except ValueError:
                pass
    return rows


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),
                p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)),
                min=float(a.min()),max=float(a.max()))


def save_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output=None):
    source=Path(source).resolve()
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if m['status']!='COMPLETE' or m['new_branches']!=520 or m['reused_branches']!=520:
        raise ValueError('Comparison is incomplete')
    rows=read(source/'per_branch.csv')
    if len(rows)!=1040:
        raise ValueError('Unexpected result count')
    index={(int(r['seed']),int(r['start_s']),r['action'],int(r['bandwidth_mhz'])):r for r in rows}
    if len(index)!=1040:
        raise ValueError('Duplicate result keys')
    out=Path(output).resolve() if output else source/'analysis'
    out.mkdir(exist_ok=False)
    base_keys=('C60','C60_holding','C60_energy','J60','full_energy_j','power_w',
        'energy_propulsion_j','energy_compute_j','energy_u2u_j','energy_upload_j',
        'actual_rate_mbps_active','actual_rate_mbps_all_links_service','u2u_tx_time_s','u2u_served_mbit',
        'u2u_admitted_mbit','u2u_active_link_seconds','u2u_tx_time_fraction_of_active',
        'allocated_tx_capacity_utilization','mean_queue_transfer_mbit','mean_queue_compute_mbit',
        'mean_queue_upload_mbit','queue_transfer_mbit','queue_compute_mbit','queue_upload_mbit','queue_total_mbit',
        'unfinished','unfinished_high_count','unfinished_low_count','completed_high_count','completed_low_count',
        'completed_weighted_delay_ms','completed_high_delay_ms','completed_low_delay_ms')
    response_keys=('rate_response_abs_mbps_service','tx_response_abs_mbit','tx_time_response_abs_s',
        'active_pattern_response_link_seconds')
    effect_keys=('C60','C60_holding','C60_energy','J60','actual_rate_mbps_active',
        'actual_rate_mbps_all_links_service','u2u_tx_time_s','u2u_served_mbit',
        'mean_queue_transfer_mbit','mean_queue_compute_mbit','mean_queue_upload_mbit',
        'queue_transfer_mbit','queue_compute_mbit','queue_total_mbit','unfinished','unfinished_high_count',
        'unfinished_low_count','completed_weighted_delay_ms','completed_high_delay_ms','completed_low_delay_ms',
        'full_energy_j','energy_propulsion_j','energy_compute_j','energy_u2u_j','energy_upload_j')
    states,paired,baselines=[],[],[]
    for seed in protocol['scenes']:
        for t in protocol['branch_times_s']:
            zero={b:index[seed,t,'zero',b] for b in (3,6)}
            nonzero={b:[index[seed,t,a['name'],b] for a in protocol['actions'][1:]] for b in (3,6)}
            best={b:min(nonzero[b],key=lambda r:r['delta_C60']) for b in (3,6)}
            state=dict(seed=seed,start_s=t,
                best3_action=best[3]['action'],best6_action=best[6]['action'],
                best3_delta_C60=best[3]['delta_C60'],best6_delta_C60=best[6]['delta_C60'],
                best3_relative_percent=100*best[3]['delta_C60']/zero[3]['C60'],
                best6_relative_percent=100*best[6]['delta_C60']/zero[6]['C60'],
                best6_action_at3_delta_C60=index[seed,t,best[6]['action'],3]['delta_C60'],
                best3_action_at6_delta_C60=index[seed,t,best[3]['action'],6]['delta_C60'],
                best_action_equal=best[3]['action']==best[6]['action'],
                zero3_C60=zero[3]['C60'],zero6_C60=zero[6]['C60'],
                **{f'best{b}_delta_{k}':best[b]['delta_'+k] for b in (3,6) for k in effect_keys if k!='C60'})
            states.append(state)
            baselines.append(dict(seed=seed,start_s=t,
                **{f'b{b}_{k}':zero[b][k] for b in (3,6) for k in base_keys},
                **{'b3_minus_b6_'+k:zero[3][k]-zero[6][k] for k in base_keys}))
            for action in protocol['actions'][1:]:
                pair={b:index[seed,t,action['name'],b] for b in (3,6)}
                row=dict(seed=seed,start_s=t,action=action['name'],
                    **{f'b{b}_{k}':pair[b][k] for b in (3,6) for k in response_keys},
                    **{f'b{b}_delta_{k}':pair[b]['delta_'+k] for b in (3,6) for k in effect_keys},
                    **{f'interaction_{k}':pair[3]['delta_'+k]-pair[6]['delta_'+k] for k in effect_keys})
                for b in (3,6):
                    row[f'b{b}_rate_response_relative_percent']=100*pair[b]['rate_response_abs_mbps_service']/max(zero[b]['actual_rate_mbps_all_links_service'],1e-12)
                paired.append(row)
    if len(states)!=40 or len(paired)!=480:
        raise ValueError('Incomplete state/action matching')
    bands={}
    for b in (3,6):
        selected=[r for r in rows if r['bandwidth_mhz']==b and r['action']!='zero']
        zero=[r for r in rows if r['bandwidth_mhz']==b and r['action']=='zero']
        bands[str(b)]=dict(
            B_means={k:float(np.mean([r[k] for r in zero])) for k in base_keys},
            states_with_improvement=sum(s[f'best{b}_delta_C60']< -1e-8 for s in states),
            improving_nonzero_actions=sum(r['delta_C60']< -1e-8 for r in selected),
            best_delta_C60=stats([s[f'best{b}_delta_C60'] for s in states]),
            best_relative_percent=stats([s[f'best{b}_relative_percent'] for s in states]),
            all_action_response={k:stats([r[k] for r in selected]) for k in response_keys},
            all_action_effect={k:dict(signed_mean=float(np.mean([r['delta_'+k] for r in selected])),
                absolute_mean=float(np.mean([abs(r['delta_'+k]) for r in selected])),
                negative_count=sum(r['delta_'+k]< -1e-8 for r in selected)) for k in effect_keys},
            best_candidate_effect_mean={k:float(np.mean([s[f'best{b}_delta_'+k] for s in states])) for k in effect_keys},
            best_candidate_holding_reduced=sum(s[f'best{b}_delta_C60_holding']< -1e-8 for s in states),
            best_candidate_energy_increased=sum(s[f'best{b}_delta_C60_energy']>1e-8 for s in states),
            best_candidate_unfinished_increased=sum(s[f'best{b}_delta_unfinished']>0 for s in states))
    paired_summary={k:dict(b3_greater_absolute=sum(abs(r['b3_delta_'+k])>abs(r['b6_delta_'+k])+1e-8 for r in paired),
        mean_interaction=float(np.mean([r['interaction_'+k] for r in paired])),
        same_effect_sign=int(sum(np.sign(r['b3_delta_'+k])==np.sign(r['b6_delta_'+k]) for r in paired))) for k in effect_keys}
    sensitivity_ratios={k:bands['3']['all_action_response'][k]['mean']/bands['6']['all_action_response'][k]['mean']
                        if bands['6']['all_action_response'][k]['mean'] else None for k in response_keys}
    report=dict(bandwidths=bands,
        sensitivity_ratio_3_to_6=sensitivity_ratios,
        paired_effect_summary=paired_summary,
        actual_rate_response_larger_at3_count=sum(r['b3_rate_response_abs_mbps_service']>r['b6_rate_response_abs_mbps_service']+1e-10 for r in paired),
        best_action_agreement_count=sum(s['best_action_equal'] for s in states),
        best3_more_negative_than_best6_count=sum(s['best3_delta_C60']<s['best6_delta_C60']-1e-8 for s in states),
        best6_selected_action_at3=stats([s['best6_action_at3_delta_C60'] for s in states]),
        best6_selected_action_at3_improvement_count=sum(s['best6_action_at3_delta_C60']< -1e-8 for s in states),
        per_scene=[dict(seed=seed,
            **{f'b{b}_best_delta_mean':float(np.mean([s[f'best{b}_delta_C60'] for s in states if s['seed']==seed])) for b in (3,6)},
            **{f'b{b}_best_relative_mean_percent':float(np.mean([s[f'best{b}_relative_percent'] for s in states if s['seed']==seed])) for b in (3,6)}) for seed in protocol['scenes']],
        scope=protocol['initial_distribution']+'; '+protocol['SAC_scope'],
        rate_scope=protocol['rate_resolution']+'; includes active-set and policy-response changes, not an isolated geometric derivative',
        temporal_resolution='Rates resolve0.5s averages, so cancellations within a frame are not recoverable at6MHz. Sent times are sums across concurrent directed links, not elapsed wall time.',
        input_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('manifest.json','protocol.json','per_branch.csv','checks.json')})
    save_csv(out/'state_comparison.csv',states)
    save_csv(out/'same_action_interactions.csv',paired)
    save_csv(out/'B_bandwidth_comparison.csv',baselines)
    save_csv(out/'scene_summary.csv',report['per_scene'])
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    compact=dict(sensitivity_ratio_3_to_6=sensitivity_ratios,
        actual_rate_response_larger_at3_count=report['actual_rate_response_larger_at3_count'],
        best_action_agreement_count=report['best_action_agreement_count'],
        best3_more_negative_than_best6_count=report['best3_more_negative_than_best6_count'],
        best6_selected_action_at3=report['best6_selected_action_at3'],
        per_scene=report['per_scene'])
    for b in ('3','6'):
        compact[b]={k:v for k,v in bands[b].items() if k not in ('all_action_effect','all_action_response')}
        compact[b]['all_action_abs_holding']=bands[b]['all_action_effect']['C60_holding']['absolute_mean']
        compact[b]['all_action_abs_C60']=bands[b]['all_action_effect']['C60']['absolute_mean']
        compact[b]['mean_abs_rate_response']=bands[b]['all_action_response']['rate_response_abs_mbps_service']['mean']
    print(json.dumps(compact,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='results/trajectory_bandwidth_stress_sac101')
    parser.add_argument('--output')
    args=parser.parse_args()
    analyze(args.source,args.output)
