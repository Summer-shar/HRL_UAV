"""Frozen paired-scenario gates; no training or threshold adaptation."""
import numpy as np


def paired(candidate, reference):
    if set(candidate) != set(reference) or not candidate:
        raise ValueError('Paired gates require identical nonempty scenario IDs')
    keys=sorted(candidate,key=int)
    a=np.asarray([candidate[k]['truncated_delay_s'] for k in keys],dtype=float)
    b=np.asarray([reference[k]['truncated_delay_s'] for k in keys],dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b<=0).any():
        raise ValueError('Gate delay metrics must be finite and reference positive')
    return dict(scenarios=len(keys), wins=int((a<b).sum()),
        candidate_mean_s=float(a.mean()),reference_mean_s=float(b.mean()),
        relative_improvement=float(1-a.mean()/b.mean()),
        candidate_unfinished=sum(candidate[k]['unfinished_parents'] for k in keys),
        reference_unfinished=sum(reference[k]['unfinished_parents'] for k in keys),
        pairs=[dict(scenario=k,candidate_s=float(x),reference_s=float(y),win=bool(x<y)) for k,x,y in zip(keys,a,b)])


def g1(hover, baselines, required_wins):
    equal=paired(hover,baselines['hover_equal'])
    local=paired(hover,baselines['hover_local'])
    passed=(equal['candidate_mean_s']<equal['reference_mean_s'] and equal['wins']>=required_wins
            and equal['candidate_unfinished']<=equal['reference_unfinished']
            and local['candidate_mean_s']<local['reference_mean_s'])
    return dict(gate='G1',passed=bool(passed),required_wins=required_wins,equal=equal,local=local,
                action='continue' if passed else 'pause_before_PPO')


def g2(moving, hover):
    result=paired(moving,hover)
    weak=result['relative_improvement']<.01 or result['wins']/result['scenarios']<.6
    return dict(gate='G2',weak_upper_signal=bool(weak),paired=result,action='continue_to_600')
