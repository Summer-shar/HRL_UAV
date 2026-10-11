"""Freeze paired script baselines before a fresh run; no learner is created."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .checkpoint import atomic_json,identity,json_digest
from .evaluate import gate_manifest
from .baselines import Routing
from .policy import Policy
from .randomness import Streams
from .physics_adapter import PhysicsAdapter
from .collector import collect_metrics
from .metrics import PooledMetrics
from .timing import ClockConfig
from .paths import source_root
from .baseline_protocol import is_paper_caliber,reference_combinations,labeled_results


def measure_case(config,scene,binding,output,scenario,stream,label,mode,route,master):
    """One paired causal baseline window; shared by serial and batch runners."""
    sid=str(scenario['scenario'])
    output=Path(output)
    item=dict(identity=binding,master=master,scenario=scenario,stream=stream,combination=label,
              rollout_seconds=config['task']['rollout_seconds'])
    path=output/f'{label}_{int(sid):02d}_{stream}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=item: raise ValueError('Baseline cache identity mismatch')
        metrics=saved['metrics']
    else:
        adapter=PhysicsAdapter(scene,timing=ClockConfig.from_config(config))
        adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
        policy=Policy(None,None,None,Streams(master,episode=int(sid),split='dev',policy_stream=stream),mode=mode,
                      deployment=scene['deployment'])
        if route=='equal':
            policy.lower_override=lambda report,history:(np.full((2,3,3),1/3,dtype=np.float64),np.full(6,6))
        else: policy.lower_override=Routing(route,scene,master)
        tick=time.perf_counter()
        metrics=collect_metrics(adapter,policy,identity=item,reward_scale=config['objective']['reward_scale'])
        diagnostics={}
        if route=='cvx':
            records=policy.lower_override.rule.records
            diagnostics=dict(solver_calls=len(records),fallback_calls=sum(r.get('fallback',False) for r in records))
        atomic_json(path,dict(identity=item,metrics=metrics,wall_seconds=time.perf_counter()-tick,diagnostics=diagnostics))
        print(f'{label} scenario={sid} stream={stream} delay={metrics["truncated_delay_s"]:.6f}',flush=True)
    return metrics


def freeze(config_path, output, scenario_count, required_wins, master=4101,prepare_only=False,include_greedy=False):
    from .run import read_config
    from .runtime import configure
    configure(test_cpu=True)
    config=read_config(config_path,output)
    binding=identity(output,config_path)
    scene=json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text())
    manifest=gate_manifest(scenario_count)
    if prepare_only and scenario_count!=6: raise ValueError('Only existing dev scenarios can be prepared without a gate decision')
    if not prepare_only and required_wins != {6:4,20:12}[scenario_count]: raise ValueError('Unapproved gate threshold')
    contract=dict(identity=binding,manifest=manifest,required_wins=required_wins,master=master)
    if include_greedy:
        contract['additional_baselines']=['hover_greedy']
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    contract_path=output/'contract.json'
    if not prepare_only and contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise ValueError('Frozen baseline contract differs; choose a new directory')
    if not prepare_only: atomic_json(contract_path,contract)
    combinations=reference_combinations(config,include_greedy=include_greedy)
    totals={k:PooledMetrics() for k in combinations}
    scenarios={k:{} for k in combinations}
    started=time.perf_counter()
    for scenario in manifest['scenarios']:
        sid=str(scenario['scenario'])
        for stream in manifest['policy_streams']:
            for label,(mode,route) in combinations.items():
                metrics=measure_case(config,scene,binding,output,scenario,stream,label,mode,route,master)
                totals[label].add(metrics)
                scenarios[label].setdefault(sid,PooledMetrics()).add(metrics)
    summary=dict(**contract,by_mode={k:v.result() for k,v in totals.items()},
                 per_scenario={mode:{k:v.result() for k,v in rows.items()} for mode,rows in scenarios.items()},
                 wall_seconds=time.perf_counter()-started,learning_updates=0)
    if is_paper_caliber(config['objective']):
        summary.update(labeled_results(summary['by_mode']))
    atomic_json(output/('prepared_existing_dev.json' if prepare_only else 'summary.json'),summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenarios',type=int,choices=(6,20),required=True)
    parser.add_argument('--required-wins',type=int,choices=(4,12))
    parser.add_argument('--prepare-only',action='store_true',help='Compute existing six dev workloads without choosing a gate or freezing a final summary')
    parser.add_argument('--include-greedy',action='store_true',help='Also measure hover_greedy; does not alter G1')
    args=parser.parse_args()
    freeze(args.config.resolve(),args.output.resolve(),args.scenarios,args.required_wins,prepare_only=args.prepare_only,
           include_greedy=args.include_greedy)


if __name__=='__main__': main()
