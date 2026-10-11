"""Run the five frozen script baselines, then verify the independent paper reference.

No learner/checkpoint is created or loaded. Artifacts are confined to this version.
python tools/run_paper_caliber.py --workers 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np

from hrl_v2.baseline_reference import measure_case, freeze
from hrl_v2.checkpoint import atomic_json, digest, identity
from hrl_v2.evaluate import gate_manifest
from hrl_v2.paths import source_root
from hrl_v2.run import read_config
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv

ROOT=source_root().parent
OUTPUT=ROOT/'outputs/lite_v2_3_0'
CONFIG=source_root()/'hrl_v2/configs/v2_3_0_lite.json'
REFERENCE=ROOT/'outputs/scene_optimality_20260928/u2u_contraction.json'
COMBINATIONS={'hover_equal':('hover','equal'),'hover_local':('hover','local'),
    'hover_cvx':('hover','cvx'),'centroid_equal':('centroid','equal'),'hover_greedy':('hover','greedy')}


def work(job):
    scenario,stream,label=job
    config=read_config(CONFIG,OUTPUT)
    scene=json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text())
    result=measure_case(config,scene,identity(OUTPUT,CONFIG),OUTPUT/'baselines',scenario,stream,
                        label,*COMBINATIONS[label],4101)
    return dict(scenario=scenario['scenario'],stream=stream,baseline=label,
                physical_steps=result['physical_steps'],delay=result['truncated_delay_s'])


def compare():
    reference=json.loads(REFERENCE.read_text())
    scene=json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text())
    ph=HotspotIndustrialEnv(scene).physics
    expected_prop=float(3*180*ph._compute_tcom_flying_power(np.array([0.]))[0])
    rows=[]
    for item in reference['rows']:
        sid=item['scenario']; ref=item['hover']['paper_literal']
        for stream in gate_manifest(20)['policy_streams']:
            record=json.loads((OUTPUT/f'baselines/hover_equal_{sid:02d}_{stream}.json').read_text())
            actual=record['metrics']
            error=abs(actual['holding_task_s']/ref['env_holding_task_s']-1)
            if error>=1e-6: raise AssertionError(('holding mismatch',sid,stream,error))
            for key in ('upload','cpu','u2u'):
                np.testing.assert_allclose(actual['energy_j'][key],ref['energy_j'][key],rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(actual['energy_j']['propulsion'],expected_prop,rtol=1e-10,atol=1e-7)
            if stream==1:
                rows.append(dict(scenario=sid,reference_holding_task_s=ref['env_holding_task_s'],
                    actual_holding_task_s=actual['holding_task_s'],relative_error=error,
                    truncated_delay_s=actual['truncated_delay_s'],**actual['energy_j']))
    rows.sort(key=lambda r:r['scenario'])
    mean=float(np.mean([r['truncated_delay_s'] for r in rows]))
    if round(mean,4)!=.5994: raise AssertionError(('mean delay',mean))
    result=dict(passed=True,reference=str(REFERENCE),reference_sha256=digest(REFERENCE),
        mean_scenario_truncated_delay_s=mean,expected_propulsion_j=expected_prop,rows=rows)
    atomic_json(OUTPUT/'acceptance_3.json',result)
    with (OUTPUT/'acceptance_3.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=10: parser.error('workers must be 1..10')
    (OUTPUT/'baselines').mkdir(parents=True,exist_ok=True)
    config=read_config(CONFIG,OUTPUT)
    binding=identity(OUTPUT,CONFIG)
    jobs=[(scene,stream,label) for label in COMBINATIONS for scene in gate_manifest(20)['scenarios']
          for stream in gate_manifest(20)['policy_streams']]
    start=time.perf_counter();completed=[]
    atomic_json(OUTPUT/'baseline_run.json',dict(status='running',identity=binding,windows=len(jobs),
        workers=args.workers,reference_sha256=digest(REFERENCE),research_training_episodes=0))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,job) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
            atomic_json(OUTPUT/'baseline_progress.json',dict(completed_windows=len(completed),total_windows=len(jobs),
                physical_steps=sum(r['physical_steps'] for r in completed),wall_seconds=time.perf_counter()-start))
    if identity(OUTPUT,CONFIG)!=binding: raise RuntimeError('Source or config changed during baseline measurement')
    summary=freeze(CONFIG,OUTPUT/'baselines',20,12,include_greedy=True)
    comparison=compare()
    rows=[]
    for name,total in summary['by_mode'].items():
        per_scene=summary['per_scenario'][name]
        rows.append(dict(baseline=name,scenario_mean_truncated_delay_s=float(np.mean([r['truncated_delay_s'] for r in per_scene.values()])),
            pooled_truncated_delay_s=total['truncated_delay_s'],
            mean_cost_sum=total['cost_sum']/total['windows'],cost_per_s=total['cost_per_s'],
            **{k+'_j':v/total['windows'] for k,v in total['energy_j'].items()}))
    atomic_json(OUTPUT/'baseline_means.json',rows)
    atomic_json(OUTPUT/'baseline_run.json',dict(status='passed',identity=binding,windows=len(jobs),
        workers=args.workers,physical_steps=sum(r['physical_steps'] for r in completed),
        wall_seconds=time.perf_counter()-start,research_training_episodes=0,
        reference_sha256=digest(REFERENCE),acceptance_3_passed=comparison['passed']))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__': main()
