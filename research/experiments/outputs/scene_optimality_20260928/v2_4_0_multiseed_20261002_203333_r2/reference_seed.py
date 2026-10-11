# Derived from D:\HRL_UAV_V2_speedup\tools\run_v2_4_0_reference.py; changes restricted to seed/path and approved D1-D9.
"""Freeze the v2.4.0 paired script baselines (G1 reference) for seed 4101.

No learner/checkpoint is created or loaded. Artifacts are confined to outputs/lite_v2_4_0.
Unlike tools/run_paper_caliber.py there is no earlier-version reference to reproduce: the
check is that scripted baselines are identical across policy streams and that hover
propulsion equals the analytical hover charge.
python tools/run_v2_4_0_reference.py --workers 6
"""
import sys
sys.dont_write_bytecode=True
from common import SEED,WORK,envpath,assert_imports,REF4101,read
assert_imports()
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

import numpy as np

from hrl_v2.baseline_reference import measure_case, freeze
from hrl_v2.checkpoint import atomic_json, identity
from hrl_v2.evaluate import gate_manifest
from hrl_v2.paths import source_root
from hrl_v2.run import read_config
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv

ROOT=source_root().parent
OUTPUT=envpath('OUT',WORK/f'references/seed_{SEED}')
CONFIG=source_root()/'hrl_v2/configs/v2_4_0_lite.json'
COMBINATIONS={'hover_equal':('hover','equal'),'hover_local':('hover','local'),
    'hover_cvx':('hover','cvx'),'hover_greedy':('hover','greedy')}


def work(job):
    scenario,stream,label=job
    config=read_config(CONFIG,OUTPUT)
    scene=json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text())
    result=measure_case(config,scene,identity(OUTPUT,CONFIG),OUTPUT/'baselines',scenario,stream,
                        label,*COMBINATIONS[label],SEED)
    return dict(scenario=scenario['scenario'],stream=stream,baseline=label,
                physical_steps=result['physical_steps'],delay=result['truncated_delay_s'])


def check():
    scene=json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text())
    ph=HotspotIndustrialEnv(scene).physics
    expected_prop=float(3*180*ph._compute_tcom_flying_power(np.array([0.]))[0])
    streams=gate_manifest(20)['policy_streams']
    for label in COMBINATIONS:
        for scenario in gate_manifest(20)['scenarios']:
            sid=scenario['scenario']
            rows=[json.loads((OUTPUT/f'baselines/{label}_{sid:02d}_{s}.json').read_text())['metrics'] for s in streams]
            if any(r['event_sha256']!=rows[0]['event_sha256'] or r['truncated_delay_s']!=rows[0]['truncated_delay_s'] for r in rows):
                raise AssertionError(('scripted baseline differs across policy streams',label,sid))
            np.testing.assert_allclose(rows[0]['energy_j']['propulsion'],expected_prop,rtol=1e-10,atol=1e-7)
    return dict(passed=True,expected_propulsion_j=expected_prop)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=10: parser.error('workers must be 1..10')
    (OUTPUT/'baselines').mkdir(parents=True,exist_ok=True)
    read_config(CONFIG,OUTPUT)
    binding=identity(OUTPUT,CONFIG)
    jobs=[(scene,stream,label) for label in COMBINATIONS for scene in gate_manifest(20)['scenarios']
          for stream in gate_manifest(20)['policy_streams']]
    start=time.perf_counter();completed=[]
    atomic_json(OUTPUT/'baseline_run.json',dict(status='running',identity=binding,windows=len(jobs),
        workers=args.workers,research_training_episodes=0))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,job) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
            atomic_json(OUTPUT/'baseline_progress.json',dict(completed_windows=len(completed),total_windows=len(jobs),
                physical_steps=sum(r['physical_steps'] for r in completed),wall_seconds=time.perf_counter()-start))
    if identity(OUTPUT,CONFIG)!=binding: raise RuntimeError('Source or config changed during baseline measurement')
    summary=freeze(CONFIG,OUTPUT/'baselines',20,12,master=SEED,include_greedy=True)
    checked=check()
    if not (summary['identity']==binding and summary['master']==SEED and summary['manifest']==gate_manifest(20) and summary['required_wins']==12 and checked['passed']):raise AssertionError('Reference hard checks failed')
    same=0
    for scenario,stream,label in jobs:
        name=f'{label}_{scenario["scenario"]:02d}_{stream}.json'
        same+=int(read(OUTPUT/'baselines'/name)['metrics']==read(REF4101/name)['metrics'])
    atomic_json(OUTPUT/'checks.json',dict(status='passed',check=checked,matching_REF4101_full_metrics=same,total=160))
    rows=[]
    for name,total in summary['by_mode'].items():
        per_scene=summary['per_scenario'][name]
        rows.append(dict(baseline=name,scenario_mean_truncated_delay_s=float(np.mean([r['truncated_delay_s'] for r in per_scene.values()])),
            pooled_truncated_delay_s=total['truncated_delay_s'],unfinished_per_window=total['unfinished_parents']/total['windows'],
            mean_cost_sum=total['cost_sum']/total['windows'],cost_per_s=total['cost_per_s'],
            **{k+'_j':v/total['windows'] for k,v in total['energy_j'].items()}))
    atomic_json(OUTPUT/'baseline_means.json',rows)
    atomic_json(OUTPUT/'baseline_run.json',dict(status='passed',identity=binding,windows=len(jobs),
        workers=args.workers,physical_steps=sum(r['physical_steps'] for r in completed),
        wall_seconds=time.perf_counter()-start,research_training_episodes=0,check=checked))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__': main()
