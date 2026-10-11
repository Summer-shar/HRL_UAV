import json,time,traceback,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import hrl_v2
from hrl_v2.runtime import configure,Learner,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.checkpoint import identity,digest,json_digest,load
from hrl_v2.evaluate import gate_manifest
from hrl_v2.policy import Policy
from hrl_v2.baselines import centroid
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.collector import collect_metrics
from hrl_v2.timing import ClockConfig
from hrl_v2.metrics import PooledMetrics
WT=Path('D:/HRL_UAV_V2_speedup')
RUN=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
OUT=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_diag')
CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def setup():
 assert Path(hrl_v2.__file__).resolve().is_relative_to(WT/'src')
 device=configure(); assert str(device)=='cuda:0'
 config=read_config(CONFIG,WT); binding=identity(WT,CONFIG)
 assert binding==read(RUN/'contract.json')['identity']
 return device,config,binding
CACHE={}
def learner_at(n,snapshot=True):
 device,config,binding=setup()
 path=RUN/(f'snapshots/episode-{n:04d}.pt' if snapshot else (f'checkpoints/episode-{n:04d}-'+('initial.pt' if n==0 else 'pending.pt')))
 if snapshot:
  assert read(path.with_suffix('.json'))['sha256']==digest(path)
  payload=torch.load(path,map_location='cpu',weights_only=False)
  assert payload['kind']=='network_snapshot_not_resumable' and payload['episode']==n and payload['identity']==binding
 else: payload=load(path,binding)
 learner=Learner(4101,device,config=config); learner.load_state_dict(payload['learner'])
 return learner
class RecordingPolicy(Policy):
 def __init__(self,*a,**k):
  super().__init__(*a,**k); assert self.observer is not None; self.records=[]
 @torch.no_grad()
 def upper(self,report,history):
  obs=self.observer.update(report,history)
  c=centroid(report,history,max_target_step_m=self.deployment['max_target_step_m'])
  action,aux=super().upper(report,history)
  assert [u.key for u in report.uavs]==[0,1,2]
  terms=sorted(report.terminals,key=lambda t:t.key)
  assert all(t.source==t.key//13 for t in terms)
  nan6=np.full(6,np.nan)
  self.records.append(dict(time=float(report.time),uav=np.asarray([u.position for u in report.uavs],float),term=np.asarray([t.position for t in terms],float),centroid=np.asarray(c,float),action=np.asarray(action,float).copy(),mean=np.asarray(aux.get('mean',nan6),float).ravel(),logstd=np.asarray(aux.get('logstd',nan6),float).ravel(),obs=np.asarray(obs.upper).copy()))
  return action,aux
class RecordingAdapter(PhysicsAdapter):
 def __init__(self,*a,**k): super().__init__(*a,**k); self.moves=[]
 def start_macro(self,request):
  motion=super().start_macro(request); self.moves.append(np.asarray(motion['displacement_m'],float).copy()); return motion

def worker(job):
 n,mode,scenario=job
 device,config,binding=setup()
 if n not in CACHE: CACHE[n]=learner_at(n)
 learner=CACHE[n]; sid=scenario['scenario']; scene=read(WT/'src/hrl_v2/configs/physical_scene.json'); timing=ClockConfig.from_config(config)
 item=dict(checkpoint=digest(RUN/f'checkpoints/episode-{n:04d}-pending.pt'),manifest=json_digest(gate_manifest(20)),policy='deterministic',rollout_seconds=timing.rollout_seconds,reward_scale=config['objective']['reward_scale'],scenario=sid,stream=0,mode=mode,master=4101)
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  streams=Streams(4101,episode=sid,split='dev',policy_stream=0)
  adapter=RecordingAdapter(scene,timing=timing); adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
  policy=RecordingPolicy(learner.sac.actor,learner.ppo.actor,None,streams,mode=mode,stochastic=False,deployment=scene['deployment'])
  metrics=collect_metrics(adapter,policy,identity=item,reward_scale=config['objective']['reward_scale'])
  metrics.update(split='dev',policy='deterministic',mode=mode,training_seed=4101,learning_updates=0,replay_writes=0)
 assert len(policy.records)==len(adapter.moves)==181,(n,mode,sid,len(policy.records),len(adapter.moves))
 metrics=json.loads(json.dumps(metrics,allow_nan=False))
 check={}
 if mode in ('hover','moving'):
  old=read(RUN/f'evaluations/episode-{n:04d}/deterministic_{mode}_{sid:02d}_0.json')
  check['identity_equal']=item==old['identity']; check['metrics_equal']=metrics==old['metrics']
  if not all(check.values()):
   diff={k:{'new':str(metrics.get(k))[:1500],'old':str(old['metrics'].get(k))[:1500]} for k in sorted(set(metrics)|set(old['metrics'])) if metrics.get(k)!=old['metrics'].get(k)}
   write(OUT/f'FAIL_N{n}_{mode}_{sid:02d}.json',dict(check=check,differences=dict(list(diff.items())[:20])))
   raise AssertionError(f'Exact replay failed: N{n} {mode} {sid}; see FAIL JSON')
 if n==400 and mode=='hover':
  old=read(RUN/f'evaluations/episode-0300/deterministic_hover_{sid:02d}_0.json')['metrics']
  check['matches_300']={k:metrics[k]==old[k] for k in ('event_sha256','truncated_delay_s')}
 arrays={k:np.asarray([r[k] for r in policy.records]) for k in policy.records[0] if k!='obs' or (n==400 and mode=='hover')}; arrays['moves']=np.asarray(adapter.moves)
 np.savez_compressed(OUT/f'windows/N{n}_{mode}_{sid:02d}.npz',**arrays)
 result=dict(item=item,check=check,**{k:metrics[k] for k in ('truncated_delay_s','stage_delay_s','cost_per_s','unfinished_parents','event_sha256','terminal_mobility')})
 if mode=='centroid': result['metrics']=metrics
 write(OUT/f'windows/N{n}_{mode}_{sid:02d}.json',result)
 return dict(n=n,mode=mode,sid=sid,check=check,per_scenario=PooledMetrics().add(metrics).result())

if __name__=='__main__':
 start=time.perf_counter()
 try:
  setup()
  manifest=json_digest(gate_manifest(20))
  assert manifest==read(RUN/'evaluations/episode-0400/summary_deterministic.json')['identity']['manifest']==json_digest(read(RUN/'development_manifest.json'))
  for n in (400,600):
   assert digest(RUN/f'checkpoints/episode-{n:04d}-pending.pt')==read(RUN/f'evaluations/episode-{n:04d}/summary_deterministic.json')['identity']['checkpoint']
   a=learner_at(n,False); b=learner_at(n,True)
   for kind in ('sac','ppo'): assert parameter_hash(getattr(a,kind).actor)==parameter_hash(getattr(b,kind).actor)
   del a,b
   print('snapshot_pending_parameters PASS',n,flush=True)
  (OUT/'windows').mkdir()
  jobs=[(n,mode,s) for n in (400,600) for mode in ('hover','moving','centroid') for s in gate_manifest(20)['scenarios']]
  results=[]
  with ProcessPoolExecutor(max_workers=6) as pool:
   futures=[pool.submit(worker,j) for j in jobs]
   try:
    for f in as_completed(futures):
     r=f.result(); results.append(r); print('DONE',len(results),r['n'],r['mode'],r['sid'],flush=True)
   except BaseException:
    for f in futures: f.cancel()
    raise
  write(OUT/'d2_raw_results.json',results)
  print('D2_REPLAYS_PASS',time.perf_counter()-start,flush=True)
 except BaseException:
  traceback.print_exc()
  write(OUT/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start))
  sys.exit(1)

