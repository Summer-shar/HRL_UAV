# Derived from NEW/eval_diag.py and diag_shared.py. Changes: arbitrary separate
# upper/lower checkpoints, PILOT paths, seed 4104/4105 only, dev-only labels.
from pilot_common import *
import torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from hrl_v2.checkpoint import identity,load,json_digest
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.upper_baselines import UpperLearner,load_typed,fixed_upper_policy
from hrl_v2.policy import Policy
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics
from hrl_v2.run import read_config
from hrl_v2.evaluate import gate_manifest
from hrl_v2.timing import ClockConfig
CACHE={};CACHE_GROUP=None

def window(t):
 global CACHE_GROUP
 start=time.perf_counter();assert_imports();binding=read(PILOT/'pilot_contract.json')['identity'];equal(identity(WT,CONFIG),binding,'worker_identity')
 check(t['seed'] in (4104,4105) and t['split']=='dev','authorized_seed_dev_only')
 config=read_config(CONFIG,WT);device=configure();scene=read(SCENE);key=(t['seed'],t['group'])
 if CACHE_GROUP!=key:CACHE.clear();gc.collect();torch.cuda.empty_cache();CACHE_GROUP=key
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  for owner in ('upper','lower'):
   pp=t.get(owner)
   if pp and pp not in CACHE:
    kind=t.get('upper_kind') if owner=='upper' else None
    saved=load_typed(pp,binding,'TFOO') if kind=='TFOO' else load(pp,binding)
    l=(UpperLearner if kind=='TFOO' else Learner)(t['seed'],device,config=config);l.load_state_dict(saved['learner']);del saved;gc.collect();CACHE[pp]=l
  upper=CACHE[t['upper']];sc=t['scenario'];streams=Streams(t['seed'],episode=sc['scenario'],split='dev',policy_stream=t['stream']);adapter=PhysicsAdapter(scene,timing=ClockConfig.from_config(config));adapter.reset(**{k:v for k,v in sc.items() if k.endswith('_seed')})
  if t.get('route')=='cvx':policy=fixed_upper_policy(upper.ppo,scene,streams,'cvx',stochastic=False,mode='moving');actors={'upper':upper.ppo.actor}
  else:
   lower=CACHE[t['lower']];policy=Policy(lower.sac.actor,upper.ppo.actor,None,streams,mode='moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment']);actors={'upper':upper.ppo.actor,'lower':lower.sac.actor}
  before={k:parameter_hash(v) for k,v in actors.items()};metrics=collect_metrics(adapter,policy,identity=t['identity'],reward_scale=config['objective']['reward_scale']);after={k:parameter_hash(v) for k,v in actors.items()};equal(before,after,'eval_actor_parameters_unchanged')
  diag={}
  if t.get('route')=='cvx':rr=policy.lower_override.rule.records;diag=dict(solver_calls=len(rr),fallback_calls=sum(x.get('fallback',False) for x in rr))
  put(t['output'],normalize(dict(identity=t['identity'],metrics=metrics,diagnostics=diag,learning_updates=0,replay_writes=0,actor_hash_check=dict(before=before,after=after,passed=True))))
 return dict(scenario=sc['scenario'],stream=t['stream'],path=t['output'],wall_seconds=time.perf_counter()-start)

def group(seed,label,upper,lower=None,route=None,upper_kind=None,root=None):
 start=time.perf_counter();root=Path(root) if root else PILOT/f'seed_{seed}/evaluations/{label}'
 check(not root.exists(),'new_eval_group_'+str(seed)+'_'+label,str(root));root.mkdir(parents=True);m=gate_manifest(20);contract=read(PILOT/'pilot_contract.json');tasks=[]
 for sc in m['scenarios']:
  for st in (1,2):
   ident=dict(protocol='v2_4_0_randpre_pilot_dev',seed=seed,group=label,manifest=json_digest(m),scenario=sc,stream=st,split='dev',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=100,checkpoints={'upper':sha(upper),'lower':sha(lower) if lower else None})
   tasks.append(dict(seed=seed,group=label,upper=str(upper),lower=str(lower) if lower else None,route=route,upper_kind=upper_kind,scenario=sc,stream=st,split='dev',identity=ident,output=str(root/f'windows/{sc["scenario"]:02d}_{st}.json')))
 results=[]
 with ProcessPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(window,t) for t in tasks]
  try:
   for f in as_completed(futures):results.append(f.result())
  except BaseException:
   for f in futures:f.cancel()
   for p in list(pool._processes.values()):
    if p.is_alive():p.terminate()
   raise
 acc=PooledMetrics();per={};files=[];events={}
 for sid in range(1,21):
  one=PooledMetrics();previous=None
  for st in (1,2):
   pp=root/f'windows/{sid:02d}_{st}.json';r=read(pp);metrics=r['metrics']
   if previous is not None:equal(metrics,previous,'deterministic_stream_metrics_'+str(seed)+'_'+label+'_'+str(sid))
   previous=metrics;acc.add(metrics);one.add(metrics);files.append(dict(scenario=sid,stream=st,path=str(pp),sha256=sha(pp)));events[str(sid)]=metrics['event_sha256'];del r
  per[str(sid)]=one.result();del previous,metrics,one
 total=acc.result();summary=dict(seed=seed,group=label,mean_s=avg([r['truncated_delay_s'] for r in per.values()]),per_scenario=per,pooled=total,unfinished_high_per_window=total['priority'][0]['unfinished']/total['episodes'],unfinished_low_per_window=total['priority'][1]['unfinished']/total['episodes'],cost_per_s=total['cost_per_s'],stage_delay_s=total['stage_delay_s'],stream_consistency=dict(passed=True,scenarios=20),external_events=events,window_files=files,wall_seconds=time.perf_counter()-start,upper_path=str(upper),lower_path=str(lower) if lower else None,upper_kind=upper_kind,route=route)
 refpath=PILOT/f'seed_{seed}/rp/RP-E/E5/summary.json'
 if refpath.exists():equal(events,read(refpath)['external_events'],'paired_external_events_'+str(seed)+'_'+label)
 put(root/'summary.json',summary);print('DEV_GROUP_PASS',seed,label,summary['mean_s'],summary['wall_seconds'],flush=True)
 return summary

def repro_eval(seed):
 s300=old_run(seed)/'checkpoints/episode-0300-evaluated.pt';sd=old_sd(seed)/'checkpoints/episode-0200.pt';tf=old_tfoo(seed)/'checkpoints/episode-0200-anchor.pt';out={}
 for label,up,kind in [('E5',sd,None),('TFOO-200',tf,'TFOO')]:
  r=group(seed,label,up,route='cvx',upper_kind=kind,root=PILOT/f'seed_{seed}/rp/RP-E/{label}');ref=hist(seed,label);equal(r['per_scenario'],ref['per_scenario'],'RP-E_'+str(seed)+'_'+label+'_per_scenario');equal(r['mean_s'],ref['mean_s'],'RP-E_'+str(seed)+'_'+label+'_mean');out[label]=dict(status='PASS',mean_s=r['mean_s'],summary=str(PILOT/f'seed_{seed}/rp/RP-E/{label}/summary.json'))
 put(PILOT/f'seed_{seed}/rp/RP-E/checks.json',out);return out
if __name__=='__main__':
 try:repro_eval(int(sys.argv[1]))
 except BaseException as e:failure('RP-E',e);raise