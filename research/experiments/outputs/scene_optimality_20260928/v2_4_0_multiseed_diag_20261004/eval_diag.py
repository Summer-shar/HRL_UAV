# Dev-only window implementation copied/adapted from evaluate_stage3.py,
# evaluate_sacdet.py, and evaluate_test_seed.py; no old output modules imported.
from diag_shared import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from hrl_v2.policy import Policy
from hrl_v2.upper_baselines import fixed_upper_policy
from hrl_v2.paper_evaluation import comparison_policy
from hrl_v2.evaluate import gate_manifest
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics
from hrl_v2.timing import ClockConfig
CACHE={};CACHE_GROUP=None
SUMMARIES={};WINDOW_INDEX={};WINDOW_TIMINGS=[];REUSE_CHECKS=[]

def owners(group,cycle=None):
 if group.startswith('S3-c'):return [f'C{int(group[4:])}']
 if group in ('E2','E5'):return ['SD200']
 if group=='TFOO-200':return ['TFOO']
 return [f'C{cycle}' if group=='X3' and cycle else 'C10' if group=='X2' else 'SD200','TFOO']
def window(task):
 global CACHE_GROUP
 started=time.perf_counter();key=(task['seed'],task['group']);config=read_config(CONFIG,WT);binding=read(NEW/'preflight.json')['identity']
 check(task.get('split','dev')=='dev','never_evaluate_test');equal(identity(WT,CONFIG),binding,'worker_identity')
 device=configure();scene=read(SCENE);scenario=task['scenario'];stream=task['stream'];seed=task['seed'];group=task['group']
 if CACHE_GROUP!=key:CACHE.clear();clean_gpu();CACHE_GROUP=key
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  used={}
  for name in owners(group,task.get('cycle')):
   if name not in CACHE:CACHE[name]=model(paths(seed)[name],seed,binding,config,device,'TFOO' if name=='TFOO' else None)
   used[name]=CACHE[name]
  adapter=PhysicsAdapter(scene,timing=ClockConfig.from_config(config));adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
  streams=Streams(seed,episode=scenario['scenario'],split='dev',policy_stream=stream)
  if group=='TFOO-200':policy=comparison_policy(used['TFOO'],scene,streams,'moving_cvx',stochastic=False)
  elif group=='E5':policy=fixed_upper_policy(used['SD200'].ppo,scene,streams,'cvx',stochastic=False,mode='moving')
  else:
   lower=used[owners(group,task.get('cycle'))[0]];upper=used['TFOO'] if group in ('X1','X2','X3') else lower
   policy=Policy(lower.sac.actor,upper.ppo.actor,None,streams,mode='moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  actors={'upper':policy.upper_actor} if hasattr(policy,'upper_actor') else {}
  # Use loaded modules directly: these are precisely the actors passed to the policy.
  upper=used['TFOO'] if group in ('TFOO-200','X1','X2','X3') else used[owners(group,task.get('cycle'))[0]]
  actors={'upper':upper.ppo.actor}
  if group not in ('E5','TFOO-200'):actors['lower']=used[owners(group,task.get('cycle'))[0]].sac.actor
  before={k:parameter_hash(v) for k,v in actors.items()}
  metrics=collect_metrics(adapter,policy,identity=task['identity'],reward_scale=config['objective']['reward_scale'])
  after={k:parameter_hash(v) for k,v in actors.items()};equal(after,before,'window_actor_hash_unchanged')
  diagnostics={}
  if group in ('E5','TFOO-200'):
   records=policy.lower_override.rule.records;diagnostics=dict(solver_calls=len(records),fallback_calls=sum(r.get('fallback',False) for r in records))
  result=normalize(dict(identity=task['identity'],metrics=metrics,diagnostics=diagnostics,learning_updates=0,replay_writes=0,actor_hash_check=dict(before=before,after=after,passed=True)))
  path=Path(task['output']);check(not path.exists(),'new_window_file',str(path));put(path,result)
 return dict(seed=seed,group=group,scenario=scenario['scenario'],stream=stream,path=str(path),wall_seconds=time.perf_counter()-started,actor_hash_unchanged=True)

def task_for(seed,group,scenario,stream,cycle=None,output=None,oldidentity=None):
 item=oldidentity or dict(protocol='v2_4_0_multiseed_dev_diagnosis',group=group,checkpoints={k:read(NEW/'preflight.json')['checkpoints'][str(seed)][k]['sha256'] for k in owners(group,cycle)},manifest=json_digest(gate_manifest(20)),scenario=scenario,stream=stream,master=seed,split='dev',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=read_config(CONFIG,WT)['objective']['reward_scale'])
 return dict(seed=seed,group=group,cycle=cycle,scenario=scenario,stream=stream,split='dev',identity=item,output=str(output or NEW/f'windows/seed_{seed}/{group}_{scenario["scenario"]:02d}_{stream}.json'))
def pool_group(seed,group,cycle=None):
 start=time.perf_counter();m=gate_manifest(20);tasks=[task_for(seed,group,sc,st,cycle) for sc in m['scenarios'] for st in (1,2)];results=[]
 # Sequential checkpoint groups; six workers, each loads the group's checkpoint once.
 with ProcessPoolExecutor(max_workers=6) as executor:
  futures={executor.submit(window,t):t for t in tasks}
  for future in as_completed(futures):
   try:r=future.result();results.append(r);WINDOW_TIMINGS.append(r)
   except BaseException:
    for pending in futures:pending.cancel()
    for process in list(executor._processes.values()):
     if process.is_alive():process.terminate()
    raise
 WINDOW_INDEX[(seed,group)]={f'{r["scenario"]}_{r["stream"]}':r['path'] for r in results}
 result=aggregate(seed,group)
 put(NEW/'progress.json',dict(status='dev_evaluation',completed_groups=len(SUMMARIES),new_windows=len(WINDOW_TIMINGS),last_group=dict(seed=seed,group=group,mean_s=result['mean_s'],wall_seconds=time.perf_counter()-start),updated_at=now()))
 put(NEW/'window_timings.json',WINDOW_TIMINGS);print('DEV_GROUP_COMPLETE',seed,group,result['mean_s'],len(WINDOW_TIMINGS),flush=True)
 return result

def replay():
 manifest=gate_manifest(20);panel=Path(read(B4101/'launch_contract.json')['panels']['A']['output']);checks=[]
 for group,oldpath in [('S3-c10',S34101/'windows/S3-c10_01_1.json'),('TFOO-200',panel/'deterministic_moving_cvx_01_1.json'),('E2',R24101/'windows/E2_01_1.json'),('E5',R24101/'windows/E5_01_1.json'),('X1',R24101/'windows/E4_01_1.json')]:
  old=read(oldpath);t=task_for(4101,group,manifest['scenarios'][0],1,output=NEW/f'replay/{group}_01_1.json',oldidentity=old['identity']);r=window(t);new=read(r['path']);d=firstdiff(normalize(new['metrics']),normalize(old['metrics']))
  checks.append(dict(group=group,reference=str(oldpath),reference_sha256=sha(oldpath),status='PASS' if d is None else 'FAIL',first_difference=d,actor_hash_unchanged=True,wall_seconds=r['wall_seconds']));put(NEW/'replay/checks.json',checks)
  check(d is None,'replay_full_metrics_equal',checks[-1]);print('REPLAY_PASS',group,flush=True)
  del old,new;CACHE.clear();clean_gpu()
 CHECKS['replay']=dict(status='PASS',windows=5);return checks

def aggregate(seed,group,oldsummary=None,oldgroup=None):
 acc=PooledMetrics();ss={};consistency=[];files=[];diags={}
 for sid in range(1,21):
  one=PooledMetrics();previous=None
  for stream in (1,2):
   path=Path(WINDOW_INDEX[(seed,group)][f'{sid}_{stream}']);r=read(path);metrics=r['metrics']
   if previous is not None:equal(metrics,previous,'deterministic_two_stream_metrics_equal')
   previous=metrics;acc.add(metrics);one.add(metrics)
   for k,v in r.get('diagnostics',{}).items():diags[k]=diags.get(k,0)+v
   files.append(dict(scenario=sid,stream=stream,path=str(path),sha256=sha(path),reused=not path.is_relative_to(NEW)))
   del r
  ss[str(sid)]=one.result();consistency.append(sid);del one,previous,metrics
 total=acc.result();mean=avg([v['truncated_delay_s'] for v in ss.values()]);summary=dict(seed=seed,group=group,mean_s=mean,per_scenario=ss,pooled=total,unfinished_high_per_window=total['priority'][0]['unfinished']/total['episodes'],unfinished_low_per_window=total['priority'][1]['unfinished']/total['episodes'],cost_per_s=total['cost_per_s'],stage_delay_s=total['stage_delay_s'],stream_consistency=dict(passed=True,scenarios=len(consistency)),diagnostics=diags,window_files=files)
 if oldsummary is not None:
  old=read(oldsummary);equal(ss,old['per_scenario'][oldgroup],'4101_reused_per_scenario_equal');oldmean=avg([r['truncated_delay_s'] for r in old['per_scenario'][oldgroup].values()]);delta=mean-oldmean
  check(abs(delta)<=1e-12,'4101_reused_group_mean',delta);REUSE_CHECKS.append(dict(group=group,summary=str(oldsummary),old_group=oldgroup,per_scenario_equal=True,mean_difference_s=delta,status='PASS'))
 SUMMARIES[(seed,group)]=summary;put(NEW/f'summaries/seed_{seed}/{group}.json',summary)
 save_csv(NEW/f'summaries/seed_{seed}/{group}.csv',[dict(seed=seed,group=group,scenario=int(sid),truncated_delay_s=r['truncated_delay_s'],cost_per_s=r['cost_per_s'],unfinished_high_per_window=r['priority'][0]['unfinished']/r['episodes'],unfinished_low_per_window=r['priority'][1]['unfinished']/r['episodes'],**{f'stage_{k}_s':v for k,v in r['stage_delay_s'].items()}) for sid,r in ss.items()])
 return summary

def reused4101():
 panel=Path(read(B4101/'launch_contract.json')['panels']['A']['output'])
 for group in BASEGROUPS:
  if group=='X2':continue
  if group.startswith('S3-c'):root=S34101/'windows';prefix=group;summary=S34101/'evaluation_summary.json';oldgroup=group
  elif group=='TFOO-200':root=panel;prefix='deterministic_moving_cvx';summary=panel/'summary_deterministic.json';oldgroup='moving_cvx'
  else:root=R24101/'windows';prefix='E4' if group=='X1' else group;summary=R24101/'evaluation_summary.json';oldgroup=prefix
  WINDOW_INDEX[(4101,group)]={f'{sid}_{stream}':str(root/f'{prefix}_{sid:02d}_{stream}.json') for sid in range(1,21) for stream in (1,2)}
  aggregate(4101,group,summary,oldgroup)
 put(NEW/'reuse_4101_checks.json',REUSE_CHECKS);CHECKS['reuse_4101']=dict(status='PASS',groups=len(REUSE_CHECKS),windows=40*len(REUSE_CHECKS))

def evaluate_all(logs):
 for seed in ORDER:
  for group in BASEGROUPS:pool_group(seed,group)
 reused4101();pool_group(4101,'X2')
 selection={}
 for seed in SEEDS:
  candidates=[dict(cycle=i,group='E2' if i==0 else f'S3-c{i}',dev_mean_s=SUMMARIES[(seed,'E2' if i==0 else f'S3-c{i}')]['mean_s'],training_block_mean_s=None if i==0 else logs['stage3_blocks'][str(seed)][i-1]['delay_s']) for i in range(11)]
  ranked=sorted(candidates,key=lambda r:(r['dev_mean_s'],r['cycle']));best=ranked[0];selection[str(seed)]=dict(candidates=candidates,ranking=ranked,Cstar=best,posthoc=True,optimistically_selected_on_same_dev=True)
 put(NEW/'summaries/selection.json',selection)
 for seed in SEEDS:
  n=selection[str(seed)]['Cstar']['cycle']
  if n in (0,10):
   alias='X1' if n==0 else 'X2';SUMMARIES[(seed,'X3')]=dict(SUMMARIES[(seed,alias)],group='X3',alias=alias);put(NEW/f'summaries/seed_{seed}/X3.json',SUMMARIES[(seed,'X3')])
  else:pool_group(seed,'X3',n)
 CHECKS['dev_windows']=dict(status='PASS',new_windows=len(WINDOW_TIMINGS),reused_windows=560,stream_comparisons=(len(WINDOW_TIMINGS)+560)//2,actor_hash_unchanged_new_windows=len(WINDOW_TIMINGS))
 put(NEW/'summaries/all_groups.json',{str(seed):{g:SUMMARIES[(seed,g)] for g in BASEGROUPS+['X3']} for seed in SEEDS})
 return selection,SUMMARIES
