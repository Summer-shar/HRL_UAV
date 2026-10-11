# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_test\evaluate_test.py; changes restricted to seed/path and approved D1-D9.
"""Derived from S3 evaluate_stage3.py and R2 evaluate_sacdet.py; no training."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import gc,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy.stats as stats
import torch
from test_seed import WT,MAIN,OUT,P,B,R2,S3,CONFIG,SCENE,PATHS,GROUPS,CONSTRUCTIONS,OWNERS,CHECKS,TIMING,read,require,firstdiff,now
from common import SEED,assert_imports
assert_imports()
from hrl_v2.checkpoint import identity,load,atomic_json,json_digest,digest
from hrl_v2.runtime import configure,Learner
from hrl_v2.run import read_config
from hrl_v2.upper_baselines import UpperLearner,load_typed,fixed_upper_policy
from hrl_v2.paper_evaluation import comparison_policy
from hrl_v2.policy import Policy
from hrl_v2.evaluate import gate_manifest
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics
from hrl_v2.gates import paired,unfinished_per_episode
from hrl_v2.timing import ClockConfig
CACHE={};CACHE_GROUP=None

def cached(name,binding,config,device):
 if name not in CACHE:
  path=PATHS[name];kind=name if name in ('NC','TOO','TFOO') else 'Proposed' if name=='P600' else None
  saved=load_typed(path,binding,kind) if kind else load(path,binding)
  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(SEED,device,config)
  learner.load_state_dict(saved['learner']);del saved;gc.collect();CACHE[name]=learner
 return CACHE[name]
def window(task):
 global CACHE_GROUP
 started=time.perf_counter();started_at=now();config=read_config(CONFIG,WT);binding=read(OUT/'preflight.json')['identity']
 require(identity(WT,CONFIG)==binding,'worker_identity');device=configure();scene=read(SCENE);clocks=ClockConfig.from_config(config)
 group=task['group'];scenario=task['scenario'];stream=task['stream'];split=task['split']
 if CACHE_GROUP!=group:CACHE.clear();gc.collect();torch.cuda.empty_cache();CACHE_GROUP=group
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  learners={name:cached(name,binding,config,device) for name in OWNERS[group]}
  adapter=PhysicsAdapter(scene,timing=clocks);adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
  streams=Streams(SEED,episode=scenario['scenario'],split=split,policy_stream=stream)
  if group=='Proposed':
   l=learners['C10'];policy=Policy(l.sac.actor,l.ppo.actor,None,streams,mode='moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  elif group in ('NC','CA','TOO','TFOO'):
   owner=OWNERS[group][0];internal=dict(NC='moving_local',CA='hover_equal',TOO='moving_equal',TFOO='moving_cvx')[group]
   policy=comparison_policy(learners[owner],scene,streams,internal,stochastic=False)
  elif group in ('Abl_TwoStage','Abl_HoverSAC'):
   policy=Policy(learners['S300'].sac.actor,learners['SD200'].ppo.actor,None,streams,mode='hover' if group=='Abl_HoverSAC' else 'moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  elif group=='Abl_UpperCVX':policy=fixed_upper_policy(learners['SD200'].ppo,scene,streams,'cvx',stochastic=False,mode='moving')
  else:raise ValueError('Unknown group '+group)
  metrics=collect_metrics(adapter,policy,identity=task['identity'],reward_scale=config['objective']['reward_scale'])
  diagnostics={}
  if group in ('TFOO','Abl_UpperCVX'):
   records=policy.lower_override.rule.records;diagnostics=dict(solver_calls=len(records),fallback_calls=sum(r.get('fallback',False) for r in records))
  result=json.loads(json.dumps(dict(identity=task['identity'],metrics=metrics,diagnostics=diagnostics,learning_updates=0,replay_writes=0),allow_nan=False))
  if task.get('replay'):return dict(result=result,started_at=started_at,wall_seconds=time.perf_counter()-started)
  path=Path(task['output']);require(not path.exists(),'new_window_path',str(path));atomic_json(path,result)
  return dict(group=group,scenario=scenario['scenario'],stream=stream,path=str(path),started_at=started_at,wall_seconds=time.perf_counter()-started)
def pool_tasks(tasks,on_done):
 with ProcessPoolExecutor(max_workers=6) as executor:
  futures={executor.submit(window,t):t for t in tasks}
  for future in as_completed(futures):
   try:r=future.result();on_done(futures[future],r)
   except BaseException:
    for pending in futures:pending.cancel()
    for process in list(executor._processes.values()):
     if process.is_alive():process.terminate()
    raise

def test_panel(config,binding,m,freeze,metadata):
 require(digest(freeze['path'])==freeze['sha256'],'protocol_hash_before_test')
 cpids={name:metadata[name]['sha256'] for name in PATHS};contract=dict(protocol_sha256=freeze['sha256'],manifest_sha256=digest(P/'test_manifest.json'),manifest_digest=json_digest(m),constructions=CONSTRUCTIONS,checkpoint_sha256=cpids,windows=640,groups=GROUPS,created_at=now())
 require(not (OUT/'test_contract.json').exists(),'new_test_contract');atomic_json(OUT/'test_contract.json',contract)
 tasks=[]
 for group in GROUPS:
  for scenario in m['scenarios']:
   for stream in m['policy_streams']:
    item=dict(protocol='v2_4_0_three_stage_test_1',protocol_sha256=freeze['sha256'],group=group,checkpoints={name:metadata[name]['sha256'] for name in OWNERS[group]},manifest=json_digest(m),scenario=scenario,stream=stream,master=SEED,split='test',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=config['objective']['reward_scale'])
    tasks.append(dict(group=group,scenario=scenario,stream=stream,split='test',identity=item,output=str(OUT/f'windows/{group}_{scenario["scenario"]:02d}_{stream}.json')))
 require(len(tasks)==640,'test_window_budget');start=time.perf_counter();completed=[];first=None
 def done(task,r):
  nonlocal first
  completed.append(r)
  if first is None or r['started_at']<first:first=r['started_at']
  atomic_json(OUT/'evaluation_progress.json',dict(completed_windows=len(completed),total_windows=640,last_completed=r,first_test_window_started_at=first))
  if len(completed)%80==0:print('TEST_EVALUATION',len(completed),'/640',flush=True)
 pool_tasks(tasks,done);TIMING['test_evaluation_seconds']=time.perf_counter()-start
 TIMING['test_per_group_mean_window_wall_seconds']={g:float(np.mean([r['wall_seconds'] for r in completed if r['group']==g])) for g in GROUPS}
 TIMING['first_test_window_started_at']=first;atomic_json(OUT/'window_timings.json',completed)

def summarize(m,freeze):
 totals={};per_scenario={};distances={};diagnostics={};consistency=[]
 for group in GROUPS:
  acc=PooledMetrics();ss={};distance=[];diags=dict(solver_calls=0,fallback_calls=0);same=0
  for scenario in m['scenarios']:
   sid=scenario['scenario'];one=PooledMetrics();reference=None;row_checks=[]
   for stream in m['policy_streams']:
    row=read(OUT/f'windows/{group}_{sid:02d}_{stream}.json');metrics=row['metrics'];acc.add(metrics);one.add(metrics);distance.append(metrics['terminal_mobility']['final_mean_terminal_to_serving_uav_m'])
    for k,v in row['diagnostics'].items():diags[k]+=v
    if stream==1:reference=metrics
    else:
     d=firstdiff(metrics,reference);same+=int(d is None);row_checks.append(dict(stream=stream,equal=d is None,first_difference=d))
    del row,metrics
   ss[str(sid)]=one.result();consistency.append(dict(group=group,scenario=sid,comparisons=row_checks));del one,reference
  totals[group]=acc.result();per_scenario[group]=ss;distances[group]=float(np.mean(distance));diagnostics[group]=diags;del acc
 summary=dict(protocol_sha256=freeze['sha256'],manifest=m,master=SEED,groups=GROUPS,windows=640,by_mode=totals,per_scenario=per_scenario,final_distance_m=distances,diagnostics=diagnostics,stream_consistency=dict(per_group={g:dict(equal=sum(int(c['equal']) for r in consistency if r['group']==g for c in r['comparisons']),total=60) for g in GROUPS},comparisons=consistency),learning_updates=0,replay_writes=0)
 atomic_json(OUT/'test_summary.json',summary);return summary

def compare(summary):
 comparisons=[]
 def add(cname,rname,category):
  c=summary['per_scenario'][cname];r=summary['per_scenario'][rname];p=paired(c,r)
  keys=sorted(c,key=int);a=np.array([c[k]['truncated_delay_s'] for k in keys]);b=np.array([r[k]['truncated_delay_s'] for k in keys]);d=a-b;nonzero=d!=0
  t=stats.ttest_rel(a,b,alternative='two-sided');w=stats.wilcoxon(d,alternative='two-sided',zero_method='wilcox');ranks=stats.rankdata(np.abs(d[nonzero]));wplus=float(ranks[d[nonzero]>0].sum());sign=stats.binomtest(p['wins'],int(nonzero.sum()),0.5,alternative='two-sided')
  row=dict(candidate=cname,reference=rname,category=category,primary_endpoint=cname=='Proposed' and rname=='TFOO',scenarios=p['scenarios'],candidate_mean_s=p['candidate_mean_s'],reference_mean_s=p['reference_mean_s'],relative_difference_percent=100*(p['candidate_mean_s']/p['reference_mean_s']-1),wins=p['wins'],pairs=p['pairs'],candidate_unfinished_per_episode=unfinished_per_episode(c),reference_unfinished_per_episode=unfinished_per_episode(r),statistics=dict(mean_d=float(d.mean()),sd_d=float(d.std(ddof=1)),nonzero_differences=int(nonzero.sum()),ttest_rel=dict(statistic=float(t.statistic),pvalue=float(t.pvalue),alternative='two-sided'),wilcoxon=dict(statistic=float(w.statistic),W_plus=wplus,pvalue=float(w.pvalue),alternative='two-sided',zero_method='wilcox'),binomtest=dict(wins=p['wins'],n=int(nonzero.sum()),pvalue=float(sign.pvalue),alternative='two-sided')))
  if category=='main':
   ca=np.array([c[k]['cost_per_s'] for k in keys]);rb=np.array([r[k]['cost_per_s'] for k in keys]);row['cost_per_s']=dict(candidate_mean=float(ca.mean()),reference_mean=float(rb.mean()),relative_difference_percent=float(100*(ca.mean()/rb.mean()-1)),wins=int((ca<rb).sum()))
  comparisons.append(row)
 for ref in ('TFOO','NC','CA','TOO'):add('Proposed',ref,'main')
 for ref in ('Abl_TwoStage','Abl_HoverSAC','Abl_UpperCVX'):add('Proposed',ref,'ablation')
 add('Abl_TwoStage','TFOO','supplementary');atomic_json(OUT/'test_comparisons.json',comparisons);return comparisons

def report(summary,comparisons,freeze,git):
 atomic_json(OUT/'timing.json',TIMING)
 lines=[f'# v2.4.0 seed {SEED} test 面板','',f"协议SHA256：{freeze['sha256']}",'','检查：'+json.dumps(CHECKS,ensure_ascii=False),'','|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|','|---|---|---|---|---|---|---|']
 for r in comparisons:lines.append('| '+' | '.join(map(str,(r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue'])))+' |')
 lines+=['','## 实测耗时','',json.dumps(TIMING,ensure_ascii=False,indent=2),'','## 最终Git状态','',json.dumps(git,ensure_ascii=False,indent=2),'','脚本错误与修正：无。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
