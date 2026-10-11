"""Derived from S3 evaluate_stage3.py and R2 evaluate_sacdet.py; no training."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import gc,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy.stats as stats
import torch
from test_4101 import WT,MAIN,OUT,P,B,A,R2,S3,CONFIG,SCENE,PATHS,GROUPS,CONSTRUCTIONS,OWNERS,CHECKS,TIMING,read,require,firstdiff,now
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
  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(4101,device,config)
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
  streams=Streams(4101,episode=scenario['scenario'],split=split,policy_stream=stream)
  if group=='Proposed':
   l=learners['C10'];policy=Policy(l.sac.actor,l.ppo.actor,None,streams,mode='moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  elif group in ('NC','CA','TOO','TFOO','Abl_Alternating'):
   owner=OWNERS[group][0];internal=dict(NC='moving_local',CA='hover_equal',TOO='moving_equal',TFOO='moving_cvx',Abl_Alternating='moving')[group]
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

def replay_checks(config,binding,freeze,metadata):
 start=time.perf_counter();scenario=gate_manifest(20)['scenarios'][0]
 references=dict(Proposed=S3/'windows/S3-c10_01_1.json',NC=A/'deterministic_moving_local_01_1.json',CA=A/'deterministic_hover_equal_01_1.json',TOO=A/'deterministic_moving_equal_01_1.json',TFOO=A/'deterministic_moving_cvx_01_1.json',Abl_TwoStage=R2/'windows/E2_01_1.json',Abl_Alternating=A/'deterministic_moving_01_1.json',Abl_HoverSAC=R2/'windows/E3_01_1.json',Abl_UpperCVX=R2/'windows/E5_01_1.json')
 tasks=[]
 for i,group in enumerate(GROUPS,1):
  ref=read(references[group]);tasks.append(dict(check=f'R{i}',group=group,scenario=scenario,stream=1,split='dev',identity=ref['identity'],reference=str(references[group]),replay=True))
 checks=[]
 def done(task,r):
  old=read(task['reference']);candidate=r['result'];d=firstdiff(candidate['metrics'],old['metrics']);dd=firstdiff(candidate['diagnostics'],old['diagnostics']) if task['group'] in ('TFOO','Abl_UpperCVX') else None
  row=dict(check=task['check'],group=task['group'],reference=task['reference'],metrics_equal=d is None,metrics_first_difference=d,diagnostics_equal=dd is None,diagnostics_checked=task['group'] in ('TFOO','Abl_UpperCVX'),diagnostics_first_difference=dd,wall_seconds=r['wall_seconds'])
  checks.append(row);atomic_json(OUT/'replay_checks.json',dict(status='checking',rows=sorted(checks,key=lambda x:int(x['check'][1:]))))
  require(d is None and dd is None,'dev_replay',row);CHECKS[task['check']]='PASS';print('REPLAY_PASS',task['check'],task['group'],flush=True)
 pool_tasks(tasks,done);require(len(checks)==9,'replay_check_count');TIMING['replay_seconds']=time.perf_counter()-start
 atomic_json(OUT/'replay_checks.json',dict(status='PASS',rows=sorted(checks,key=lambda x:int(x['check'][1:])),wall_seconds=TIMING['replay_seconds']))
 require(not (OUT/'windows').exists(),'no_test_window_before_replay_pass')

def test_panel(config,binding,m,freeze,metadata):
 require(digest(freeze['path'])==freeze['sha256'],'protocol_hash_before_test')
 cpids={name:metadata[name]['sha256'] for name in PATHS};contract=dict(protocol_sha256=freeze['sha256'],manifest_sha256=digest(P/'test_manifest.json'),manifest_digest=json_digest(m),constructions=CONSTRUCTIONS,checkpoint_sha256=cpids,windows=720,groups=GROUPS,created_at=now())
 require(not (OUT/'test_contract.json').exists(),'new_test_contract');atomic_json(OUT/'test_contract.json',contract)
 tasks=[]
 for group in GROUPS:
  for scenario in m['scenarios']:
   for stream in m['policy_streams']:
    item=dict(protocol='v2_4_0_three_stage_test_1',protocol_sha256=freeze['sha256'],group=group,checkpoints={name:metadata[name]['sha256'] for name in OWNERS[group]},manifest=json_digest(m),scenario=scenario,stream=stream,master=4101,split='test',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=config['objective']['reward_scale'])
    tasks.append(dict(group=group,scenario=scenario,stream=stream,split='test',identity=item,output=str(OUT/f'windows/{group}_{scenario["scenario"]:02d}_{stream}.json')))
 require(len(tasks)==720,'test_window_budget');start=time.perf_counter();completed=[];first=None
 def done(task,r):
  nonlocal first
  completed.append(r)
  if first is None or r['started_at']<first:first=r['started_at']
  atomic_json(OUT/'evaluation_progress.json',dict(completed_windows=len(completed),total_windows=720,last_completed=r,first_test_window_started_at=first))
  if len(completed)%80==0:print('TEST_EVALUATION',len(completed),'/720',flush=True)
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
 summary=dict(protocol_sha256=freeze['sha256'],manifest=m,master=4101,groups=GROUPS,windows=720,by_mode=totals,per_scenario=per_scenario,final_distance_m=distances,diagnostics=diagnostics,stream_consistency=dict(per_group={g:dict(equal=sum(int(c['equal']) for r in consistency if r['group']==g for c in r['comparisons']),total=60) for g in GROUPS},comparisons=consistency),learning_updates=0,replay_writes=0)
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
 for ref in ('Abl_TwoStage','Abl_Alternating','Abl_HoverSAC','Abl_UpperCVX'):add('Proposed',ref,'ablation')
 add('Abl_TwoStage','TFOO','supplementary');atomic_json(OUT/'test_comparisons.json',comparisons);return comparisons

def report(summary,comparisons,freeze,git):
 lines=['# v2.4.0 seed 4101 三阶段协议冻结与 test 面板','', '状态：全部步骤完成；仅评估，learning_updates=0，replay_writes=0。','', '## 检查结果','']
 fmt=lambda x:'缺失' if x is None else f'{x:.9g}' if isinstance(x,float) else str(x)
 def table(headers,rs):
  lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']);lines.extend('| '+' | '.join(fmt(v) for v in r)+' |' for r in rs);lines.append('')
 replay=read(OUT/'replay_checks.json');pre=read(OUT/'preflight.json');main=comparisons[0];st=main['statistics']
 checkrows=[[f'P{i}',CHECKS[f'P{i}']] for i in range(1,7)]+[['协议冻结',CHECKS['protocol_freeze']]]+[[r['check'],f'{CHECKS[r["check"]]}；{r["group"]}；完整metrics相等'+('；diagnostics相等' if r['diagnostics_checked'] else '')] for r in replay['rows']]
 equal=sum(r['equal'] for r in summary['stream_consistency']['per_group'].values());total=sum(r['total'] for r in summary['stream_consistency']['per_group'].values())
 checkrows += [['跨stream一致性（只报告）',f'{equal}/{total} 完整metrics逐位相同'],['协议哈希复核',CHECKS['protocol_hash_recheck']]];table(['检查','结果'],checkrows)
 lines.extend(['P1：两棵工作树分支、HEAD、status。P2：WT/src导入、cuda:0、运行时、scipy。P3：五份identity相等且validation passed。P4：排除自身后的GPU/进程检查。P5：test manifest完全一致且与dev种子三元组不相交。P6：checkpoint身份、SHA256、类型、权重哈希及基线预算。','', f'解释器：{pre["runtime"]["executable"]}；torch={pre["runtime"]["torch"]}；CUDA={pre["runtime"]["cuda"]}；scipy={pre["scipy_version"]}；torch线程数={pre["torch_threads"]}。','', '## 协议冻结','',f'路径：{freeze["path"]}','',f'SHA256：`{freeze["sha256"]}`','',f'冻结时间：{freeze["frozen_at"]}','',f'冻结时 OUT/windows 存在：{freeze["windows_exists_at_freeze"]}','',f'第一个 test 窗口开始时间：{TIMING["first_test_window_started_at"]}','', '本协议在看过 4101 dev 结果后制定；4101 test 为 test 划分的首次使用。多seed方案仅写入协议，本任务未执行其他seed。','', '## 主终点：Proposed 对 TFOO-200',''])
 table(['Proposed均值s','TFOO均值s','相对差%','胜场/20','Wilcoxon统计量','W+','Wilcoxon p','t检验p','符号检验p'],[[main['candidate_mean_s'],main['reference_mean_s'],main['relative_difference_percent'],main['wins'],st['wilcoxon']['statistic'],st['wilcoxon']['W_plus'],st['wilcoxon']['pvalue'],st['ttest_rel']['pvalue'],st['binomtest']['pvalue']]])
 lines.extend(['Wilcoxon p<0.05：'+('是' if st['wilcoxon']['pvalue']<0.05 else '否')+'。','',"20场景配对差；Wilcoxon双侧，zero_method='wilcox'，α=0.05。次要比较只作报告，不做多重比较校正。",'', '## 主比较',''])
 headers=['候选','参考','候选均值s','参考均值s','相对差%','胜场/20','mean d','sd d','Wilcoxon统计量','W+','Wilcoxon p','t统计量','t p','符号p']
 def comparisonrows(rs):return [[r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['mean_d'],r['statistics']['sd_d'],r['statistics']['wilcoxon']['statistic'],r['statistics']['wilcoxon']['W_plus'],r['statistics']['wilcoxon']['pvalue'],r['statistics']['ttest_rel']['statistic'],r['statistics']['ttest_rel']['pvalue'],r['statistics']['binomtest']['pvalue']] for r in rs]
 table(headers,comparisonrows([r for r in comparisons if r['category']=='main']))
 lines.extend(['### 主比较 cost/s（场景等权）','']);table(['参考','Proposed均值','参考均值','相对差%','胜场/20'],[[r['reference'],r['cost_per_s']['candidate_mean'],r['cost_per_s']['reference_mean'],r['cost_per_s']['relative_difference_percent'],r['cost_per_s']['wins']] for r in comparisons if r['category']=='main'])
 lines.extend(['## 消融比较','']);table(headers,comparisonrows([r for r in comparisons if r['category']=='ablation']))
 lines.extend(['## 补充比较','']);table(headers,comparisonrows([r for r in comparisons if r['category']=='supplementary']))
 lines.extend(['gates.unfinished_per_episode 是20个场景各自每窗口未完成数之和，不是原始窗口总数。下表按80窗口归一。','', '## 各组指标（80窗口池化）',''])
 table(['组','截断s','upload s','ready s','u2u s','cpu s','cost/s','高优未完成/窗口','低优未完成/窗口','最终距离m','solver_calls','fallback_calls'],[[g,m['truncated_delay_s'],*[m['stage_delay_s'][k] for k in ('upload','ready','u2u','cpu')],m['cost_per_s'],*[p['unfinished']/m['windows'] for p in m['priority']],summary['final_distance_m'][g],summary['diagnostics'][g]['solver_calls'] if g in ('TFOO','Abl_UpperCVX') else '不适用',summary['diagnostics'][g]['fallback_calls'] if g in ('TFOO','Abl_UpperCVX') else '不适用'] for g,m in summary['by_mode'].items()])
 lines.extend(['## 主终点逐场景','']);table(['场景','Proposed s','TFOO s','差s','胜'],[[r['scenario'],r['candidate_s'],r['reference_s'],r['candidate_s']-r['reference_s'],r['win']] for r in main['pairs']])
 lines.extend(['## 跨stream一致性','', '对每组每场景，将stream 2/3/4完整metrics与stream 1逐位比较；此项只报告，不作为停止条件。',''])
 table(['组','相同数量','比较总数'],[[g,r['equal'],r['total']] for g,r in summary['stream_consistency']['per_group'].items()])
 lines.extend(['## 耗时','']);atomic_json(OUT/'timing.json',TIMING)
 table(['阶段','墙钟秒'],[[k,TIMING[k]] for k in ('preflight_seconds','protocol_seconds','replay_seconds','test_evaluation_seconds','summary_seconds','total_wall_seconds_before_report_write')])
 lines.extend(['每窗口墙钟是worker内从窗口开始到写完窗口JSON的实测时间（含该窗口的读取、初始化与计算）；并行运行，不能将窗口时间求和当成面板墙钟。',''])
 table(['组','窗口数','平均每窗口墙钟秒'],[[g,80,TIMING['test_per_group_mean_window_wall_seconds'][g]] for g in GROUPS])
 lines.extend(['## 脚本实现错误与修正','', '无。','', '## 最终工作树状态','']);table(['工作树','git status','branch','HEAD'],[[root,'空' if not r['status'] else r['status'],r['branch'],r['head']] for root,r in git.items()])
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
