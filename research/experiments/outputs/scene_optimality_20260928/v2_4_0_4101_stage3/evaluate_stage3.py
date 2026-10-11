"""Stage3 replay check, fixed 10-cycle dev evaluation, comparisons and factual report."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,gc,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from stage3_4101 import WT,MAIN,OUT,R2,P,B,CONFIG,SD,S300,read,rows,require,git_states,hashes,firstdiff,CHECKS
from hrl_v2.checkpoint import identity,load,atomic_json,json_digest
from hrl_v2.runtime import configure,Learner,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.policy import Policy
from hrl_v2.evaluate import gate_manifest
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics
from hrl_v2.gates import paired,unfinished_per_episode
from hrl_v2.timing import ClockConfig
CACHE={}
def cached(path,binding,config,device):
 key=str(path)
 if key not in CACHE:
  saved=load(path,binding);learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner']);CACHE[key]=learner;del saved;gc.collect()
 return CACHE[key]
def window(task):
 config=read_config(CONFIG,WT);binding=read(OUT/'contract.json')['identity'];require(identity(WT,CONFIG)==binding,'evaluation_identity')
 device=configure();scene=read(WT/'src/hrl_v2/configs/physical_scene.json');clocks=ClockConfig.from_config(config)
 scenario=task['scenario'];stream=task['stream'];item=task['identity']
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  if task.get('replay'):
   lower=cached(S300,binding,config,device);upper=cached(SD,binding,config,device)
  else:
   path=OUT/f'runs/stage3/checkpoints/cycle-{task["cycle"]:02d}.pt'
   # Keep only the current cycle learner; no growing GPU cache across 10 groups.
   if str(path) not in CACHE:CACHE.clear();gc.collect();torch.cuda.empty_cache()
   lower=upper=cached(path,binding,config,device)
  require(parameter_hash(upper.ppo.actor)==read(OUT/'preflight.json')['frozen_upper']['hashes']['ppo_actor'],'evaluation_frozen_upper_actor')
  adapter=PhysicsAdapter(scene,timing=clocks);adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
  streams=Streams(4101,episode=scenario['scenario'],split='dev',policy_stream=stream)
  policy=Policy(lower.sac.actor,upper.ppo.actor,None,streams,mode='moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  metrics=collect_metrics(adapter,policy,identity=item,reward_scale=config['objective']['reward_scale'])
  result=json.loads(json.dumps(dict(identity=item,metrics=metrics,diagnostics={},learning_updates=0,replay_writes=0),allow_nan=False))
  if task.get('replay'):return result
  path=Path(task['output']);require(not path.exists(),'new_window_path',str(path));atomic_json(path,result)
  return dict(path=str(path),group=task['group'],scenario=scenario['scenario'],stream=stream)
def evaluate(config,binding):
 start=time.perf_counter();manifest=gate_manifest(20);previous=read(R2/'evaluation_summary.json')
 require(manifest==previous['manifest'] and manifest['policy_streams']==[1,2] and manifest['split']=='dev','paired_manifest')
 old=read(R2/'windows/E2_01_1.json')
 new=window(dict(replay=True,scenario=manifest['scenarios'][0],stream=1,identity=old['identity']))
 difference=firstdiff(new['metrics'],old['metrics'])
 check=dict(status='PASS' if difference is None else 'FAIL',reference=str(R2/'windows/E2_01_1.json'),metrics_equal=difference is None,first_difference=difference,wall_seconds=time.perf_counter()-start)
 atomic_json(OUT/'replay_checks.json',check);require(difference is None,'E2_replay_bitwise',check);CHECKS['E2_replay']='PASS';print('E2_REPLAY_PASS',flush=True)
 del new,old;CACHE.clear();gc.collect();torch.cuda.empty_cache()
 tasks=[];names={f'S3-c{n}':f'SACDET200_upper + SAC_cycle_{n:02d}' for n in range(1,11)}
 for cycle in range(1,11):
  group=f'S3-c{cycle}'
  for scenario in manifest['scenarios']:
   for stream in (1,2):
    item=dict(protocol='frozen_upper_stage3_repair',group=group,name=names[group],manifest=json_digest(manifest),scenario=scenario,stream=stream,master=4101,split='dev',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=config['objective']['reward_scale'])
    tasks.append(dict(cycle=cycle,group=group,scenario=scenario,stream=stream,identity=item,output=str(OUT/f'windows/{group}_{scenario["scenario"]:02d}_{stream}.json')))
 completed=0
 with ProcessPoolExecutor(max_workers=6) as executor:
  futures={executor.submit(window,t):t for t in tasks}
  for future in as_completed(futures):
   try:r=future.result()
   except BaseException:
    for pending in futures:pending.cancel()
    for process in list(executor._processes.values()):
     if process.is_alive():process.terminate()
    raise
   completed+=1;atomic_json(OUT/'evaluation_progress.json',dict(completed_windows=completed,total_windows=400,last_completed=r))
   if completed%40==0:print('EVALUATION',completed,'/400',flush=True)
 totals={};scenarios={};distances={};dev=[]
 # Read/aggregate in deterministic group/scenario/stream order, not completion order.
 for cycle in range(1,11):
  group=f'S3-c{cycle}';acc=PooledMetrics();ss={};distance=[]
  for sid in range(1,21):
   one=PooledMetrics()
   for stream in (1,2):
    r=read(OUT/f'windows/{group}_{sid:02d}_{stream}.json');m=r['metrics'];acc.add(m);one.add(m);distance.append(m['terminal_mobility']['final_mean_terminal_to_serving_uav_m']);del r,m
   ss[str(sid)]=one.result();del one
  totals[group]=acc.result();scenarios[group]=ss;distances[group]=float(np.mean(distance));del acc
 summary=dict(names=names,manifest=manifest,master=4101,windows=400,by_mode=totals,per_scenario=scenarios,final_distance_m=distances,evaluation_wall_seconds=time.perf_counter()-start)
 atomic_json(OUT/'evaluation_summary.json',summary);return summary

def comparisons(summary):
 previous=read(R2/'evaluation_summary.json');oldcomp=read(R2/'comparisons.json');launch=read(B/'launch_contract.json')
 a=read(Path(launch['panels']['A']['output'])/'summary_deterministic.json')
 e2=previous['per_scenario']['E2'];e3=previous['per_scenario']['E3'];tf=a['per_scenario']['moving_cvx']
 require(a['identity']['manifest']==json_digest(summary['manifest']),'TFOO200_manifest')
 def make(cname,rname,c,r):
  p=paired(c,r)
  return dict(candidate=cname,reference=rname,scenarios=p['scenarios'],wins=p['wins'],candidate_mean_s=p['candidate_mean_s'],reference_mean_s=p['reference_mean_s'],relative_difference_percent=100*(p['candidate_mean_s']/p['reference_mean_s']-1),candidate_unfinished_per_episode=unfinished_per_episode(c),reference_unfinished_per_episode=unfinished_per_episode(r),pairs=p['pairs'])
 rechecks=[]
 for rname,reference,pct,wins in (('E3',e3,-27.735420899,20),('TFOO-200',tf,-4.499045329,13)):
  new=make('E2',rname,e2,reference);old=next(r for r in oldcomp if r['candidate']=='E2' and r['reference']==rname)
  exact=firstdiff(new,old);okay=exact is None and round(new['relative_difference_percent'],9)==pct and new['wins']==wins
  rechecks.append(dict(reference=rname,equal_to_R2=exact is None,first_difference=exact,relative_difference_percent=new['relative_difference_percent'],expected_rounded_percent=pct,wins=new['wins'],passed=okay))
  atomic_json(OUT/'recompute_checks.json',dict(status='PASS' if all(r['passed'] for r in rechecks) else 'FAIL',rows=rechecks));require(okay,'paired_recompute',rechecks[-1])
 CHECKS['paired_recompute']='PASS';print('PAIRED_RECOMPUTE_PASS',flush=True)
 results=[make(f'S3-c{k}','E2',summary['per_scenario'][f'S3-c{k}'],e2) for k in range(1,11)]
 final=summary['per_scenario']['S3-c10']
 for name,ref in (('TFOO-200',tf),('CA-hover',a['per_scenario']['hover_equal']),('E3',e3),('E5',previous['per_scenario']['E5'])):results.append(make('S3-c10',name,final,ref))
 atomic_json(OUT/'comparisons.json',results);main=next(r for r in results if r['candidate']=='S3-c10' and r['reference']=='E2');criterion=read(OUT/'contract.json')['criterion']
 helpful=main['candidate_mean_s']<criterion['mean_less_than_s'] and main['wins']>=criterion['minimum_wins']
 verdict=dict(verdict='STAGE3_HELPS' if helpful else 'STAGE3_NOT_HELPFUL',criterion=criterion,mean_condition=main['candidate_mean_s']<criterion['mean_less_than_s'],wins_condition=main['wins']>=criterion['minimum_wins'],main_comparison=main)
 atomic_json(OUT/'verdict.json',verdict);return results,verdict

def training_curves():
 candidate=rows(OUT/'runs/stage3/train_metrics.jsonl');original={r['global_episode']:r for r in rows(P/'train_metrics.jsonl')};table=[];blocks=[]
 for i,r in enumerate(candidate,1):
  old=original[r['global_episode']]
  table.append(dict(lower_episode=i,cycle=(i-1)//10+1,global_episode=r['global_episode'],S3_truncated_delay_s=r['truncated_delay_s'],P_truncated_delay_s=old['truncated_delay_s'],S3_work_seconds=r['work_seconds'],S3_wall_seconds=r['wall_seconds'],P_work_seconds=old['work_seconds']))
 for i in range(0,100,10):
  rr=table[i:i+10];blocks.append(dict(cycle=i//10+1,global_episodes=[r['global_episode'] for r in rr],S3_mean_s=float(np.mean([r['S3_truncated_delay_s'] for r in rr])),P_mean_s=float(np.mean([r['P_truncated_delay_s'] for r in rr]))))
 with (OUT/'training_curves.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 atomic_json(OUT/'training_curves.json',dict(per_episode=table,by_cycle=blocks));return table,blocks

def plots(table,results):
 os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig');(OUT/'mplconfig').mkdir(exist_ok=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,5))
 for label,key in (('S3','S3_truncated_delay_s'),('Original Proposed','P_truncated_delay_s')):ax.plot([r['lower_episode'] for r in table],[r[key] for r in table],label=label)
 ax.set_xlabel('Lower episode (original stage3 lower phases)');ax.set_ylabel('Training truncated delay (s)');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
 rs=results[:10];tf=next(r['reference_mean_s'] for r in results if r['reference']=='TFOO-200');e2=rs[0]['reference_mean_s']
 fig,ax=plt.subplots(figsize=(10,5));ax.plot(range(1,11),[r['candidate_mean_s'] for r in rs],'o-',label='S3')
 ax.axhline(e2,label='E2',color='green',linestyle='--');ax.axhline(tf,label='TFOO-200',color='orange',linestyle=':');ax.set_xticks(range(1,11));ax.set_xlabel('Cycle');ax.set_ylabel('Dev scenario-equal mean truncated delay (s)');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'dev_by_cycle.png',dpi=160);plt.close(fig)

def report(summary,results,verdict,table,blocks,total_start):
 lines=['# v2.4.0 seed 4101 第三阶段修复试验','',f'全部规定步骤完成。VERDICT={verdict["verdict"]}。','', '## 检查结果','']
 fmt=lambda x:'缺失' if x is None else f'{x:.9f}' if isinstance(x,float) else str(x)
 def tab(headers,data):
  lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
  lines.extend('| '+' | '.join(fmt(x) for x in row)+' |' for row in data);lines.append('')
 training=read(OUT/'training_summary.json');loop=read(OUT/'loop_equiv.json');pre=read(OUT/'preflight.json');replay=read(OUT/'replay_checks.json');recheck=read(OUT/'recompute_checks.json')
 tab(['检查','结果'],[['预检',pre['status']],['LOOP_EQUIV',f'{loop["status"]}；591–593 共270行学习日志逐位相等'],['冻结上层 actor/value、counts、version','PASS'],['SAC计数','PASS'],['经验池清空次数增量',training['replay_flushes_delta']],['E2_01_1完整metrics回放',replay['status']],['E2/E3及E2/TFOO-200复算',recheck['status']],['dev评估窗口',summary['windows']]])
 lines.extend(['唯一协议差异：上层冻结为 SACDET-200，因此经验池只在第一块开头因上层版本变化清空一次；其余规则与原第三阶段下层块相同。','',f'解释器：{pre["runtime"]["executable"]}；Python={pre["runtime"]["python"]}；torch={pre["runtime"]["torch"]}；CUDA={pre["runtime"]["cuda"]}；设备={pre["runtime"]["device"]}；torch线程数={pre["torch_threads"]}。','', '## 预注册判据与主比较','', '主判据只用 S3-c10，在 dev 20 个场景上与 E2 配对比较。每场景取 stream 1、2 的平均，配对和胜场定义与 hrl_v2.gates.paired 相同。','', 'STAGE3_HELPS 当且仅当：S3-c10 场景等权均值 < E2 场景等权均值 0.912816585 s，且胜场 ≥ 11/20；否则 STAGE3_NOT_HELPFUL。S3-c1 到 S3-c9 只作报告曲线，不参与判定，也不得据此挑选模型。','', f'VERDICT={verdict["verdict"]}；均值条件={verdict["mean_condition"]}；胜场条件={verdict["wins_condition"]}。',''])
 main=verdict['main_comparison']
 tab(['候选','参考','候选场景均值s','参考场景均值s','相对差%','胜场/20'],[[main['candidate'],main['reference'],main['candidate_mean_s'],main['reference_mean_s'],main['relative_difference_percent'],main['wins']]])
 lines.extend(['相对差=(candidate/reference−1)×100%。负值表示候选时延较低。','', '### 逐场景主配对',''])
 tab(['场景','S3-c10 s','E2 s','胜'],[[r['scenario'],r['candidate_s'],r['reference_s'],r['win']] for r in main['pairs']])
 lines.extend(['## dev曲线（中间节点不参与判定）',''])
 tab(['块','场景等权均值s','E2均值s','相对差%','胜场/20'],[[r['candidate'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins']] for r in results[:10]])
 lines.extend(['## 报告用的最终配对',''])
 tab(['候选','参考','候选均值s','参考均值s','相对差%','胜场/20'],[[r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins']] for r in results[10:]])
 lines.extend(['## 各组评估指标','', '下表截断时延与四段时延为40窗口池化值；主比较与dev曲线为20场景等权值。高、低优先级未完成数按40窗口归一。最终距离为窗口最终UAV到关联终端距离的均值。',''])
 tab(['组','截断s','upload s','ready s','u2u s','cpu s','cost/s','高优未完成/窗口','低优未完成/窗口','最终距离m'],[[g,m['truncated_delay_s'],*[m['stage_delay_s'][k] for k in ('upload','ready','u2u','cpu')],m['cost_per_s'],*[p['unfinished']/40 for p in m['priority']],summary['final_distance_m'][g]] for g,m in summary['by_mode'].items()])
 lines.extend(['配对JSON的 candidate_unfinished_per_episode/reference_unfinished_per_episode 原样采用 gates.unfinished_per_episode，表示20场景各自每窗口未完成数之和，不是原始40窗口未完成总数；上表另明确除以40。','', '## 训练曲线','', '每块10个下层轮次的截断时延均值。P取原训练中完全相同的global轮次。逐轮完整数据见 training_curves.csv。',''])
 tab(['块','global轮次','S3均值s','P原训练均值s'],[[r['cycle'],','.join(map(str,r['global_episodes'])),r['S3_mean_s'],r['P_mean_s']] for r in blocks])
 lines.extend(['![逐轮训练曲线](training_curves.png)','', '![dev曲线](dev_by_cycle.png)','', '## 参数哈希、计数与复算',''])
 tab(['网络','初始SHA256','最终SHA256'],[[k,training['initial_hashes'][k],training['final_hashes'][k]] for k in ('ppo_actor','ppo_value')])
 tab(['SAC计数','实测增量','预注册增量'],[[k,v,training['expected_sac_delta'][k]] for k,v in training['sac_delta'].items()])
 lines.extend([f'上层初始/最终counts：`{json.dumps(training["initial_ppo_counts"],ensure_ascii=False)}` / `{json.dumps(training["final_ppo_counts"],ensure_ascii=False)}`。','',f'上层初始/最终version：{training["initial_ppo_version"]} / {training["final_ppo_version"]}；发生跳过的global轮次：{training["skipped_global_episodes"]}。',''])
 tab(['复算','相对差%','胜场','与R2完整配对对象相同'],[[r['reference'],r['relative_difference_percent'],r['wins'],r['equal_to_R2']] for r in recheck['rows']])
 lines.extend(['## 耗时',''])
 total=time.perf_counter()-total_start
 timing=dict(loop_equiv_seconds=loop['wall_seconds'],formal_training_seconds=training['wall_seconds'],evaluation_seconds=summary['evaluation_wall_seconds'],total_wall_seconds_before_report_write=total)
 atomic_json(OUT/'timing.json',timing)
 tab(['阶段','墙钟秒'],[[k,v] for k,v in timing.items()])
 lines.extend(['### LOOP_EQUIV逐轮',''])
 tab(['global','work秒','wall秒'],[[r['global_episode'],r['work_seconds'],r['wall_seconds']] for r in rows(OUT/'runs/loop_equiv/train_metrics.jsonl')])
 lines.extend(['### 正式训练逐轮',''])
 tab(['下层轮次','global','work秒','wall秒'],[[r['lower_episode'],r['global_episode'],r['S3_work_seconds'],r['S3_wall_seconds']] for r in table])
 lines.extend(['## 脚本实现错误与修正','', '本次未发生脚本实现错误，未使用一次修正机会。','', '## 最终工作树状态',''])
 states=git_states();atomic_json(OUT/'git_after.json',states)
 tab(['工作树','status','branch','HEAD'],[[root,'空' if not r['status'] else r['status'],r['branch'],r['head']] for root,r in states.items()])
 lines.extend(['## 事实与推断','', '本报告仅列实测与检查结果',''])
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def finish(config,binding,device,total_start):
 summary=evaluate(config,binding);results,verdict=comparisons(summary);table,blocks=training_curves();plots(table,results);report(summary,results,verdict,table,blocks,total_start)
 print('REPORT_COMPLETE',verdict['verdict'],flush=True)
