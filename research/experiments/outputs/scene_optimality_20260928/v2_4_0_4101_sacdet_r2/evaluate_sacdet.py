"""Replay checks, six E panels, fixed actor probes and factual SACDET report."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,gc,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from sacdet_4101 import WT,MAIN,OUT,P,B,CONFIG,PREVIOUS,read,require,git_states,hashes
from hrl_v2.checkpoint import identity,digest,json_digest,load,atomic_json
from hrl_v2.runtime import configure,Learner,parameter_hash
from hrl_v2.upper_baselines import UpperLearner,load_typed,upper_phases,fixed_upper_policy
from hrl_v2.policy import Policy
from hrl_v2.paper_evaluation import comparison_policy
from hrl_v2.evaluate import gate_manifest
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics
from hrl_v2.gates import paired,unfinished_per_episode
from hrl_v2.lite_observation import UPPER_DIM
from hrl_v2.timing import ClockConfig
from hrl_v2.run import read_config
BASE_ANALYSIS=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_baselines'
R2=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_diag_r2'
CACHE={}

def cached_learner(path,kind,binding,config,device):
 key=(str(path),kind)
 if key not in CACHE:
  saved=load_typed(path,binding,kind) if kind in ('Proposed','TFOO') else load(path,binding)
  learner=(UpperLearner if kind=='TFOO' else Learner)(4101,device,config=config)
  learner.load_state_dict(saved['learner']);CACHE[key]=learner
  del saved;gc.collect()
 return CACHE[key]

def window(task):
 config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);device=configure()
 scene=read(WT/'src/hrl_v2/configs/physical_scene.json');clocks=ClockConfig.from_config(config)
 scenario=task['scenario'];stream=task['stream'];item=task['identity']
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  adapter=PhysicsAdapter(scene,timing=clocks)
  adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
  streams=Streams(4101,episode=scenario['scenario'],split='dev',policy_stream=stream)
  if task.get('replay'):
   owner=task['owner'];learner=cached_learner(Path(task['checkpoint']),owner,binding,config,device)
   policy=comparison_policy(learner,scene,streams,task['internal'],stochastic=False)
  else:
   group=task['group'];lower=cached_learner(P/'checkpoints/episode-0300-evaluated.pt','Proposed',binding,config,device)
   if group=='E4':upper=cached_learner(B/'runs/TFOO/seed_4101/checkpoints/episode-0200-anchor.pt','TFOO',binding,config,device)
   else:
    n=100 if group=='E1' else 200
    upper=cached_learner(OUT/f'runs/sacdet/checkpoints/episode-{n:04d}.pt','SACDET',binding,config,device)
   if group in ('E5','E6'):
    policy=fixed_upper_policy(upper.ppo,scene,streams,'cvx' if group=='E5' else 'equal',stochastic=False,mode='moving')
   else:policy=Policy(lower.sac.actor,upper.ppo.actor,None,streams,mode='hover' if group=='E3' else 'moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  metrics=collect_metrics(adapter,policy,identity=item,reward_scale=config['objective']['reward_scale'])
  diagnostics={}
  if getattr(policy,'lower_override',None) is not None and hasattr(policy.lower_override,'rule'):
   records=policy.lower_override.rule.records;diagnostics=dict(solver_calls=len(records),fallback_calls=sum(r.get('fallback',False) for r in records))
  result=json.loads(json.dumps(dict(identity=item,metrics=metrics,diagnostics=diagnostics,learning_updates=0,replay_writes=0),allow_nan=False))
  if 'output' in task:
   path=Path(task['output']);require(not path.exists(),'new_window_file',str(path));atomic_json(path,result)
  return result

def diff_values(a,b,prefix=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:out.append(dict(field=prefix+k,candidate=a.get(k),reference=b.get(k)))
   else:out.extend(diff_values(a[k],b[k],prefix+k+'.'))
  return out
 if a==b:return []
 return [dict(field=prefix.rstrip('.'),candidate=a,reference=b)]

def evaluation(config,binding):
 manifest=gate_manifest(20);launch=read(B/'launch_contract.json')
 require(json_digest(manifest)==read(Path(launch['panels']['A']['output'])/'summary_deterministic.json')['identity']['manifest'],'E_manifest_digest')
 require(manifest['policy_streams']==[1,2] and manifest['split']=='dev','E_stream_contract')
 checks=[];evalstart=time.perf_counter()
 for panel,internal,owner in (('A','moving','Proposed'),('A','moving_cvx','TFOO'),('A','shared_upper_cvx','Proposed'),('B','moving','Proposed')):
  definition=launch['panels'][panel];oldpath=Path(definition['output'])/f'deterministic_{internal}_01_1.json';old=read(oldpath)
  checkpoint=definition['proposed_checkpoint'] if owner=='Proposed' else definition['baseline_checkpoints'][owner]
  new=window(dict(replay=True,owner=owner,checkpoint=checkpoint,internal=internal,scenario=manifest['scenarios'][0],stream=1,identity=old['identity']))
  difference=diff_values(new['metrics'],old['metrics'])
  checks.append(dict(panel=panel,internal_name=internal,scenario=1,stream=1,metrics_equal=not difference,differences=difference,reference=str(oldpath)))
  atomic_json(OUT/'replay_checks.json',dict(status='checking' if difference else 'passed_so_far',rows=checks))
  require(not difference,'replay_metrics_not_bitwise_equal',checks[-1]);print('REPLAY_PASS',panel,internal,flush=True)
 atomic_json(OUT/'replay_checks.json',dict(status='PASS',rows=checks))
 CACHE.clear();gc.collect();torch.cuda.empty_cache()
 names={'E1':'SACDET-100','E2':'SACDET-200','E3':'hover_SAC300','E4':'TFOO200_upper+SAC300','E5':'SACDET200_upper+cvx','E6':'SACDET200_upper+equal'}
 tasks=[]
 for group in names:
  for scenario in manifest['scenarios']:
   for stream in (1,2):
    item=dict(protocol='SACDET_diagnostic_E_r2',group=group,name=names[group],manifest=json_digest(manifest),scenario=scenario,stream=stream,master=4101,split='dev',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=config['objective']['reward_scale'])
    tasks.append(dict(group=group,scenario=scenario,stream=stream,identity=item,output=str(OUT/f'windows/{group}_{scenario["scenario"]:02d}_{stream}.json')))
 rows=[]
 with ProcessPoolExecutor(max_workers=6) as executor:
  futures={executor.submit(window,t):(t['group'],t['scenario']['scenario'],t['stream']) for t in tasks}
  for future in as_completed(futures):
   try:result=future.result()
   except BaseException:
    for queued in futures:queued.cancel()
    for process in list(executor._processes.values()):
     if process.is_alive():process.terminate()
    raise
   rows.append(result)
   atomic_json(OUT/'evaluation_progress.json',dict(completed_windows=len(rows),total_windows=240,last_completed=dict(zip(('group','scenario','stream'),futures[future]))))
   if len(rows)%20==0:print('EVALUATION',len(rows),'/240',flush=True)
 rows.sort(key=lambda r:(r['identity']['group'],r['identity']['scenario']['scenario'],r['identity']['stream']))
 totals={g:PooledMetrics() for g in names};scenarios={g:{} for g in names};distances={g:[] for g in names};diags={g:dict(solver_calls=0,fallback_calls=0) for g in names}
 for r in rows:
  g=r['identity']['group'];sid=str(r['identity']['scenario']['scenario']);totals[g].add(r['metrics']);scenarios[g].setdefault(sid,PooledMetrics()).add(r['metrics'])
  distances[g].append(r['metrics']['terminal_mobility']['final_mean_terminal_to_serving_uav_m'])
  for key,value in r['diagnostics'].items():diags[g][key]+=value
 summary=dict(names=names,manifest=manifest,master=4101,windows=240,by_mode={g:t.result() for g,t in totals.items()},per_scenario={g:{sid:t.result() for sid,t in ss.items()} for g,ss in scenarios.items()},final_distance_m={g:float(np.mean(v)) for g,v in distances.items()},diagnostics=diags,evaluation_wall_seconds=time.perf_counter()-evalstart)
 atomic_json(OUT/'evaluation_summary.json',summary)
 return summary

def request_displacement(actions):
 c=np.clip(np.asarray(actions).reshape(-1,3,2),-1,1)
 return 20*c/np.maximum(np.linalg.norm(c,axis=-1,keepdims=True),1)
def direction_metrics(a,b):
 na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1);mask=(na>1e-9)&(nb>1e-9)
 cos=np.sum(a[mask]*b[mask],axis=-1)/(na[mask]*nb[mask])
 return dict(valid=int(mask.sum()),mean=float(cos.mean()) if cos.size else None,positive_fraction=float((cos>0).mean()) if cos.size else None)

def probes(config,binding,device):
 obs=[];uav=[];term=[];centroid=[]
 for sid in range(1,21):
  with np.load(R2/f'windows/N400_hover_{sid:02d}.npz') as z:
   obs.append(z['obs'][:180]);uav.append(z['uav'][:180,:,:2]);term.append(z['term'][:180,:,:2]);centroid.append(z['centroid'][:180])
 X=np.concatenate(obs);U=np.concatenate(uav);T=np.concatenate(term).reshape(3600,3,13,2);C=np.concatenate(centroid).reshape(3600,3,2)
 require(UPPER_DIM==729 and X.shape==(3600,729),'probe_shape',list(X.shape))
 def distance_delta(displacement):
  before=np.linalg.norm(T-U[:,:,None,:],axis=-1).mean(axis=-1)
  after=np.linalg.norm(T-(U+displacement)[:,:,None,:],axis=-1).mean(axis=-1)
  return float((after-before).mean())
 centroid_delta=distance_delta(request_displacement(C.reshape(3600,6)));phases=upper_phases()
 def compute(name,n,learner,initial,previous):
  vec=np.concatenate([p.detach().cpu().numpy().ravel().astype(np.float64) for p in learner.ppo.actor.parameters()])
  if initial is None:initial=vec.copy()
  with torch.no_grad():mean,logstd=learner.ppo.actor.decode(torch.as_tensor(X,dtype=torch.float32,device='cuda:0'))
  mean=mean.cpu().numpy().astype(np.float64);logstd=logstd.cpu().numpy().astype(np.float64);require(mean.shape==logstd.shape==(3600,6),'probe_forward_shape')
  disp=request_displacement(np.tanh(mean));norm=np.linalg.norm(disp,axis=-1);avg=disp.mean(0);denom=norm.mean(0)
  drift=np.divide(np.linalg.norm(avg,axis=-1),denom,out=np.zeros(3),where=denom>0)
  row=dict(baseline=name,episode=n,global_episode=phases[n-1].global_episode if n else 300,kind='snapshot' if n else 'full',ppo_actor_hash=parameter_hash(learner.ppo.actor),l2_from_initial=float(np.linalg.norm(vec-initial)),l2_from_previous=float(np.linalg.norm(vec-previous)) if previous is not None else None,displacement_mean_m=float(norm.mean()),displacement_p50_m=float(np.percentile(norm,50)),displacement_p90_m=float(np.percentile(norm,90)),stationary_fraction=float((norm<1).mean()),logstd_mean=float(logstd.mean()),mean_state_std=float(mean.std(axis=0).mean()),mean_abs=float(np.abs(mean).mean()),drift_ratio_per_uav=[float(drift[j]) if denom[j]>0 else None for j in range(3)],mean_displacement_vector_per_uav_m=avg.tolist(),centroid_direction=direction_metrics(disp,C),unweighted_terminal_centroid_direction=direction_metrics(disp,T.mean(axis=2)-U),distance_delta_m=distance_delta(disp),centroid_distance_delta_m=centroid_delta)
  return row,initial,vec
 reference=next(r for r in read(BASE_ANALYSIS/'probe_baselines.json')['rows'] if r['baseline']=='TFOO' and r['episode']==200)
 init=None;prev=None;row=None
 for n in (0,175,200):
  path=B/'runs/TFOO/seed_4101'
  if n==0:saved=load_typed(path/'checkpoints/episode-0000-initial.pt',binding,'TFOO')
  else:
   checkpoint=path/f'snapshots/episode-{n:04d}.pt';require(read(checkpoint.with_suffix('.json'))['sha256']==digest(checkpoint),'TFOO_probe_snapshot_sha');saved=torch.load(checkpoint,map_location='cpu',weights_only=False);require(saved['identity']==binding,'TFOO_probe_identity')
  learner=UpperLearner(4101,device,config);learner.load_state_dict(saved['learner']);row,init,vec=compute('TFOO',n,learner,init,prev);prev=vec.copy();del saved,learner;gc.collect();torch.cuda.empty_cache()
 def close(a,b):
  if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
  if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
  if isinstance(a,float) and isinstance(b,(float,int)):return abs(a-b)<1e-12 or a==b
  return a==b
 same=close(row,reference);atomic_json(OUT/'probe_recheck.json',dict(status='PASS' if same else 'FAIL',reference=reference,recomputed=row,tolerance=1e-12));require(same,'TFOO_probe_recheck',diff_values(row,reference));print('PROBE_RECHECK_PASS',flush=True)
 rows=[];init=None;prev=None
 for n in (0,*range(25,201,25)):
  path=OUT/f'runs/sacdet/snapshots/episode-{n:04d}.pt';saved=load(path,binding);learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner'])
  row,init,vec=compute('SACDET',n,learner,init,prev);rows.append(row);prev=vec.copy();del saved,learner;gc.collect();torch.cuda.empty_cache()
 existing=[r for r in read(BASE_ANALYSIS/'probe_baselines.json')['rows'] if (r['baseline']=='Proposed' and r['episode'] in (400,600)) or (r['baseline'] in ('TOO','TFOO') and r['episode'] in (100,200))]
 atomic_json(OUT/'probe_sacdet.json',dict(probe_shape=list(X.shape),rows=rows,existing_comparisons=existing));return rows,existing

def comparisons(summary):
 launch=read(B/'launch_contract.json');old={key:read(Path(v['output'])/'summary_deterministic.json') for key,v in launch['panels'].items()};results=[]
 def add(cname,rname,candidate,reference):
  pair=paired(candidate,reference)
  results.append(dict(candidate=cname,reference=rname,scenarios=pair['scenarios'],wins=pair['wins'],candidate_mean_s=pair['candidate_mean_s'],reference_mean_s=pair['reference_mean_s'],relative_difference_percent=100*(pair['candidate_mean_s']/pair['reference_mean_s']-1),candidate_unfinished_per_episode=unfinished_per_episode(candidate),reference_unfinished_per_episode=unfinished_per_episode(reference),pairs=pair['pairs']))
 for g,panel,n in (('E1','B',100),('E2','A',200)):
  c=summary['per_scenario'][g];add(g,'E3',c,summary['per_scenario']['E3'])
  for mode,name in (('moving','Proposed'),('moving_equal','TOO'),('moving_cvx','TFOO'),('hover_equal','CA')):add(g,f'{name}-{400 if g=="E1" and name=="Proposed" else 600 if name=="Proposed" else n if name!="CA" else "hover"}',c,old[panel]['per_scenario'][mode])
 add('E4','TFOO-200',summary['per_scenario']['E4'],old['A']['per_scenario']['moving_cvx'])
 for ref in ('E5','E6'):add('E2',ref,summary['per_scenario']['E2'],summary['per_scenario'][ref])
 atomic_json(OUT/'comparisons.json',results);return results

def curves():
 def rows(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
 proposed={r['global_episode']:r for r in rows(P/'train_metrics.jsonl')};others={name:rows(B/f'runs/{name}/seed_4101/train_metrics.jsonl') for name in ('TOO','TFOO')};sacdet=rows(OUT/'runs/sacdet/train_metrics.jsonl');phases=upper_phases();bins=[]
 for i in range(0,200,5):
  selected=phases[i:i+5];g=[p.global_episode for p in selected]
  bins.append(dict(first_episode=i+1,last_episode=i+5,global_episodes=g,SACDET=float(np.mean([r['truncated_delay_s'] for r in sacdet[i:i+5]])),Proposed=float(np.mean([proposed[n]['truncated_delay_s'] for n in g])),**{name:float(np.mean([r['truncated_delay_s'] for r in v[i:i+5]])) for name,v in others.items()}))
 atomic_json(OUT/'training_curves.json',bins)
 with (OUT/'training_curves.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(bins[0]));writer.writeheader();writer.writerows(bins)
 return bins

def report(summary,probes,existing,pairs,bins):
 fmt=lambda x:'缺失' if x is None else f'{x:.9f}' if isinstance(x,float) else str(x)
 lines=['# v2.4.0 seed 4101 SACDET 诊断 E（r2）','',f'previous_failed_attempt={PREVIOUS}。上一轮原因：GPU 检查未排除自身进程。','', '## 检查结果','']
 def table(headers,rows):
  lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']);lines.extend('| '+' | '.join(fmt(v) for v in r)+' |' for r in rows);lines.append('')
 loop=read(OUT/'loop_equiv.json');training=read(OUT/'training_summary.json')
 table(['检查','结果'],[['预检','PASS'],['LOOP_EQUIV',loop['status']],['SAC actor哈希不变',training['sacdet']['sac_actor_hash_unchanged']],['SAC计数不变',training['sacdet']['sac_counts_unchanged']],['PPO更新次数',training['sacdet']['ppo_updates']],['已有论文面板窗口回放','4/4 逐位相等'],['TFOO-200探针复算','PASS；逐位或绝对差 < 1e-12'],['评估窗口',summary['windows']]])
 if loop['first_difference'] is not None:lines.extend(['LOOP_EQUIV第一处不一致：','```json',json.dumps(loop['first_difference'],ensure_ascii=False,indent=2),'```',''])
 lines.extend(['PPO日志比较全部epoch、KL及accepted字段；原attempt_id属于原训练运行标记。','', '## 脚本实现错误与修正','', 'r2 执行中未发生检查或脚本实现错误；未使用允许的一次修正机会。上一轮原始报错为 AssertionError: other GPU Python process（preflight.py:44）；GPU 检查差异见 gpu_check_changes_vs_attempt1.diff。修正理由是旧检查把自身算作其他进程。本次检查记录见 contract.json / preflight.json：显式排除 os.getpid() 和 os.getppid()，同时核对 CIM 命令行及 nvidia-smi 计算进程列表。','', '## 训练曲线','', '每5个上层轮次的截断时延均值；Proposed取 upper_phases 对应全局轮次。401轮后，Proposed含交替更新下层；SACDET保持SAC-300冻结。',''])
 table(['上层轮次','对应全局轮次','SACDET s','Proposed s','TOO s','TFOO s'],[[f'{r["first_episode"]}–{r["last_episode"]}',','.join(map(str,r['global_episodes'])),r['SACDET'],r['Proposed'],r['TOO'],r['TFOO']] for r in bins])
 lines.extend(['## 上层固定探针','', '同一批3600个Proposed-400悬停状态上的请求位移；不含地图裁剪、安全投影和闭环执行。零位移下方向余弦未定义。',''])
 table(['组','上层/快照轮次','global','位移均值m','p50 m','p90 m','<1m占比','centroid cos','cos>0占比','距离变化m'],[[r['baseline'],r['episode'],r['global_episode'],r['displacement_mean_m'],r['displacement_p50_m'],r['displacement_p90_m'],r['stationary_fraction'],r['centroid_direction']['mean'],r['centroid_direction']['positive_fraction'],r['distance_delta_m']] for r in probes+existing])
 lines.extend(['## 配对比较','', '每场景取两条stream的平均，共20个dev场景。相对差=(candidate/reference−1)×100%；负值表示candidate时延较低。',''])
 table(['candidate','reference','candidate均值s','reference均值s','相对差%','胜场/20'],[[r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins']] for r in pairs])
 lines.extend(['## 各组评估指标','', '截断时延及分段时延采用池化口径；配对表采用场景等权口径。未完成数按40窗口归一，分优先级列出。',''])
 table(['组','名称','截断s','upload s','ready s','u2u s','cpu s','cost/s','高优未完成/窗口','低优未完成/窗口','最终UAV关联终端距离m'],[[g,summary['names'][g],m['truncated_delay_s'],*[m['stage_delay_s'][k] for k in ('upload','ready','u2u','cpu')],m['cost_per_s'],*[p['unfinished']/m['episodes'] for p in m['priority']],summary['final_distance_m'][g]] for g,m in summary['by_mode'].items()])
 table(['组','CVX solver_calls','fallback_calls'],[[g,d['solver_calls'],d['fallback_calls']] for g,d in summary['diagnostics'].items() if g=='E5'])
 lines.extend(['## 耗时',''])
 table(['运行','轮数','总墙钟s'],[[key,r['episodes'],r['wall_seconds']] for key,r in training.items()])
 times=[]
 for name in ('control_stoch','sacdet'):
  rows=[json.loads(l) for l in (OUT/f'runs/{name}/train_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
  times.extend([name,r['episode'],r['global_episode'],r['wall_seconds']] for r in rows)
 table(['运行','轮次','global','每轮秒数'],times)
 lines.extend([f'评估总时长（含4个核对窗口）：{summary["evaluation_wall_seconds"]:.9f} s。','', '## 事实与推断','', '本报告列出实测指标与检查结果；未作机制或原因推断。','', '## 最终工作树状态',''])
 states=git_states();atomic_json(OUT/'git_after.json',states);table(['工作树','git status --porcelain','branch','HEAD'],[[root,'空' if not r['status'] else r['status'],r['branch'],r['head']] for root,r in states.items()])
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def plot_results(bins,probe,existing):
 import os
 os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig');(OUT/'mplconfig').mkdir(exist_ok=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,5))
 for name in ('SACDET','Proposed','TOO','TFOO'):
  ax.plot([r['last_episode'] for r in bins],[r[name] for r in bins],label=name)
 ax.set_xlabel('Upper episode (mean over 5 episodes)');ax.set_ylabel('Training truncated delay (s)');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
 for name in ('SACDET','Proposed','TOO','TFOO'):
  rs=[r for r in probe+existing if r['baseline']==name]
  axes[0].plot([r['global_episode'] for r in rs],[r['displacement_mean_m'] for r in rs],'o-',label=name)
  axes[1].plot([r['global_episode'] for r in rs],[r['centroid_direction']['mean'] for r in rs],'o-',label=name)
 axes[0].set_ylabel('Mean requested displacement (m)');axes[1].set_ylabel('Mean cosine to centroid');axes[1].set_xlabel('Proposed global episode')
 for ax in axes:ax.grid(alpha=.3);ax.legend()
 fig.tight_layout();fig.savefig(OUT/'probe_sacdet.png',dpi=160);plt.close(fig)

def finish(config,binding,device):
 summary=evaluation(config,binding);probe,existing=probes(config,binding,device);pair=comparisons(summary);bins=curves();plot_results(bins,probe,existing);report(summary,probe,existing,pair,bins)
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  config=read_config(CONFIG,WT);finish(config,identity(WT,CONFIG),configure())
 except BaseException as e:
  atomic_json(OUT/'failure.json',dict(stage='evaluation_and_analysis',error=type(e).__name__,message=str(e),traceback=traceback.format_exc()));traceback.print_exc();sys.exit(1)