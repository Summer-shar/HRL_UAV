"""Read-only analysis of frozen independent upper baselines and dev panels."""
from pathlib import Path
import csv,gc,json,shutil,subprocess,sys,time,traceback
import numpy as np
import torch
import hrl_v2
from hrl_v2.checkpoint import identity,digest,json_digest
from hrl_v2.evaluate import gate_manifest
from hrl_v2.gates import paired,unfinished_per_episode
from hrl_v2.lite_observation import UPPER_DIM
from hrl_v2.runtime import configure,parameter_hash,Learner
from hrl_v2.run import read_config
from hrl_v2.upper_baselines import UpperLearner,load_typed,upper_phases
ROOT=Path('D:/HRL_UAV_V2_speedup')
CONFIG=ROOT/'src/hrl_v2/configs/v2_4_0_lite.json'
WORK=ROOT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'
RUN=ROOT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
FROZEN=ROOT/'outputs/lite_v2_4_0/baselines'
R2=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_diag_r2')
OUT=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_baselines')
NAMES=('NC','TOO','TFOO')
MODES=('moving','moving_local','hover_equal','moving_equal','moving_cvx','hover_local','shared_upper_equal','shared_upper_cvx')
LABELS={'moving':'Proposed','moving_local':'NC','hover_equal':'CA','moving_equal':'TOO','moving_cvx':'TFOO','hover_local':'hover_local','shared_upper_equal':'shared_upper_equal','shared_upper_cvx':'shared_upper_cvx'}
CHECKS={}
class HardCheckError(RuntimeError):pass
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def require(condition,label,details=None):
 if not condition:
  raise HardCheckError(json.dumps(dict(check=label,details=details),ensure_ascii=False,default=str))
def differences(a,b):
 return {k:dict(candidate=a.get(k),reference=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
# Copied from the r2 diagnostic; only array-derived diagnostics, no simulation.
def request_displacement(actions):
 c=np.clip(np.asarray(actions).reshape(-1,3,2),-1,1)
 return 20*c/np.maximum(np.linalg.norm(c,axis=-1,keepdims=True),1)
def direction_metrics(a,b):
 na=np.linalg.norm(a,axis=-1); nb=np.linalg.norm(b,axis=-1)
 mask=(na>1e-9)&(nb>1e-9)
 cos=np.sum(a[mask]*b[mask],axis=-1)/(na[mask]*nb[mask])
 return dict(valid=int(mask.sum()),mean=float(cos.mean()) if cos.size else None,positive_fraction=float((cos>0).mean()) if cos.size else None)

def analyze():
 start=time.perf_counter()
 require(read(WORK/'queue_status.json')['status']=='complete','queue_complete')
 require(Path(hrl_v2.__file__).resolve().is_relative_to(ROOT/'src'),'import_path')
 for root in (ROOT,Path('D:/HRL_UAV_V2')):
  require(not subprocess.check_output(['git','status','--porcelain'],cwd=root).strip(),'git_clean_before',str(root))
 device=configure();require(str(device)=='cuda:0','cuda')
 config=read_config(CONFIG,ROOT);binding=identity(WORK,CONFIG)
 launch=read(WORK/'launch_contract.json')
 require(binding==launch['identity']==read(RUN/'contract.json')['identity'],'H1_binding')
 panels={key:Path(value['output']) for key,value in launch['panels'].items()}
 summaries={key:read(path/'summary_deterministic.json') for key,path in panels.items()}
 anchor_checks={};anchor_hashes={}
 for name in NAMES:
  run=WORK/'runs'/name/'seed_4101'
  require(read(run/'contract.json')['identity']==binding,'H1_run_identity',name)
  result=read(run/'result.json')
  require(result['status']=='complete_performance_not_evaluated' and result['completed_episode']==200,'H1_completion',result)
  anchor_checks[name]=[];anchor_hashes[name]={}
  for n,batches in ((100,20),(200,40)):
   p=run/f'checkpoints/episode-{n:04d}-anchor.pt'
   require(p.is_file(),'H1_anchor_exists',str(p))
   saved=load_typed(p,binding,name)
   require(saved['completed_episode']==n and saved['pending_evaluation'] is None and saved['learner']['ppo']['counts']['batches']==batches,'H1_anchor_budget',dict(name=name,episode=n))
   learner=UpperLearner(4101,device,config);learner.load_state_dict(saved['learner'])
   anchor_hashes[name][n]=parameter_hash(learner.ppo.actor)
   anchor_checks[name].append(dict(episode=n,ppo_batches=batches,sha256=digest(p),pending_evaluation=None,actor_hash=anchor_hashes[name][n]))
   del saved,learner;gc.collect();torch.cuda.empty_cache()
 checkpoint_paths={};manifest_sha=json_digest(gate_manifest(20))
 require(manifest_sha==json_digest(read(RUN/'development_manifest.json')),'H2_manifest_source')
 for key,definition in launch['panels'].items():
  paths={'Proposed':Path(definition['proposed_checkpoint']),**{name:Path(p) for name,p in definition['baseline_checkpoints'].items()}}
  checkpoint_paths[key]=paths
  for name,p in paths.items():
   saved=load_typed(p,binding,name);require(saved['identity']==binding,'H1_panel_checkpoint_identity',dict(panel=key,name=name));del saved
  expected_sha={name:digest(p) for name,p in paths.items()}
  require(summaries[key]['identity']['checkpoints']==expected_sha,'H2_checkpoint_digests',dict(panel=key,actual=summaries[key]['identity']['checkpoints'],expected=expected_sha))
  require(summaries[key]['identity']['manifest']==manifest_sha,'H2_panel_manifest',key)
  expected={f'deterministic_{mode}_{sid:02d}_{stream}.json' for mode in MODES for sid in range(1,21) for stream in (1,2)}
  actual={p.name for p in panels[key].glob('deterministic_*.json')}
  require(actual==expected,'H2_windows',dict(panel=key,count=len(actual),missing=sorted(expected-actual),extra=sorted(actual-expected)))
 CHECKS['H1']=dict(passed=True,anchors=anchor_checks,identity=binding)
 CHECKS['H2']=dict(passed=True,manifest_sha256=manifest_sha,windows={k:320 for k in panels},checkpoint_sha256={k:{n:digest(p) for n,p in paths.items()} for k,paths in checkpoint_paths.items()})
 frozen=read(FROZEN/'summary.json')
 raw={};h3={}
 for key,path in panels.items():
  raw[key]={}
  for mode in MODES:
   raw[key][mode]={(sid,stream):read(path/f'deterministic_{mode}_{sid:02d}_{stream}.json') for sid in range(1,21) for stream in (1,2)}
  h3[key]={}
  for mode in ('hover_equal','hover_local'):
   count=0
   for (sid,stream),window in raw[key][mode].items():
    old=read(FROZEN/f'{mode}_{sid:02d}_{stream}.json')['metrics']
    require(window['metrics']==old,'H3_full_metrics',dict(panel=key,mode=mode,scenario=sid,stream=stream,differences=differences(window['metrics'],old)))
    count+=1
   require(summaries[key]['per_scenario'][mode]==frozen['per_scenario'][mode],'H3_per_scenario',dict(panel=key,mode=mode,differences=differences(summaries[key]['per_scenario'][mode],frozen['per_scenario'][mode])))
   h3[key][mode]=dict(equal_windows=count,per_scenario_equal=True)
 CHECKS['H3']=dict(passed=True,panels=h3)
 print('H1_H2_H3_PASS',flush=True)
 CHECKS['R1']={};CHECKS['R2']={}
 drop=('split','policy','mode','training_seed','learning_updates','replay_writes')
 for key in panels:
  n=600 if key=='A' else 400;equal=0;diff_rows=[]
  for (sid,stream),window in raw[key]['moving'].items():
   new={k:v for k,v in window['metrics'].items() if k not in drop}
   old={k:v for k,v in read(RUN/f'evaluations/episode-{n:04d}/deterministic_moving_{sid:02d}_0.json')['metrics'].items() if k not in drop}
   if new==old:equal+=1
   else:diff_rows.append(dict(scenario=sid,stream=stream,differences=dict(list(differences(new,old).items())[:10])))
  CHECKS['R1'][key]=dict(equal_windows=equal,total=40,different_windows=diff_rows)
  CHECKS['R2'][key]={mode:sum(raw[key][mode][(sid,1)]['metrics']==raw[key][mode][(sid,2)]['metrics'] for sid in range(1,21)) for mode in MODES}
 saved=load_typed(RUN/'checkpoints/episode-0000-initial.pt',binding,'Proposed')
 proposed=Learner(4101,device,config=config);proposed.load_state_dict(saved['learner'])
 initial_proposed={k:parameter_hash(getattr(proposed.ppo,k)) for k in ('actor','value')}
 del saved,proposed;gc.collect();torch.cuda.empty_cache()
 CHECKS['R3']={}
 for name in NAMES:
  saved=load_typed(WORK/'runs'/name/'seed_4101/checkpoints/episode-0000-initial.pt',binding,name)
  learner=UpperLearner(4101,device,config);learner.load_state_dict(saved['learner'])
  hashes={k:parameter_hash(getattr(learner.ppo,k)) for k in ('actor','value')}
  CHECKS['R3'][name]=dict(hashes=hashes,proposed_hashes=initial_proposed,equal={k:hashes[k]==initial_proposed[k] for k in hashes})
  del saved,learner;gc.collect();torch.cuda.empty_cache()
 write(OUT/'checks.json',CHECKS)
 tables={}
 for key in panels:
  tables[key]=[]
  for mode in MODES:
   m=summaries[key]['by_mode'][mode];windows=list(raw[key][mode].values());e=m['episodes']
   row=dict(internal=mode,label=LABELS[mode],truncated_delay_s=m['truncated_delay_s'],stage_delay_s=m['stage_delay_s'],cost_per_s=m['cost_per_s'],unfinished_per_window=m['unfinished_parents']/e,energy_j_per_window={k:m.get('energy_j',{}).get(k)/e if k in m.get('energy_j',{}) else None for k in ('upload','cpu','u2u','propulsion')},final_distance_m=float(np.mean([w['metrics']['terminal_mobility']['final_mean_terminal_to_serving_uav_m'] for w in windows])),solver_calls=sum(w['diagnostics'].get('solver_calls',0) for w in windows) if mode in ('moving_cvx','shared_upper_cvx') else None,fallback_calls=sum(w['diagnostics'].get('fallback_calls',0) for w in windows) if mode in ('moving_cvx','shared_upper_cvx') else None)
   tables[key].append(row)
 comparisons=[]
 def add_comparison(panel,candidate,reference,crows,rrows):
  result=paired(crows,rrows)
  comparisons.append(dict(panel=panel,candidate=candidate,reference=reference,**{k:result[k] for k in ('scenarios','wins','candidate_mean_s','reference_mean_s','relative_improvement')},candidate_unfinished_per_episode=unfinished_per_episode(crows),reference_unfinished_per_episode=unfinished_per_episode(rrows)))
 for key in panels:
  p=summaries[key]['per_scenario']
  for mode in ('moving_local','hover_equal','moving_equal','moving_cvx'):add_comparison(key,'moving',mode,p['moving'],p[mode])
  add_comparison(key,'moving_equal','hover_equal',p['moving_equal'],p['hover_equal'])
  add_comparison(key,'moving_cvx','frozen_hover_cvx',p['moving_cvx'],frozen['per_scenario']['hover_cvx'])
  for c,r in (('moving_local','hover_local'),('shared_upper_equal','moving_equal'),('shared_upper_cvx','moving_cvx')):add_comparison(key,c,r,p[c],p[r])
 for mode in MODES:add_comparison('B_vs_A',mode,mode,summaries['B']['per_scenario'][mode],summaries['A']['per_scenario'][mode])
 write(OUT/'comparisons.json',dict(tables=tables,comparisons=comparisons))
 probe=[];probe_u=[];probe_t=[];probe_c=[]
 for sid in range(1,21):
  with np.load(R2/f'windows/N400_hover_{sid:02d}.npz') as z:
   probe.append(z['obs'][:180]);probe_u.append(z['uav'][:180,:,:2]);probe_t.append(z['term'][:180,:,:2]);probe_c.append(z['centroid'][:180])
 X=np.concatenate(probe);U=np.concatenate(probe_u);T=np.concatenate(probe_t).reshape(3600,3,13,2);C=np.concatenate(probe_c).reshape(3600,3,2)
 require(UPPER_DIM==729 and X.shape==(3600,729),'probe_shape',list(X.shape))
 def distance_delta(displacements):
  before=np.linalg.norm(T-U[:,:,None,:],axis=-1).mean(axis=-1)
  after=np.linalg.norm(T-(U+displacements)[:,:,None,:],axis=-1).mean(axis=-1)
  return float((after-before).mean())
 centroid_delta=distance_delta(request_displacement(C.reshape(3600,6)))
 probe_rows=[];phases=upper_phases();snapshot_anchor_checks={}
 for name in NAMES:
  initial_vector=None;previous_vector=None;snapshot_anchor_checks[name]={}
  for n in (0,*range(25,201,25)):
   run=WORK/'runs'/name/'seed_4101'
   if n==0:payload=load_typed(run/'checkpoints/episode-0000-initial.pt',binding,name)
   else:
    path=run/f'snapshots/episode-{n:04d}.pt';side=read(path.with_suffix('.json'))
    require(side['sha256']==digest(path),'snapshot_digest',dict(name=name,episode=n))
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require(payload['kind']=='network_snapshot_not_resumable' and payload['episode']==n and payload['identity']==binding,'snapshot_metadata',dict(name=name,episode=n))
   learner=UpperLearner(4101,device,config);learner.load_state_dict(payload['learner']);del payload
   actor_hash=parameter_hash(learner.ppo.actor)
   if n in (100,200):
    require(actor_hash==anchor_hashes[name][n],'snapshot_anchor_actor_hash',dict(name=name,episode=n));snapshot_anchor_checks[name][str(n)]=True
   vec=np.concatenate([p.detach().cpu().numpy().ravel().astype(np.float64) for p in learner.ppo.actor.parameters()])
   if initial_vector is None:initial_vector=vec.copy()
   with torch.no_grad():mean,logstd=learner.ppo.actor.decode(torch.as_tensor(X,dtype=torch.float32,device='cuda:0'))
   mean=mean.cpu().numpy().astype(np.float64);logstd=logstd.cpu().numpy().astype(np.float64)
   require(mean.shape==logstd.shape==(3600,6),'probe_forward_shape',dict(name=name,episode=n))
   disp=request_displacement(np.tanh(mean));norm=np.linalg.norm(disp,axis=-1);avg=disp.mean(0);denom=norm.mean(0)
   drift=np.divide(np.linalg.norm(avg,axis=-1),denom,out=np.zeros(3),where=denom>0)
   row=dict(baseline=name,episode=n,global_episode=phases[n-1].global_episode if n else 300,kind='snapshot' if n else 'full',ppo_actor_hash=actor_hash,l2_from_initial=float(np.linalg.norm(vec-initial_vector)),l2_from_previous=float(np.linalg.norm(vec-previous_vector)) if previous_vector is not None else None,displacement_mean_m=float(norm.mean()),displacement_p50_m=float(np.percentile(norm,50)),displacement_p90_m=float(np.percentile(norm,90)),stationary_fraction=float((norm<1).mean()),logstd_mean=float(logstd.mean()),mean_state_std=float(mean.std(axis=0).mean()),mean_abs=float(np.abs(mean).mean()),drift_ratio_per_uav=[float(drift[j]) if denom[j]>0 else None for j in range(3)],mean_displacement_vector_per_uav_m=avg.tolist(),centroid_direction=direction_metrics(disp,C),unweighted_terminal_centroid_direction=direction_metrics(disp,T.mean(axis=2)-U),distance_delta_m=distance_delta(disp),centroid_distance_delta_m=centroid_delta)
   probe_rows.append(row);previous_vector=vec.copy()
   del learner,mean,logstd,vec;gc.collect();torch.cuda.empty_cache()
   print('PROBE',name,n,row['global_episode'],flush=True)
 proposed_rows=[]
 for r in read(R2/'d1_actor_drift.json')['rows']:
  row={k:v for k,v in r.items() if k!='sac_actor_hash'};row.update(baseline='Proposed',global_episode=r['episode']);proposed_rows.append(row)
 all_probe=proposed_rows+probe_rows
 write(OUT/'probe_baselines.json',dict(probe_shape=list(X.shape),snapshot_anchor_checks=snapshot_anchor_checks,rows=all_probe,off_policy_note='探针状态来自 Proposed-400 悬停轨迹，对基线属于离策略探针（推断其方向指标仍有参考价值）'))
 with (OUT/'probe_baselines.csv').open('w',encoding='utf-8',newline='') as f:
  fields=list(probe_rows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in all_probe:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
 CHECKS['probe_snapshot_anchor_checks']=snapshot_anchor_checks
 ppo={}
 for name in NAMES:
  rows=[]
  with (WORK/'runs'/name/'seed_4101/learning.jsonl').open(encoding='utf-8') as f:
   for line in f:
    r=json.loads(line)
    if 'ppo' not in r:continue
    accepted=[x for x in r['ppo'] if x['accepted']];kls=[x['kl'] for x in accepted]
    rows.append(dict(episode=r['episode'],proposed_global_episode=r['proposed_global_episode'],attempt_id=r.get('attempt_id'),accepted_epochs=len(accepted),epochs=len(r['ppo']),accepted_kl_mean=float(np.mean(kls)) if kls else None,accepted_kl_max=max(kls) if kls else None,has_rejected_epoch=any(not x['accepted'] for x in r['ppo'])))
  count=sum(r['accepted_epochs'] for r in rows);ks=[]
  with (WORK/'runs'/name/'seed_4101/learning.jsonl').open(encoding='utf-8') as f:
   for line in f:ks.extend(x['kl'] for x in json.loads(line).get('ppo',[]) if x['accepted'])
  ppo[name]=dict(batches=len(rows),expected_40=len(rows)==40,accepted_epochs=count,accepted_kl_mean=float(np.mean(ks)) if ks else None,accepted_kl_max=max(ks) if ks else None,has_rejected_epoch=any(r['has_rejected_epoch'] for r in rows),rows=rows)
 write(OUT/'ppo_log_summary.json',ppo)
 process_times={p.stem:read(p) for p in (WORK/'logs').glob('*.process.json')}
 episode_times={}
 for name in NAMES:
  with (WORK/'runs'/name/'seed_4101/train_metrics.jsonl').open(encoding='utf-8') as f:values=[json.loads(l)['wall_seconds'] for l in f]
  episode_times[name]=dict(episodes=len(values),mean_s=float(np.mean(values)),median_s=float(np.median(values)),p90_s=float(np.percentile(values,90)))
 CHECKS['timing']=dict(processes=process_times,episodes=episode_times)
 for key,path in panels.items():
  dest=OUT/'panels'/key;dest.mkdir(parents=True)
  for file in ('summary_deterministic.json','comparison_deterministic.md'):shutil.copyfile(path/file,dest/file)
 import matplotlib
 (OUT/'mplconfig').mkdir(exist_ok=True);matplotlib.get_cachedir=lambda:str(OUT/'mplconfig');matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
 for name in ('Proposed',*NAMES):
  rows=sorted([r for r in all_probe if r['baseline']==name and r['global_episode']>=300],key=lambda r:r['global_episode'])
  axes[0].plot([r['global_episode'] for r in rows],[r['displacement_mean_m'] for r in rows],'o-',label=name)
  axes[1].plot([r['global_episode'] for r in rows],[r['centroid_direction']['mean'] for r in rows],'o-',label=name)
 axes[0].set_ylabel('Mean requested displacement (m)');axes[1].set_ylabel('Mean cosine to centroid');axes[1].set_xlabel('Proposed global episode')
 for ax in axes:ax.grid(alpha=.3);ax.legend()
 fig.tight_layout();fig.savefig(OUT/'probe_baselines.png',dpi=160);plt.close(fig)
 for root in (ROOT,Path('D:/HRL_UAV_V2')):require(not subprocess.check_output(['git','status','--porcelain'],cwd=root).strip(),'git_clean_after',str(root))
 CHECKS['git_clean_before_after']=True;CHECKS['analysis_wall_seconds']=time.perf_counter()-start;write(OUT/'checks.json',CHECKS)
 make_report(tables,comparisons,all_probe,ppo,process_times,episode_times)
 print('ANALYSIS_COMPLETE',CHECKS['analysis_wall_seconds'],flush=True)

def make_report(tables,comparisons,probes,ppo,process_times,episode_times):
 def fmt(v):
  if v is None:return '缺'
  if isinstance(v,float):return f'{v:.6f}'
  if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
  return str(v)
 lines=['# v2.4.0 seed 4101 对照实验分析','', '## 检查结果','', '| 检查 | 结果 |','|---|---|','| 队列 | complete（2026-10-01 00:53:38 +08:00） |','| ROOT / 主工作树执行前后git status | 均为空 |','| H1 身份链、200轮完成、100/200 anchor预算与类型 | 全部通过 |','| H2 manifest、checkpoint摘要、窗口完整性 | A/B均通过，各320窗口 |','| H3 冻结hover_equal/hover_local完整metrics | A/B各80/80相等 |','| H3 冻结per_scenario | A/B两组均逐位相等 |','| 基线100/200快照与anchor的actor哈希 | NC/TOO/TFOO均相等 |',f'| 探针 | (3600,729)，cuda:0 |', '| R3 初始actor/value与Proposed哈希 | '+json.dumps({n:v['equal'] for n,v in CHECKS['R3'].items()},ensure_ascii=False)+' |','']
 lines.extend(['| 非拦截检查 | A | B |','|---|---|---|',f"| R1 Proposed完整指标（去掉指定6键）相等 | {CHECKS['R1']['A']['equal_windows']}/40 | {CHECKS['R1']['B']['equal_windows']}/40 |"])
 for mode in MODES:lines.append(f"| R2 {mode}两stream完整metrics相等 | {CHECKS['R2']['A'][mode]}/20 | {CHECKS['R2']['B'][mode]}/20 |")
 lines.append('')
 for key in ('A','B'):
  if CHECKS['R1'][key]['different_windows']:lines.extend([f'R1 {key}不等窗口及前10个差异键：', '```json',json.dumps(CHECKS['R1'][key]['different_windows'],ensure_ascii=False,indent=2),'```',''])
 def table(headers,rows):
  lines.append('| '+' | '.join(headers)+' |');lines.append('|'+'|'.join(['---']*len(headers))+'|')
  lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in rows);lines.append('')
 lines.extend(['## 面板指标','', 'A：Proposed-600 / 基线-200。B：Proposed-400 / 基线-100。基线100轮对应Proposed全局400轮，基线200轮对应全局590轮。','', '本节时延和四段时延取summary.by_mode池化指标；未完成及能耗按窗口归一。配对表使用20场景等权均值，因此均值口径与池化表不同。',''])
 for key in ('A','B'):
  lines.extend([f'### 面板{key}',''])
  table(['论文名/内部名','截断s','upload s','ready s','u2u s','cpu s','cost/s','未完成/窗口','终点距离m'],[[r['label']+'/'+r['internal'],r['truncated_delay_s'],*[r['stage_delay_s'][k] for k in ('upload','ready','u2u','cpu')],r['cost_per_s'],r['unfinished_per_window'],r['final_distance_m']] for r in tables[key]])
  table(['内部名','upload J/窗口','cpu J/窗口','u2u J/窗口','propulsion J/窗口','solver_calls','fallback_calls'],[[r['internal'],*[r['energy_j_per_window'][k] for k in ('upload','cpu','u2u','propulsion')],r['solver_calls'],r['fallback_calls']] for r in tables[key]])
 lines.extend(['## 配对比较','', 'unfinished_per_episode按每场景的episodes归一后求20场景合计；by_priority=[高优,低优]。正相对改善表示candidate更低。',''])
 for key in ('A','B','B_vs_A'):
  lines.extend([f'### {key}',''])
  table(['candidate/reference','scenarios','wins','candidate s','reference s','相对改善','candidate未完成归一化','reference未完成归一化'],[[r['candidate']+'/'+r['reference'],r['scenarios'],r['wins'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_improvement'],r['candidate_unfinished_per_episode'],r['reference_unfinished_per_episode']] for r in comparisons if r['panel']==key])
 lines.extend(['## 上层固定探针','', '探针状态来自 Proposed-400 悬停轨迹，对基线属于离策略探针（推断其方向指标仍有参考价值）。','', '下表为固定3600状态上的请求位移；不包含地图裁剪、安全投影或闭环执行。零位移时方向余弦和漂移比无定义，输出null。全字段、参数哈希、每UAV漂移比与平均向量见probe_baselines.json/.csv。',''])
 table(['组','baseline轮次','global轮次','L2 initial','位移均值m','p50 m','p90 m','<1m占比','centroid cos','cos>0','距离变化m'],[[r['baseline'],r['episode'],r['global_episode'],r['l2_from_initial'],r['displacement_mean_m'],r['displacement_p50_m'],r['displacement_p90_m'],r['stationary_fraction'],r['centroid_direction']['mean'],r['centroid_direction']['positive_fraction'],r['distance_delta_m']] for r in probes])
 lines.extend(['## PPO日志',''])
 table(['基线','batch数','accepted epochs','accepted KL均值','accepted KL最大','有rejected epoch'],[[name,ppo[name]['batches'],ppo[name]['accepted_epochs'],ppo[name]['accepted_kl_mean'],ppo[name]['accepted_kl_max'],ppo[name]['has_rejected_epoch']] for name in NAMES])
 all_global=sorted({r['proposed_global_episode'] for value in ppo.values() for r in value['rows']})
 def batch_cell(name,g):
  rs=[r for r in ppo[name]['rows'] if r['proposed_global_episode']==g]
  return '; '.join(f"{r['accepted_epochs']}/{r['epochs']}; {r['accepted_kl_mean']:.6f}/{r['accepted_kl_max']:.6f}; reject={r['has_rejected_epoch']}" for r in rs) if rs else '缺'
 lines.extend(['逐批单元：accepted/总epochs；accepted KL均值/最大；是否有rejected。attempt_id原值保存在ppo_log_summary.json。',''])
 table(['proposed_global_episode','NC','TOO','TFOO'],[[g,*[batch_cell(name,g) for name in NAMES]] for g in all_global])
 lines.extend(['## 耗时',''])
 table(['子进程','退出码','wall_seconds','结束时间'],[[name,value['exit_code'],value['wall_seconds'],value['ended_at']] for name,value in process_times.items()])
 table(['基线','训练轮数','每轮wall均值s','中位数s','p90 s'],[[name,value['episodes'],value['mean_s'],value['median_s'],value['p90_s']] for name,value in episode_times.items()])
 lines.append(f"本次只读分析墙钟：{CHECKS['analysis_wall_seconds']:.6f} s。")
 lines.extend(['','## 事实回答与推断','', '### Q1：基线上层是否移动、方向如何',''])
 for name in NAMES:
  rows=[r for r in probes if r['baseline']==name]
  for e in (100,200):
   r=next(r for r in rows if r['episode']==e)
   lines.append(f"- {name}-{e}（全局{r['global_episode']}）：相对initial参数L2={r['l2_from_initial']:.6f}，请求位移均值={r['displacement_mean_m']:.6f} m，centroid方向cos均值={fmt(r['centroid_direction']['mean'])}，cos>0占比={fmt(r['centroid_direction']['positive_fraction'])}，一步距离变化={r['distance_delta_m']:.6f} m。")
 lines.extend(['', '推断：固定探针中的方向余弦可用于描述对终端方向的对齐程度；这些离策略前向指标不等于基线闭环轨迹统计。','', '### Q2：TOO/CA、TFOO/hover_cvx、NC/hover_local',''])
 for key in ('B','A'):
  for c,r in (('moving_equal','hover_equal'),('moving_cvx','frozen_hover_cvx'),('moving_local','hover_local')):
   v=next(x for x in comparisons if x['panel']==key and x['candidate']==c and x['reference']==r)
   lines.append(f"- {key} {LABELS[c]}对{r}：candidate/reference={v['candidate_mean_s']:.6f}/{v['reference_mean_s']:.6f} s，胜场={v['wins']}/20，相对改善={v['relative_improvement']:.6f}；场景均值{'更低' if v['relative_improvement']>0 else '不更低'}。")
 lines.extend(['','### Q3：Proposed对独立基线',''])
 table(['面板','对照','胜场/20','相对改善'],[[r['panel'],LABELS[r['reference']],r['wins'],r['relative_improvement']] for key in ('B','A') for r in comparisons if r['panel']==key and r['candidate']=='moving' and r['reference'] in ('moving_local','hover_equal','moving_equal','moving_cvx')])
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
 try:analyze()
 except BaseException as error:
  write(OUT/'failure.json',dict(error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS))
  traceback.print_exc();sys.exit(1)
