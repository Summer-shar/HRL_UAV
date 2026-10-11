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
OUT=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_diag_r2')
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
 acc=PooledMetrics(); acc.add(metrics)
 return dict(n=n,mode=mode,sid=sid,check=check,per_scenario=acc.result())

if __name__=='__main__':
 print('INTERPRETER',dict(executable=sys.executable,prefix=sys.prefix,torch=torch.__version__,cuda=torch.version.cuda),flush=True)
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
  # New stages begin only after every D2 window has completed its replay check.
  import csv,gc
  from hrl_v2.gates import paired,unfinished_per_episode
  from hrl_v2.lite_observation import UPPER_DIM
  by_mode={}
  per_checks=[]
  for n in (400,600):
   stored=read(RUN/f'evaluations/episode-{n:04d}/summary_deterministic.json')['per_scenario']
   for mode in ('hover','moving','centroid'):
    rows={str(r['sid']):r['per_scenario'] for r in results if r['n']==n and r['mode']==mode}
    assert len(rows)==20
    by_mode[f'{n}/{mode}']=rows
    if mode!='centroid':
     for sid,row in rows.items():
      value=json.loads(json.dumps(row,allow_nan=False)); old=stored[mode][sid]
      if value!=old:
       difference={k:dict(new=value.get(k),old=old.get(k)) for k in sorted(set(value)|set(old)) if value.get(k)!=old.get(k)}
       write(OUT/f'FAIL_per_scenario_N{n}_{mode}_{sid}.json',difference)
       print('PER_SCENARIO_FAIL',n,mode,sid,json.dumps(difference,ensure_ascii=False),flush=True)
       raise AssertionError('per_scenario mismatch')
      per_checks.append(dict(n=n,mode=mode,sid=int(sid),equal=True))
  print('PER_SCENARIO_PASS',len(per_checks),flush=True)
  def request_displacement(actions):
   c=np.clip(np.asarray(actions).reshape(-1,3,2),-1,1)
   return 20*c/np.maximum(np.linalg.norm(c,axis=-1,keepdims=True),1)
  def direction_metrics(a,b):
   na=np.linalg.norm(a,axis=-1); nb=np.linalg.norm(b,axis=-1)
   mask=(na>1e-9)&(nb>1e-9)
   cos=np.sum(a[mask]*b[mask],axis=-1)/(na[mask]*nb[mask])
   return dict(valid=int(mask.sum()),mean=float(cos.mean()) if cos.size else None,positive_fraction=float((cos>0).mean()) if cos.size else None)
  derived={}; grouped={}
  for n in (400,600):
   for mode in ('hover','moving','centroid'):
    all_u=[]; all_t=[]; all_m=[]; all_req=[]; all_c=[]; paths=[]; final=[]; terminal_disp=[]; dist=[]; hover_zeros=[]
    for sid in range(1,21):
     with np.load(OUT/f'windows/N{n}_{mode}_{sid:02d}.npz') as z:
      print('SHAPES',n,mode,sid,{k:z[k].shape for k in ('uav','term','centroid','action','moves')},flush=True)
      u=z['uav'][:180,:,:2].copy(); t=z['term'][:180,:,:2].copy(); moves=z['moves'][:180,:,:2].copy()
      req=request_displacement(z['action'][:180]); c=z['centroid'][:180].reshape(180,3,2).copy()
     d=np.linalg.norm(t.reshape(180,3,13,2)-u[:,:,None,:],axis=-1).mean(axis=-1)
     norms=np.linalg.norm(moves,axis=-1)
     j=read(OUT/f'windows/N{n}_{mode}_{sid:02d}.json')
     row=dict(distance_time_mean_m=float(d.mean()),distance_k0_m=float(d[0].mean()),distance_k179_m=float(d[179].mean()),distance_time_mean_per_uav_m=d.mean(0).tolist(),path_length_per_uav_m=norms.sum(0).tolist(),mean_requested_displacement_m=float(np.linalg.norm(req,axis=-1).mean()),mean_executed_displacement_m=float(norms.mean()),stationary_fraction=float((norms<1).mean()),terminal_mobility=j['terminal_mobility'],hover_moves_all_zero=bool(np.all(moves==0)) if mode=='hover' else None,direction=direction_metrics(moves,c) if mode!='hover' else None)
     derived[f'{n}/{mode}/{sid}']=row
     all_u.append(u); all_t.append(t); all_m.append(moves); all_req.append(req); all_c.append(c); paths.append(norms.sum(0)); dist.append(d)
     final.append(j['terminal_mobility']['final_mean_terminal_to_serving_uav_m']); terminal_disp.append(j['terminal_mobility']['mean_displacement_m'])
     if mode=='hover': hover_zeros.append(row['hover_moves_all_zero'])
    rows=by_mode[f'{n}/{mode}']; vals=list(rows.values()); norms=np.linalg.norm(np.asarray(all_m),axis=-1)
    ds=np.asarray(dist)
    grouped[f'{n}/{mode}']=dict(n=n,mode=mode,scenarios=20,truncated_delay_s=float(np.mean([v['truncated_delay_s'] for v in vals])),stage_delay_s={k:float(np.mean([v['stage_delay_s'][k] for v in vals])) for k in ('upload','ready','u2u','cpu')},cost_per_s=float(np.mean([v['cost_per_s'] for v in vals])),unfinished_per_window=float(np.mean([v['unfinished_parents']/v['episodes'] for v in vals])),distance_time_mean_m=float(ds.mean()),distance_k0_m=float(ds[:,0].mean()),distance_k179_m=float(ds[:,-1].mean()),final_mean_terminal_to_serving_uav_m=float(np.mean(final)),terminal_mean_displacement_m=float(np.mean(terminal_disp)),path_length_per_uav_m=np.mean(paths,axis=0).tolist(),mean_requested_displacement_m=float(np.linalg.norm(np.asarray(all_req),axis=-1).mean()),mean_executed_displacement_m=float(norms.mean()),stationary_fraction=float((norms<1).mean()),direction=direction_metrics(np.asarray(all_m),np.asarray(all_c)) if mode!='hover' else None,hover_moves_all_zero=all(hover_zeros) if mode=='hover' else None)
  ref=read(RUN/'baseline_reference.json')['per_scenario']; comparisons=[]
  def compare(n,candidate,reference,crows,rrows):
   p=paired(crows,rrows)
   row=dict(n=n,candidate=candidate,reference=reference,**{k:p[k] for k in ('scenarios','wins','candidate_mean_s','reference_mean_s','relative_improvement')},candidate_unfinished_per_episode=unfinished_per_episode(crows),reference_unfinished_per_episode=unfinished_per_episode(rrows))
   comparisons.append(row)
  for n in (400,600):
   for c,r in (('moving','hover'),('centroid','hover'),('centroid','moving')): compare(n,c,r,by_mode[f'{n}/{c}'],by_mode[f'{n}/{r}'])
   for c in ('hover','moving','centroid'):
    for r in ('hover_equal','hover_cvx','hover_greedy'): compare(n,c,r,by_mode[f'{n}/{c}'],ref[r])
  d2=dict(per_scenario=by_mode,per_scenario_checks=per_checks,window_derived=derived,groups=grouped,comparisons=comparisons)
  write(OUT/'d2_summary.json',d2)
  print('D2_SUMMARY_PASS',flush=True)
  probe=[]; probe_u=[]; probe_t=[]; probe_c=[]
  for sid in range(1,21):
   with np.load(OUT/f'windows/N400_hover_{sid:02d}.npz') as z:
    probe.append(z['obs'][:180]); probe_u.append(z['uav'][:180,:,:2]); probe_t.append(z['term'][:180,:,:2]); probe_c.append(z['centroid'][:180])
  X=np.concatenate(probe); U=np.concatenate(probe_u); T=np.concatenate(probe_t).reshape(3600,3,13,2); C=np.concatenate(probe_c).reshape(3600,3,2)
  assert UPPER_DIM==729 and X.shape==(3600,UPPER_DIM)
  print('D1_PROBE',X.shape,flush=True)
  def distance_delta(displacements):
   before=np.linalg.norm(T-U[:,:,None,:],axis=-1).mean(axis=-1)
   after=np.linalg.norm(T-(U+displacements)[:,:,None,:],axis=-1).mean(axis=-1)
   return float((after-before).mean())
  centroid_delta=distance_delta(request_displacement(C.reshape(3600,6)))
  d1=[]; initial_vector=None; previous_vector=None
  nodes=[(0,False)]+[(n,True) for n in range(25,601,25)]+[(n,False) for n in (440,480,520,560)]
  nodes.sort()
  for n,is_snapshot in nodes:
   learner=learner_at(n,is_snapshot)
   vec=np.concatenate([p.detach().cpu().numpy().ravel().astype(np.float64) for p in learner.ppo.actor.parameters()])
   if initial_vector is None: initial_vector=vec.copy()
   with torch.no_grad():
    mean,logstd=learner.ppo.actor.decode(torch.as_tensor(X,dtype=torch.float32,device='cuda:0'))
   mean=mean.cpu().numpy().astype(np.float64); logstd=logstd.cpu().numpy().astype(np.float64)
   assert mean.shape==logstd.shape==(3600,6)
   disp=request_displacement(np.tanh(mean)); norm=np.linalg.norm(disp,axis=-1); avg=disp.mean(0); denom=norm.mean(0)
   drift=np.divide(np.linalg.norm(avg,axis=-1),denom,out=np.zeros(3),where=denom>0)
   row=dict(episode=n,kind='snapshot' if is_snapshot else 'full',ppo_actor_hash=parameter_hash(learner.ppo.actor),sac_actor_hash=parameter_hash(learner.sac.actor),l2_from_initial=float(np.linalg.norm(vec-initial_vector)),l2_from_previous=float(np.linalg.norm(vec-previous_vector)) if previous_vector is not None else None,displacement_mean_m=float(norm.mean()),displacement_p50_m=float(np.percentile(norm,50)),displacement_p90_m=float(np.percentile(norm,90)),stationary_fraction=float((norm<1).mean()),logstd_mean=float(logstd.mean()),mean_state_std=float(mean.std(axis=0).mean()),mean_abs=float(np.abs(mean).mean()),drift_ratio_per_uav=[float(drift[j]) if denom[j]>0 else None for j in range(3)],mean_displacement_vector_per_uav_m=avg.tolist(),centroid_direction=direction_metrics(disp,C),unweighted_terminal_centroid_direction=direction_metrics(disp,T.mean(axis=2)-U),distance_delta_m=distance_delta(disp),centroid_distance_delta_m=centroid_delta)
   d1.append(row); previous_vector=vec.copy()
   del learner,mean,logstd,vec; gc.collect(); torch.cuda.empty_cache()
   print('D1_DONE',n,row['kind'],flush=True)
  initial_hash=d1[0]['ppo_actor_hash']; lookup={r['episode']:r for r in d1}
  d1_checks=dict(probe_shape=list(X.shape),upper_dim=UPPER_DIM,ppo_hash_25_to_300_equals_initial=all(r['ppo_actor_hash']==initial_hash for r in d1 if 25<=r['episode']<=300),sac_hash_300_equals_400=lookup[300]['sac_actor_hash']==lookup[400]['sac_actor_hash'])
  write(OUT/'d1_actor_drift.json',dict(checks=d1_checks,rows=d1))
  with (OUT/'d1_actor_drift.csv').open('w',encoding='utf-8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(d1[0])); writer.writeheader()
   for row in d1: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
  ppo=[]
  with (RUN/'learning.jsonl').open(encoding='utf-8') as f:
   for line in f:
    row=json.loads(line)
    if 'ppo' not in row: continue
    epochs=row['ppo']; accepted=[r for r in epochs if r['accepted']]; kls=[r['kl'] for r in accepted]
    ppo.append(dict(global_episode=row['global_episode'],attempt_id=row.get('attempt_id'),epoch_count=len(epochs),accepted_epoch_count=len(accepted),accepted_kl_mean=float(np.mean(kls)) if kls else None,accepted_kl_max=max(kls) if kls else None,has_rejected_epoch=any(not r['accepted'] for r in epochs)))
  duplicate={str(e):[r['attempt_id'] for r in ppo if r['global_episode']==e] for e in sorted({r['global_episode'] for r in ppo}) if sum(r['global_episode']==e for r in ppo)>1}
  ppo_summary=dict(batch_count=len(ppo),batch_count_equals_40=len(ppo)==40,duplicate_global_episodes=duplicate,ppo_only=[r for r in ppo if 301<=r['global_episode']<=400],alternating=[r for r in ppo if r['global_episode']>=401])
  write(OUT/'ppo_log_summary.json',ppo_summary)
  plot_status={}
  try:
   import matplotlib
  except ImportError as exc:
   plot_status=dict(status='skipped',message=str(exc))
  else:
   (OUT/'mplconfig').mkdir(exist_ok=True)
   matplotlib.get_cachedir=lambda:str(OUT/'mplconfig')
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
   x=[r['episode'] for r in d1]
   axes[0].plot(x,[r['displacement_mean_m'] for r in d1],'o-'); axes[0].set_ylabel('Mean requested displacement (m)')
   axes[1].plot(x,[r['centroid_direction']['mean'] for r in d1],'o-'); axes[1].set_ylabel('Mean cosine to centroid')
   axes[2].plot(x,[r['distance_delta_m'] for r in d1],'o-',label='Actor'); axes[2].axhline(centroid_delta,color='gray',linestyle='--',label='Centroid'); axes[2].set_ylabel('One-step distance delta (m)'); axes[2].set_xlabel('Checkpoint episode'); axes[2].legend()
   for ax in axes: ax.grid(alpha=.3)
   fig.tight_layout(); fig.savefig(OUT/'d1_actor_drift.png',dpi=160); plt.close(fig)
   fig,axes=plt.subplots(2,2,figsize=(12,11))
   for sid,ax in enumerate(axes.flat,1):
    with np.load(OUT/f'windows/N400_hover_{sid:02d}.npz') as z: terms=z['term'][:,:,:2].copy()
    ax.scatter(terms[0,:,0],terms[0,:,1],marker='.',color='gray',label='Terminal start'); ax.scatter(terms[180,:,0],terms[180,:,1],marker='x',color='black',label='Terminal end')
    for n,mode,color in ((400,'moving','tab:blue'),(600,'moving','tab:red'),(400,'centroid','tab:green')):
     with np.load(OUT/f'windows/N{n}_{mode}_{sid:02d}.npz') as z: u=z['uav'][:,:,:2].copy()
     for j in range(3): ax.plot(u[:,j,0],u[:,j,1],color=color,linestyle=('-','--',':')[j],label=f'{mode}-{n}' if j==0 else None)
    ax.set_title(f'Scenario {sid}'); ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
   fig.tight_layout(); fig.savefig(OUT/'d2_trajectories.png',dpi=160); plt.close(fig)
   plot_status=dict(status='created',files=['d1_actor_drift.png','d2_trajectories.png'])
  wall=time.perf_counter()-start
  write(OUT/'completion.json',dict(wall_seconds=wall,windows=len(results),plot_status=plot_status))
  def fmt(v):
   if v is None: return '缺失'
   if isinstance(v,float): return f'{v:.6f}'
   if isinstance(v,(dict,list)): return json.dumps(v,ensure_ascii=False,separators=(',',':'))
   return str(v)
  lines=['# v2.4.0 seed 4101 Proposed D1/D2 第2次执行','', '这是第2次执行；第1次停在D2，原因是prompt写错的PooledMetrics链式调用。本次对原脚本的改动见changes_vs_attempt1.diff。','', '## 预检与解释器','', '| 检查 | 结果 |','|---|---|', '| HEAD | 25d59ff9f7c3e27c060c63d462235a30ce28930a |', '| 执行前src状态 | 空 |', '| 新OUT执行前 | 不存在 |', '| 执行前相关Python进程 | 未发现 |',f'| 导入路径 | {hrl_v2.__file__} |','| 主进程及worker设备 | cuda:0 |','| contract identity / 三份manifest | 相等 / 相等 |','| 400、600 pending哈希 | 均与原评估identity相等 |','| 400、600 snapshot与pending的SAC/PPO actor哈希 | 均相等 |',f'| sys.executable | {sys.executable} |',f'| sys.prefix | {sys.prefix} |',f'| torch / CUDA | {torch.__version__} / {torch.version.cuda} |', '| 学习更新 / replay写入 | 0 / 0 |','', '## D2逐位核对','', '| 节点/模式 | 窗口 | identity相等 | 完整metrics相等 | per_scenario相等 |','|---|---|---|---|---|']
  for n in (400,600):
   for mode in ('hover','moving'):
    rr=[r for r in results if r['n']==n and r['mode']==mode]
    lines.append(f"| {n}/{mode} | {len(rr)} | {sum(r['check']['identity_equal'] for r in rr)}/20 | {sum(r['check']['metrics_equal'] for r in rr)}/20 | 20/20 |")
  match=[r for r in results if r['n']==400 and r['mode']=='hover']
  lines.extend(['',f"400 hover对300 hover：event_sha256相等 {sum(r['check']['matches_300']['event_sha256'] for r in match)}/20；truncated_delay_s相等 {sum(r['check']['matches_300']['truncated_delay_s'] for r in match)}/20。", '',f'全部{len(results)}个窗口：records与moves长度均为181；派生统计只取0–179。距离仅使用(x,y)。数组shape逐窗口见run_log.txt。', '', '## D2指标',''])
  def table(headers,rows):
   lines.append('| '+' | '.join(headers)+' |'); lines.append('|'+'|'.join(['---']*len(headers))+'|')
   lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in rows); lines.append('')
  table(['节点/模式','截断s','upload s','ready s','u2u s','cpu s','cost/s','未完成/窗口'],[[k,g['truncated_delay_s'],*[g['stage_delay_s'][s] for s in ('upload','ready','u2u','cpu')],g['cost_per_s'],g['unfinished_per_window']] for k,g in grouped.items()])
  table(['节点/模式','时间平均距离m','k0距离m','k179距离m','终点距离m','终端位移m'],[[k,*[g[v] for v in ('distance_time_mean_m','distance_k0_m','distance_k179_m','final_mean_terminal_to_serving_uav_m','terminal_mean_displacement_m')]] for k,g in grouped.items()])
  table(['节点/模式','每UAV路径长度m','请求位移均值m','执行位移均值m','静止占比','方向cos均值','cos>0占比','hover moves全0'],[[k,g['path_length_per_uav_m'],g['mean_requested_displacement_m'],g['mean_executed_displacement_m'],g['stationary_fraction'],g['direction']['mean'] if g['direction'] else None,g['direction']['positive_fraction'] if g['direction'] else None,g['hover_moves_all_zero']] for k,g in grouped.items()])
  lines.extend(['方向仅统计双方范数均大于1e-9的决策；hover跳过。静止定义为执行位移小于1 m。','', '## 配对','', '未完成数为gates.unfinished_per_episode返回的20场景归一化合计；不是单窗口均值。高/低优顺序为[高,低]。不报告paired的原始未完成数合计。',''])
  table(['节点','candidate/reference','wins/20','candidate s','reference s','改善比例','candidate未完成归一化','reference未完成归一化'],[[r['n'],r['candidate']+'/'+r['reference'],r['wins'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_improvement'],r['candidate_unfinished_per_episode'],r['reference_unfinished_per_episode']] for r in comparisons])
  lines.extend(['## D1固定探针','', f'X.shape={X.shape}；UPPER_DIM={UPPER_DIM}。0025–0300 PPO actor哈希全部等于initial：{d1_checks["ppo_hash_25_to_300_equals_initial"]}；0300与0400 SAC actor哈希相等：{d1_checks["sac_hash_300_equals_400"]}。',''])
  table(['轮次','L2 initial','L2 previous','位移均值m','p50 m','p90 m','<1m占比','logstd均值','mean状态std','abs(mean)'],[[r['episode'],*[r[k] for k in ('l2_from_initial','l2_from_previous','displacement_mean_m','displacement_p50_m','displacement_p90_m','stationary_fraction','logstd_mean','mean_state_std','mean_abs')]] for r in d1])
  table(['轮次','漂移比(UAV0/1/2)','centroid cos','centroid cos>0','非加权质心cos','非加权质心cos>0','距离变化m','centroid距离变化m'],[[r['episode'],r['drift_ratio_per_uav'],r['centroid_direction']['mean'],r['centroid_direction']['positive_fraction'],r['unweighted_terminal_centroid_direction']['mean'],r['unweighted_terminal_centroid_direction']['positive_fraction'],r['distance_delta_m'],r['centroid_distance_delta_m']] for r in d1])
  table(['轮次','每UAV平均位移向量m'],[[r['episode'],r['mean_displacement_vector_per_uav_m']] for r in d1])
  lines.extend(['参数哈希完整值见d1_actor_drift.json及.csv。距离变化为一步请求位移后的几何距离变化。零位移的漂移比分母为0，记为null；initial没有前一checkpoint，L2 previous记为缺失。','', '## PPO日志','',f'批次数={len(ppo)}；等于40：{len(ppo)==40}；重复global_episode及attempt_id：{fmt(duplicate)}。',''])
  for label,key in (('PPO-only 301–400','ppo_only'),('交替阶段401–600','alternating')):
   lines.extend([f'### {label}',''])
   table(['global_episode','accepted epochs','epochs','accepted KL均值','accepted KL最大','有accepted=false','attempt_id'],[[r['global_episode'],r['accepted_epoch_count'],r['epoch_count'],r['accepted_kl_mean'],r['accepted_kl_max'],r['has_rejected_epoch'],r['attempt_id']] for r in ppo_summary[key]])
  lines.extend(['## 图与墙钟','', f'作图检查：{fmt(plot_status)}。',f'总墙钟时间：{wall:.6f} s。',''])
  (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
  print('COMPLETE',wall,flush=True)
 except BaseException:
  traceback.print_exc()
  write(OUT/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start))
  sys.exit(1)


