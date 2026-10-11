from d_guard import install
install()
"""Read-only observation and frozen evaluations; policies and service kernel are unchanged."""
from j_common import *
import copy, multiprocessing, psutil, torch
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from hrl_v2.checkpoint import identity,load,json_digest
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.upper_baselines import UpperLearner,load_typed,fixed_upper_policy
from hrl_v2.policy import Policy
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics,stage_residence
from hrl_v2.run import read_config
from hrl_v2.evaluate import gate_manifest
from hrl_v2.timing import ClockConfig
from j_eval import distribution,frozen_reference
from d_compare import exact_equal as whole_metrics_equal
from d_pool import run_tasks
from d_storage import load_evaluation
CACHE={}


def exclusive(path,data):
 p=output_path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(data,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')


def safe_ratio(a,b):return float(a/b) if b else None

class Probe:
 def __init__(self,a):
  e=a.env;self.start=e.physics.uav_coords.copy();self.prev=self.start.copy();self.prev_users=e.physics.user_coords.copy()
  self.period=round(5/e.dt);self.count_parents=0;self.count_fragments=0;self.upload_previous=set();self.samples=[];self.total=np.zeros(3);self.window_begin=0.
  self.window_parents=set();self.reset()
 def reset(self):
  self.move=np.zeros(3);self.nearest=np.zeros(39);self.ticks=0;self.queue_sum=np.zeros((2,3));self.queue_max=np.zeros((2,3),dtype=int)
  self.upload_tasks={};self.birth_dist=[[],[]];self.assign_cycles=np.zeros((2,3,3));self.assign_fractions=np.zeros((2,3,3));self.fragment_count=np.zeros((2,3,3),dtype=int)
  self.cost=0.;self.energy=dict(upload=0.,ready=0.,u2u=0.,cpu=0.,propulsion=0.)
 def __call__(self,a,reading):
  e=a.env;current=e.physics.uav_coords.copy();users=e.physics.user_coords
  distances=np.linalg.norm(self.prev_users[:,None,:]-self.prev[None,:,:],axis=2)
  self.nearest+=distances.min(axis=1);self.ticks+=1
  newids=range(self.count_parents,len(e.parents))
  service_ids=self.upload_previous.copy()
  for pid in newids:
   p=e.parents[pid];ar=p.arrival;service_ids.add(pid);self.window_parents.add(pid)
   self.birth_dist[ar.priority].append(float(distances[ar.user_id,ar.source]))
  self.count_parents=len(e.parents)
  for pid in service_ids:
   ar=e.parents[pid].arrival;v=float(distances[ar.user_id,ar.source])
   rec=self.upload_tasks.setdefault(pid,[0.,0,ar.priority]);rec[0]+=v;rec[1]+=1
  self.upload_previous={pid for q in e.upload for pid in q}
  step=np.linalg.norm(current-self.prev,axis=1);self.move+=step;self.total+=step
  self.prev=current;self.prev_users=users.copy()
  lengths=np.array([[len(q) for q in row] for row in e.compute]);self.queue_sum+=lengths;self.queue_max=np.maximum(self.queue_max,lengths)
  for fid in range(self.count_fragments,len(e.fragments)):
   f=e.fragments[fid];idx=(f.priority,f.source,f.destination);self.assign_cycles[idx]+=f.compute_cycles;self.fragment_count[idx]+=1
   self.assign_fractions[idx]+=f.compute_cycles/e.parents[f.parent_id].arrival.compute_cycles
  self.count_fragments=len(e.fragments)
  self.cost+=reading['cost']
  for k,v in reading['energy_j'].items():self.energy[k]+=v
  if e.step_index%self.period:return
  end=e.time_s
  task_means=[[r[0]/r[1] for r in self.upload_tasks.values() if r[2]==pr] for pr in (0,1)]
  active=list(e._active_parents.values());stage_counts={k:np.zeros((2,3),dtype=int) for k in ('upload','ready','u2u','cpu')}
  for p in active:
   ar=p.arrival
   if p.upload_finish_s is None:stage_counts['upload'][ar.priority,ar.source]+=1
   elif p.route_time_s is None:stage_counts['ready'][ar.priority,ar.source]+=1
  for f in e._active_fragments.values():stage_counts[f.stage][f.priority,f.destination]+=1
  residence={k:0. for k in ('upload','ready','u2u','cpu')};res_by_priority=[dict(residence),dict(residence)]
  for pid in self.window_parents:
   p=e.parents[pid]
   for k,v in stage_residence(e,p,self.window_begin,end).items():res_by_priority[p.arrival.priority][k]+=v
  for k in residence:residence[k]=10*res_by_priority[0][k]+res_by_priority[1][k]
  cycles_dest=self.assign_cycles.sum(axis=(0,1));fraction_dest=self.assign_fractions.sum(axis=(0,1));total_cycles=float(cycles_dest.sum());total_fractions=float(fraction_dest.sum())
  local=sum(self.assign_cycles[:,i,i].sum() for i in range(3));local_f=sum(self.assign_fractions[:,i,i].sum() for i in range(3))
  self.samples.append(dict(begin_s=self.window_begin,end_s=end,executed_distance_m=self.move.tolist(),distance_from_start_m=np.linalg.norm(current-self.start,axis=1).tolist(),movement_ge_19_5m=(self.move>=19.5).tolist(),uav_positions_m=current.tolist(),user_to_nearest_uav_mean_m_per_user=(self.nearest/self.ticks).tolist(),upload_task_distance_mean_m=[float(np.mean(v)) if v else None for v in task_means],upload_task_count=[len(v) for v in task_means],birth_task_distance_mean_m=[float(np.mean(v)) if v else None for v in self.birth_dist],birth_task_count=[len(v) for v in self.birth_dist],cpu_queue_mean=(self.queue_sum/self.ticks).tolist(),cpu_queue_max=self.queue_max.tolist(),assigned_cycles=self.assign_cycles.tolist(),assigned_task_fractions=self.assign_fractions.tolist(),assigned_fragment_count=self.fragment_count.tolist(),compute_cycles_share_per_uav=[safe_ratio(v,total_cycles) for v in cycles_dest],compute_task_share_per_uav=[safe_ratio(v,total_fractions) for v in fraction_dest],routing_cycles_share=dict(local=safe_ratio(local,total_cycles),forward=safe_ratio(total_cycles-local,total_cycles),other=0. if total_cycles else None),routing_task_share=dict(local=safe_ratio(local_f,total_fractions),forward=safe_ratio(total_fractions-local_f,total_fractions),other=0. if total_fractions else None),unfinished_high_low=[sum(p.arrival.priority==pr for p in active) for pr in (0,1)],backlog_by_stage_priority_uav={k:v.tolist() for k,v in stage_counts.items()},stage_residence_task_s=res_by_priority,weighted_stage_residence_task_s=residence,cost=self.cost,cost_per_s=self.cost/5,energy_j=self.energy.copy()))
  self.window_begin=end;self.window_parents=set(e._active_parents);self.reset()
 def result(self):
  equal(len(self.samples),180,'expected_180_five_second_windows')
  return dict(definitions=dict(motion='Executed physical-tick path length and end-of-5s displacement from reset; threshold >=19.5m per UAV interval.',upload_target='Read-only parent.arrival.source; fixed user-region upload association.',upload_distance='Within each 5s interval: each parent waiting/uploading during a tick contributes its mean start-of-tick horizontal distance; parents then weighted equally, priorities separate. A parent spanning intervals appears once in each such interval.',birth_distance='Each newly admitted parent counted once at first service tick start, priorities separate.',nearest='Every user equally, nearest UAV horizontal distance averaged across start-of-tick positions within 5s.',cpu_queue='Post-tick queued fragment count, per priority/UAV; per-5s mean and max.',compute_task_share='Allocated compute fraction summed per parent (fragment.compute_cycles / parent.compute_cycles); split parents can contribute fractional shares to multiple UAVs.',routing='Shares by allocated cycles and by per-parent compute fractions; local source==destination, forward source!=destination, other zero in this 3-UAV environment.',backlog='Upload/ready count parents by upload source; u2u/cpu count fragments by compute destination. These counts have different units.',stage_residence='Existing pure stage_residence function over parents active/arrived within each 5s interval; same censored last-finishing-branch definition. Each interval independently censored.'),initial_positions_m=self.start.tolist(),final_positions_m=self.prev.tolist(),executed_total_m_per_uav=self.total.tolist(),samples_5s=self.samples)

def state_hashes(l):
 result={}
 for owner,names in (('ppo',('actor','value')),('sac',('actor','q1','q2','target1','target2'))):
  if hasattr(l,owner):
   for name in names:result[owner+'.'+name]=parameter_hash(getattr(getattr(l,owner),name))
 return result

def worker_init():set_below_normal();assert_imports()

def window(t):
 begin=time.perf_counter();c=contract();binding=c['identity'];assert_imports()
 check(t['seed'] in (4103,4104) and t['stream'] in (1,2) and 1<=t['scenario']['scenario']<=20,'only_authorized_seed_stream_scene')
 config=read_config(CONFIG,WT);scene=read(SCENE);device=configure();equal(str(device),'cuda:0','device');equal(torch.get_num_threads(),10,'same_eval_threads')
 key=(t['upper'],t['lower'],t['upper_kind'])
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  if key not in CACHE:
   CACHE.clear();gc.collect();torch.cuda.empty_cache()
   u=load_evaluation(t['upper'],binding,'TFOO') if t['upper_kind']=='TFOO' else load_evaluation(t['upper'],binding)
   l=(UpperLearner if t['upper_kind']=='TFOO' and t['route']=='cvx' else Learner)(t['seed'],device,config=config)
   state=copy.deepcopy(u['learner'])
   if t['lower'] and t['lower']!=t['upper']:
    lower=load_evaluation(t['lower'],binding)
    if t['upper_kind']=='TFOO':
     combined=copy.deepcopy(lower['learner']);combined['ppo']=state['ppo'];state=combined
    else:state['sac']=lower['learner']['sac']
    del lower
   l.load_state_dict(state)
   for owner in ('ppo','sac'):
    if owner in state:
     live=l.state_dict()[owner]
     for field,value in state[owner].items():equal(live[field],value,'loaded_'+owner+'_'+field)
   rng={k:u[k] for k in ('global_cpu_rng','global_cuda_rng') if k in u};del u,state;gc.collect();CACHE[key]=(l,rng)
  l,rng=CACHE[key]
  if 'global_cpu_rng' in rng:torch.set_rng_state(rng['global_cpu_rng'])
  if 'global_cuda_rng' in rng:torch.cuda.set_rng_state_all(rng['global_cuda_rng'])
  before=state_hashes(l);counts={owner:dict(counts=dict(getattr(l,owner).counts),version=getattr(l,owner).version) for owner in ('ppo','sac') if hasattr(l,owner)}
  sc=t['scenario'];a=PhysicsAdapter(scene,timing=ClockConfig.from_config(config));a.reset(**{k:v for k,v in sc.items() if k.endswith('_seed')})
  streams=Streams(t['seed'],episode=sc['scenario'],split='dev',policy_stream=t['stream'])
  if t['route']=='cvx':policy=fixed_upper_policy(l.ppo,scene,streams,'cvx',stochastic=False,mode='moving')
  else:policy=Policy(l.sac.actor,l.ppo.actor,None,streams,mode='moving',stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
  probe=Probe(a);metrics=collect_metrics(a,policy,identity=t['identity'],on_tick=probe,reward_scale=config['objective']['reward_scale'])
  after=state_hashes(l);equal(before,after,'all_parameters_unchanged');equal(counts,{owner:dict(counts=dict(getattr(l,owner).counts),version=getattr(l,owner).version) for owner in counts},'update_counts_unchanged')
  equal({k:metrics[k] for k in t['reference']['hashes']},t['reference']['hashes'],'frozen_external_events')
  anchor_details=[]
  for anchor_info in t['anchors']:
   old=read(anchor_info['path']);anchor_details.append(dict(reference=anchor_info['path'],**whole_metrics_equal(metrics,old['metrics'],'anchor_'+t['label']+'_'+str(sc['scenario'])+'_'+str(t['stream']),record=True)))
  extra={}
  if t['route']=='cvx':rr=policy.lower_override.rule.records;extra=dict(solver_calls=len(rr),fallback_calls=sum(r.get('fallback',False) for r in rr))
  result=dict(identity=t['identity'],metrics=metrics,observations=probe.result(),diagnostics=extra,anchor_checks=dict(passed=True,count=len(t['anchors']),comparisons=anchor_details),parameter_hash_check=dict(before=before,after=after,passed=True),learning_updates=0,replay_writes=0,rng_source=t['upper'],wall_seconds=time.perf_counter()-begin)
 return result

def pool(tasks):
 return run_tasks(window,tasks,initializer=worker_init,timeout_seconds=contract()['task_timeout']['task_timeout_seconds'],max_workers=int(os.environ.get('D_EVAL_WORKERS','6')),resource_check=guard_resources)


def group(g,scenario_ids=None,output_root=None):
 start=time.perf_counter();root=output_path((Path(output_root) if output_root else CPIL/'groups')/g['label']);check(not root.exists() or not any(root.iterdir()),'new_or_empty_group_directory',str(root));root.mkdir(parents=True,exist_ok=True)
 print('GROUP_PREPARE_REFERENCE',g['label'],flush=True);refs=lean_reference();print('GROUP_REFERENCE_READY',g['label'],flush=True);manifest=gate_manifest(20);tasks=[];anchor_windows=[]
 for p in g['anchors']:anchor_windows.append({(w['scenario'],w['stream']):w for w in read(p)['window_files']})
 scenario_ids=list(range(1,21)) if scenario_ids is None else list(scenario_ids)
 for sc in manifest['scenarios']:
  if sc['scenario'] not in scenario_ids:continue
  for stream in (1,2):
   key=f'{sc["scenario"]:02d}_{stream}';ident=dict(protocol='jointD_development_evaluation',group=g['label'],seed=g['seed'],scenario=sc,stream=stream,split='dev',rollout_seconds=900,upper_policy='deterministic',lower_policy='deterministic')
   tasks.append(dict(g,scenario=sc,stream=stream,identity=ident,reference=refs['windows'][key],anchors=[x[(sc['scenario'],stream)] for x in anchor_windows],output=str(root/'windows'/f'{key}.json')))
 print('GROUP_WORKERS_START',g['label'],len(tasks),flush=True);values=pool(tasks)
 memory={(v['identity']['scenario']['scenario'],v['identity']['stream']):v for v in values}
 return summarize_group(g,root,scenario_ids,start,memory)

def summarize_group(g,root,scenario_ids,start,memory):
 acc=PooledMetrics();per={};files=[];motion=[];displacement=[];nearest=[];task_sum=np.zeros(2);task_count=np.zeros(2);birth_sum=np.zeros(2);birth_count=np.zeros(2);cycles=np.zeros((2,3,3));fractions=np.zeros((2,3,3));qsum=np.zeros((2,3));qmax=np.zeros((2,3));windows=0
 for sid in scenario_ids:
  one=PooledMetrics();previous=None;prevobs=None
  for st in (1,2):
   p=root/'scenarios'/f'{sid:02d}_{st}.json';d=memory[(sid,st)]
   if previous is not None:whole_metrics_equal(d['metrics'],previous,'two_streams_metrics');whole_metrics_equal(d['observations'],prevobs,'two_streams_observations')
   previous=d['metrics'];prevobs=d['observations'];one.add(d['metrics']);acc.add(d['metrics'])
   exclusive(p,dict(identity=d['identity'],metrics=d['metrics'],observations=observation_summary(d['observations']),diagnostics=d['diagnostics'],anchor_checks=d['anchor_checks'],parameter_hash_check=d['parameter_hash_check'],learning_updates=0,replay_writes=0,wall_seconds=d['wall_seconds']))
   files.append(dict(scenario=sid,stream=st,path=str(p)))
   for w in d['observations']['samples_5s']:
    windows+=1;motion.extend(w['executed_distance_m']);displacement.extend(w['distance_from_start_m']);nearest.extend(w['user_to_nearest_uav_mean_m_per_user']);cycles+=np.array(w['assigned_cycles']);fractions+=np.array(w['assigned_task_fractions']);qsum+=np.array(w['cpu_queue_mean']);qmax=np.maximum(qmax,w['cpu_queue_max'])
    for pr in (0,1):
     n=w['upload_task_count'][pr];task_count[pr]+=n
     if n:task_sum[pr]+=n*w['upload_task_distance_mean_m'][pr]
     n=w['birth_task_count'][pr];birth_count[pr]+=n
     if n:birth_sum[pr]+=n*w['birth_task_distance_mean_m'][pr]
  per[str(sid)]=one.result()
 total=acc.result();mean=avg([v['truncated_delay_s'] for v in per.values()])
 for ap in g['anchors']:
  original=read(ap);expected={str(i):original['per_scenario'][str(i)] for i in scenario_ids};whole_metrics_equal(per,expected,'anchor_full_per_scenario');whole_metrics_equal(mean,avg([v['truncated_delay_s'] for v in expected.values()]),'anchor_mean')
 summary=dict(group=g,mean_s=mean,per_scenario=per,pooled=total,window_files=[],scenario_files=files,raw_windows_written=False,stream_consistency=dict(passed=True,scenarios=len(scenario_ids),metrics='exact Python float values; numeric repr recorded',observations='exact'),anchor_checks=dict(passed=True,references=g['anchors']),learning_updates=0,replay_writes=0,observations=dict(executed_5s=distribution(motion),distance_from_start=distribution(displacement),movement_ge_19_5m_fraction=float(np.mean(np.array(motion)>=19.5)),nearest_uav_distance=distribution(nearest),upload_task_distance_mean_m=[safe_ratio(task_sum[i],task_count[i]) for i in (0,1)],upload_task_window_count=task_count.tolist(),birth_task_distance_mean_m=[safe_ratio(birth_sum[i],birth_count[i]) for i in (0,1)],birth_task_count=birth_count.tolist(),assigned_cycles=cycles.tolist(),assigned_task_fractions=fractions.tolist(),cpu_queue_mean=(qsum/windows).tolist(),cpu_queue_max=qmax.tolist()),wall_seconds=time.perf_counter()-start)
 exclusive(root/'summary.json',summary);print('GROUP_PASS',g['label'],mean,summary['wall_seconds'],flush=True);return summary


def observation_summary(obs):
    samples=obs['samples_5s'];motion=[];distance=[];nearest=[];cycles=np.zeros((2,3,3));fractions=np.zeros((2,3,3));qs=np.zeros((2,3));qm=np.zeros((2,3));ts=np.zeros(2);tc=np.zeros(2);bs=np.zeros(2);bc=np.zeros(2)
    for w in samples:
        motion.extend(w['executed_distance_m']);distance.extend(w['distance_from_start_m']);nearest.extend(w['user_to_nearest_uav_mean_m_per_user']);cycles+=np.asarray(w['assigned_cycles']);fractions+=np.asarray(w['assigned_task_fractions']);qs+=np.asarray(w['cpu_queue_mean']);qm=np.maximum(qm,w['cpu_queue_max'])
        for pr in (0,1):
            n=w['upload_task_count'][pr];tc[pr]+=n
            if n:ts[pr]+=n*w['upload_task_distance_mean_m'][pr]
            n=w['birth_task_count'][pr];bc[pr]+=n
            if n:bs[pr]+=n*w['birth_task_distance_mean_m'][pr]
    return dict(definitions=obs['definitions'],initial_positions_m=obs['initial_positions_m'],final_positions_m=obs['final_positions_m'],executed_total_m_per_uav=obs['executed_total_m_per_uav'],windows_aggregated=len(samples),executed_5s=distribution(motion),distance_from_start=distribution(distance),movement_ge_19_5m_fraction=float(np.mean(np.asarray(motion)>=19.5)),nearest_uav_distance=distribution(nearest),upload_task_distance_mean_m=[safe_ratio(ts[i],tc[i]) for i in (0,1)],upload_task_window_count=tc.tolist(),birth_task_distance_mean_m=[safe_ratio(bs[i],bc[i]) for i in (0,1)],birth_task_count=bc.tolist(),assigned_cycles=cycles.tolist(),assigned_task_fractions=fractions.tolist(),compute_task_share_per_uav=[safe_ratio(v,float(fractions.sum())) for v in fractions.sum(axis=(0,1))],cpu_queue_mean=(qs/len(samples)).tolist(),cpu_queue_max=qm.tolist())


def lean_reference():
    manifest=gate_manifest(20);files={(w['scenario'],w['stream']):w for w in read(NEW/'summaries/seed_4104/TFOO-200.json')['window_files']};windows={}
    for sc in manifest['scenarios']:
        for stream in (1,2):
            w=files[(sc['scenario'],stream)];old=read(w['path']);equal(old['identity']['scenario'],sc,'reference_scenario');equal(old['identity']['stream'],stream,'reference_stream');equal(old['metrics']['seconds'],900,'reference_900s')
            hashes={k:old['metrics'][k] for k in ('event_sha256','canonical_event_sha256','realization_sha256') if k in old['metrics']}
            windows[f"{sc['scenario']:02d}_{stream}"]=dict(hashes=hashes)
    return dict(windows=windows)
