# Derived from OUTM/stage2_seed.py (upper), OUTM/stage3_seed.py (lower),
# and WT/src/hrl_v2/run.py (stage1). Only P1-P6 wrappers and PILOT outputs.
from pilot_common import *
import copy
import torch
from hrl_v2 import SCHEMA
from hrl_v2.checkpoint import identity,load,save,runtime_record
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.run import read_config,append,collection_policy
from hrl_v2.randomness import Streams,derive,PURPOSES
from hrl_v2.schedule import phase
from hrl_v2.upper_baselines import upper_phases
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect
from hrl_v2.policy import Policy
from hrl_v2.telemetry import compact,resources
from hrl_v2.replay import Replay
from hrl_v2.timing import ClockConfig
# P1-only named RNG extension: original identifiers and all old derivations stay exact.
ORIGINAL_PURPOSES=dict(PURPOSES)
check('random_upper' not in PURPOSES and max(PURPOSES.values())==12,'P1_new_rng_identifier')
PURPOSES['random_upper']=13
check({k:PURPOSES[k] for k in ORIGINAL_PURPOSES}==ORIGINAL_PURPOSES,'P1_existing_rng_ids_unchanged')
EXPECTED1=dict(triggers=27000,skipped=90,events=26910,critic_fits=53820,target_updates=26910,actor_updates=13410)
EXPECTEDL=dict(triggers=4500,skipped=90,events=4410,critic_fits=8820,target_updates=4410,actor_updates=1800)
EXPECTEDK=dict(triggers=4500,skipped=0,events=4500,critic_fits=9000,target_updates=4500,actor_updates=1800)

def binding():return read(PILOT/'pilot_contract.json')['identity']
def cleanup():gc.collect();torch.cuda.empty_cache()
def learner_from(path,seed,device,config):
 z=load(path,binding());l=Learner(seed,device,config=config);l.load_state_dict(z['learner']);return l,z

def restore_rng(z):
 torch.set_rng_state(z['global_cpu_rng']);torch.cuda.set_rng_state_all(z['global_cuda_rng'])

def checkpoint(path,l,p,*,replay=None,cycle=None,label=None):
 check(not Path(path).exists(),'new_checkpoint_'+str(path))
 z=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding(),completed_episode=p.global_episode,pending_evaluation=None,learner=l.state_dict(),global_cpu_rng=torch.get_rng_state(),global_cuda_rng=torch.cuda.get_rng_state_all(),runtime=runtime_record(),resumable=False,kind='pilot_safe_boundary',cycle=cycle,label=label)
 if replay is not None:z.update(replay=replay,seed=int(label.split('_')[-1]) if label and label.startswith('seed_') else None,ppo_batch=[],streams=None,development_results={},baseline_reference=None)
 save(output_path(path),z)

def hashes(l):return {f'{prefix}_{n}':parameter_hash(getattr(getattr(l,prefix),n)) for prefix,names in [('sac',('actor','q1','q2','target1','target2')),('ppo',('actor','value'))] for n in names}

class RandomUpperPolicy(Policy):
 def __init__(self,*args,random_seed,**kwargs):
  super().__init__(*args,**kwargs);self.random_upper=np.random.default_rng(random_seed);self.vectors=[]
 @torch.no_grad()
 def upper(self,report,history):
  if self.observer is not None:self.observer.update(report,history)
  u=self.random_upper.uniform(-1,1,6);self.vectors.append(u.copy());return u,{}

class Coverage:
 def __init__(self,a):
  self.start=a.env.physics.uav_coords.copy();self.centers=np.asarray(a.env._scene.realization['cluster_centers_m'],dtype=float);self.samples=[self.start.copy()]
 def tick(self,a,reading):
  if a.env.step_index%a.timing.physical_per_upper==0:self.samples.append(a.env.physics.uav_coords.copy())
 def result(self):
  x=np.asarray(self.samples);return dict(samples=len(x),distance_start=np.linalg.norm(x-self.start,axis=-1).tolist(),distance_cluster_center=np.linalg.norm(x-self.centers,axis=-1).tolist(),pair_distance=np.stack([np.linalg.norm(x[:,i]-x[:,j],axis=-1) for i,j in ((0,1),(0,2),(1,2))],axis=-1).tolist(),definitions=dict(position='initial and every completed physical macro boundary; no RNG draws',cluster_center='causal initial realization cluster center for associated region',pair_order=[[0,1],[0,2],[1,2]]))

def quantiles(x):
 a=np.asarray(x);return [dict(p10=float(np.quantile(v,.1)),p50=float(np.quantile(v,.5)),p90=float(np.quantile(v,.9)),max=float(np.max(v))) for v in a.T]

def summarize_positions(run,width=25):
 ff=sorted((run/'positions').glob('*.json'));out=[]
 for start in range(0,len(ff),width):
  rr=[read(p) for p in ff[start:start+width]];row=dict(first=rr[0]['global_episode'],last=rr[-1]['global_episode'],episodes=len(rr),position_sampling=rr[0]['coverage']['definitions'])
  for k in ('distance_start','distance_cluster_center','pair_distance'):row[k]=quantiles([v for r in rr for v in r['coverage'][k]])
  vec=[v for r in rr for v in r.get('random_vectors',[])];row['random_vectors_includes_committed_bootstrap_successor']=True
  if vec:
   a=np.asarray(vec);row['random_vector_stats']=dict(samples=len(a),mean=a.mean(0).tolist(),std=a.std(0).tolist(),min=a.min(0).tolist(),max=a.max(0).tolist())
  row['max_actual_step_m']=max(r['max_actual_step_m'] for r in rr);out.append(row)
 put(run/'position_25_bins.json',out);return out

def original_episode(seed,p,kind):
 if kind=='stage1':root=old_run(seed);n=p.global_episode
 elif kind=='upper':root=old_sd(seed);n=next(i for i,q in enumerate(upper_phases(),1) if q.global_episode==p.global_episode)
 else:root=old_l(seed);n=p.global_episode
 path=root/f'episodes/episode-{n:04d}.json';check(path.exists(),'original_episode_present_'+str(seed)+'_'+str(p.global_episode),str(path));return read(path)

def run_lower(seed,l,replay,phases,run,*,random_upper=False,stage1=False,keep_replay=False,rp=False):
 run=output_path(run);check(not run.exists(),'new_lower_run_'+str(run));run.mkdir(parents=True)
 device=configure();config=read_config(CONFIG,WT);scene=read(SCENE);clocks=ClockConfig.from_config(config);pre=copy.deepcopy(l.state_dict()['ppo']);sc=dict(l.sac.counts);initial_flush=replay.flushes;timing=time.perf_counter();available=[];skipped=[];saved=[]
 version='random-uniform-0' if random_upper else 'hover-0' if stage1 else f'ppo-{l.ppo.version}-stochastic'
 if keep_replay:replay.upper_version=version
 for index,p in enumerate(phases,1):
  equal(identity(WT,CONFIG),binding(),'training_identity')
  external={k+'_seed':derive(seed,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};old=original_episode(seed,p,'stage1' if stage1 else 'lower');equal(external,{k:v for k,v in next(x for x in read(old_run(seed)/'training_manifest.json') if x['global_episode']==p.global_episode).items() if k.endswith('_seed')},'external_seeds_'+str(seed)+'_'+str(p.global_episode))
  streams=Streams(seed,p.stage,p.cycle,p.block_episode);a=PhysicsAdapter(scene,timing=clocks);a.reset(**external);replay.set_upper_version(version);available.append(len(replay.episodes))
  if random_upper:pol=RandomUpperPolicy(l.sac.actor,l.ppo.actor,None,streams,mode='moving',stochastic=True,random_seed=derive(seed,1,0,p.global_episode,purpose='random_upper'))
  else:pol=collection_policy(l,streams,p)
  before=dict(l.sac.counts);res=[];started=time.perf_counter();cov=Coverage(a);torch.cuda.reset_peak_memory_stats()
  def event():
   rr=l.sac.event(replay,p.alpha,p.actor_enabled,streams);res.extend(v for f in rr.get('critics',[]) for v in f['residual_mse']);append(output_path(run/'learning.jsonl'),dict(attempt_id=run.name,global_episode=p.global_episode,physical_tick=a.env.step_index,event=l.sac.counts['triggers'],alpha=p.alpha,**rr))
  e=collect(a,pol,upper_version=version,lower_version=l.sac.version,identity=dict(global_episode=p.global_episode,external=external),on_event=event,on_tick=cov.tick,reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro']);replay.add(e);metrics=e.metrics;work=time.perf_counter()-started
  row=dict(attempt_id=run.name,global_episode=p.global_episode,layer=p.layer,stage=p.stage,reward_physical=float(sum(e.rewards)),reward_learning_macro_sum=float(sum(h['reward_learning'] for h in e.high)),work_seconds=work,checkpoint_seconds=None,evaluation_seconds=0.,snapshot_seconds=0.,wall_seconds=None,alpha=p.alpha,critic_residual_mse=avg(res) if res else None,actor_relative_entropy=pol.entropy_sum/pol.entropy_count if pol.entropy_count else None,entropy_samples=pol.entropy_count,entropy_scope='executed lower samples plus real bootstrap successor',sac_delta={k:v-before[k] for k,v in l.sac.counts.items()},resources=resources(),event_sha256=metrics['event_sha256'],replay_completed_before=available[-1],replay_completed_after=len(replay.episodes),replay_flushes=replay.flushes,**compact(metrics))
  if row['sac_delta']['skipped']:skipped.append(p.global_episode)
  if rp:
   equal(row['event_sha256'],old['metrics']['event_sha256'],'RP_lower_event_'+str(p.global_episode))
   if stage1:equal(row['sac_delta'],old['train_metric']['sac_delta'],'RP1_counts_'+str(p.global_episode))
  moves=np.asarray([h['displacement'] for h in e.high]);maximum=float(np.linalg.norm(moves,axis=-1).max());check(maximum<=20+1e-6,'motion_cap_'+str(p.global_episode),maximum)
  put(run/f'positions/episode-{p.global_episode:04d}.json',dict(global_episode=p.global_episode,coverage=cov.result(),random_vectors=np.asarray(getattr(pol,'vectors',[])).tolist(),max_actual_step_m=maximum))
  put(run/f'episodes/episode-{p.global_episode:04d}.json',dict(attempt_id=run.name,phase=p.__dict__,metrics=metrics,wall_seconds=work,storage_bytes=e.storage_bytes,sac=dict(l.sac.counts),ppo=dict(l.ppo.counts),train_metric=row))
  save_at=time.perf_counter()
  if not rp:
   if stage1 and p.global_episode in (100,200,300):
    pp=run/f'checkpoints/S{p.global_episode}r.pt';checkpoint(pp,l,p,replay=replay if p.global_episode==300 else None,label=f'seed_{seed}');saved.append(str(pp))
   elif not stage1 and p.block_episode==10:
    n=p.cycle//2;prefix='K' if keep_replay else 'c';pp=run/f'checkpoints/{prefix}{n}.pt';checkpoint(pp,l,p,cycle=n,label=prefix+str(n));saved.append(str(pp))
  row.update(checkpoint_seconds=time.perf_counter()-save_at,wall_seconds=time.perf_counter()-started,resources=resources());append(output_path(run/'train_metrics.jsonl'),row)
  put(run/'progress.json',dict(status='complete' if index==len(phases) else 'running',completed_episodes=index,total_episodes=len(phases),global_episode=p.global_episode,sac=dict(l.sac.counts),ppo=dict(l.ppo.counts),replay_completed=len(replay.episodes),replay_flushes=replay.flushes))
  put(PILOT/'progress.json',dict(status='training',seed=seed,stage=run.name,completed=index,total=len(phases),global_episode=p.global_episode,time=now()))
  print('LOWER',seed,run.name,index,len(phases),p.global_episode,row['truncated_delay_s'],round(row['wall_seconds'],3),flush=True)
  del e,a,pol,cov,old,metrics
  if index%25==0:summarize_positions(run)
 equal(l.state_dict()['ppo'],pre,'frozen_complete_PPO_'+run.name)
 delta={k:v-sc[k] for k,v in l.sac.counts.items()};fd=replay.flushes-initial_flush
 if not rp:
  expect=EXPECTED1 if stage1 else EXPECTEDK if keep_replay else EXPECTEDL;equal(delta,expect,'SAC_delta_'+run.name);equal(fd,0 if keep_replay else 1,'flush_delta_'+run.name)
  equal(skipped,[1] if stage1 else [] if keep_replay else [phases[0].global_episode],'skipped_episodes_'+run.name)
  if stage1:
   reference=load(old_run(seed)/'checkpoints/episode-0300-evaluated.pt',binding())
   equal(l.sac.counts,reference['learner']['sac']['counts'],'S300_original_counts_'+str(seed))
   del reference;cleanup()
  else:equal(available,[55]*50 if keep_replay else list(range(50)),'available_episodes_'+run.name)
 result=dict(status='passed',wall_seconds=time.perf_counter()-timing,episodes=len(phases),SAC_delta=delta,replay_flushes_delta=fd,available_completed=available,skipped_episodes=skipped,checkpoints=saved,PPO_complete_unchanged=True,position_bins=summarize_positions(run));put(run/'result.json',result);return result

def run_upper(seed,l,phases,run,*,start_n=0,rp=False):
 run=output_path(run);check(not run.exists(),'new_upper_run_'+str(run));run.mkdir(parents=True);config=read_config(CONFIG,WT);scene=read(SCENE);clocks=ClockConfig.from_config(config);sac=copy.deepcopy(l.state_dict()['sac']);count0=l.ppo.counts['batches'];timing=time.perf_counter();batch=[];saved=[]
 for i,p in enumerate(phases,1):
  equal(identity(WT,CONFIG),binding(),'training_identity');started=time.perf_counter();streams=Streams(seed,p.stage,p.cycle,p.block_episode);external={k+'_seed':derive(seed,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};old=original_episode(seed,p,'upper');equal(external,{k:v for k,v in next(x for x in read(old_run(seed)/'training_manifest.json') if x['global_episode']==p.global_episode).items() if k.endswith('_seed')},'upper_external_'+str(seed)+'_'+str(p.global_episode));a=PhysicsAdapter(scene,timing=clocks);a.reset(**external)
  version=f'ppo-{l.ppo.version}-stochastic';pol=Policy(l.sac.actor,l.ppo.actor,l.ppo.value,streams,mode='moving',stochastic=False,upper_stochastic=True);cov=Coverage(a)
  e=collect(a,pol,upper_version=version,lower_version=l.sac.version,identity=dict(global_episode=p.global_episode,external=external),on_tick=cov.tick,reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro']);batch.append(e)
  if len(batch)==5:logs=l.ppo.update(batch,streams);batch.clear();append(output_path(run/'learning.jsonl'),dict(episode=start_n+i,global_episode=p.global_episode,ppo=logs))
  metrics=e.metrics
  if rp:equal(metrics['event_sha256'],old['metrics']['event_sha256'],'RP-U_event_'+str(p.global_episode))
  row=dict(episode=start_n+i,global_episode=p.global_episode,reward_physical=float(sum(e.rewards)),event_sha256=metrics['event_sha256'],work_seconds=time.perf_counter()-started,**compact(metrics));moves=np.asarray([h['displacement'] for h in e.high]);maximum=float(np.linalg.norm(moves,axis=-1).max());check(maximum<=20+1e-6,'upper_motion_cap_'+str(p.global_episode),maximum)
  put(run/f'positions/episode-{i:04d}.json',dict(global_episode=p.global_episode,coverage=cov.result(),max_actual_step_m=maximum))
  put(run/f'episodes/episode-{i:04d}.json',dict(episode=start_n+i,global_episode=p.global_episode,phase=p.__dict__,metrics=metrics,reward_physical=row['reward_physical'],ppo=dict(l.ppo.counts)))
  save_at=time.perf_counter()
  if not rp and (start_n+i)%25==0:pp=run/f'checkpoints/U{start_n+i}.pt';checkpoint(pp,l,p,label=f'U{start_n+i}');saved.append(str(pp))
  row.update(checkpoint_seconds=time.perf_counter()-save_at,wall_seconds=time.perf_counter()-started);append(output_path(run/'train_metrics.jsonl'),row);put(run/'progress.json',dict(status='complete' if i==len(phases) else 'running',completed_episode=start_n+i,global_episode=p.global_episode,ppo_batches=l.ppo.counts['batches'],sac_updates=0,replay_writes=0));put(PILOT/'progress.json',dict(status='training',seed=seed,stage=run.name,completed=i,total=len(phases),global_episode=p.global_episode,time=now()))
  print('UPPER',seed,run.name,i,len(phases),p.global_episode,row['truncated_delay_s'],round(row['wall_seconds'],3),flush=True);del e,a,pol,cov,old,metrics
  if i%25==0:summarize_positions(run)
 equal(l.state_dict()['sac'],sac,'frozen_complete_SAC_'+run.name);check(not batch and l.ppo.counts['batches']-count0==len(phases)//5,'PPO_budget_'+run.name)
 result=dict(status='passed',wall_seconds=time.perf_counter()-timing,episodes=len(phases),PPO_batches_delta=l.ppo.counts['batches']-count0,PPO_batches_total=l.ppo.counts['batches'],SAC_complete_unchanged=True,checkpoints=saved,position_bins=summarize_positions(run));put(run/'result.json',result);return result

def repro_train(seed):
 config=read_config(CONFIG,WT);device=configure();root=PILOT/f'seed_{seed}/rp';s300=old_run(seed)/'checkpoints/episode-0300-evaluated.pt';saved=load(s300,binding());l=Learner(seed,device,config=config)
 equal(l.state_dict()['ppo'],saved['learner']['ppo'],'RP1_fresh_complete_PPO_vs_S300_'+str(seed))
 tf=load(old_tfoo(seed)/'checkpoints/episode-0000-initial.pt',binding());tfhash={k:__import__('hashlib').sha256(b''.join(v.cpu().numpy().tobytes() for v in tf['learner']['ppo'][k].values())).hexdigest() for k in ('actor','value')};initialhash={k:__import__('hashlib').sha256(b''.join(v.cpu().numpy().tobytes() for v in l.state_dict()['ppo'][k].values())).hexdigest() for k in ('actor','value')};put(root/'RP-1/TFOO_initial_PPO_comparison.json',dict(report_only=True,initial_hash=initialhash,TFOO_initial_hash=tfhash,equal=initialhash==tfhash));del tf,saved
 r1=run_lower(seed,l,Replay(),[phase(n) for n in range(1,6)],root/'RP-1/run',stage1=True,rp=True);del l;cleanup()
 l,z=learner_from(s300,seed,device,config);restore_rng(z);del z;ru=run_upper(seed,l,upper_phases()[:10],root/'RP-U/run',rp=True);del l;cleanup()
 l,z=learner_from(old_sd(seed)/'checkpoints/episode-0200.pt',seed,device,config);del z;z=load(s300,binding());r=z['replay'];restore_rng(z);del z;rl=run_lower(seed,l,r,[phase(n) for n in (411,412,413)],root/'RP-L/run',rp=True);del l,r;cleanup()
 result={'RP-1':r1,'RP-U':ru,'RP-L':rl};put(root/'training_checks.json',dict(status='PASS',results=result,checks=CHECKS));return result
if __name__=='__main__':
 try:repro_train(int(sys.argv[1]))
 except BaseException as e:failure('RP-training',e);raise