# Derived from read-only PILOT/pilot_train_r2.py and WT/src/hrl_v2/run.py.
# Curriculum: 200 hover + 100 random, no replay clearing, exact audits,
# replay-bearing A-S200h/A-S300c/A-c1..5, and CPIL-only output paths.
from cp_common import *
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


EXPECTED200=dict(triggers=18000,skipped=90,events=17910,critic_fits=35820,target_updates=17910,actor_updates=8910)
EXPECTED100R=dict(triggers=9000,skipped=0,events=9000,critic_fits=18000,target_updates=9000,actor_updates=4500)

def rng_state():
 return dict(cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])

def rng_assert(before,label):
 equal(torch.get_rng_state(),before['cpu'],'RNG_CPU_unchanged_'+label)
 equal(torch.cuda.get_rng_state_all(),before['cuda'],'RNG_CUDA_unchanged_'+label)

def replay_summary(replay):
 return dict(upper_version=replay.upper_version,writes=replay.writes,flushes=replay.flushes,episodes=[x.identity['global_episode'] for x in replay.episodes])

def compare_snapshot(seed,n,l):
 before=rng_state()
 path=old_run(seed)/f'snapshots/episode-{n:04d}.pt'
 side=read(path.with_suffix('.json'))
 equal(sha(path),side['sha256'],'snapshot_sha_'+str(n))
 equal(side['identity'],binding(),'snapshot_sidecar_identity_'+str(n))
 z=torch.load(path,map_location='cpu',weights_only=False)
 equal(z['identity'],binding(),'snapshot_payload_identity_'+str(n))
 equal(z['episode'],n,'snapshot_episode_'+str(n))
 equal(z['kind'],'network_snapshot_not_resumable','snapshot_kind_'+str(n))
 equal(z['resumable'],False,'snapshot_not_resumable_'+str(n))
 candidate=l.state_dict()
 for family,names in (('sac',('actor','q1','q2','target1','target2')),('ppo',('actor','value'))):
  for name in names:equal(candidate[family][name],z['learner'][family][name],f'snapshot_network_{n}_{family}_{name}')
 del candidate,z
 rng_assert(before,'snapshot_comparison_'+str(n))

def verify_full_checkpoint(path,l,replay,expected_label):
 before=rng_state()
 z=load(path,binding())
 equal(l.state_dict(),z['learner'],'checkpoint_state_dict_readback_'+Path(path).stem)
 equal(len(z['replay'].episodes),55,'checkpoint_replay55_'+Path(path).stem)
 equal(replay_summary(replay),replay_summary(z['replay']),'checkpoint_replay_metadata_'+Path(path).stem)
 equal(z['replay'].upper_version,expected_label,'checkpoint_replay_label_'+Path(path).stem)
 equal(z['replay_metadata'],dict(upper_version=replay.upper_version,flushes=replay.flushes,completed=len(replay.episodes),writes=replay.writes,episode_ids=[x.identity['global_episode'] for x in replay.episodes]),'checkpoint_replay_saved_metadata_'+Path(path).stem)
 for a,b in zip(replay.episodes,z['replay'].episodes):
  equal(a.__dict__,b.__dict__,'checkpoint_replay_episode_'+Path(path).stem+'_'+str(a.identity['global_episode']))
 equal(z['global_cpu_rng'],before['cpu'],'checkpoint_saved_CPU_RNG_'+Path(path).stem)
 equal(z['global_cuda_rng'],before['cuda'],'checkpoint_saved_CUDA_RNG_'+Path(path).stem)
 del z
 rng_assert(before,'checkpoint_readback_'+Path(path).stem)

def run_curriculum(seed,run):
 check(seed==4104,'curriculum_only_seed_4104')
 config=read_config(CONFIG,WT);device=configure();l=Learner(seed,device,config=config);replay=Replay()
 before=rng_state()
 z=load(old_run(seed)/'checkpoints/episode-0300-evaluated.pt',binding())
 equal(l.state_dict()['ppo'],z['learner']['ppo'],'RP1_fresh_complete_PPO_vs_S300_'+str(seed))
 del z
 rng_assert(before,'fresh_PPO_check')
 result=run_lower(seed,l,replay,[phase(n) for n in range(1,301)],run,stage1=True,curriculum=True)
 del l,replay;cleanup()
 return result

def binding():return read(CPIL/'cp_contract.json')['identity']
def cleanup():gc.collect();torch.cuda.empty_cache()
def learner_from(path,seed,device,config):
 z=load(path,binding());l=Learner(seed,device,config=config);l.load_state_dict(z['learner']);return l,z

def restore_rng(z):
 torch.set_rng_state(z['global_cpu_rng']);torch.cuda.set_rng_state_all(z['global_cuda_rng'])

def checkpoint(path,l,p,*,replay=None,cycle=None,label=None):
 check(not Path(path).exists(),'new_checkpoint_'+str(path))
 z=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding(),completed_episode=p.global_episode,pending_evaluation=None,learner=l.state_dict(),global_cpu_rng=torch.get_rng_state(),global_cuda_rng=torch.cuda.get_rng_state_all(),runtime=runtime_record(),resumable=False,kind='curriculum_safe_boundary',cycle=cycle,label=label)
 if replay is not None:z.update(replay_metadata=dict(upper_version=replay.upper_version,flushes=replay.flushes,completed=len(replay.episodes),writes=replay.writes,episode_ids=[x.identity['global_episode'] for x in replay.episodes]),replay=replay,seed=int(label.split('_')[-1]) if label and label.startswith('seed_') else None,ppo_batch=[],streams=None,development_results={},baseline_reference=None)
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

def run_lower(seed,l,replay,phases,run,*,random_upper=False,stage1=False,keep_replay=False,rp=False,checkpoint_replay=False,checkpoint_prefix="A-c",curriculum=False):
 run=output_path(run);check(not run.exists(),'new_lower_run_'+str(run));run.mkdir(parents=True)
 device=configure();config=read_config(CONFIG,WT);scene=read(SCENE);clocks=ClockConfig.from_config(config);pre=copy.deepcopy(l.state_dict()['ppo']);sc=dict(l.sac.counts);initial_flush=replay.flushes;timing=time.perf_counter();available=[];skipped=[];saved=[];segments={};replay_mix=[]
 version='random-uniform-0' if random_upper else 'hover-0' if stage1 else f'ppo-{l.ppo.version}-stochastic'
 if keep_replay:replay.upper_version=version
 if curriculum:
  equal([p.global_episode for p in phases],list(range(1,301)),'curriculum_exact_300_phases')
  pilot_first=Path(read(PILOT_REF/f'seed_{seed}/pipeline_data.json')['runs']['step1'])
  equal(replay_summary(replay),dict(upper_version=None,writes=0,flushes=0,episodes=[]),'curriculum_fresh_replay')
 for index,p in enumerate(phases,1):
  audit_before=rng_state()
  if curriculum:
   random_upper=p.global_episode>200
   if p.global_episode==201:
    equal(len(replay.episodes),55,'curriculum_transition_55_completed')
    equal(replay.flushes-initial_flush,1,'curriculum_transition_one_initial_flush')
    old_flush=replay.flushes
    replay.upper_version='random-uniform-0'
    equal(replay.flushes,old_flush,'curriculum_transition_no_flush')
   version='random-uniform-0' if random_upper else 'hover-0'
  equal(identity(WT,CONFIG),binding(),'training_identity')
  external={k+'_seed':derive(seed,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};old=original_episode(seed,p,'stage1' if stage1 else 'lower');equal(external,{k:v for k,v in next(x for x in read(old_run(seed)/'training_manifest.json') if x['global_episode']==p.global_episode).items() if k.endswith('_seed')},'external_seeds_'+str(seed)+'_'+str(p.global_episode))
  streams=Streams(seed,p.stage,p.cycle,p.block_episode);a=PhysicsAdapter(scene,timing=clocks);a.reset(**external);replay.set_upper_version(version);available.append(len(replay.episodes))
  if random_upper:pol=RandomUpperPolicy(l.sac.actor,l.ppo.actor,None,streams,mode='moving',stochastic=True,random_seed=derive(seed,1,0,p.global_episode,purpose='random_upper'))
  else:pol=collection_policy(l,streams,p)
  rng_assert(audit_before,'pre_episode_checks_'+str(p.global_episode))
  before=dict(l.sac.counts);res=[];started=time.perf_counter();cov=Coverage(a);torch.cuda.reset_peak_memory_stats()
  def event():
   rr=l.sac.event(replay,p.alpha,p.actor_enabled,streams);res.extend(v for f in rr.get('critics',[]) for v in f['residual_mse']);append(output_path(run/'learning.jsonl'),dict(attempt_id=run.name,global_episode=p.global_episode,physical_tick=a.env.step_index,event=l.sac.counts['triggers'],alpha=p.alpha,**rr))
  e=collect(a,pol,upper_version=version,lower_version=l.sac.version,identity=dict(global_episode=p.global_episode,external=external),on_event=event,on_tick=cov.tick,reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro']);replay.add(e);metrics=e.metrics;work=time.perf_counter()-started; audit_before=rng_state()
  row=dict(attempt_id=run.name,global_episode=p.global_episode,layer=p.layer,stage=p.stage,reward_physical=float(sum(e.rewards)),reward_learning_macro_sum=float(sum(h['reward_learning'] for h in e.high)),work_seconds=work,checkpoint_seconds=None,evaluation_seconds=0.,snapshot_seconds=0.,wall_seconds=None,alpha=p.alpha,critic_residual_mse=avg(res) if res else None,actor_relative_entropy=pol.entropy_sum/pol.entropy_count if pol.entropy_count else None,entropy_samples=pol.entropy_count,entropy_scope='executed lower samples plus real bootstrap successor',sac_delta={k:v-before[k] for k,v in l.sac.counts.items()},resources=resources(),event_sha256=metrics['event_sha256'],replay_completed_before=available[-1],replay_completed_after=len(replay.episodes),replay_flushes=replay.flushes,**compact(metrics))
  if row['sac_delta']['skipped']:skipped.append(p.global_episode)
  if rp or (curriculum and p.global_episode<=200):
   equal(row['event_sha256'],old['metrics']['event_sha256'],'RP_lower_event_'+str(p.global_episode))
   if stage1:equal(row['sac_delta'],old['train_metric']['sac_delta'],'RP1_counts_'+str(p.global_episode))
  moves=np.asarray([h['displacement'] for h in e.high]);maximum=float(np.linalg.norm(moves,axis=-1).max());check(maximum<=20+1e-6,'motion_cap_'+str(p.global_episode),maximum)
  if curriculum:
   if p.global_episode>200:
    old_position=read(pilot_first/f'positions/episode-{p.global_episode:04d}.json')
    equal(np.asarray(pol.vectors),np.asarray(old_position['random_vectors']),'curriculum_random_vectors_'+str(p.global_episode))
   if 201<=p.global_episode<=255:
    mix=dict(global_episode=p.global_episode,hover=sum(x.upper_version=='hover-0' for x in replay.episodes),random=sum(x.upper_version=='random-uniform-0' for x in replay.episodes),episodes=[x.identity['global_episode'] for x in replay.episodes],when='after completed episode added')
    equal(mix['hover'],max(0,255-p.global_episode),'curriculum_hover_FIFO_mix_'+str(p.global_episode))
    equal(mix['random'],min(55,p.global_episode-200),'curriculum_random_FIFO_mix_'+str(p.global_episode))
    replay_mix.append(mix)
   if p.global_episode<=200 and p.global_episode%25==0:compare_snapshot(seed,p.global_episode,l)
   if p.global_episode in (200,300):
    delta_now={k:v-sc[k] for k,v in l.sac.counts.items()}
    segments[str(p.global_episode)]=delta_now
    equal(delta_now,EXPECTED200 if p.global_episode==200 else EXPECTED1,'curriculum_counts_through_'+str(p.global_episode))
    if p.global_episode==300:equal({k:delta_now[k]-segments['200'][k] for k in delta_now},EXPECTED100R,'curriculum_random100_counts')
  put(run/f'positions/episode-{p.global_episode:04d}.json',dict(global_episode=p.global_episode,coverage=cov.result(),random_vectors=np.asarray(getattr(pol,'vectors',[])).tolist(),max_actual_step_m=maximum))
  put(run/f'episodes/episode-{p.global_episode:04d}.json',dict(attempt_id=run.name,phase=p.__dict__,metrics=metrics,wall_seconds=work,storage_bytes=e.storage_bytes,sac=dict(l.sac.counts),ppo=dict(l.ppo.counts),train_metric=row))
  save_at=time.perf_counter()
  if not rp:
   if curriculum and p.global_episode in (200,300):
    pp=run/('checkpoints/A-S200h.pt' if p.global_episode==200 else 'checkpoints/A-S300c.pt')
    checkpoint(pp,l,p,replay=replay,label=f'seed_{seed}')
    verify_full_checkpoint(pp,l,replay,expected_label=version)
    saved.append(str(pp))
   elif not stage1 and p.block_episode==10:
    n=p.cycle//2;prefix=checkpoint_prefix;pp=run/f'checkpoints/{prefix}{n}.pt'
    checkpoint(pp,l,p,replay=replay if checkpoint_replay else None,cycle=n,label=prefix+str(n))
    if checkpoint_replay:verify_full_checkpoint(pp,l,replay,expected_label=version)
    saved.append(str(pp))
  row.update(checkpoint_seconds=time.perf_counter()-save_at,wall_seconds=time.perf_counter()-started,resources=resources());append(output_path(run/'train_metrics.jsonl'),row)
  put(run/'progress.json',dict(status='complete' if index==len(phases) else 'running',completed_episodes=index,total_episodes=len(phases),global_episode=p.global_episode,sac=dict(l.sac.counts),ppo=dict(l.ppo.counts),replay_completed=len(replay.episodes),replay_flushes=replay.flushes))
  put(CPIL/'progress.json',dict(status='training',seed=seed,stage=run.name,completed=index,total=len(phases),global_episode=p.global_episode,time=now()))
  print('LOWER',seed,run.name,index,len(phases),p.global_episode,row['truncated_delay_s'],round(row['wall_seconds'],3),flush=True)
  rng_assert(audit_before,'post_episode_checks_and_checkpoint_'+str(p.global_episode))
  del e,a,pol,cov,old,metrics
  if index%25==0:summarize_positions(run)
 audit_before=rng_state()
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
 if curriculum:
  equal([x.identity['global_episode'] for x in replay.episodes],list(range(246,301)),'curriculum_final_FIFO_246_300')
  equal([x.upper_version for x in replay.episodes],['random-uniform-0']*55,'curriculum_final_only_random')
  put(run/'replay_mix_201_255.json',replay_mix)
 rng_assert(audit_before,'final_lower_checks_'+run.name)
 result=dict(curriculum=curriculum,segment_cumulative_SAC_delta=segments,replay_mixture=replay_mix,final_replay=replay_summary(replay),status='passed',wall_seconds=time.perf_counter()-timing,episodes=len(phases),SAC_delta=delta,replay_flushes_delta=fd,available_completed=available,skipped_episodes=skipped,checkpoints=saved,PPO_complete_unchanged=True,position_bins=summarize_positions(run));put(run/'result.json',result);return result

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
  if not rp and (start_n+i)%25==0:pp=run/f'checkpoints/A-U{start_n+i}.pt';checkpoint(pp,l,p,label=f'U{start_n+i}');saved.append(str(pp))
  row.update(checkpoint_seconds=time.perf_counter()-save_at,wall_seconds=time.perf_counter()-started);append(output_path(run/'train_metrics.jsonl'),row);put(run/'progress.json',dict(status='complete' if i==len(phases) else 'running',completed_episode=start_n+i,global_episode=p.global_episode,ppo_batches=l.ppo.counts['batches'],sac_updates=0,replay_writes=0));put(CPIL/'progress.json',dict(status='training',seed=seed,stage=run.name,completed=i,total=len(phases),global_episode=p.global_episode,time=now()))
  print('UPPER',seed,run.name,i,len(phases),p.global_episode,row['truncated_delay_s'],round(row['wall_seconds'],3),flush=True);del e,a,pol,cov,old,metrics
  if i%25==0:summarize_positions(run)
 equal(l.state_dict()['sac'],sac,'frozen_complete_SAC_'+run.name);check(not batch and l.ppo.counts['batches']-count0==len(phases)//5,'PPO_budget_'+run.name)
 result=dict(status='passed',wall_seconds=time.perf_counter()-timing,episodes=len(phases),PPO_batches_delta=l.ppo.counts['batches']-count0,PPO_batches_total=l.ppo.counts['batches'],SAC_complete_unchanged=True,checkpoints=saved,position_bins=summarize_positions(run));put(run/'result.json',result);return result

def repro_train(seed):
 check(seed==4104,'curriculum_only_seed_4104')
 config=read_config(CONFIG,WT);device=configure();root=CPIL/f'seed_{seed}/rp';s300=old_run(seed)/'checkpoints/episode-0300-evaluated.pt'
 l,z=learner_from(s300,seed,device,config);restore_rng(z);del z;ru=run_upper(seed,l,upper_phases()[:10],root/'RP-U/run',rp=True);del l;cleanup()
 l,z=learner_from(old_sd(seed)/'checkpoints/episode-0200.pt',seed,device,config);del z;z=load(s300,binding());r=z['replay'];restore_rng(z);del z;rl=run_lower(seed,l,r,[phase(n) for n in (411,412,413)],root/'RP-L/run',rp=True);del l,r;cleanup()
 result={'RP-U':ru,'RP-L':rl};put(root/'training_checks.json',dict(status='PASS',results=result,checks=CHECKS));return result

if __name__=='__main__':
 raise RuntimeError('Use cp_main.py; this module has no standalone training entry.')
