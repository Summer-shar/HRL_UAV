# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_stage3\stage3_4101.py; changes restricted to seed/path and approved D1-D9.
"""Authorized Stage3 lower-only experiment; all writes confined to OUT."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,gc,json,os,subprocess,time,traceback
import torch,hrl_v2,hrl_uav
from common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports
assert_imports()
from hrl_v2.checkpoint import identity,load,save,atomic_json,digest,runtime_record
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.run import read_config,append,collection_policy
from hrl_v2.schedule import phase
from hrl_v2.randomness import Streams,derive
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect
from hrl_v2.telemetry import compact
from hrl_v2.timing import ClockConfig
from hrl_v2 import SCHEMA
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
OUT=envpath('OUT',stage_path('stage3'))
R2=envpath('R2',stage_path('stage2'))
P=envpath('P',run_path())
B=envpath('VALIDATION_ROOT',WORK)
CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json'
SD=R2/'runs/sacdet/checkpoints/episode-0200.pt'
S300=P/'checkpoints/episode-0300-evaluated.pt'
STAGE='initialization';CHECKS={};TOTAL_START=time.perf_counter()
EXPECTED=dict(triggers=9000,skipped=90,events=8910,critic_fits=17820,target_updates=8910,actor_updates=3600)
class HardCheckError(RuntimeError):pass
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
def require(ok,label,details=None):
 if not ok:raise HardCheckError(json.dumps(dict(check=label,details=details),ensure_ascii=False,default=str))
def normalize(x):return json.loads(json.dumps(x,allow_nan=False))
def firstdiff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:return dict(field=path+k,candidate=a.get(k),reference=b.get(k))
   d=firstdiff(a[k],b[k],path+k+'.')
   if d:return d
  return None
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):return dict(field=path+'length',candidate=len(a),reference=len(b))
  for i,(x,y) in enumerate(zip(a,b)):
   d=firstdiff(x,y,path+str(i)+'.')
   if d:return d
  return None
 return None if a==b else dict(field=path.rstrip('.'),candidate=a,reference=b)
def git_states():
 states={}
 for root,branch,head in ((MAIN,'codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),(WT,'codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')):
  r={k:subprocess.check_output(['git','-C',str(root),*cmd],text=True).strip() for k,cmd in dict(status=['status','--porcelain'],branch=['branch','--show-current'],head=['rev-parse','HEAD']).items()}
  require(r==dict(status='',branch=branch,head=head),'git_state',dict(root=str(root),actual=r));states[str(root)]=r
 return states
def hashes(learner):
 return {f'{prefix}_{name}':parameter_hash(getattr(getattr(learner,prefix),name)) for prefix,names in (('sac',('actor','q1','q2','target1','target2')),('ppo',('actor','value'))) for name in names}
def new_learner(path,binding,config,device):
 saved=load(path,binding);learner=Learner(SEED,device,config=config);learner.load_state_dict(saved['learner']);return learner,saved
def restore_rng(saved):
 torch.set_rng_state(saved['global_cpu_rng']);torch.cuda.set_rng_state_all(saved['global_cuda_rng'])
def preflight():
 global STAGE
 STAGE='preflight';record=dict(status='checking')
 record['git_before']=git_states()
 for package in (hrl_v2,hrl_uav):require(Path(package.__file__).resolve().is_relative_to(WT/'src'),'import_path',package.__file__)
 device=configure();require(str(device)=='cuda:0','device')
 config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);validation=read(B/'validation/validation.json')
 require(binding==read(P/'contract.json')['identity']==validation['identity'],'identity_chain')
 require(validation['status']=='passed','validation_status')
 record.update(identity=binding,validation_status=validation['status'],runtime=runtime_record(),sys_prefix=sys.prefix,torch_threads=torch.get_num_threads(),self_pid=os.getpid(),parent_pid=os.getppid(),imports={p.__name__:p.__file__ for p in (hrl_v2,hrl_uav)})
 record.update(**gpu_check());atomic_json(OUT/'preflight.json',record)
 learner,sd=new_learner(SD,binding,config,device);hs=hashes(learner);summary=read(R2/'training_summary.json')['sacdet']['final_hashes']
 require(all(hs[k]==summary[k] for k in ('ppo_actor','ppo_value')),'SACDET200_upper_hashes',hs)
 sd_counts=dict(learner.sac.counts);sd_version=learner.sac.version;upper=dict(hashes={k:hs[k] for k in ('ppo_actor','ppo_value')},counts=dict(learner.ppo.counts),version=learner.ppo.version)
 del learner,sd;gc.collect();torch.cuda.empty_cache()
 learner,s300=new_learner(S300,binding,config,device);h300=hashes(learner)
 require(all(hs[k]==h300[k] for k in hs if k.startswith('sac_')) and sd_counts==learner.sac.counts and sd_version==learner.sac.version,'SAC300_full_match',dict(sd=hs,s300=h300,sd_counts=sd_counts,s300_counts=learner.sac.counts,sd_version=sd_version,s300_version=learner.sac.version))
 require(s300['replay'].upper_version=='hover-0','S300_replay_upper_version',s300['replay'].upper_version)
 del learner,s300;gc.collect();torch.cuda.empty_cache()
 lower=[phase(n) for n in range(401,601) if phase(n).layer=='lower']
 require(len(lower)==100 and lower[0].global_episode==411 and lower[-1].global_episode==600,'LOWER_endpoints')
 for block in range(10):
  pp=lower[block*10:(block+1)*10]
  require([p.block_episode for p in pp]==list(range(1,11)) and len({p.cycle for p in pp})==1,'LOWER_block',block+1)
  require(all(p.alpha==(0.0005 if block<5 else 0.0) and p.actor_enabled==(p.block_episode>2) for p in pp),'LOWER_alpha_actor',block+1)
 table=[dict(**p.__dict__,actor_enabled=p.actor_enabled) for p in lower]
 record.update(status='passed',source_hashes=hs,S300_hashes=h300,sac_counts=sd_counts,sac_version=sd_version,frozen_upper=upper,LOWER=table);atomic_json(OUT/'preflight.json',record)
 require(not (OUT/'contract.json').exists(),'new_contract')
 atomic_json(OUT/'contract.json',dict(identity=binding,source_checkpoints={label:dict(path=str(path),sha256=digest(path)) for label,path in (('SACDET-200',SD),('S300',S300))},LOWER=table,expected_sac_delta=EXPECTED,seed=SEED,intermediate_models="S3-c1..S3-c9 report only; no selection",only_protocol_difference='上层冻结为 SACDET-200，因此经验池只在第一块开头因上层版本变化清空一次；其余规则与原第三阶段下层块相同。',created_at=time.strftime('%Y-%m-%d %H:%M:%S')))
 CHECKS['preflight']='PASS';print('PREFLIGHT_PASS',flush=True);print(json.dumps(record['runtime']),flush=True)
 return config,binding,device,lower

def cycle_save(run,learner,p,binding):
 n=p.cycle//2;hs=hashes(learner)
 payload=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding,completed_episode=p.global_episode,pending_evaluation=None,cycle=n,learner=learner.state_dict(),parameter_hashes=hs,sac_counts=dict(learner.sac.counts),resumable=False,kind='stage3_full_learner_without_replay_not_resumable')
 path=run/f'checkpoints/cycle-{n:02d}.pt';require(not path.exists(),'new_cycle_checkpoint',str(path));save(path,payload)
 side=read(path.with_suffix('.json'));side.update(parameter_hashes=hs,sac_counts=dict(learner.sac.counts),resumable=False,cycle=n);atomic_json(path.with_suffix('.json'),side)

def run_lower(learner,replay,phases,run_dir):
 global STAGE
 run_dir.mkdir(parents=True,exist_ok=False);config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);scene=read(WT/'src/hrl_v2/configs/physical_scene.json');clocks=ClockConfig.from_config(config)
 attempt_id='stage3_'+run_dir.name;start=time.perf_counter()
 for index,p in enumerate(phases,1):
  STAGE=f'{run_dir.name}_global_{p.global_episode}';require(identity(WT,CONFIG)==binding,'identity_during_training')
  streams=Streams(SEED,p.stage,p.cycle,p.block_episode);adapter=PhysicsAdapter(scene,timing=clocks)
  external={k+'_seed':derive(SEED,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};adapter.reset(**external)
  version=f'ppo-{learner.ppo.version}-stochastic';replay.set_upper_version(version)
  policy=collection_policy(learner,streams,p);started=time.perf_counter();before=dict(learner.sac.counts);residuals=[]
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
  def event():
   row=learner.sac.event(replay,p.alpha,p.actor_enabled,streams)
   residuals.extend(v for fit in row.get('critics',[]) for v in fit['residual_mse'])
   append(run_dir/'learning.jsonl',dict(attempt_id=attempt_id,global_episode=p.global_episode,physical_tick=adapter.env.step_index,event=learner.sac.counts['triggers'],alpha=p.alpha,**row))
  episode=collect(adapter,policy,upper_version=version,lower_version=learner.sac.version,identity=dict(global_episode=p.global_episode,external=external),on_event=event,reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro'])
  replay.add(episode);metrics=episode.metrics;work=time.perf_counter()-started
  row=dict(attempt_id=attempt_id,global_episode=p.global_episode,layer=p.layer,stage=p.stage,reward_physical=float(sum(episode.rewards)),reward_learning_macro_sum=float(sum(h['reward_learning'] for h in episode.high)),work_seconds=work,alpha=p.alpha,critic_residual_mse=sum(residuals)/len(residuals) if residuals else None,actor_relative_entropy=policy.entropy_sum/policy.entropy_count if policy.entropy_count else None,entropy_samples=policy.entropy_count,entropy_scope='executed lower samples plus real bootstrap successor',sac_delta={k:v-before[k] for k,v in learner.sac.counts.items()},event_sha256=metrics['event_sha256'],**compact(metrics))
  atomic_json(run_dir/f'episodes/episode-{p.global_episode:04d}.json',dict(attempt_id=attempt_id,phase=p.__dict__,metrics=metrics,wall_seconds=work,storage_bytes=episode.storage_bytes,sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts),train_metric=row))
  save_at=time.perf_counter()
  if p.block_episode==10:cycle_save(run_dir,learner,p,binding)
  row.update(checkpoint_seconds=time.perf_counter()-save_at,wall_seconds=time.perf_counter()-started);append(run_dir/'train_metrics.jsonl',row)
  atomic_json(run_dir/'progress.json',dict(status='complete' if index==len(phases) else 'running',completed_lower_episodes=index,total_lower_episodes=len(phases),global_episode=p.global_episode,sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts),replay_flushes=replay.flushes))
  print(f'{run_dir.name} [{index}/{len(phases)}] global={p.global_episode} delay={row["truncated_delay_s"]:.9f} work={work:.3f}s events={row["sac_delta"]["events"]} actor={row["sac_delta"]["actor_updates"]}',flush=True)
  del adapter,policy,episode
 return dict(episodes=len(phases),wall_seconds=time.perf_counter()-start)

def formal(config,binding,device,lower,phases_limit=None):
 global STAGE
 STAGE='formal_initialization';learner,sd=new_learner(SD,binding,config,device);before=hashes(learner);sac=dict(learner.sac.counts);ppo=dict(learner.ppo.counts);version=learner.ppo.version;del sd
 saved=load(S300,binding);replay=saved['replay'];flush=replay.flushes;restore_rng(saved);del saved;gc.collect()
 selected=lower if phases_limit is None else lower[:phases_limit]
 timing=run_lower(learner,replay,selected,OUT/'runs/stage3');STAGE='formal_checks';after=hashes(learner);delta={k:v-sac[k] for k,v in learner.sac.counts.items()}
 result=dict(**timing,initial_hashes=before,final_hashes=after,initial_ppo_counts=ppo,final_ppo_counts=dict(learner.ppo.counts),initial_ppo_version=version,final_ppo_version=learner.ppo.version,sac_delta=delta,expected_sac_delta=EXPECTED,replay_flushes_delta=replay.flushes-flush,skipped_global_episodes=[r['global_episode'] for r in rows(OUT/'runs/stage3/train_metrics.jsonl') if r['sac_delta']['skipped']])
 result.update(upper_frozen=all(before[k]==after[k] for k in ('ppo_actor','ppo_value')) and ppo==learner.ppo.counts and version==learner.ppo.version)
 atomic_json(OUT/'training_summary.json',result)
 require(result['upper_frozen'],'upper_frozen',result);CHECKS['frozen_upper']='PASS'
 if phases_limit is None:
  require(delta==EXPECTED and result['skipped_global_episodes']==[411],'SAC_counts',result);CHECKS['SAC_counts']='PASS'
  side=read(OUT/'runs/stage3/checkpoints/cycle-10.json');require(side['completed_episode']==600 and side['pending_evaluation'] is None,'cycle10_sidecar')
 metrics=rows(OUT/'runs/stage3/train_metrics.jsonl')
 result['cycle_mean_truncated_delay_s']={str(c):sum(r['truncated_delay_s'] for r in metrics if phase(r['global_episode']).cycle//2==c)/sum(phase(r['global_episode']).cycle//2==c for r in metrics) for c in sorted({phase(r['global_episode']).cycle//2 for r in metrics})}
 atomic_json(OUT/'training_summary.json',result)
 require(result['replay_flushes_delta']==1,'replay_flushes',result);CHECKS['replay_flushes']='PASS'
 print('FORMAL_TRAINING_PASS',json.dumps(dict(sac_delta=delta,replay_flushes_delta=replay.flushes-flush)),flush=True)
 del learner,replay;gc.collect();torch.cuda.empty_cache();return result

def main():
 config,binding,device,lower=preflight();result=formal(config,binding,device,lower)
 atomic_json(OUT/'checks.json',dict(status='passed',checks=CHECKS,result=result))
 atomic_json(OUT/'git_after.json',git_states())
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));raise
