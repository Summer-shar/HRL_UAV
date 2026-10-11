"""SACDET r2: explicit user-authorized frozen-SAC upper training, no source edits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,csv,gc,json,os,subprocess,time,traceback
import torch,hrl_v2
from hrl_v2.checkpoint import identity,load,save,atomic_json,digest,runtime_record
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.run import read_config,append
from hrl_v2.upper_baselines import upper_phases
from hrl_v2.randomness import Streams,derive
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect
from hrl_v2.policy import Policy
from hrl_v2.telemetry import compact
from hrl_v2.timing import ClockConfig
from hrl_v2 import SCHEMA
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'
PREVIOUS=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet'
P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'
CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json'
STAGE='initialization';CHECKS={}
class HardCheckError(RuntimeError):pass
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(ok,label,details=None):
 if not ok:raise HardCheckError(json.dumps(dict(check=label,details=details),ensure_ascii=False,default=str))
def git_states():
 rows={}
 for root,branch,head in ((MAIN,'codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),(WT,'codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')):
  r={k:subprocess.check_output(['git','-C',str(root),*command],text=True).strip() for k,command in dict(status=['status','--porcelain'],branch=['rev-parse','--abbrev-ref','HEAD'],head=['rev-parse','HEAD']).items()}
  require(r==dict(status='',branch=branch,head=head),'git_state',dict(root=str(root),actual=r));rows[str(root)]=r
 return rows
def hashes(learner):return dict(sac_actor=parameter_hash(learner.sac.actor),ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value))
def preflight():
 global STAGE
 STAGE='preflight';record=dict(status='checking',previous_failed_attempt=str(PREVIOUS),previous_failure_reason='GPU 检查未排除自身进程')
 record['git_before']=git_states()
 require(Path(hrl_v2.__file__).resolve().is_relative_to(WT/'src'),'import_path',hrl_v2.__file__)
 device=configure();require(str(device)=='cuda:0','device')
 config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);validation=read(B/'validation/validation.json')
 require(binding==read(P/'contract.json')['identity']==validation['identity'],'identity_chain')
 require(validation['status']=='passed','validation_status')
 record.update(identity=binding,validation_status='passed',runtime=runtime_record(),torch_threads=torch.get_num_threads(),self_pid=os.getpid(),parent_pid=os.getppid())
 command="Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4"
 raw=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,encoding='utf-8',errors='replace')
 procs=json.loads(raw) if raw.strip() else [];procs=procs if isinstance(procs,list) else [procs]
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
 gpu_rows=list(csv.DictReader(gpu.splitlines()));gpu_pids={int(next(iter(row.values())).strip()) for row in gpu_rows}
 excluded={os.getpid(),os.getppid()};keywords=('hrl_v2','train','launch','diag','evaluate','paper')
 other=[]
 for proc in procs:
  if int(proc['ProcessId']) in excluded:continue
  cmd=(proc.get('CommandLine') or '').lower()
  if int(proc['ProcessId']) in gpu_pids or any(k in cmd for k in keywords):other.append(proc)
 record.update(cim_raw=raw,python_processes=procs,nvidia_smi_raw=gpu,nvidia_smi_rows=gpu_rows,excluded_pids=sorted(excluded),other_training_processes=other)
 atomic_json(OUT/'preflight.json',record)
 require(not other,'other_training_processes',other)
 hs={}
 for label,filename in (('evaluated','episode-0300-evaluated.pt'),('pending','episode-0300-pending.pt'),('initial','episode-0000-initial.pt')):
  saved=load(P/'checkpoints'/filename,binding);learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner']);hs[label]=hashes(learner)
  del saved,learner;gc.collect();torch.cuda.empty_cache()
 require(hs['evaluated']==hs['pending'],'300_pending_evaluated_hashes',hs)
 require(all(hs['evaluated'][k]==hs['initial'][k] for k in ('ppo_actor','ppo_value')),'initial_upper_hashes',hs)
 phases=upper_phases();require(len(phases)==200,'upper_phases_count');require([p.global_episode for p in phases[:100]]==list(range(301,401)),'first_100_mapping');require(phases[99].global_episode==400 and phases[199].global_episode==590,'phase_endpoints')
 record.update(status='passed',parameter_hashes=hs,upper_phases=[p.__dict__ for p in phases]);atomic_json(OUT/'preflight.json',record)
 CHECKS['preflight']='PASS'
 atomic_json(OUT/'contract.json',dict(previous_failed_attempt=str(PREVIOUS),previous_failure_reason='GPU 检查未排除自身进程',identity=binding,config_path=str(CONFIG),seed=4101,source_checkpoint=str(P/'checkpoints/episode-0300-evaluated.pt'),source_checkpoint_sha256=digest(P/'checkpoints/episode-0300-evaluated.pt'),runtime=record['runtime'],created_at=time.strftime('%Y-%m-%d %H:%M:%S'),initial_hashes=hs['evaluated'],upper_phases=record['upper_phases'],control_episodes=25,sacdet_episodes=200,ppo_batches=40,only_collection_protocol_difference='SAC lower deterministic decoding; PPO upper remains stochastic',implementation_repair_limit=1,auto_resume=False))
 print('PREFLIGHT_PASS',flush=True)
 return config,binding,device

def payload_save(run,learner,n,binding,kind):
 hs=hashes(learner)
 state=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding,completed_episode=n,pending_evaluation=None,episode=n,learner=learner.state_dict(),parameter_hashes=hs,resumable=False,kind=kind,seed=4101)
 path=run/('snapshots' if kind=='network_snapshot_not_resumable' else 'checkpoints')/f'episode-{n:04d}.pt'
 require(not path.exists(),'new_snapshot_path',str(path));save(path,state)
 return hs

def original_ppo():
 rows={}
 for line in (P/'learning.jsonl').read_text(encoding='utf-8').splitlines():
  row=json.loads(line)
  if 'ppo' in row:rows.setdefault(row['global_episode'],[]).append(row)
 return rows

def train_run(label,lower_stochastic,total,config,binding,device):
 global STAGE
 STAGE='training_'+label;run=OUT/'runs'/label;run.mkdir(parents=True,exist_ok=False)
 saved=load(P/'checkpoints/episode-0300-evaluated.pt',binding)
 learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner'])
 torch.set_rng_state(saved['global_cpu_rng']);torch.cuda.set_rng_state_all(saved['global_cuda_rng'])
 initial_hashes=hashes(learner);initial_counts=dict(learner.sac.counts);del saved;gc.collect()
 scene=read(WT/'src/hrl_v2/configs/physical_scene.json');clocks=ClockConfig.from_config(config)
 phases=upper_phases()[:total];batch=[];comparisons=[];oldppo=original_ppo() if lower_stochastic else {};first_difference=None;updates_count=0
 def compare(episode,key,new,old):
  nonlocal first_difference
  equal=new==old;row=dict(episode=episode,field=key,equal=equal)
  if not equal:
   row.update(candidate=new,reference=old)
   if first_difference is None:first_difference=row.copy()
  comparisons.append(row)
 payload_save(run,learner,0,binding,'network_snapshot_not_resumable')
 start=time.perf_counter()
 for n,p in enumerate(phases,1):
  STAGE=f'training_{label}_episode_{n}'
  require(identity(WT,CONFIG)==binding,'identity_during_training')
  started=time.perf_counter();streams=Streams(4101,p.stage,p.cycle,p.block_episode);adapter=PhysicsAdapter(scene,timing=clocks)
  external={k+'_seed':derive(4101,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};adapter.reset(**external)
  version=f'ppo-{learner.ppo.version}-stochastic'
  policy=Policy(learner.sac.actor,learner.ppo.actor,learner.ppo.value,streams,mode='moving',stochastic=lower_stochastic,upper_stochastic=True)
  episode=collect(adapter,policy,upper_version=version,lower_version=learner.sac.version,identity=dict(global_episode=p.global_episode,external=external),reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro'])
  batch.append(episode);logs=None
  if len(batch)==5:
   logs=learner.ppo.update(batch,streams);batch.clear();updates_count+=1
   append(run/'learning.jsonl',dict(episode=n,global_episode=p.global_episode,ppo=logs))
  metrics=episode.metrics;reward=float(sum(episode.rewards))
  row=dict(episode=n,global_episode=p.global_episode,reward_physical=reward,**compact(metrics),event_sha256=metrics['event_sha256'],work_seconds=time.perf_counter()-started)
  atomic_json(run/'episodes'/f'episode-{n:04d}.json',dict(episode=n,global_episode=p.global_episode,phase=p.__dict__,metrics=metrics,reward_physical=reward,ppo=dict(learner.ppo.counts)))
  if lower_stochastic:
   old=read(P/'episodes'/f'episode-{p.global_episode:04d}.json')
   for key in ('event_sha256','truncated_delay_s'):compare(n,key,metrics[key],old['metrics'][key])
   compare(n,'reward_physical',reward,old['train_metric']['reward_physical'])
   if logs is not None:
    expected=oldppo.get(p.global_episode,[]);compare(n,'ppo_log_row_count',len(expected),1)
    if len(expected)==1:compare(n,'ppo_update_log',logs,expected[0]['ppo'])
  if n%25==0:payload_save(run,learner,n,binding,'network_snapshot_not_resumable')
  if n in (100,200):payload_save(run,learner,n,binding,'sacdet_full_learner_not_resumable')
  row['wall_seconds']=time.perf_counter()-started;append(run/'train_metrics.jsonl',row)
  atomic_json(run/'progress.json',dict(status='running' if n<total else 'complete',completed_episode=n,total_episodes=total,global_episode=p.global_episode,ppo_batches=updates_count,sac_updates=0,sac_replay_writes=0))
  print(f'{label} [{n}/{total}] global={p.global_episode} delay={row["truncated_delay_s"]:.9f} wall={row["wall_seconds"]:.3f}',flush=True)
  del adapter,policy,episode
 hs=hashes(learner);require(hs['sac_actor']==initial_hashes['sac_actor'],'frozen_sac_actor_hash',dict(initial=initial_hashes,current=hs));require(learner.sac.counts==initial_counts,'sac_counters_unchanged')
 require(not batch and updates_count==total//5,'ppo_batch_budget',dict(actual=updates_count,expected=total//5));require(learner.ppo.counts['batches']==total//5,'ppo_count_budget')
 if lower_stochastic:
  path=P/'snapshots/episode-0325.pt';side=read(path.with_suffix('.json'));require(side['sha256']==digest(path),'control_reference_snapshot_sha')
  old=torch.load(path,map_location='cpu',weights_only=False);reference=Learner(4101,device,config=config);reference.load_state_dict(old['learner']);rh=hashes(reference)
  compare(25,'ppo_actor_hash',hs['ppo_actor'],rh['ppo_actor']);compare(25,'ppo_value_hash',hs['ppo_value'],rh['ppo_value']);del old,reference
  equiv=dict(status='PASS' if first_difference is None else 'FAIL',first_difference=first_difference,comparisons=comparisons,metadata_note='PPO update entries compared in full; original attempt_id is an old run marker, not a PPO output')
  atomic_json(OUT/'loop_equiv.json',equiv);CHECKS['LOOP_EQUIV']=equiv['status'];print('LOOP_EQUIV',equiv['status'],flush=True)
 result=dict(status='complete',episodes=total,ppo_updates=updates_count,sac_actor_hash_unchanged=True,sac_counts_unchanged=True,initial_hashes=initial_hashes,final_hashes=hs,wall_seconds=time.perf_counter()-start)
 atomic_json(run/'result.json',result);del learner;gc.collect();torch.cuda.empty_cache();return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('preflight','train','all'),default='all');args=parser.parse_args()
 config,binding,device=preflight()
 if args.stage=='preflight':return
 control=train_run('control_stoch',True,25,config,binding,device)
 mainrun=train_run('sacdet',False,200,config,binding,device)
 atomic_json(OUT/'training_summary.json',dict(control=control,sacdet=mainrun))
 if args.stage=='all':
  global STAGE
  STAGE='evaluation_and_analysis'
  from evaluate_sacdet import finish
  finish(config,binding,device)
 git=git_states();atomic_json(OUT/'git_after.json',git)
 print('SACDET_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS));traceback.print_exc();sys.exit(1)