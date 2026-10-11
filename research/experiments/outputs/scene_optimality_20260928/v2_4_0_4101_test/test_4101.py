"""Derived from S3 stage3_4101.py and R2 sacdet_4101.py; evaluation only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,gc,json,os,subprocess,time,traceback
from datetime import datetime
import torch,hrl_v2,hrl_uav
from hrl_v2.checkpoint import identity,load,atomic_json,digest,json_digest,runtime_record
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.evaluate import manifest,gate_manifest
from hrl_v2.upper_baselines import UpperLearner,load_typed,upper_phases
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900';A=B/'paper_dev_A_P600_B200'
R2=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'
S3=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_stage3'
OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_test'
CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json';SCENE=WT/'src/hrl_v2/configs/physical_scene.json'
PATHS=dict(C10=S3/'runs/stage3/checkpoints/cycle-10.pt',SD200=R2/'runs/sacdet/checkpoints/episode-0200.pt',S300=P/'checkpoints/episode-0300-evaluated.pt',P600=P/'checkpoints/episode-0600-evaluated.pt',P600_pending=P/'checkpoints/episode-0600-pending.pt',**{name:B/f'runs/{name}/seed_4101/checkpoints/episode-0200-anchor.pt' for name in ('NC','TOO','TFOO')})
GROUPS=['Proposed','NC','CA','TOO','TFOO','Abl_TwoStage','Abl_Alternating','Abl_HoverSAC','Abl_UpperCVX']
CONSTRUCTIONS=dict(Proposed="Policy(C10.sac.actor, C10.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",NC="comparison_policy(NC learner, scene, streams, 'moving_local', stochastic=False)",CA="comparison_policy(C10 learner, scene, streams, 'hover_equal', stochastic=False)",TOO="comparison_policy(TOO learner, scene, streams, 'moving_equal', stochastic=False)",TFOO="comparison_policy(TFOO learner, scene, streams, 'moving_cvx', stochastic=False)",Abl_TwoStage="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_Alternating="comparison_policy(P600 learner, scene, streams, 'moving', stochastic=False)",Abl_HoverSAC="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='hover', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_UpperCVX="fixed_upper_policy(SD200.ppo, scene, streams, 'cvx', stochastic=False, mode='moving')")
OWNERS=dict(Proposed=['C10'],NC=['NC'],CA=['C10'],TOO=['TOO'],TFOO=['TFOO'],Abl_TwoStage=['S300','SD200'],Abl_Alternating=['P600'],Abl_HoverSAC=['S300','SD200'],Abl_UpperCVX=['SD200'])
CHECKS={};TIMING={};STAGE='initialization';TOTAL_START=time.perf_counter()
class HardCheckError(RuntimeError):pass
def now():return datetime.now().astimezone().isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(ok,label,details=None):
 if not ok:raise HardCheckError(json.dumps(dict(check=label,details=details),ensure_ascii=False,default=str))
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
 result=dict(ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value))
 if hasattr(learner,'sac'):result['sac_actor']=parameter_hash(learner.sac.actor)
 return result

def preflight():
 global STAGE
 STAGE='preflight';start=time.perf_counter();record=dict(status='checking')
 record['git_before']=git_states();CHECKS['P1']='PASS';atomic_json(OUT/'preflight.json',record)
 for package in (hrl_v2,hrl_uav):require(Path(package.__file__).resolve().is_relative_to(WT/'src'),'import_path',package.__file__)
 device=configure();require(str(device)=='cuda:0','device')
 import scipy,scipy.stats
 record.update(runtime=runtime_record(),torch_threads=torch.get_num_threads(),scipy_version=scipy.__version__,imports={p.__name__:p.__file__ for p in (hrl_v2,hrl_uav)},sys_prefix=sys.prefix);CHECKS['P2']='PASS'
 config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);validation=read(B/'validation/validation.json')
 bindings=[binding,read(P/'contract.json')['identity'],validation['identity'],read(R2/'contract.json')['identity'],read(S3/'contract.json')['identity']]
 require(all(x==binding for x in bindings) and validation['status']=='passed','identity_chain_and_validation')
 record.update(identity=binding,validation_status=validation['status'],identity_chain_count=5);CHECKS['P3']='PASS';atomic_json(OUT/'preflight.json',record)
 cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4"
 raw=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True,encoding='utf-8',errors='replace')
 procs=json.loads(raw) if raw.strip() else [];procs=procs if isinstance(procs,list) else [procs]
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
 gpuro=list(csv.DictReader(gpu.splitlines()));gpids={int(next(iter(r.values())).strip()) for r in gpuro};excluded={os.getpid(),os.getppid()};other=[]
 for proc in procs:
  pid=int(proc['ProcessId']);command=(proc.get('CommandLine') or '').lower()
  if pid not in excluded and (pid in gpids or any(k in command for k in ('hrl_v2','train','launch','diag','evaluate','paper'))):other.append(proc)
 record.update(cim_raw=raw,python_processes=procs,nvidia_smi_raw=gpu,nvidia_smi_rows=gpuro,excluded_pids=sorted(excluded),other_training_processes=other);atomic_json(OUT/'preflight.json',record)
 require(not other,'other_training_or_evaluation_processes',other);CHECKS['P4']='PASS'
 m=read(P/'test_manifest.json');require(m==manifest('test'),'test_manifest_exact')
 require(m['split']=='test' and [r['scenario'] for r in m['scenarios']]==list(range(1,21)) and m['policy_streams']==[1,2,3,4],'test_manifest_structure')
 triple=lambda r:(r['layout_seed'],r['load_seed'],r['arrival_seed'])
 overlap=set(map(triple,m['scenarios']))&set(map(triple,gate_manifest(20)['scenarios']))
 require(not overlap,'test_dev_seed_overlap',list(overlap));CHECKS['P5']='PASS'
 record['test_manifest']=dict(path=str(P/'test_manifest.json'),sha256=digest(P/'test_manifest.json'),json_digest=json_digest(m),dev_overlap_count=len(overlap))
 metadata={};panel=read(A/'summary_deterministic.json')
 for name,path in PATHS.items():
  side=read(path.with_suffix('.json'));sha=digest(path)
  if 'sha256' in side:require(sha==side['sha256'],'checkpoint_sha',name)
  kind=name if name in ('NC','TOO','TFOO') else 'Proposed' if name in ('P600','P600_pending') else None
  saved=load_typed(path,binding,kind) if kind else load(path,binding)
  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(4101,device,config)
  learner.load_state_dict(saved['learner']);hs=hashes(learner)
  metadata[name]=dict(path=str(path),sha256=sha,parameter_hashes=hs,completed_episode=saved['completed_episode'],pending_evaluation=saved['pending_evaluation'],ppo_counts=dict(learner.ppo.counts),typed_loader=kind)
  if name=='C10':
   require(side['completed_episode']==600 and side['pending_evaluation'] is None,'C10_sidecar_endpoint')
   require(hs['sac_actor']==side['parameter_hashes']['sac_actor'],'C10_sac_actor_sidecar')
  if name in ('NC','TOO','TFOO'):
   require(saved['completed_episode']==200 and saved['pending_evaluation'] is None and learner.ppo.counts['batches']==40,'baseline_endpoint_and_budget',name)
   require(sha==panel['identity']['checkpoints'][name],'baseline_A_checkpoint_sha',name)
  del saved,learner;gc.collect();torch.cuda.empty_cache()
 frozen=read(S3/'preflight.json')['frozen_upper']['hashes']
 for k in ('ppo_actor','ppo_value'):
  require(metadata['C10']['parameter_hashes'][k]==frozen[k]==metadata['SD200']['parameter_hashes'][k],'frozen_upper_hash_chain',k)
 require(metadata['SD200']['parameter_hashes']['sac_actor']==metadata['S300']['parameter_hashes']['sac_actor'],'SD200_SAC_frozen')
 require(metadata['P600']['parameter_hashes']==metadata['P600_pending']['parameter_hashes'],'P600_pending_evaluated_hashes')
 CHECKS['P6']='PASS';TIMING['preflight_seconds']=time.perf_counter()-start
 record.update(status='passed',checkpoint_metadata=metadata,checks=dict(CHECKS),wall_seconds=TIMING['preflight_seconds']);atomic_json(OUT/'preflight.json',record)
 print('PREFLIGHT_PASS',flush=True);print(json.dumps(record['runtime']),flush=True)
 return config,binding,device,m,metadata

def freeze_protocol(config,binding,m,metadata):
 global STAGE
 STAGE='protocol_freeze';start=time.perf_counter();require(not (OUT/'windows').exists(),'freeze_before_windows')
 stamp=now();cp=lambda name:{k:metadata[name][k] for k in ('path','sha256','parameter_hashes')}
 s3contract=read(S3/'contract.json');r2contract=read(R2/'contract.json')
 protocol=dict(schema='hrl_v2_4_0_three_stage_protocol_1',frozen_at=stamp,frozen_before_any_test_window=True,
  code=dict(worktree=str(WT),branch='codex/v2-lite-speedup',HEAD=subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip(),identity=binding,CONFIG=dict(path=str(CONFIG),sha256=digest(CONFIG)),SCENE=dict(path=str(SCENE),sha256=digest(SCENE))),
  history=dict(original_preregistration='600 轮：1–300 悬停训练 SAC；301–400 固定 SAC 训练 PPO；401–600 上下层每 10 轮交替',deviations=['第二阶段：上层 200 轮在 upper_phases() 身份上连续训练，SAC 冻结且确定性解码，PPO 随机采集','第三阶段：原 401–600 中 100 个下层轮次连续训练，PPO 冻结；经验池只在第一块开头清空一次','最终模型取第三阶段结束时的 learner，不再用 dev_best'],decided_after=['SACDET r2 的 dev 结果','第三阶段修复试验的 dev 结果'],note='本协议在看过 4101 dev 结果后制定；4101 test 为 test 划分的首次使用'),
  stages=dict(stage1=dict(name='SAC 悬停训练',global_episodes='1–300',procedure='hrl_v2.run 原第一阶段，语义不变',seed_4101_end_checkpoint=cp('S300')),
   stage2=dict(name='上层 PPO 训练（下层冻结、确定性）',episode_identities=[p.global_episode for p in upper_phases()],init='第一阶段末 learner，以及该 checkpoint 的 global CPU/CUDA RNG 状态',lower='SAC 冻结，stochastic=False',upper_collection='upper_stochastic=True',ppo_update='每 5 轮一次，共 40 批',driver=dict(path=str(R2/'sacdet_4101.py'),sha256=digest(R2/'sacdet_4101.py'),entry="train_run('sacdet', lower_stochastic=False, total=200)"),seed_4101_output=cp('SD200')),
   stage3=dict(name='下层 SAC 微调（上层冻结）',episode_identities=s3contract['LOWER'],rules=dict(only_protocol_difference=s3contract['only_protocol_difference'],clarification='经验池按原规则在上层版本变化时清空；因上层冻结，只在第一块（global 411）开头清空一次，之后累积'),driver=dict(path=str(S3/'stage3_4101.py'),sha256=digest(S3/'stage3_4101.py')),seed_4101_output=cp('C10'))),
  final_model=dict(rule='第三阶段 100 个下层轮次全部结束后的 learner（cycle-10）；不按 dev 挑选；cycle-1..9 只作报告曲线',seed_4101={k:metadata['C10'][k] for k in ('path','sha256')}),
  evaluation=dict(policy='上下层均为确定性',proposed_construction=CONSTRUCTIONS['Proposed'],streams="Streams(master, episode=scenario, split='test', policy_stream=stream)",test_manifest=dict(path=str(P/'test_manifest.json'),sha256=digest(P/'test_manifest.json'),json_digest=json_digest(m)),primary_metric='截断时延；先把每个场景全部 stream 窗口池化，再对 20 个场景等权平均，配对口径同 hrl_v2.gates.paired',test_scenarios_master_independent=True),
  baselines={name:dict(internal_name=internal,training='各自独立训练 200 个上层轮次（upper_phases 身份），配固定的确定性下层，取 episode-0200-anchor.pt',evaluation="paper_evaluation.comparison_policy(..., stochastic=False)",seed_4101=dict(path=str(PATHS[name]),sha256=metadata[name]['sha256'])) for name,internal in (('NC','moving_local'),('TOO','moving_equal'),('TFOO','moving_cvx'))},
  ablations=dict(TwoStage='SD200 上层 + SAC-300',Alternating='P600 原交替（仅 seed 4101）',HoverSAC='SAC-300 悬停',UpperCVX='SD200 上层（与 C10 上层哈希相同）+ cvx 路由'),
  statistics=dict(primary_endpoint="test 上 Proposed 对 TFOO-200：20 个场景的配对差，scipy.stats.wilcoxon 双侧，zero_method='wilcox'，α=0.05",also_reported='配对 t 检验（ttest_rel）、符号检验（binomtest）、相对差、胜场',secondary='Proposed 对 NC/CA/TOO，以及四个消融；只作报告，不做多重比较校正'),
  multi_seed_plan=dict(seeds=[4102,4103,4104,4105],pipeline=['stage1：hrl_v2.run 跑到第 300 轮（G1 只报告）','stage2：同上驱动，只做 seed 参数化','stage3：同上驱动，只做 seed 参数化','NC/TOO/TFOO：按 v2.4.0 原基线协议各训练 200 个上层轮次','test 面板：与 4101 同构造（不含 Alternating）'],driver_rule='只允许 seed 参数化；其他任何差异须在启动前逐条列出并获批',aggregate_primary='4101–4105 共 5 个 seed：每个 test 场景先取 5 个 seed 的场景值均值，得到 20 个配对差，Proposed 对 TFOO 做 Wilcoxon 双侧检验，α=0.05；同时报告每个 seed 的场景等权均值、胜场，以及 Proposed 更优的 seed 数'),
  source_contracts=dict(R2=r2contract,S3=s3contract))
 protocol['baselines']['CA']=dict(internal_name='hover_equal',description='悬停 + 均分，不用上层')
 path=OUT/'protocol_v2_4_0_three_stage.json';require(not path.exists(),'new_protocol_file');atomic_json(path,protocol)
 freeze=dict(path=str(path),sha256=digest(path),frozen_at=stamp,windows_exists_at_freeze=(OUT/'windows').exists());atomic_json(OUT/'protocol_freeze.json',freeze)
 require(not freeze['windows_exists_at_freeze'],'freeze_before_windows');CHECKS['protocol_freeze']='PASS';TIMING['protocol_seconds']=time.perf_counter()-start
 print('PROTOCOL_FROZEN',freeze['sha256'],flush=True);return freeze

def main():
 sys.modules['test_4101']=sys.modules[__name__]
 config,binding,device,m,metadata=preflight();freeze=freeze_protocol(config,binding,m,metadata)
 from evaluate_test import replay_checks,test_panel,summarize,compare,report
 global STAGE
 STAGE='dev_replay';replay_checks(config,binding,freeze,metadata)
 STAGE='test_panel';test_panel(config,binding,m,freeze,metadata)
 STAGE='summary';start=time.perf_counter();summary=summarize(m,freeze);comparisons=compare(summary);TIMING['summary_seconds']=time.perf_counter()-start
 STAGE='final_checks';require(digest(freeze['path'])==freeze['sha256'],'protocol_hash_unchanged');CHECKS['protocol_hash_recheck']='PASS'
 git=git_states();atomic_json(OUT/'git_after.json',git);TIMING['total_wall_seconds_before_report_write']=time.perf_counter()-TOTAL_START
 report(summary,comparisons,freeze,git);print('TEST_PANEL_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));traceback.print_exc();sys.exit(1)

