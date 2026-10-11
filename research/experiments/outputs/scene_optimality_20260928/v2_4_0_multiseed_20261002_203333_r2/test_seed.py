# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_test\test_4101.py; changes restricted to seed/path and approved D1-D9.
"""Derived from S3 stage3_4101.py and R2 sacdet_4101.py; evaluation only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,gc,json,os,subprocess,time,traceback
from datetime import datetime
import torch,hrl_v2,hrl_uav
from common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports,PROTO,PROTO_SHA,protocol_check,P4101
assert_imports()
from hrl_v2.checkpoint import identity,load,atomic_json,digest,json_digest,runtime_record
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.evaluate import manifest,gate_manifest
from hrl_v2.upper_baselines import UpperLearner,load_typed,upper_phases
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
P=envpath('P',run_path())
B=envpath('B',WORK)
R2=envpath('R2',stage_path('stage2'))
S3=envpath('S3',stage_path('stage3'))
OUT=envpath('OUT',stage_path('test'))
CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json';SCENE=WT/'src/hrl_v2/configs/physical_scene.json'
PATHS=dict(C10=S3/'runs/stage3/checkpoints/cycle-10.pt',SD200=R2/'runs/sacdet/checkpoints/episode-0200.pt',S300=P/'checkpoints/episode-0300-evaluated.pt',**{name:B/f'runs/{name}/seed_{SEED}/checkpoints/episode-0200-anchor.pt' for name in ('NC','TOO','TFOO')})
GROUPS=['Proposed','NC','CA','TOO','TFOO','Abl_TwoStage','Abl_HoverSAC','Abl_UpperCVX']
CONSTRUCTIONS=dict(Proposed="Policy(C10.sac.actor, C10.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",NC="comparison_policy(NC learner, scene, streams, 'moving_local', stochastic=False)",CA="comparison_policy(C10 learner, scene, streams, 'hover_equal', stochastic=False)",TOO="comparison_policy(TOO learner, scene, streams, 'moving_equal', stochastic=False)",TFOO="comparison_policy(TFOO learner, scene, streams, 'moving_cvx', stochastic=False)",Abl_TwoStage="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_HoverSAC="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='hover', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_UpperCVX="fixed_upper_policy(SD200.ppo, scene, streams, 'cvx', stochastic=False, mode='moving')")
OWNERS=dict(Proposed=['C10'],NC=['NC'],CA=['C10'],TOO=['TOO'],TFOO=['TFOO'],Abl_TwoStage=['S300','SD200'],Abl_HoverSAC=['S300','SD200'],Abl_UpperCVX=['SD200'])
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
 config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);validation=read(envpath('VALIDATION_ROOT',WORK)/'validation/validation.json')
 bindings=[binding,read(P/'contract.json')['identity'],validation['identity'],read(R2/'contract.json')['identity'],read(S3/'contract.json')['identity']]
 require(all(x==binding for x in bindings) and validation['status']=='passed','identity_chain_and_validation')
 record.update(identity=binding,validation_status=validation['status'],identity_chain_count=5);CHECKS['P3']='PASS';atomic_json(OUT/'preflight.json',record)
 record.update(**gpu_check());atomic_json(OUT/'preflight.json',record)
 m=read(P/'test_manifest.json');require(m==manifest('test')==read(P4101/'test_manifest.json'),'test_manifest_exact')
 require(m['split']=='test' and [r['scenario'] for r in m['scenarios']]==list(range(1,21)) and m['policy_streams']==[1,2,3,4],'test_manifest_structure')
 triple=lambda r:(r['layout_seed'],r['load_seed'],r['arrival_seed'])
 overlap=set(map(triple,m['scenarios']))&set(map(triple,gate_manifest(20)['scenarios']))
 require(not overlap,'test_dev_seed_overlap',list(overlap));CHECKS['P5']='PASS'
 record['test_manifest']=dict(path=str(P/'test_manifest.json'),sha256=digest(P/'test_manifest.json'),json_digest=json_digest(m),dev_overlap_count=len(overlap))
 metadata={}
 for name,path in PATHS.items():
  side=read(path.with_suffix('.json'));sha=digest(path)
  if 'sha256' in side:require(sha==side['sha256'],'checkpoint_sha',name)
  kind=name if name in ('NC','TOO','TFOO') else None
  saved=load_typed(path,binding,kind) if kind else load(path,binding)
  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(SEED,device,config)
  learner.load_state_dict(saved['learner']);hs=hashes(learner)
  metadata[name]=dict(path=str(path),sha256=sha,parameter_hashes=hs,completed_episode=saved['completed_episode'],pending_evaluation=saved['pending_evaluation'],ppo_counts=dict(learner.ppo.counts),typed_loader=kind)
  if name=='C10':
   require(side['completed_episode']==600 and side['pending_evaluation'] is None,'C10_sidecar_endpoint')
   require(hs['sac_actor']==side['parameter_hashes']['sac_actor'],'C10_sac_actor_sidecar')
  if name in ('NC','TOO','TFOO'):
   require(saved['completed_episode']==200 and saved['pending_evaluation'] is None and learner.ppo.counts['batches']==40,'baseline_endpoint_and_budget',name)
  del saved,learner;gc.collect();torch.cuda.empty_cache()
 frozen=read(S3/'preflight.json')['frozen_upper']['hashes']
 for k in ('ppo_actor','ppo_value'):
  require(metadata['C10']['parameter_hashes'][k]==frozen[k]==metadata['SD200']['parameter_hashes'][k],'frozen_upper_hash_chain',k)
 require(metadata['SD200']['parameter_hashes']['sac_actor']==metadata['S300']['parameter_hashes']['sac_actor'],'SD200_SAC_frozen')
 CHECKS['P6']='PASS';TIMING['preflight_seconds']=time.perf_counter()-start
 record.update(status='passed',checkpoint_metadata=metadata,checks=dict(CHECKS),wall_seconds=TIMING['preflight_seconds']);atomic_json(OUT/'preflight.json',record)
 print('PREFLIGHT_PASS',flush=True);print(json.dumps(record['runtime']),flush=True)
 return config,binding,device,m,metadata

def main():
 sys.modules['test_seed']=sys.modules[__name__]
 config,binding,device,m,metadata=preflight();freeze=dict(path=str(PROTO),sha256=protocol_check());CHECKS['protocol_hash']='PASS'
 from evaluate_test_seed import test_panel,summarize,compare,report
 global STAGE
 STAGE='test_panel';test_panel(config,binding,m,freeze,metadata)
 STAGE='summary';start=time.perf_counter();summary=summarize(m,freeze);comparisons=compare(summary);TIMING['summary_seconds']=time.perf_counter()-start
 STAGE='final_checks';require(digest(freeze['path'])==freeze['sha256'],'protocol_hash_unchanged');CHECKS['protocol_hash_recheck']='PASS'
 git=git_states();atomic_json(OUT/'git_after.json',git);TIMING['total_wall_seconds_before_report_write']=time.perf_counter()-TOTAL_START
 report(summary,comparisons,freeze,git);atomic_json(OUT/'checks.json',dict(status='passed',checks=CHECKS));print('TEST_PANEL_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));traceback.print_exc();sys.exit(1)

