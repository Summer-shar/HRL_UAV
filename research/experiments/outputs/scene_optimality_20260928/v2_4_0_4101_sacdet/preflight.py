import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,subprocess,traceback,gc,time
import torch,hrl_v2
from hrl_v2.checkpoint import identity,load,atomic_json,digest,runtime_record
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.upper_baselines import upper_phases
OUT=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_sacdet')
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'
checks={}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def require(ok,label):
 if not ok:raise AssertionError(label)
try:
 for root,branch,head in ((MAIN,'codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),(WT,'codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')):
  commands={'status':['status','--porcelain'],'branch':['rev-parse','--abbrev-ref','HEAD'],'head':['rev-parse','HEAD']}
  values={k:subprocess.check_output(['git','-C',str(root),*v],text=True).strip() for k,v in commands.items()}
  require(values==dict(status='',branch=branch,head=head),'git state '+str(root));checks[str(root)]=values
 require(Path(hrl_v2.__file__).resolve().is_relative_to(WT/'src'),'import path')
 device=configure();require(str(device)=='cuda:0','cuda device')
 config=read_config(WT/'src/hrl_v2/configs/v2_4_0_lite.json',WT)
 binding=identity(WT,WT/'src/hrl_v2/configs/v2_4_0_lite.json')
 validation=read(B/'validation/validation.json')
 require(binding==read(P/'contract.json')['identity']==validation['identity'],'identity chain')
 require(validation['status']=='passed','validation passed')
 checks['binding']=binding;checks['validation_status']='passed'
 hashes={}
 for label,filename in (('evaluated','episode-0300-evaluated.pt'),('pending','episode-0300-pending.pt'),('initial','episode-0000-initial.pt')):
  saved=load(P/'checkpoints'/filename,binding)
  learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner'])
  hashes[label]=dict(sac_actor=parameter_hash(learner.sac.actor),ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value))
  del saved,learner;gc.collect();torch.cuda.empty_cache()
 require(hashes['evaluated']==hashes['pending'],'300 pending/evaluated hashes')
 require(all(hashes['evaluated'][k]==hashes['initial'][k] for k in ('ppo_actor','ppo_value')),'initial PPO hashes')
 checks['parameter_hashes']=hashes
 phases=upper_phases();require(len(phases)==200,'upper phase length');require([p.global_episode for p in phases[:100]]==list(range(301,401)),'first 100 global episodes');require(phases[99].global_episode==400 and phases[199].global_episode==590,'phase endpoints')
 checks['upper_phases']=dict(length=200,first=301,hundredth=400,two_hundredth=590)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
 checks['nvidia_smi']=gpu
 require(not any('python' in line.lower() for line in gpu.splitlines()[1:]),'other GPU Python process')
 checks['runtime']=runtime_record();checks['status']='passed';atomic_json(OUT/'preflight.json',checks)
 print('PREFLIGHT_PASS',json.dumps(dict(hashes=hashes,upper_phases=checks['upper_phases']),ensure_ascii=False),flush=True)
except BaseException as error:
 atomic_json(OUT/'failure.json',dict(stage='preflight',error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=checks));traceback.print_exc();sys.exit(1)