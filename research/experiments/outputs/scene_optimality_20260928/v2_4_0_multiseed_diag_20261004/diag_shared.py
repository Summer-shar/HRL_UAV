# Dev diagnosis paths and read-only model utilities. No imports from old output directories.
import sys
sys.dont_write_bytecode=True
from common import *
import gc,time
import numpy as np
import torch
from hrl_v2.checkpoint import identity,load,json_digest,runtime_record
from hrl_v2.runtime import configure,Learner,parameter_hash
from hrl_v2.upper_baselines import UpperLearner,load_typed
from hrl_v2.run import read_config
NEW=Path(__file__).resolve().parent
SCENE=WT/'src/hrl_v2/configs/physical_scene.json'
SEEDS=[4101,4102,4103,4104,4105]
ORDER=[4103,4102,4104,4105]
BINDING=None
CHECKS={};TIMING={}
BASEGROUPS=[*[f'S3-c{i}' for i in range(1,11)],'E2','E5','TFOO-200','X1','X2']
def put(path,value):
 path=Path(path).resolve();check(NEW in path.parents,'authorized_write_path',str(path));write(path,value)
def jsonrows(path):
 path=Path(path)
 if not path.exists():return []
 with path.open(encoding='utf-8-sig') as f:return [json.loads(l) for l in f if l.strip()]
def paths(seed):
 stage2=R24101 if seed==4101 else OUTM/f'seed_{seed}/stage2'
 stage3=S34101 if seed==4101 else OUTM/f'seed_{seed}/stage3'
 run=P4101 if seed==4101 else WORK/f'runs/Proposed/seed_{seed}'
 b=B4101 if seed==4101 else WORK
 # Actual E4 path is established by evaluate_sacdet.py and sacdet_4101.py.
 tfoo=b/f'runs/TFOO/seed_{seed}/checkpoints/episode-0200-anchor.pt'
 return dict(S300=run/'checkpoints/episode-0300-evaluated.pt',SD200=stage2/'runs/sacdet/checkpoints/episode-0200.pt',TFOO=tfoo,**{f'C{i}':stage3/f'runs/stage3/checkpoints/cycle-{i:02d}.pt' for i in range(1,11)})
def roots(seed):
 return dict(stage1=P4101 if seed==4101 else WORK/f'runs/Proposed/seed_{seed}',stage2=(R24101 if seed==4101 else OUTM/f'seed_{seed}/stage2')/'runs/sacdet',stage3=(S34101 if seed==4101 else OUTM/f'seed_{seed}/stage3')/'runs/stage3',TFOO=paths(seed)['TFOO'].parent.parent)
def hashes(learner):
 out={f'ppo_{n}':parameter_hash(getattr(learner.ppo,n)) for n in ('actor','value')}
 if hasattr(learner,'sac'):out.update({f'sac_{n}':parameter_hash(getattr(learner.sac,n)) for n in ('actor','q1','q2','target1','target2')})
 return out
def model(path,seed,binding,config,device,kind=None):
 saved=load_typed(path,binding,'TFOO') if kind=='TFOO' else load(path,binding)
 l=(UpperLearner if kind=='TFOO' else Learner)(seed,device,config=config)
 l.load_state_dict(saved['learner']);del saved;gc.collect();return l
def save_csv(path,rows):
 import csv
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clean_gpu():gc.collect();torch.cuda.empty_cache()
def avg(values):
 v=[float(x) for x in values if x is not None and np.isfinite(x)]
 return float(np.mean(v)) if v else None
