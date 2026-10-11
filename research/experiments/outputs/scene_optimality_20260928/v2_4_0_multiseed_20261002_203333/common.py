import sys
sys.dont_write_bytecode=True
import os,json,hashlib,subprocess,csv,shutil
from pathlib import Path
from datetime import datetime
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
WORK=WT/'outputs/lite_v2_4_0/multiseed_20261002_203333'
OUTM=MAIN/'outputs/scene_optimality_20260928/v2_4_0_multiseed_20261002_203333'
PY=MAIN/'.venv/Scripts/python.exe';CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json'
P4101=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
B4101=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'
REF4101=WT/'outputs/lite_v2_4_0/baselines'
R24101=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'
S34101=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_stage3'
T4101=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_test'
PROTO=T4101/'protocol_v2_4_0_three_stage.json'
PROTO_SHA='6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a'
SEED=int(os.environ.get('MULTI_SEED','4101'))
def now():return datetime.now().astimezone().isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
class CheckFailure(RuntimeError):pass
def check(ok,label,details=None):
 if not ok:raise CheckFailure(json.dumps(dict(check=label,details=details),ensure_ascii=False,default=str))
def protocol_check():
 actual=sha(PROTO);check(actual==PROTO_SHA,'protocol_sha256',actual);return actual

def git_states():
 states={}
 for root,branch,head in ((MAIN,'codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),(WT,'codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')):
  r={k:subprocess.check_output(['git','-C',str(root),*cmd],text=True).strip() for k,cmd in dict(status=['status','--porcelain'],branch=['branch','--show-current'],head=['rev-parse','HEAD']).items()}
  check(r==dict(status='',branch=branch,head=head),'git_state',dict(root=str(root),actual=r));states[str(root)]=r
 return states

def gpu_check():
 cmd='Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4'
 raw=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True,encoding='utf-8',errors='replace')
 allprocs=json.loads(raw) if raw.strip() else [];allprocs=allprocs if isinstance(allprocs,list) else [allprocs]
 parents={int(p['ProcessId']):int(p['ParentProcessId']) for p in allprocs}
 excluded=set();pid=os.getpid()
 while pid and pid not in excluded:excluded.add(pid);pid=parents.get(pid,0)
 procs=[p for p in allprocs if (p.get('Name') or '').lower()=='python.exe']
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
 gpurows=list(csv.DictReader(gpu.splitlines()));gpids={int(next(iter(r.values())).strip()) for r in gpurows}
 other=[p for p in procs if int(p['ProcessId']) not in excluded and (int(p['ProcessId']) in gpids or any(k in (p.get('CommandLine') or '').lower() for k in ('hrl_v2','train','launch','diag','evaluate','paper')))]
 record=dict(cim_raw=raw,python_processes=procs,nvidia_smi_raw=gpu,nvidia_smi_rows=gpurows,excluded_pids=sorted(excluded),other_training_processes=other)
 check(not other,'other_training_processes',record);return record

def assert_imports():
 import hrl_v2,hrl_uav
 for p in (hrl_v2,hrl_uav):check(Path(p.__file__).resolve().is_relative_to(WT/'src'),'import_path',p.__file__)
 return {p.__name__:p.__file__ for p in (hrl_v2,hrl_uav)}
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
def equal(a,b,label):
 d=firstdiff(normalize(a),normalize(b));check(d is None,label,d)
def envpath(name,default):return Path(os.environ.get('MULTI_'+name,str(default)))
def run_path(seed=SEED):return WORK/f'runs/Proposed/seed_{seed}'
def stage_path(stage,seed=SEED):return OUTM/f'seed_{seed}/{stage}'
