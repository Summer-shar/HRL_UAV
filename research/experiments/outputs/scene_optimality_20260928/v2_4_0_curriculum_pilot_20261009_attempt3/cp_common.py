# Attempt3; derived from read-only attempt2/cp_common.py. Resolve both paths and
# compare normalized Windows anchors; no numerical/training/RNG changes.
# Derived from read-only PILOT/pilot_common.py; only output guard renamed to CPIL.
# Derived from OUTM/common.py and NEW/diag_shared.py; writes confined to this CPIL.
# New P1 RNG purpose is checked separately; no source or old-output module is changed.
import sys
sys.dont_write_bytecode=True
import os,json,hashlib,subprocess,csv,gc,time,shutil,traceback
from pathlib import Path,PureWindowsPath
from datetime import datetime,timezone,timedelta
import numpy as np
CPIL=Path(__file__).resolve().parent
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
WORK=WT/'outputs/lite_v2_4_0/multiseed_20261002_203333_r2'
OUTM=MAIN/'outputs/scene_optimality_20260928/v2_4_0_multiseed_20261002_203333_r2'
NEW=MAIN/'outputs/scene_optimality_20260928/v2_4_0_multiseed_diag_20261004'
PY=MAIN/'.venv/Scripts/python.exe';CONFIG=WT/'src/hrl_v2/configs/v2_4_0_lite.json';SCENE=WT/'src/hrl_v2/configs/physical_scene.json'
PROTO=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_test/protocol_v2_4_0_three_stage.json'
PROTO_SHA='6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a'
CHECKS={};STAGE='initialization'
class HardCheckError(RuntimeError):pass
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
 with Path(path).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def _windows_guard_key(resolved):
 # Only normalize equivalent extended-path syntax after filesystem resolution.
 # Keep the resolved path itself for I/O, including its long-path prefix.
 value=str(resolved)
 if value[:8].upper()=='\\\\?\\UNC\\':value='\\\\'+value[8:]
 elif value.startswith('\\\\?\\'):value=value[4:]
 return PureWindowsPath(value)

def output_path(path):
 p=Path(path).resolve()
 root=CPIL.resolve()
 if not _windows_guard_key(p).is_relative_to(_windows_guard_key(root)):
  raise RuntimeError('Write outside CPIL: '+str(p))
 return p
def put(path,value):
 p=output_path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def textfile(path,value):
 p=output_path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
def check(ok,label,details=None):
 if not ok:raise HardCheckError(json.dumps(dict(check=label,details=details),ensure_ascii=False,default=str))
 CHECKS[label]={'status':'PASS','details':details}
def normalize(v):return json.loads(json.dumps(v,allow_nan=False))
def firstdiff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b),key=str):
   if k not in a or k not in b:return dict(field=path+str(k),candidate=str(a.get(k)),reference=str(b.get(k)))
   d=firstdiff(a[k],b[k],path+str(k)+'.')
   if d:return d
  return None
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  if len(a)!=len(b):return dict(field=path+'length',candidate=len(a),reference=len(b))
  for i,(x,y) in enumerate(zip(a,b)):
   d=firstdiff(x,y,path+str(i)+'.')
   if d:return d
  return None
 if isinstance(a,np.generic):
  return None if isinstance(b,np.generic) and a.dtype==b.dtype and a.tobytes()==b.tobytes() else dict(field=path,candidate=str(a),reference=str(b))
 if hasattr(a,'shape'):
  import torch
  if torch.is_tensor(a):return None if torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()) else dict(field=path,candidate='tensor',reference='different tensor')
  return None if isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b) else dict(field=path,candidate='array',reference='different array')
 return None if a==b else dict(field=path.rstrip('.'),candidate=a,reference=b)
def equal(a,b,label):check(firstdiff(a,b) is None,label,firstdiff(a,b))
def git_states():
 out={}
 for root,branch,head in ((MAIN,'codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),(WT,'codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')):
  actual={k:subprocess.check_output(['git','--no-optional-locks','-C',str(root),*args],text=True).strip() for k,args in dict(status=['status','--porcelain'],branch=['branch','--show-current'],head=['rev-parse','HEAD']).items()}
  check(actual==dict(status='',branch=branch,head=head),'git_'+str(root),actual);out[str(root)]=actual
 return out
def gpu_check():
 raw=subprocess.check_output(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4'],text=True,encoding='utf-8',errors='replace')
 pp=json.loads(raw) if raw.strip() else [];pp=pp if isinstance(pp,list) else [pp];parents={int(x['ProcessId']):int(x['ParentProcessId']) for x in pp};excluded=set();pid=os.getpid()
 while pid and pid not in excluded:excluded.add(pid);pid=parents.get(pid,0)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
 gpids={int(next(iter(r.values())).strip()) for r in csv.DictReader(gpu.splitlines())};py=[x for x in pp if (x.get('Name') or '').lower()=='python.exe']
 other=[x for x in py if int(x['ProcessId']) not in excluded and (int(x['ProcessId']) in gpids or any(k in (x.get('CommandLine') or '').lower() for k in ('hrl_v2','train','launch','diag','evaluate','paper')))]
 r=dict(cim_raw=raw,python_processes=py,nvidia_smi_raw=gpu,excluded_process_chain=sorted(excluded),other_training_processes=other)
 check(not other,'gpu_no_other_training',r);return r
def assert_imports():
 import hrl_v2,hrl_uav
 out={x.__name__:x.__file__ for x in (hrl_v2,hrl_uav)}
 check(all(Path(p).resolve().is_relative_to(WT/'src') for p in out.values()),'import_paths',out);return out
def old_run(s):return WORK/f'runs/Proposed/seed_{s}'
def old_sd(s):return OUTM/f'seed_{s}/stage2/runs/sacdet'
def old_l(s):return OUTM/f'seed_{s}/stage3/runs/stage3'
def old_tfoo(s):return WORK/f'runs/TFOO/seed_{s}'
def hist(s,g):return read(NEW/f'summaries/seed_{s}/{g}.json')
def avg(xs):return float(np.mean(xs))
def tail_label(xs):
 improvement=1-xs[-1]/xs[-2]
 return dict(improvement=improvement,label='仍在明显下降（可能没训够）' if improvement>=.02 else '基本平稳' if improvement<=.005 else '略有下降')
def failure(stage,error):
 record=dict(stage=stage,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),completed_checks=CHECKS,substantive_check=isinstance(error,HardCheckError),time=now())
 put(CPIL/'failure.json',record);put(CPIL/'progress.json',dict(status='failed',stage=stage,time=now()));return record
PILOT_REF=MAIN/'outputs/scene_optimality_20260928/v2_4_0_randpre_pilot_20261005'
PILOT=CPIL
PREVIOUS_FAILED_ATTEMPT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_curriculum_pilot_20261008_attempt2'
