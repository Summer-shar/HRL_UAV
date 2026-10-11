"""Read-only access and Win32 console devices are allowed; real writes stay in attempt2."""
import os,sys,re
from pathlib import Path,PureWindowsPath
ROOT=Path(__file__).resolve().parent.parent
_INSTALLED=False
class WriteBoundaryError(RuntimeError):pass
def device(path):
 raw=os.fsdecode(path).replace('/','\\')
 # Extended filesystem syntax can name real files called NUL, so never exempt it.
 if raw.startswith('\\\\?\\'):return False
 if raw.startswith('\\\\.\\'):raw=raw[4:]
 name=raw.rstrip('\\').rsplit('\\',1)[-1].split('.')[0].rstrip(': ').upper()
 return bool(re.fullmatch(r'(NUL|CON|CONIN\$|CONOUT\$|PRN|AUX|COM[1-9]|LPT[1-9])',name))
def key(path):
 raw=str(path)
 if raw[:8].upper()=='\\\\?\\UNC\\':raw='\\\\'+raw[8:]
 elif raw.startswith('\\\\?\\'):raw=raw[4:]
 return PureWindowsPath(raw)
def allowed(path):
 if path is None or isinstance(path,int):return
 if device(path):return
 resolved=Path(os.fsdecode(path)).resolve()
 if not key(resolved).is_relative_to(key(ROOT.resolve())):
  raise WriteBoundaryError('Write outside attempt2: '+str(resolved))
def audit(event,args):
 if event=='open':
  p,mode,flags=args
  if (mode and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):allowed(p)
 elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'):
  allowed(args[0])
 elif event in ('os.rename','os.link','os.symlink'):
  allowed(args[0]);allowed(args[1])
def install():
 global _INSTALLED
 if _INSTALLED:return
 sys.dont_write_bytecode=True
 for k,name in [('MPLCONFIGDIR','mplconfig'),('TMP','tmp'),('TEMP','tmp'),('TORCH_HOME','torchcache'),('CUDA_CACHE_PATH','cudacache')]:
  p=ROOT/name;p.mkdir(exist_ok=True);os.environ[k]=str(p)
 os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONIOENCODING']='utf-8'
 os.environ['PYTHONPATH']='D:/HRL_UAV_V2_speedup/src'
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 sys.addaudithook(audit);_INSTALLED=True
