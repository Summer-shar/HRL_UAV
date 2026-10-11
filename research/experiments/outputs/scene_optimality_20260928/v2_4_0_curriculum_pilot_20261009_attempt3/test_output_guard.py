# Lightweight path-only checks. No project/model imports or training/evaluation.
import sys
sys.dont_write_bytecode=True
import json,subprocess,traceback
from pathlib import Path,PureWindowsPath
from unittest.mock import patch
import cp_common as common
ROOT=Path(__file__).resolve().parent
RESULTS=[]
BS=chr(92);PREFIX=BS*2+'?'+BS;UNC=BS*2

def record(name,fn):
 try:
  details=fn();RESULTS.append(dict(name=name,status='PASS',details=details));print('PASS',name,flush=True)
 except BaseException as e:
  RESULTS.append(dict(name=name,status='FAIL',error=str(e),traceback=traceback.format_exc()));print('FAIL',name,str(e),flush=True)

def extended(path):
 s=str(path)
 if s.startswith(PREFIX):return s
 return PREFIX+'UNC'+BS+s[2:] if s.startswith(UNC) else PREFIX+s

def accepts(path):
 actual=common.output_path(path);expected=Path(path).resolve()
 assert actual==expected,'I/O path must preserve resolved representation'
 assert common._windows_guard_key(actual).is_relative_to(common._windows_guard_key(common.CPIL.resolve()))
 return dict(input=str(path),resolved=str(actual))

def rejects(path):
 try:common.output_path(path)
 except RuntimeError as e:
  assert str(e).startswith('Write outside CPIL: ');return dict(input=str(path),rejection=str(e))
 raise AssertionError('Escaping path accepted: '+str(path))

def synthetic_unc(case):
 original=common.CPIL
 try:
  common.CPIL=Path(UNC+'server/share/allowed')
  with patch.object(Path,'resolve',lambda self,*args,**kwargs:self):
   if case=='inside':
    candidate=Path(PREFIX+'UNC'+BS+'SERVER/SHARE/allowed/future/a.json')
    result=common.output_path(candidate);assert str(result)==str(candidate)
    return dict(input=str(candidate),resolved=str(result),filesystem_resolution='isolated no-network representation test')
   candidate=Path(PREFIX+'UNC'+BS+'server/other_share/allowed/a.json')
   return dict(**rejects(candidate),filesystem_resolution='isolated no-network representation test')
 finally:common.CPIL=original

def synthetic_dos_prefix():
 candidate=Path(extended(ROOT/'future/path.json'));original=common.CPIL
 try:
  common.CPIL=Path(str(ROOT))
  with patch.object(Path,'resolve',lambda self,*args,**kwargs:self):
   result=common.output_path(candidate);assert str(result)==str(candidate)
   return dict(input=str(candidate),resolved=str(result),extended_prefix_preserved=str(result).startswith(PREFIX))
 finally:common.CPIL=original

def unc_wrong_root():
 with patch.object(Path,'resolve',lambda self,*args,**kwargs:self):
  return dict(**rejects(Path(PREFIX+'UNC'+BS+'server/share/a.json')),filesystem_resolution='isolated no-network foreign-anchor test')

def junction_escape():
 original=common.CPIL;fixture=ROOT/'guard_test_entities';allowed=fixture/'allowed';outside=fixture/'outside';link=allowed/'escape_link'
 allowed.mkdir(parents=True,exist_ok=False);outside.mkdir(parents=True,exist_ok=False)
 command="New-Item -ItemType Junction -Path '"+str(link)+"' -Target '"+str(outside)+"' -ErrorAction Stop | Out-Null"
 completed=subprocess.run(['powershell','-NoProfile','-Command',command],text=True,capture_output=True,encoding='utf-8',errors='replace')
 if completed.returncode:
  return dict(status='SKIPPED_junction_creation_not_available',returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,all_entities_inside_CPIL=True)
 try:
  common.CPIL=allowed
  accepted=accepts(allowed/'safe.json');rejected=rejects(link/'escape.json')
  target=link.resolve();assert common._windows_guard_key(target)==common._windows_guard_key(outside.resolve())
  return dict(accepted=accepted,rejected=rejected,junction=str(link),target=str(target),all_entities_inside_CPIL=True)
 finally:common.CPIL=original

if __name__=='__main__':
 record('ordinary_inside',lambda:accepts(ROOT/'future/normal.json'))
 record('extended_DOS_inside',lambda:accepts(extended(ROOT/'future/normal.json')))
 record('case_insensitive_inside',lambda:accepts(str(ROOT/'future/normal.json').swapcase()))
 record('nonexistent_deep_long_inside',lambda:accepts(ROOT/('x'*90)/('y'*90)/('z'*90)/'payload.json'))
 record('dotdot_within_root',lambda:accepts(ROOT/'future/../inside.json'))
 record('dotdot_escape_rejected',lambda:rejects(ROOT/'../outside.json'))
 record('similar_directory_prefix_rejected',lambda:rejects(Path(str(ROOT)+'_other')/'payload.json'))
 record('other_drive_rejected',lambda:rejects(Path('C:/guard_test_no_file.json')))
 record('other_drive_extended_rejected',lambda:rejects(Path(PREFIX+'C:/guard_test_no_file.json')))
 record('extended_DOS_return_prefix_preserved',synthetic_dos_prefix)
 record('UNC_extended_equivalent_inside',lambda:synthetic_unc('inside'))
 record('UNC_different_share_rejected',lambda:synthetic_unc('outside'))
 record('UNC_foreign_anchor_rejected',unc_wrong_root)
 record('resolved_directory_junction_escape_rejected',junction_escape)
 result=dict(status='PASS' if all(x['status']=='PASS' for x in RESULTS) else 'FAIL',tests=RESULTS,no_project_model_imports=True,no_training_or_evaluation=True,all_created_entities_inside_CPIL=True)
 (ROOT/'output_guard_tests.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
 assert not any(x.startswith('hrl_v2') or x.startswith('hrl_uav') or x=='torch' for x in sys.modules),'Project/model import occurred'
 print('OUTPUT_GUARD_TESTS',result['status'],len(RESULTS),flush=True)
 sys.exit(0 if result['status']=='PASS' else 1)
