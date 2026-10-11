from d_guard import install,ROOT,WriteBoundaryError,allowed
install()
import sys,os,json,time,traceback
from d_pool import run_tasks,save_error
def bomb(task):raise RuntimeError('intentional_worker_failure_for_preflight')
def slow(task):time.sleep(30);return task
def good(task):return task*2
def main():
 mode=sys.argv[1]
 if mode=='worker_failure':run_tasks(bomb,[1,2],timeout_seconds=10,max_workers=2)
 elif mode=='worker_timeout':run_tasks(slow,[1],timeout_seconds=1,max_workers=1)
 elif mode=='success':assert sorted(run_tasks(good,[1,2],timeout_seconds=10,max_workers=2))==[2,4]
 elif mode=='guard':
  with open('NUL','w') as f:f.write('device test')
  allowed('CON');allowed('PRN');allowed('AUX')
  p=ROOT/'selftest'/'allowed.txt';p.parent.mkdir(exist_ok=True)
  p.write_text('inside',encoding='utf-8')
  with open('\\\\?\\'+str(p),'a') as f:f.write(' extended')
  for target in [ROOT.parent/'must_not_be_created_by_jointD.txt',PathOutside()]:
   try:
    with open(target,'w') as f:f.write('forbidden')
   except WriteBoundaryError:pass
   else:raise AssertionError('boundary did not reject '+str(target))
  with open(__file__,'r',encoding='utf-8') as f:assert f.read()
  with open(ROOT.parent/'v2_4_0_jointC_diag_20261010'/'contract.json','r',encoding='utf-8') as f:assert f.read()
  print('GUARD_PASS',flush=True)
def PathOutside():return '\\\\?\\'+str(ROOT.parent/'must_not_be_created_extended_jointD.txt')
if __name__=='__main__':
 try:main()
 except BaseException:
  print(traceback.format_exc(),file=sys.stderr,flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(1)
