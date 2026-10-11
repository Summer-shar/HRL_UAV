import sys
sys.dont_write_bytecode=True
import subprocess,time,traceback,os,msvcrt
from common import *
STATE=dict(status='starting',pid=os.getpid(),seed=None,step='startup',started_at=now(),completed_steps=[],auto_resume=False)
START=time.perf_counter()
def state(status,**fields):
 STATE.update(status=status,updated_at=now(),**fields);write(WORK/'queue_status.json',STATE);print(json.dumps({k:v for k,v in STATE.items() if k!='completed_steps'},ensure_ascii=False),flush=True)

def launch(step,arguments,seed=None):
 state('running',seed=seed,step=step,step_started_at=now(),step_ended_at=None,step_wall_seconds=None)
 log=WORK/'logs'/f"{'seed_'+str(seed)+'_' if seed is not None else ''}{step}.log"
 command=[str(PY),'-B','-u']+list(map(str,arguments));started=time.perf_counter();began=now()
 with log.open('x',encoding='utf-8') as f:
  child=subprocess.Popen(command,cwd=WT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  state('running',child_pid=child.pid,child_command=command,child_log=str(log))
  code=child.wait()
 record=dict(seed=seed,step=step,command=command,pid=child.pid,exit_code=code,started_at=began,ended_at=now(),wall_seconds=time.perf_counter()-started,log=str(log))
 write(log.with_suffix('.process.json'),record);STATE['completed_steps'].append(record)
 state('step_complete' if code==0 else 'failed',step_ended_at=record['ended_at'],step_wall_seconds=record['wall_seconds'],child_pid=None)
 if code:
  failures=[]
  for candidate in (envpath('OUT',OUTM)/'failure.json',run_path(seed or 4101)/'technical_failure.json'):
   if candidate.is_file():failures.append(dict(path=str(candidate),record=read(candidate)))
  check(False,'subprocess_exit',dict(process=record,child_failures=failures,log_tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-80:]))
 return record

def main():
 (WORK/'logs').mkdir(exist_ok=True)
 with (WORK/'queue.lock').open('a+b') as handle:
  handle.seek(0,2)
  if handle.tell()==0:handle.write(b'0');handle.flush()
  handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
  try:
   check('torch' not in sys.modules,'queue_parent_no_torch_no_cuda')
   startup=dict(git=git_states(),protocol_sha256=protocol_check(),free_bytes=shutil.disk_usage('D:/').free,gpu=gpu_check(),directories_absent_before_creation=read(OUTM/'startup.json')['directories_absent_before_creation'])
   check(startup['free_bytes']>=150*1024**3,'startup_disk_150GB',startup['free_bytes'])
   for root,path in ((MAIN,OUTM),(WT,WORK)):subprocess.check_call(['git','-C',str(root),'check-ignore',str(path)])
   check(startup['directories_absent_before_creation'],'new_directories_pre_creation')
   write(OUTM/'startup_queue.json',startup)
   launch('validation',[OUTM/'checked_entry.py','hrl_v2.validate','--config',CONFIG,'--output',WORK/'validation','--skip-benchmark'])
   launch('validation_check',[OUTM/'check_validation.py'])
   for step in ('RP-W','RP-S2','RP-S3','RP-T','RP-REF'):launch(step,[OUTM/'repro_4101.py',step])
   repro={step:read(OUTM/f'repro_4101/{step}/checks.json') for step in ('RP-W','RP-S2','RP-S3','RP-T','RP-REF')};check(all(r['status']=='passed' for r in repro.values()),'all_reproduction_pass')
   write(OUTM/'repro_4101/checks.json',dict(status='passed',results=repro))
   for seed in (4102,4103,4104,4105):
    os.environ['MULTI_SEED']=str(seed)
    for key in ('MULTI_OUT','MULTI_P','MULTI_R2','MULTI_S3','MULTI_B','MULTI_VALIDATION_ROOT'):os.environ.pop(key,None)
    state('running',seed=seed,step='seed_preflight',seed_started_at=now())
    free=shutil.disk_usage('D:/').free;check(free>=40*1024**3,'seed_disk_40GB',free)
    write(OUTM/f'seed_{seed}/startup.json',dict(protocol_sha256=protocol_check(),free_bytes=free,gpu=gpu_check(),git=git_states()))
    for step,driver in (('reference','reference_seed.py'),('stage1','stage1_wrapper.py'),('stage2','stage2_seed.py'),('stage3','stage3_seed.py'),('baselines','launch_baselines_seed.py'),('test','test_seed.py')):
     if step=='baselines':write(OUTM/f'seed_{seed}/gpu_before_baselines.json',gpu_check())
     launch(step,[OUTM/driver],seed)
    state('seed_complete',seed=seed,seed_ended_at=now())
   protocol_check();launch('aggregate',[OUTM/'aggregate.py']);protocol_check()
   check('torch' not in sys.modules,'queue_parent_never_initialized_cuda')
   write(OUTM/'final_checks.json',dict(status='passed',git=git_states(),protocol_sha256=protocol_check(),total_wall_seconds=time.perf_counter()-START,parent_cuda_never_initialized=True))
   state('complete',seed=None,step='finished',ended_at=now(),total_wall_seconds=time.perf_counter()-START)
  finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  failure=dict(seed=STATE.get('seed'),step=STATE.get('step'),check=str(error),error=type(error).__name__,details=str(error),traceback=traceback.format_exc(),at=now(),automatic_resume=False)
  write(OUTM/'failure.json',failure);state('failed',error=failure,total_wall_seconds=time.perf_counter()-START)
  (OUTM/'report.md').write_text('# 多seed队列停止\n\n```json\n'+json.dumps(failure,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
  traceback.print_exc();sys.exit(1)
