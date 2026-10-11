"""Bounded dispatch; every worker failure is persisted before it is propagated."""
from d_guard import install,ROOT
install()
import os,json,time,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
def save_error(name,data):
 root=ROOT/'errors';root.mkdir(exist_ok=True)
 path=root/(name+'_'+str(time.time_ns())+'.json')
 with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,default=str)
 return str(path)
def entry(function,task):
 try:return function(task)
 except BaseException as e:
  path=save_error('worker',dict(pid=os.getpid(),task=task,error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()))
  print('WORKER_FAILURE_RECORDED',path,flush=True);raise
def close_failed(ex,futures):
 processes=list((ex._processes or {}).values())
 for f in futures:f.cancel()
 try:ex._call_queue.cancel_join_thread()
 except (AttributeError,ValueError):pass
 ex.shutdown(wait=False,cancel_futures=True)
 for p in processes:
  if p.is_alive():p.terminate()
 for p in processes:
  p.join(timeout=1)
  if p.is_alive():p.kill()
def run_tasks(function,tasks,initializer=None,timeout_seconds=60,max_workers=6,resource_check=None):
 ex=ProcessPoolExecutor(max_workers=max_workers,initializer=initializer,mp_context=multiprocessing.get_context('spawn'))
 pending={};iterator=iter(tasks);results=[]
 def submit():
  try:t=next(iterator)
  except StopIteration:return False
  pending[ex.submit(entry,function,t)]=(time.monotonic(),t);return True
 try:
  for _ in range(min(max_workers,len(tasks))):submit()
  while pending:
   done,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
   # Check completed futures first so the original worker traceback is not hidden.
   for f in done:
    result=f.result();pending.pop(f);results.append(result);submit()
   if resource_check:resource_check()
   elapsed=time.monotonic()
   for f,(start,t) in pending.items():
    if elapsed-start>timeout_seconds:
     raise TimeoutError('Task exceeded %.6f seconds (including worker startup): %r'%(timeout_seconds,t))
 except BaseException as e:
  save_error('pool',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),active_tasks=[v[1] for v in pending.values()],timeout_seconds=timeout_seconds,time=time.time()))
  close_failed(ex,pending);raise
 ex.shutdown(wait=True)
 return results
