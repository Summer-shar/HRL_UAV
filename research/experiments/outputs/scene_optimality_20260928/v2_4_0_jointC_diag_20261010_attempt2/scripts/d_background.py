from d_guard import install,ROOT
install()
import os,sys,json,subprocess,datetime,traceback
PY='D:/HRL_UAV_V2/.venv/Scripts/python.exe'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(name,data):
 p=ROOT/'recovery'/name
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def run():
 write('supervisor.json',dict(status='running',pid=os.getpid(),time=now(),independent_background=True))
 for name in ('d_preflight.py','d_resume.py'):
  stdout=ROOT/'logs'/('resume_'+name+'.stdout.log');stderr=ROOT/'logs'/('resume_'+name+'.stderr.log')
  with stdout.open('ab',buffering=0) as out,stderr.open('ab',buffering=0) as err:
   p=subprocess.Popen([PY,'-B','-u',str(ROOT/'scripts'/name)],cwd=str(ROOT),env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
   write('supervisor.json',dict(status='running',pid=os.getpid(),child_pid=p.pid,script=name,time=now(),stdout=str(stdout),stderr=str(stderr)))
   code=p.wait()
  write(name+'.exit.json',dict(time=now(),returncode=code,stdout=str(stdout),stderr=str(stderr)))
  if code:
   write('supervisor.json',dict(status='failed',script=name,returncode=code,time=now()))
   if not (ROOT/'failure.json').exists():
    data=dict(status='failed',stage='background_child_exit',script=name,returncode=code,time=now())
    (ROOT/'failure.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
   return code
 write('supervisor.json',dict(status='complete',time=now()));return 0
if __name__=='__main__':
 try:sys.exit(run())
 except Exception:
  write('supervisor_error.json',dict(time=now(),traceback=traceback.format_exc()));raise
