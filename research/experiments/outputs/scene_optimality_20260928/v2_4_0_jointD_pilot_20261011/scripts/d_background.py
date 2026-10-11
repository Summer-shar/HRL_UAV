from d_guard import install,ROOT
install()
from j_common import *
import subprocess,sys,psutil


def child(name):
    with (ROOT/'logs/main.stdout.log').open('ab',buffering=0) as out,(ROOT/'logs/main.stderr.log').open('ab',buffering=0) as err:
        p=subprocess.Popen([str(PY),'-B','-u',str(ROOT/'scripts'/name)],cwd=str(ROOT),env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        put(ROOT/'recovery/supervisor.json',dict(status='running',pid=os.getpid(),child_pid=p.pid,script=name,time=now(),independent_background=True,full_command=[str(PY),'-B','-u',str(ROOT/'scripts'/name)]))
        code=p.wait()
    put(ROOT/f'recovery/{name}.{time.time_ns()}.exit.json',dict(time=now(),returncode=code,script=name))
    return code


def main():
    put(ROOT/'recovery/supervisor_start.json',dict(pid=os.getpid(),time=now(),independent_background=True))
    for name in ('d_setup.py','d_preflight.py','d_main.py'):
        if name=='d_setup.py' and (ROOT/'contract.json').exists():continue
        if name=='d_preflight.py' and (ROOT/'preflight.json').exists() and read(ROOT/'preflight.json')['passed']:continue
        for attempt in range(3):
            code=child(name)
            if code==0:break
            if (ROOT/'failure.json').exists():return code
            # An unexpected process exit with no substantive error may resume the nearest verified boundary.
            marker=ROOT/'recovery'/('unexpected_exit_'+str(time.time_ns())+'.json');put(marker,dict(time=now(),script=name,returncode=code,attempt=attempt+1,reason='process interrupted without recorded exception; retry same frozen script/state'))
            for proc in psutil.Process().children(recursive=True):
                if proc.pid!=os.getpid() and proc.is_running():proc.terminate()
        else:raise RuntimeError('same interruption repeated 3 times')
    put(ROOT/'recovery/supervisor.json',dict(status='complete',pid=os.getpid(),time=now()));return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:failure('background_supervisor',e);raise
