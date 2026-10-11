import pathlib,json,time,psutil,datetime,sys
from j_common import read
ROOT=pathlib.Path(__file__).resolve().parent.parent
supervisor=int(sys.argv[1])
def now():return datetime.datetime.now().astimezone().isoformat()
def put(path,data):
    assert path.resolve().is_relative_to(ROOT)
    tmp=path.with_name(path.name+'.safety_tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def stop_children():
    stopped=[]
    for proc in psutil.process_iter(['name','cmdline']):
        cmd=' '.join(proc.info['cmdline'] or []).replace('\\','/').lower()
        if proc.pid==psutil.Process().pid:continue
        if (proc.info['name'] or '').lower()=='python.exe' and str(ROOT).replace('\\','/').lower() in cmd and ('d_seed.py' in cmd or 'd_parallel.py' in cmd):
            try:
                for child in proc.children(recursive=True):
                    try:child.terminate();stopped.append(child.pid)
                    except psutil.Error:pass
                proc.terminate();stopped.append(proc.pid)
            except psutil.Error:pass
    put(ROOT/'recovery/parallel_safety_stop.json',dict(time=now(),stopped_pids=sorted(set(stopped)),decision_path_changed=False))
while True:
    p=ROOT/'progress.json'
    state=read(p) if p.exists() else {}
    if state.get('status')=='complete':break
    if (ROOT/'failure.json').exists():stop_children();break
    try:
        cmd=' '.join(psutil.Process(supervisor).cmdline())
        alive='d_parallel.py' in cmd and psutil.Process(supervisor).is_running()
    except psutil.Error:alive=False
    if not alive:
        put(ROOT/'failure.json',dict(time=now(),stage='parallel_supervisor_exit',error='UnexpectedSupervisorExit',message='Parallel supervisor exited before full completion; seed processes stopped at last verified recovery records',previous_progress=state))
        stop_children();break
    time.sleep(2)
