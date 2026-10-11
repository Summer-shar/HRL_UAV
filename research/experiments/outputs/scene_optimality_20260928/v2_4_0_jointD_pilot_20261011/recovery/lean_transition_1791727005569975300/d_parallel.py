from d_guard import install,ROOT
install()
from j_common import *
from d_setup import input_checks
from d_results import finish
import psutil,sys,subprocess


def main():
    input_checks();set_below_normal()
    amendment=read(ROOT/'speedup_amendment.json');baseline=amendment['baseline'];threshold=baseline['baseline_cycle_seconds']*1.6
    processes={};handles=[];retries={4104:0,4103:0};paused=False;tested=False;history=[]
    def launch(seed):
        directory=ROOT/f'seed_{seed}';logs=directory/'logs';logs.mkdir(parents=True,exist_ok=True)
        out=(logs/'main.stdout.log').open('ab',buffering=0);err=(logs/'main.stderr.log').open('ab',buffering=0);handles.extend([out,err])
        env=os.environ.copy();env['D_EVAL_WORKERS']='3'
        p=subprocess.Popen([str(PY),'-B','-u',str(ROOT/'scripts/d_seed.py'),str(seed)],cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        processes[seed]=p
        put(directory/'launch.json',dict(time=now(),pid=p.pid,seed=seed,command=[str(PY),'-B','-u',str(ROOT/'scripts/d_seed.py'),str(seed)],evaluation_workers=3))
        return p
    launch_time=time.time();launch(4104);launch(4103)
    put(ROOT/'recovery/supervisor.json',dict(status='parallel_running',pid=os.getpid(),time=now(),children={str(k):p.pid for k,p in processes.items()},decision_path_changed=False))
    baseline_n=baseline['verified_checkpoint']['completed_episode']
    sample_begin=time.time();psutil.cpu_percent(None);last_sample=0
    while True:
        guard_resources();states={}
        for seed,p in list(processes.items()):
            directory=ROOT/f'seed_{seed}';err=directory/'failure.json'
            if err.exists():
                record=read(err);put(ROOT/'failure.json',record)
                for q in processes.values():
                    for child in psutil.Process(q.pid).children(recursive=True) if q.poll() is None else []:child.terminate()
                    if q.poll() is None:q.terminate()
                put(ROOT/'progress.json',dict(status='failed',seed=seed,time=now(),error=record));return 1
            code=p.poll()
            if code==75 and seed==4103:paused=True
            elif code not in (None,0,75):
                retries[seed]+=1
                put(ROOT/f'recovery/parallel_unexpected_exit_{seed}_{time.time_ns()}.json',dict(time=now(),seed=seed,code=code,attempt=retries[seed]))
                if retries[seed]>=3:raise RuntimeError('same seed unexpected exit repeated 3 times')
                launch(seed)
            if (directory/'report.json').exists():states[str(seed)]={'status':'complete'}
            elif (directory/'train/training_progress.json').exists():states[str(seed)]=read(directory/'train/training_progress.json')
            else:states[str(seed)]={'status':'starting'}
        if paused and (ROOT/'seed_4104/train/result.json').exists():
            request=ROOT/'seed_4103/pause_requested.json'
            if request.exists():request.rename(ROOT/f'recovery/consumed_pause_{time.time_ns()}.json')
            launch(4103);paused=False
        if time.time()-last_sample>=15:
            usage={}
            for seed,p in processes.items():
                try:
                    actual=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
                    usage[str(seed)]=[dict(pid=q.pid,cpu_times=q.cpu_times()._asdict(),rss=q.memory_info().rss) for q in actual]
                except psutil.Error:usage[str(seed)]=[]
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
            history.append(dict(time=time.time(),system_CPU_percent=psutil.cpu_percent(None),GPU=gpu,processes=usage));last_sample=time.time()
        rows4=rows(ROOT/'seed_4104/train/train_metrics.jsonl') if (ROOT/'seed_4104/train/train_metrics.jsonl').exists() else []
        rows3=rows(ROOT/'seed_4103/train/train_metrics.jsonl') if (ROOT/'seed_4103/train/train_metrics.jsonl').exists() else []
        new4=[r for r in rows4 if r['global_episode']>baseline_n]
        if not tested and len(new4)>=5 and len(rows3)>=5 and (ROOT/'seed_4103/train/resume/latest.json').exists():
            def cycle_mean(seed,rr,start):
                end=(ROOT/f"seed_{seed}/train/episodes/episode-{rr[-1]['global_episode']:04d}.json").stat().st_mtime
                return (end-start)/len(rr)
            resumed=ROOT/'seed_4104/recovery/restore_validation.json'
            start4=__import__('datetime').datetime.fromisoformat(read(resumed)['time']).timestamp()
            cycle4=cycle_mean(4104,new4,start4);cycle3=cycle_mean(4103,rows3,launch_time)
            # Exclude the one-time verified restoration from steady training; include checkpoint pauses within these completed rounds.
            report=dict(passed=True,time=now(),rounds={"4104":len(new4),"4103":len(rows3)},baseline=baseline,cycle_seconds={"4104":cycle4,"4103":cycle3},collection_seconds={"4104":avg([r['collection_seconds'] for r in new4]),"4103":avg([r['collection_seconds'] for r in rows3])},slowdown_fraction_4104=cycle4/baseline['baseline_cycle_seconds']-1,threshold_seconds=threshold,resource_samples=history,decision_path_changed=False)
            report['pause_4103']=cycle4>threshold
            put(ROOT/'parallel_benchmark.json',report);tested=True
            if report['pause_4103']:
                put(ROOT/'seed_4103/pause_requested.json',dict(time=now(),reason='4104 cycle slowdown >60%',benchmark=str(ROOT/'parallel_benchmark.json')))
        put(ROOT/'progress.json',dict(status='running',stage='parallel_training_and_evaluation',time=now(),seeds=states,parallel_benchmark_done=tested,seed_4103_paused=paused))
        put(ROOT/'recovery/supervisor.json',dict(status='parallel_running',pid=os.getpid(),time=now(),children={str(k):p.pid for k,p in processes.items()},seed_4103_paused=paused))
        if all((ROOT/f'seed_{seed}/report.json').exists() for seed in (4104,4103)):
            finish([read(ROOT/f'seed_{seed}/report.json') for seed in (4104,4103)],read(ROOT/'execution_start.json')['unix']);break
        time.sleep(2)
    for f in handles:f.close()
    put(ROOT/'recovery/supervisor.json',dict(status='complete',time=now(),pid=os.getpid()));return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:failure('parallel_supervisor',e);raise
