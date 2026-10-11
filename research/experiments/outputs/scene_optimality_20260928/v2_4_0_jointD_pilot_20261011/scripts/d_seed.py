from d_guard import install,ROOT
install()
from j_common import *
from d_setup import input_checks
from d_train import run_seed,ControlledPause
from d_results import eval_seed
import sys

def main(seed):
    check(read(ROOT/'preflight.json')['passed'],'original_preflight_passed');input_checks();set_below_normal()
    directory=ROOT/f'seed_{seed}';directory.mkdir(exist_ok=True)
    put(directory/'active_process.json',dict(pid=os.getpid(),time=now(),command=sys.argv,status='training'))
    result=directory/'train/result.json'
    if result.exists():
        training=read(result);check(training['hardchecks_passed'],'completed_train_hardchecks')
    else:training=run_seed(seed)
    input_checks();put(directory/'active_process.json',dict(pid=os.getpid(),time=now(),command=sys.argv,status='evaluation',workers=int(os.environ.get('D_EVAL_WORKERS','6'))))
    if not (directory/'report.json').exists():eval_seed(seed,training)
    put(directory/'active_process.json',dict(pid=os.getpid(),time=now(),status='complete'))
    return 0

if __name__=='__main__':
    seed=int(sys.argv[1])
    try:sys.exit(main(seed))
    except ControlledPause as e:
        put(ROOT/f'seed_{seed}/active_process.json',dict(status='paused',time=now(),reason=str(e)));sys.exit(75)
    except Exception as e:
        put(ROOT/f'seed_{seed}/failure.json',dict(time=now(),seed=seed,error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),substantive_check=isinstance(e,HardCheckError)));raise
