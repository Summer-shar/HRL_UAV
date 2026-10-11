from d_guard import install,ROOT
install()
from j_common import *
from d_setup import input_checks
from d_train import run_seed
from d_results import eval_seed,finish
import sys


def main():
    check(read(ROOT/'preflight.json')['passed'],'preflight_must_pass');input_checks();set_below_normal();reports=[]
    if not (ROOT/'execution_start.json').exists():put(ROOT/'execution_start.json',dict(time=now(),unix=time.time(),pid=os.getpid(),command=sys.argv,OUT=str(ROOT),seed_order=[4104,4103],fresh_training=True))
    started=read(ROOT/'execution_start.json')['unix'];put(ROOT/'active_process.json',dict(pid=os.getpid(),time=now(),command=sys.argv))
    for seed in (4104,4103):
        final=ROOT/f'seed_{seed}/report.json'
        if final.exists():
            report=read(final);check(report['status']=='complete','complete_report_reuse');equal(report['contract_sha256'],sha(ROOT/'contract.json'),'report_same_contract');reports.append(report);continue
        result=ROOT/f'seed_{seed}/train/result.json'
        if result.exists():training=read(result);check(training['hardchecks_passed'],'completed_train_hardchecks')
        else:training=run_seed(seed)
        input_checks();report=eval_seed(seed,training);reports.append(report)
    finish(reports,started);print('JOINT_D_ALL_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:failure('formal_training_or_evaluation',e);raise
