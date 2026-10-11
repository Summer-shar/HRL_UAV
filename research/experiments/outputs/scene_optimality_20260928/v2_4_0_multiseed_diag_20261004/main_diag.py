"""Single sequential main: preflight -> replay -> Q4 -> Q2 -> dev -> cross -> report -> final checks."""
import sys
sys.dont_write_bytecode=True
from diag_shared import *
import traceback
from preflight_diag import preflight,fresh_manifest
from eval_diag import replay,evaluate_all
from logs_diag import log_analysis
from cross_diag import cross_analysis,dev_test_analysis
from report_diag import report
STAGE='starting'
def timed(name,fn):
 global STAGE
 STAGE=name;start=time.perf_counter();put(NEW/'progress.json',dict(status=name,started_at=now()));print('START',name,now(),flush=True)
 result=fn();TIMING[name]=time.perf_counter()-start;print('DONE',name,TIMING[name],flush=True);return result

def main():
 global STAGE
 start=time.perf_counter();config,binding,pre=timed('preflight',preflight)
 rr=timed('replay_checks',replay);fresh=timed('Q4_seed_manifest_only',fresh_manifest);logs=timed('Q2_read_only_logs',log_analysis)
 selection,summaries=timed('Q1_dev_evaluation',lambda:evaluate_all(logs))
 cross=timed('Q3_dev_cross',lambda:cross_analysis(summaries,selection));posthoc=timed('read_existing_test_summaries',lambda:dev_test_analysis(summaries))
 STAGE='report';put(NEW/'progress.json',dict(status=STAGE,updated_at=now()))
 # Initial factual report is followed by the required read-only final checks.
 report(pre,rr,fresh,logs,selection,summaries,cross,posthoc)
 def finish_checks():
  states=git_states();put(NEW/'git_after.json',states);equal(identity(WT,CONFIG),binding,'final_code_config_scene_identity');equal(sha(PROTO),PROTO_SHA,'final_protocol_sha256')
  rows=[]
  for seed,checkpoints in pre['checkpoints'].items():
   for name,meta in checkpoints.items():
    actual=sha(meta['path']);equal(actual,meta['sha256'],'final_checkpoint_unchanged');rows.append(dict(seed=int(seed),checkpoint=name,path=meta['path'],before_sha256=meta['sha256'],after_sha256=actual,passed=True))
  put(NEW/'checkpoint_sha_after.json',rows);CHECKS['final_git_identity_checkpoint_sha']=dict(status='PASS',trees=2,checkpoint_files=len(rows))
 timed('final_readonly_checks',finish_checks);TIMING['total_wall_seconds']=time.perf_counter()-start
 CHECKS['no_training_no_test']=dict(status='PASS',parameter_updates=0,test_windows=0,fresh_test_environments=0)
 put(NEW/'checks.json',CHECKS);put(NEW/'timing.json',TIMING)
 report(pre,rr,fresh,logs,selection,summaries,cross,posthoc)
 put(NEW/'progress.json',dict(status='complete',finished_at=now(),new_windows=CHECKS['dev_windows']['new_windows'],reused_windows=560,total_wall_seconds=TIMING['total_wall_seconds']));print('COMPLETE',now(),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  put(NEW/'failure.json',dict(stage=STAGE,type=type(e).__name__,substantive=isinstance(e,CheckFailure),message=str(e),traceback=traceback.format_exc(),time=now()))
  put(NEW/'progress.json',dict(status='failed',stage=STAGE,error=type(e).__name__,time=now()));traceback.print_exc();sys.exit(1)
