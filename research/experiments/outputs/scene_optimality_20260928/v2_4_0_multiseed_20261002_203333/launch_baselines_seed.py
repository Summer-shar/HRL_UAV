# Derived from launch_baselines.py; D7 no dev panels, seed/path parameterized.
import sys
sys.dont_write_bytecode=True
import time,subprocess,json
from common import *
assert_imports()
from hrl_v2.checkpoint import identity,load,atomic_json
from hrl_v2.upper_baselines import load_typed,UpperLearner
from hrl_v2.runtime import configure,parameter_hash
ORDER=('TOO','TFOO','NC')
def verify_baseline(name,binding,device):
 output=WORK/f'runs/{name}/seed_{SEED}';result=read(output/'result.json')
 check(result.get('status')=='complete_performance_not_evaluated' and result.get('completed_episode')==200,'baseline_completion',name)
 checks=[]
 for n,batches in ((100,20),(200,40)):
  path=output/f'checkpoints/episode-{n:04d}-anchor.pt';saved=load_typed(path,binding,name)
  check(saved['completed_episode']==n and saved['pending_evaluation'] is None and saved['learner']['ppo']['counts']['batches']==batches,'baseline_anchor_budget',str(path))
  checks.append(dict(episode=n,ppo_batches=batches,sha256=sha(path)));del saved
 initial=load_typed(output/'checkpoints/episode-0000-initial.pt',binding,name);learner=UpperLearner(SEED,device,read(CONFIG));learner.load_state_dict(initial['learner'])
 hs=dict(upper_actor=parameter_hash(learner.ppo.actor),value=parameter_hash(learner.ppo.value));expected=read(run_path()/'contract.json')['initial_parameters']
 return dict(anchors=checks,initial_parameters=hs,initial_upper_matches_proposed={k:hs[k]==expected[k] for k in hs})

def main():
 binding=identity(WORK,CONFIG);git_states();jobs=[];failures=[];start=time.perf_counter()
 # Queue checks GPU once before this launch; baseline child processes do not check GPU.
 try:
  for name in ORDER:
   output=WORK/f'runs/{name}/seed_{SEED}';check(not output.exists(),'new_baseline_output',str(output))
   cmd=[str(PY),'-B','-u',str(OUTM/'checked_entry.py'),'hrl_v2.train_upper','--project-root',str(WORK),'--config',str(CONFIG),'--baseline',name,'--seed',str(SEED),'--fresh','--output',str(output)]
   logfile=WORK/f'logs/seed_{SEED}_{name}.log';handle=logfile.open('x',encoding='utf-8');t=time.perf_counter()
   child=subprocess.Popen(cmd,cwd=WT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   job=dict(name=name,child=child,handle=handle,command=cmd,started=t,started_at=now());jobs.append(job)
 except BaseException:
  for j in jobs:j['child'].wait();j['handle'].close()
  raise
 pending=list(jobs);completed=[]
 while pending:
  for j in list(pending):
   code=j['child'].poll()
   if code is None:continue
   j['handle'].close();pending.remove(j)
   row=dict(name=j['name'],pid=j['child'].pid,command=j['command'],exit_code=code,started_at=j['started_at'],ended_at=now(),wall_seconds=time.perf_counter()-j['started']);completed.append(row)
   write(WORK/f"logs/seed_{SEED}_{j['name']}.process.json",row)
   if code:failures.append(row)
  write(WORK/f'baseline_progress_seed_{SEED}.json',dict(status='failed_waiting_for_children' if failures else 'running',children=[dict(name=j['name'],pid=j['child'].pid) for j in pending],completed=completed,failures=failures,no_children_killed=True))
  if pending:time.sleep(2)
 check(not failures,'baseline_children',failures)
 device=configure();checks={name:verify_baseline(name,binding,device) for name in ORDER}
 write(OUTM/f'seed_{SEED}/baseline_checks.json',dict(status='passed',checks=checks,wall_seconds=time.perf_counter()-start,processes=completed))
 git_states()
if __name__=='__main__':main()
