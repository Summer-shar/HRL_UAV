# Derived from hrl_v2.run.main; D1 wrapper, unchanged 1-300 training call.
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,gc,traceback
from common import *
assert_imports()
import torch
import hrl_v2.run as run
from hrl_v2.schedule import phase as original_phase
from hrl_v2.checkpoint import load,identity,atomic_json
from hrl_v2.runtime import Learner,configure,parameter_hash
from hrl_v2.evaluate import manifest
class StopStage1(Exception):pass

def guard_for(output):
 def guarded(number):
  if number==301 and (Path(output)/'checkpoints/episode-0300-evaluated.pt').exists():raise StopStage1()
  return original_phase(number)
 return guarded

def postcheck(root,config_path,output):
 binding=identity(root,config_path);config=run.read_config(config_path,root);device=configure()
 ev=output/'checkpoints/episode-0300-evaluated.pt';pd=output/'checkpoints/episode-0300-pending.pt'
 check(ev.is_file() and ev.with_suffix('.json').is_file(),'300_evaluated_files')
 saved=load(ev,binding);check(saved['completed_episode']==300 and saved['pending_evaluation'] is None,'300_endpoint')
 learner=Learner(SEED,device,config=config);learner.load_state_dict(saved['learner'])
 hs=dict(sac_actor=parameter_hash(learner.sac.actor),ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value));counts=dict(learner.sac.counts)
 check(learner.ppo.counts['batches']==0,'stage1_ppo_zero');del learner,saved;gc.collect();torch.cuda.empty_cache()
 pending=load(pd,binding);learner=Learner(SEED,device,config=config);learner.load_state_dict(pending['learner'])
 phs=dict(sac_actor=parameter_hash(learner.sac.actor),ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value))
 equal(hs,phs,'300_pending_evaluated_hashes');del learner,pending;gc.collect();torch.cuda.empty_cache()
 old=load(P4101/'checkpoints/episode-0300-evaluated.pt',binding);equal(counts,old['learner']['sac']['counts'],'stage1_sac_counts_vs4101');del old;gc.collect()
 initial=read(output/'contract.json')['initial_parameters'];check(hs['ppo_actor']==initial['upper_actor'] and hs['ppo_value']==initial['value'],'stage1_upper_unchanged')
 metrics=[json.loads(x) for x in (output/'train_metrics.jsonl').read_text().splitlines() if x.strip()]
 equal([r['global_episode'] for r in metrics],list(range(1,301)),'stage1_exact_episodes')
 forbidden=[]
 for folder in ('episodes','checkpoints','gates','evaluations','snapshots'):
  for path in (output/folder).glob('episode-*'):
   n=int(path.name.split('-')[1].split('.')[0]);
   if n>300:forbidden.append(str(path))
 check(not forbidden,'no_301_outputs',forbidden)
 equal(read(output/'test_manifest.json'),manifest('test'),'own_test_manifest');equal(read(output/'test_manifest.json'),read(P4101/'test_manifest.json'),'test_manifest_vs4101')
 atomic_json(output/'checks.json',dict(status='passed',sac_counts=counts,ppo_batches=0,parameter_hashes=hs,gate=read(output/'gates/episode-0300.json')))

def main():
 root=WORK.resolve();rooted=lambda p:p if p.is_absolute() else root/p
 config_path=rooted(CONFIG);config=run.read_config(config_path,root);binding=identity(root,config_path)
 run.require_research_protocol(config)
 output=rooted(envpath('P',run_path()));reference=rooted(WORK/f'references/seed_{SEED}/baselines/summary.json')
 run.phase=guard_for(output)
 try:
  result=run.train(root,config_path,SEED,output,None,reference,checkpoint_every=10)
  check(result.get('status')=='paused_G1','stage1_normal_return',result)
 except StopStage1:
  result=dict(status='stopped_after_300',completed_episode=300)
  atomic_json(output/'stage1_stop.json',result)
 except Exception as error:
  if output.exists():atomic_json(output/'technical_failure.json',dict(error=type(error).__name__,message=str(error)))
  raise
 postcheck(root,config_path,output);print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=='__main__':main()
