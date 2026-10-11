# Orchestrates approved P1-P6 only; derived from OUTM/queue_multiseed.py.
# Serial: all 4104, including diagnostic 3K/report, before any 4105 evaluation/training.
from pilot_common import *
import copy
import torch
import pilot_train as tr
import pilot_eval as ev
import pilot_report as report
from hrl_v2.checkpoint import identity,load,runtime_record,runtime_signature
from hrl_v2.runtime import configure,Learner,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.schedule import phase
from hrl_v2.upper_baselines import upper_phases,load_typed
CURRENT_STAGE='initialize';CONTRACT_SHA=None

def stage(name):
 global CURRENT_STAGE
 CURRENT_STAGE=name;check(sha(PILOT/'pilot_contract.json')==CONTRACT_SHA,'immutable_contract_'+name)
 put(PILOT/'progress.json',dict(status='stage_start',stage=name,time=now()));print('STAGE_START',name,flush=True)

def preflight(seed):
 stage(f'seed_{seed}_preflight');config=read_config(CONFIG,WT);bind=identity(WT,CONFIG);v=read(WORK/'validation/validation.json');equal(bind,binding0(),'current_contract_identity');equal(bind,read(old_run(seed)/'contract.json')['identity'],'Proposed_identity_'+str(seed));equal(bind,v['identity'],'validation_identity_'+str(seed));check(v['status']=='passed','validation_pass_'+str(seed));git=git_states();imports=assert_imports();gpu=gpu_check();device=configure();check(str(device)=='cuda:0','cuda_'+str(seed));equal(runtime_signature(runtime_record()),runtime_signature(v['runtime']),'runtime_validation_'+str(seed));free=shutil.disk_usage(PILOT).free;estimate=read(PILOT/'step0_facts/facts.json')['disk']['estimated_two_seed_bytes']//2;check(free>estimate,'seed_disk_'+str(seed),dict(free=free,estimate=estimate));check(sha(PROTO)==PROTO_SHA,'frozen_protocol_'+str(seed))
 put(PILOT/f'seed_{seed}/preflight.json',dict(status='passed',identity=bind,git=git,imports=imports,gpu=gpu,runtime=runtime_record(),disk_free=free,time=now()));return config,device

def binding0():return read(PILOT/'pilot_contract.json')['identity']

def selected(candidates):return min(enumerate(candidates),key=lambda v:(v[1]['mean_s'],v[0]))[1]

def state_hash(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def state_model(seed,upper,lower,config,device):
 u=load(upper,binding0());h=load(lower,binding0());z=copy.deepcopy(u['learner']);z['sac']=h['learner']['sac'];l=Learner(seed,device,config=config);l.load_state_dict(z)
 equal(l.state_dict()['ppo'],u['learner']['ppo'],'assemble_PPO_U100_or_U200');equal(l.state_dict()['sac'],h['learner']['sac'],'assemble_SAC_H');equal(parameter_hash(l.ppo.actor),state_hash(u['learner']['ppo']['actor']),'assembled_upper_hash_equals_U');equal(parameter_hash(l.sac.actor),state_hash(h['learner']['sac']['actor']),'assembled_lower_hash_equals_H');tr.restore_rng(u);del h,z;return l,u

def execute_seed(seed):
 config,device=preflight(seed);root=PILOT/f'seed_{seed}';data=dict(seed=seed,groups={},timing={},training_results={},runs={},RP={},TFOO_skipped=[],TFOO_checkpoints_evaluated=[])
 def evaluate(label,up,low=None,route=None,kind=None):
  stage(f'seed_{seed}_dev_{label}');r=ev.group(seed,label,up,low,route,kind);data['groups'][label]=r;data['timing']['dev_'+label]=r['wall_seconds'];return r
 stage(f'seed_{seed}_RP_E');ep=root/'rp/RP-E/checks.json'
 if not ep.exists():ev.repro_eval(seed)
 data['RP']['RP-E']=read(ep)
 for label in ('E5','TFOO-200'):
  r=read(root/f'rp/RP-E/{label}/summary.json');equal(r['per_scenario'],hist(seed,label)['per_scenario'],'RP-E_reused_validation_'+str(seed)+'_'+label);data['groups'][label]=r;data['timing']['RP-E_'+label]=r['wall_seconds']
 stage(f'seed_{seed}_RP_training');pp=root/'rp/training_checks.json'
 if not pp.exists():tr.repro_train(seed)
 rp=read(pp);check(rp['status']=='PASS','RP_training_PASS_'+str(seed));data['RP'].update({k:dict(status='PASS',episodes=v['episodes'],SAC_delta=v.get('SAC_delta'),PPO_batches=v.get('PPO_batches_delta'),wall_seconds=v['wall_seconds']) for k,v in rp['results'].items()});data['timing'].update({k:v['wall_seconds'] for k,v in rp['results'].items()})
 stage(f'seed_{seed}_TFOO_dev')
 for p in [Path(v) for v in read(PILOT/'pilot_contract.json')['selection']['TFOO_candidates'][str(seed)]]:
  n=int(p.name.split('-')[1]);label=f'TFOO-{n}'
  try:
   saved=load_typed(p,binding0(),'TFOO');del saved;tr.cleanup()
  except ValueError as e:
   if 'identity' in str(e).lower() or 'digest' in str(e).lower():check(False,'TFOO_identity_or_digest',str(e))
   data['TFOO_skipped'].append(dict(path=str(p),reason=str(e)));continue
  if n==200:r=data['groups']['TFOO-200']
  else:r=evaluate(label,p,route='cvx',kind='TFOO')
  data['groups'][label]=r;data['TFOO_checkpoints_evaluated'].append(n)
 candidates=[dict(label=f'TFOO-{n}',mean_s=data['groups'][f'TFOO-{n}']['mean_s'],checkpoint=str(next(p for p in [Path(v) for v in read(PILOT/'pilot_contract.json')['selection']['TFOO_candidates'][str(seed)]] if int(p.name.split('-')[1])==n))) for n in sorted(data['TFOO_checkpoints_evaluated'])];check(bool(candidates),'TFOO_at_least_one_evaluable');data['TFOO_dev']=selected(candidates);put(root/'TFOO_selection.json',dict(candidates=candidates,selected=data['TFOO_dev'],skipped=data['TFOO_skipped']))
 stage(f'seed_{seed}_step1');run=root/'step1_attempt1';l=Learner(seed,device,config=config);r=tr.run_lower(seed,l,tr.Replay(),[phase(n) for n in range(1,301)],run,random_upper=True,stage1=True);data['training_results']['step1']=r;data['runs']['step1']=str(run);data['timing']['step1']=r['wall_seconds'];del l;tr.cleanup();s300=run/'checkpoints/S300r.pt';s100=run/'checkpoints/S100r.pt';s200=run/'checkpoints/S200r.pt';tf200=old_tfoo(seed)/'checkpoints/episode-0200-anchor.pt';sd200=old_sd(seed)/'checkpoints/episode-0200.pt'
 for n,p in ((100,s100),(200,s200),(300,s300)):evaluate(f'D-a_TFOO_S{n}r',tf200,p,kind='TFOO')
 evaluate('D-a_SD200_S300r',sd200,s300)
 stage(f'seed_{seed}_step2');l,z=tr.learner_from(s300,seed,device,config);tr.restore_rng(z);del z;run=root/'step2_attempt1';r=tr.run_upper(seed,l,upper_phases()[:100],run);data['training_results']['step2']=r;data['runs']['step2']=str(run);data['timing']['step2']=r['wall_seconds'];del l;tr.cleanup();u100=run/'checkpoints/U100.pt'
 stage(f'seed_{seed}_step3');l,z=tr.learner_from(u100,seed,device,config);del z;z=load(s300,binding0());replay=z['replay'];tr.restore_rng(z);del z;run=root/'step3_attempt1';low=[phase(n) for n in range(401,601) if phase(n).layer=='lower'];r=tr.run_lower(seed,l,replay,low[:50],run);data['training_results']['step3']=r;data['runs']['step3']=str(run);data['timing']['step3']=r['wall_seconds'];del l,replay;tr.cleanup()
 hr=[dict(label='H_S300r',mean_s=evaluate('H_S300r',u100,s300)['mean_s'],checkpoint=str(s300))]
 for i in range(1,6):
  p=run/f'checkpoints/c{i}.pt';rr=evaluate(f'H_c{i}',u100,p);hr.append(dict(label=f'H_c{i}',mean_s=rr['mean_s'],checkpoint=str(p)))
 data['H']=selected(hr);put(root/'H_selection.json',dict(candidates=hr,selected=data['H']));h=Path(data['H']['checkpoint'])
 stage(f'seed_{seed}_step4');l,z=state_model(seed,u100,h,config,device);del z;run=root/'step4_attempt1';r=tr.run_upper(seed,l,upper_phases()[100:],run,start_n=100);check(l.ppo.counts['batches']==40,'PPO_final_40_'+str(seed));data['training_results']['step4']=r;data['runs']['step4']=str(run);data['timing']['step4']=r['wall_seconds'];del l;tr.cleanup();u200=run/'checkpoints/U200.pt'
 stage(f'seed_{seed}_step5');l,z=state_model(seed,u200,h,config,device);del z;run=root/'step5_attempt1';r=tr.run_lower(seed,l,tr.Replay(),low[50:],run);data['training_results']['step5']=r;data['runs']['step5']=str(run);data['timing']['step5']=r['wall_seconds'];del l;tr.cleanup()
 fr=[dict(label='F_H',mean_s=evaluate('F_H',u200,h)['mean_s'],checkpoint=str(h))]
 for i in range(6,11):
  p=run/f'checkpoints/c{i}.pt';rr=evaluate(f'F_c{i}',u200,p);fr.append(dict(label=f'F_c{i}',mean_s=rr['mean_s'],checkpoint=str(p)))
 data['F']=selected(fr);put(root/'F_selection.json',dict(candidates=fr,selected=data['F']))
 for n in range(25,201,25):
  p=root/f'step{2 if n<=100 else 4}_attempt1/checkpoints/U{n}.pt';evaluate(f'D-b_U{n}_cvx',p,route='cvx')
 stage(f'seed_{seed}_step3K');l,z=tr.learner_from(u100,seed,device,config);del z;z=load(s300,binding0());replay=z['replay'];tr.restore_rng(z);del z;run=root/'step3K_attempt1';r=tr.run_lower(seed,l,replay,low[:50],run,keep_replay=True);data['training_results']['step3K']=r;data['runs']['step3K']=str(run);data['timing']['step3K']=r['wall_seconds'];del l,replay;tr.cleanup()
 for i in range(1,6):evaluate(f'K{i}',u100,run/f'checkpoints/K{i}.pt')
 data['hard_checks']=dict(status='PASS',SAC_counts={k:r.get('SAC_delta') for k,r in data['training_results'].items()},flushes={k:r.get('replay_flushes_delta') for k,r in data['training_results'].items()},available_completed={k:dict(min=min(r['available_completed']),max=max(r['available_completed'])) for k,r in data['training_results'].items() if 'available_completed' in r},frozen_PPO_all_lower=True,frozen_SAC_all_upper=True,PPO_batches_40=True,all_motion_steps_within20m=True,all_external_seeds_match=True,all_deterministic_stream_metrics_equal=True,contract_unchanged=True,checks=CHECKS)
 stage(f'seed_{seed}_report');put(root/'pipeline_data.json',data);return report.report_seed(seed,data)

def main():
 global CONTRACT_SHA
 CONTRACT_SHA=sha(PILOT/'pilot_contract.json');start=time.perf_counter();put(PILOT/'start.json',dict(time=now(),contract_sha256=CONTRACT_SHA,seed_order=[4104,4105],workers=6,self_pid=os.getpid(),parent_pid=os.getppid()))
 records=[]
 for seed in (4104,4105):
  records.append(execute_seed(seed));put(PILOT/'completed_seeds.json',[r['seed'] for r in records])
 stage('final_report');result=report.final_summary(records);check(sha(PILOT/'pilot_contract.json')==CONTRACT_SHA,'final_contract_unchanged');put(PILOT/'timing.json',dict(total_wall_seconds=time.perf_counter()-start,finished_at=now()));put(PILOT/'progress.json',dict(status='complete',completed_seeds=[4104,4105],time=now()));print('PILOT_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure(CURRENT_STAGE,e)
  try:put(PILOT/'git_after_failure.json',git_states())
  except BaseException:pass
  raise