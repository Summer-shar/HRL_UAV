# Immutable curriculum contract, derived from PILOT/create_contract.py.
from cp_common import *
from hrl_v2.schedule import phase
from hrl_v2.upper_baselines import upper_phases
from hrl_v2.randomness import derive
from hrl_v2.checkpoint import identity,json_digest
from hrl_v2.evaluate import gate_manifest
import copy
facts=read(CPIL/'step0_facts/facts.json')
equal(facts['status'],'PASS','step0_PASS_before_contract')
check(not (CPIL/'cp_contract.json').exists(),'contract_new')
pilot=read(PILOT_REF/'pilot_contract.json')
equal(identity(WT,CONFIG),facts['identity'],'contract_identity_step0')
equal(pilot['identity'],facts['identity'],'contract_identity_pilot')
def table(pp,kind):
 out=[]
 for p in pp:
  row=dict(**p.__dict__,actor_enabled=p.actor_enabled,external_seeds={k+'_seed':derive(4104,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')})
  if kind=='curriculum':row.update(upper='hover' if p.global_episode<=200 else 'random_uniform',replay_label='hover-0' if p.global_episode<=200 else 'random-uniform-0')
  out.append(row)
 return out
low=[phase(n) for n in range(401,601) if phase(n).layer=='lower'];up=upper_phases()
counts1=dict(triggers=27000,skipped=90,events=26910,critic_fits=53820,target_updates=26910,actor_updates=13410)
counts200=dict(triggers=18000,skipped=90,events=17910,critic_fits=35820,target_updates=17910,actor_updates=8910)
counts100=dict(triggers=9000,skipped=0,events=9000,critic_fits=18000,target_updates=9000,actor_updates=4500)
countsK=dict(triggers=4500,skipped=0,events=4500,critic_fits=9000,target_updates=4500,actor_updates=1800)
criteria=copy.deepcopy(pilot['criteria'])
criteria['criterion1']=criteria['criterion1'].replace('U200+cvx','A-U200+cvx').replace('U200cvx','A-U200cvx')
criteria['criterion2']=criteria['criterion2'].replace('F/TFOOdev','A-F/TFOO-200')
deviations=[
'\u7b2c\u4e00\u9636\u6bb5\u6539\u4e3a\u60ac\u505c200\u8f6e+\u968f\u673a\u4e0a\u5c42100\u8f6e',
'\u6362\u4e0a\u5c42\u65f6\u4e0d\u6e05\u7a7a\u7ecf\u9a8c\u6c60',
'\u7b2c\u4e09\u3001\u4e94\u6b65\u4e0d\u6e05\u7a7a\u7ecf\u9a8c\u6c60',
'\u7b2c\u4e94\u6b65\u4eceA-H\u81ea\u5df1\u7684\u7ecf\u9a8c\u6c60\u63a5\u7740\u7ec3',
'\u6309dev\u9009\u62e9A-H\u3001A-F\uff1b\u540c\u8bd5\u70b9\uff0c\u4e0e\u51bb\u7ed3\u534f\u8bae\u4e0d\u7528development_best\u7684\u56fa\u5b9a\u9884\u7b97\u9009\u62e9\u4e0d\u540c',
'A-c1..A-c5\u5e26\u7ecf\u9a8c\u6c60\u548c\u5168\u5c40\u968f\u673a\u6570',
'DG-SAC300\u4e0eDG-v1c8\u53ea\u4f5c\u8bca\u65ad',
'\u53ea\u8dd14104\uff0c\u4e0d\u81ea\u52a8\u8fdb\u51654105']
m=gate_manifest(20)
contract=dict(created_at=now(),protocol='v2_4_0_curriculum_pilot_dev',seed=4104,seeds=[4104],identity=facts['identity'],frozen_protocol_sha256=PROTO_SHA,previous_failed_attempt=str(PREVIOUS_FAILED_ATTEMPT),previous_failure='step0 disk_free_greater_than_two_times_estimate',source_prompt='C:/Users/PC/.codex/attachments/944824f3-3d1c-4f77-bbc9-3ee2b0b57152/\u5df2\u7c98\u8d34\u7684\u6587\u672c.txt',directory_authorization='user approved fresh attempt2 and requested training after approved cache cleanup',paths={k:str(v) for k,v in dict(CPIL=CPIL,WORK=WORK,OUTM=OUTM,NEW=NEW,PILOT_REF=PILOT_REF,ORIG=old_run(4104),PY=PY,CONFIG=CONFIG,SCENE=SCENE).items()},environment=dict(PYTHONPATH=str(WT/'src'),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(CPIL/'mplconfig')),inputs_sha256=facts['protected_input_sha256'],input_models=facts['input_models'],original_training_manifest=str(old_run(4104)/'training_manifest.json'),original_episode_sha256={str(n):facts['protected_input_sha256'][str(old_run(4104)/f'episodes/episode-{n:04d}.json')] for n in range(1,301)},evaluation=dict(manifest='gate_manifest(20)',manifest_sha256=json_digest(m),split='dev',scenario_ids=list(range(1,21)),streams=[1,2],upper='PPO tanh mean',lower='argmax support then Dirichlet mean',workers=6,device='cuda:0',metric='mean of20 per_scenario.truncated_delay_s',selection_metric='minimum dev20 scenario-equal mean truncated_delay_s',tie_break='earliest candidate',test_evaluation=False,protocol='v2_4_0_curriculum_pilot_dev'),random_upper=dict(purpose_identifier=13,seed_formula="derive(4104,1,0,n,purpose='random_upper')",action='independent Uniform[-1,1]^6 at every existing macro commit including true successor',mapping='unchanged RandomUpperPolicy; max20m physical mapping',PURPOSES_existing_identifiers_unchanged=True),steps=[dict(name='step1',start='fresh Learner4104+Replay; single process without reload for1..300',phases=table([phase(n) for n in range(1,301)],'curriculum'),SAC_delta=counts1,SAC_1_200_delta=counts200,SAC_201_300_delta=counts100,flushes_delta=1,replay_capacity=55,final_replay_episodes=list(range(246,301)),PPO_unchanged=True,full_checkpoints=['A-S200h','A-S300c']),dict(name='step2',start='A-S300c network and RNG',phases=table(up[:100],'upper'),PPO_batches_delta=20,SAC_unchanged=True),dict(name='step3',start='A-U100 network; A-S300c replay/RNG; keep_replay',phases=table(low[:50],'lower'),SAC_delta=countsK,flushes_delta=0,available_completed=[55,55],full_checkpoints=[f'A-c{i}' for i in range(1,6)],PPO_unchanged=True),dict(name='step4',start='state_model(A-U100,A-H); RNG from A-U100',phases=table(up[100:],'upper'),PPO_batches_delta=20,PPO_batches_total=40,SAC_unchanged=True),dict(name='step5',start='state_model(A-U200,A-H); RNG from A-U200; replay from A-H; keep_replay',phases=table(low[50:],'lower'),SAC_delta=countsK,flushes_delta=0,available_completed=[55,55],no_replay_checkpoints=[f'A-c{i}' for i in range(6,11)],PPO_unchanged=True)],selection=dict(H_candidates=['A-U100+A-S300c',*[f'A-U100+A-c{i}' for i in range(1,6)]],F_candidates=['A-U200+A-H',*[f'A-U200+A-c{i}' for i in range(6,11)]],metric='minimum dev20 scenario-equal mean truncated_delay_s',tie='earliest; A-S300c before A-c1, A-H before A-c6',TFOO_fixed='TFOO-200',TFOO_not_reselected=True),criteria=criteria,diagnostics=dict(labels=['DG-SAC300','DG-v1c8'],statement='\u53ea\u505a\u8bca\u65ad\uff0c\u4e0d\u8fdb\u65b9\u6cd5\u3001\u4e0d\u53c2\u4e0e\u9009\u6a21\u578b',D_a=['TFOO-200+A-S200h','TFOO-200+A-S300c'],D_b=[f'A-U{i}+cvx' for i in range(25,201,25)]),deviations=deviations,branch_3K=False,disk_estimate=facts['disk'],time_estimate=facts['time_estimate'],hard_failure='immediately write failure.json and stop; do not relax or auto-resume',implementation_error='at most one documented implementation correction; new attempt directory from failing step, never overwrite completed output',hover_repro_failure='first discrepancy stops; only if proven original-loop omission may fix once and restart S1 in new attempt fromepisode1; second failure stops',no_source_config_old_output_writes=True)
put(CPIL/'cp_contract.json',contract);textfile(CPIL/'cp_contract.sha256',sha(CPIL/'cp_contract.json')+'\n')
print('CONTRACT_CREATED',sha(CPIL/'cp_contract.json'),len(contract['inputs_sha256']),flush=True)
