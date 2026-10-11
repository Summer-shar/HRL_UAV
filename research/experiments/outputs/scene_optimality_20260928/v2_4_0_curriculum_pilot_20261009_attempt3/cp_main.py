# Attempt3: verify/copy prior passed RP, restart at DG-SAC300 without RP reruns.
# Derived from PILOT/pilot_main_r2.py; selected() and state_model() unchanged.
# Changes: one 4104 curriculum run, CPIL outputs/immutable contract, RP-E/U/L,
# diagnostic groups, keep-replay S3/S5 with A-H-owned replay, no 3K or test.
import ctypes
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
from cp_common import *
import copy
import torch
import cp_train as tr
import cp_eval as ev
import cp_report as report
from hrl_v2.checkpoint import identity,load,runtime_record,runtime_signature
from hrl_v2.runtime import configure,Learner,parameter_hash
from hrl_v2.run import read_config
from hrl_v2.schedule import phase
from hrl_v2.upper_baselines import upper_phases
CURRENT_STAGE='initialize';CONTRACT_SHA=None

def stage(name):
 global CURRENT_STAGE
 CURRENT_STAGE=name
 equal(sha(CPIL/'cp_contract.json'),CONTRACT_SHA,'immutable_contract_'+name)
 equal((CPIL/'cp_contract.sha256').read_text(encoding='utf-8-sig').strip().split()[0],CONTRACT_SHA,'contract_sidecar_'+name)
 put(CPIL/'logs/gpu_stages'/f'{name}.json',gpu_check())
 put(CPIL/'progress.json',dict(status='stage_start',stage=name,time=now()))
 print('STAGE_START',name,flush=True)

def binding0():return read(CPIL/'cp_contract.json')['identity']

def preflight(seed):
 stage(f'seed_{seed}_preflight');check(seed==4104,'only_seed_4104')
 config=read_config(CONFIG,WT);bind=identity(WT,CONFIG);v=read(WORK/'validation/validation.json')
 equal(bind,binding0(),'current_contract_identity')
 equal(bind,read(PILOT_REF/'pilot_contract.json')['identity'],'PILOT_identity')
 equal(bind,read(old_run(seed)/'contract.json')['identity'],'Proposed_identity_'+str(seed))
 equal(bind,v['identity'],'validation_identity_'+str(seed));check(v['status']=='passed','validation_pass_'+str(seed))
 git=git_states();imports=assert_imports();gpu=gpu_check();device=configure();check(str(device)=='cuda:0','cuda_'+str(seed))
 equal(runtime_signature(runtime_record()),runtime_signature(v['runtime']),'runtime_validation_'+str(seed))
 check(sha(PROTO)==PROTO_SHA,'frozen_protocol_'+str(seed))
 put(CPIL/f'seed_{seed}/preflight.json',dict(status='passed',identity=bind,git=git,imports=imports,gpu=gpu,runtime=runtime_record(),disk_free=shutil.disk_usage(CPIL).free,time=now()))
 return config,device

def selected(candidates):return min(enumerate(candidates),key=lambda v:(v[1]['mean_s'],v[0]))[1]

def state_hash(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def state_model(seed,upper,lower,config,device):
 u=load(upper,binding0());h=load(lower,binding0());z=copy.deepcopy(u['learner']);z['sac']=h['learner']['sac'];l=Learner(seed,device,config=config);l.load_state_dict(z)
 equal(l.state_dict()['ppo'],u['learner']['ppo'],'assemble_PPO_U100_or_U200');equal(l.state_dict()['sac'],h['learner']['sac'],'assemble_SAC_H');equal(parameter_hash(l.ppo.actor),state_hash(u['learner']['ppo']['actor']),'assembled_upper_hash_equals_U');equal(parameter_hash(l.sac.actor),state_hash(h['learner']['sac']['actor']),'assembled_lower_hash_equals_H');tr.restore_rng(u);del h,z;return l,u

def final_readonly_checks():
 contract=read(CPIL/'cp_contract.json');checks={}
 for filename,expected in contract['inputs_sha256'].items():
  actual=sha(filename);equal(actual,expected,'readonly_input_'+filename)
  checks[filename]=dict(expected=expected,actual=actual,unchanged=True)
 equal(sha(PROTO),PROTO_SHA,'final_frozen_protocol')
 equal(sha(CPIL/'cp_contract.json'),CONTRACT_SHA,'final_contract')
 git=git_states();put(CPIL/'git_after.json',git)
 result=dict(status='PASS',inputs=checks,protocol=dict(path=str(PROTO),sha256=PROTO_SHA,unchanged=True),contract_sha256=CONTRACT_SHA,git=git,time=now())
 put(CPIL/'final_readonly_checks.json',result);return result


def reuse_rp(seed,data):
 """Verify copied prior PASS records without invoking any training or evaluation."""
 contract=read(CPIL/'cp_contract.json');meta=contract['rp_reuse']
 check(meta['enabled'] and meta['status']=='PASS','RP_reuse_contract_enabled')
 equal(seed,4104,'RP_reuse_only_seed4104')
 manifest_path=Path(meta['manifest_path'])
 equal(sha(manifest_path),meta['manifest_sha256'],'RP_reuse_manifest_contract_sha')
 manifest=read(manifest_path);equal(manifest,meta['manifest'],'RP_reuse_manifest_contract_payload')
 source_attempt=Path(meta['source_attempt'])
 equal(sha(source_attempt/'cp_contract.json'),manifest['source_contract']['sha256'],'RP_reuse_source_contract_sha')
 for record in manifest['files']:
  source=Path(record['source']);destination=Path(record['destination'])
  check(source.is_relative_to(source_attempt/'seed_4104/rp'),'RP_reuse_source_boundary_'+record['relative_path'])
  check(destination.is_relative_to(CPIL/f'seed_{seed}/rp'),'RP_reuse_copy_boundary_'+record['relative_path'])
  equal(contract['inputs_sha256'][str(source)],record['sha256'],'RP_reuse_frozen_source_sha_'+record['relative_path'])
  equal(sha(source),record['sha256'],'RP_reuse_live_source_sha_'+record['relative_path'])
  equal(sha(destination),record['sha256'],'RP_reuse_copy_sha_'+record['relative_path'])
 root=CPIL/f'seed_{seed}/rp'
 rp=read(root/'training_checks.json');equal(rp['status'],'PASS','RP_reuse_training_PASS')
 equal(sorted(rp['results']),['RP-L','RP-U'],'RP_reuse_groups')
 for name,result in rp['results'].items():equal(result['status'],'passed','RP_reuse_pass_'+name)
 equal(rp['results']['RP-U']['episodes'],10,'RP_reuse_U_episodes')
 equal(rp['results']['RP-U']['PPO_batches_delta'],2,'RP_reuse_U_batches')
 equal(rp['results']['RP-L']['episodes'],3,'RP_reuse_L_episodes')
 ec=read(root/'RP-E/checks.json');equal(sorted(ec),['E5','TFOO-200'],'RP_reuse_E_groups')
 data['RP']['RP-E']=ec
 for label in ('E5','TFOO-200'):
  equal(ec[label]['status'],'PASS','RP_reuse_E_PASS_'+label)
  r=read(root/f'RP-E/{label}/summary.json')
  equal(r['stream_consistency'],dict(passed=True,scenarios=20),'RP_reuse_E_streams_'+label)
  for origin,ref in [('PILOT',read(PILOT_REF/f'seed_{seed}/rp/RP-E/{label}/summary.json')),('NEW',hist(seed,label))]:
   equal(r['per_scenario'],ref['per_scenario'],'RP_reuse_E_per_scenario_'+origin+'_'+label)
   equal(r['mean_s'],ref['mean_s'],'RP_reuse_E_mean_'+origin+'_'+label)
  data['groups'][label]=r;data['timing']['RP-E_'+label]=r['wall_seconds']
 data['RP'].update({k:dict(status='PASS',episodes=v['episodes'],SAC_delta=v.get('SAC_delta'),PPO_batches=v.get('PPO_batches_delta'),wall_seconds=v['wall_seconds']) for k,v in rp['results'].items()})
 data['timing'].update({k:v['wall_seconds'] for k,v in rp['results'].items()})
 data['RP_reuse']=dict(status='PASS',source_attempt=str(source_attempt),copied_groups=['RP-E','RP-U','RP-L'],reused_timing_keys=['RP-E_E5','RP-E_TFOO-200','RP-U','RP-L'],manifest_path=str(manifest_path),manifest_sha256=meta['manifest_sha256'],source_contract_sha256=manifest['source_contract']['sha256'],file_count=manifest['file_count'],note='Prior attempt measured timings retained; no RP rerun in this attempt')
 put(CPIL/f'seed_{seed}/rp_reuse_checks.json',dict(status='PASS',verified_at=now(),**{k:v for k,v in data['RP_reuse'].items() if k!='status'}))
 print('RP_REUSE_PASS',seed,manifest['file_count'],'starting_next=DG-SAC300',flush=True)

def execute_seed(seed):
 config,device=preflight(seed);root=CPIL/f'seed_{seed}'
 data=dict(seed=seed,groups={},timing={},training_results={},runs={},RP={},TFOO_skipped=[],TFOO_checkpoints_evaluated=[200])
 def persist():put(root/'pipeline_data.json',data)
 def evaluate(label,up,low=None,route=None,kind=None):
  stage(f'seed_{seed}_dev_{label}')
  r=ev.group(seed,label,up,low,route,kind);data['groups'][label]=r;data['timing']['dev_'+label]=r['wall_seconds'];persist();return r
 def training_result(key,run,result):
  data['training_results'][key]=result;data['runs'][key]=str(run);data['timing'][key]=result['wall_seconds'];persist()
 if read(CPIL/'cp_contract.json').get('rp_reuse',{}).get('enabled'):
  stage(f'seed_{seed}_RP_reuse_checks');reuse_rp(seed,data);persist()
 else:
  stage(f'seed_{seed}_RP_E');data['RP']['RP-E']=ev.repro_eval(seed)
  for label in ('E5','TFOO-200'):
   r=read(root/f'rp/RP-E/{label}/summary.json');data['groups'][label]=r;data['timing']['RP-E_'+label]=r['wall_seconds']
  persist()
  stage(f'seed_{seed}_RP_training');tr.repro_train(seed);rp=read(root/'rp/training_checks.json');check(rp['status']=='PASS','RP_training_PASS_'+str(seed))
  equal(sorted(rp['results']),['RP-L','RP-U'],'RP_only_upper_lower')
  data['RP'].update({k:dict(status='PASS',episodes=v['episodes'],SAC_delta=v.get('SAC_delta'),PPO_batches=v.get('PPO_batches_delta'),wall_seconds=v['wall_seconds']) for k,v in rp['results'].items()})
  data['timing'].update({k:v['wall_seconds'] for k,v in rp['results'].items()});persist()
 tf200=old_tfoo(seed)/'checkpoints/episode-0200-anchor.pt'
 data['TFOO_dev']=dict(label='TFOO-200',mean_s=data['groups']['TFOO-200']['mean_s'],checkpoint=str(tf200),selection='fixed previously selected; not reselected',source=str(PILOT_REF/f'seed_{seed}/TFOO_selection.json'))
 previous=read(PILOT_REF/f'seed_{seed}/pipeline_data.json');pu200=Path(previous['runs']['step4'])/'checkpoints/U200.pt'
 orig_s300=old_run(seed)/'checkpoints/episode-0300-evaluated.pt';v1c8=old_l(seed)/'checkpoints/cycle-08.pt'
 evaluate('DG-SAC300',pu200,orig_s300);evaluate('DG-v1c8',pu200,v1c8)
 stage(f'seed_{seed}_step1');run=root/'step1_attempt1';r=tr.run_curriculum(seed,run);training_result('step1',run,r);tr.cleanup()
 s200=run/'checkpoints/A-S200h.pt';s300=run/'checkpoints/A-S300c.pt'
 evaluate('D-a_TFOO_A-S200h',tf200,s200,kind='TFOO');evaluate('D-a_TFOO_A-S300c',tf200,s300,kind='TFOO')
 stage(f'seed_{seed}_step2');l,z=tr.learner_from(s300,seed,device,config);tr.restore_rng(z);del z
 run=root/'step2_attempt1';r=tr.run_upper(seed,l,upper_phases()[:100],run);training_result('step2',run,r);del l;tr.cleanup();u100=run/'checkpoints/A-U100.pt'
 stage(f'seed_{seed}_step3');l,z=tr.learner_from(u100,seed,device,config);del z;z=load(s300,binding0());replay=z['replay'];tr.restore_rng(z);del z
 run=root/'step3_attempt1';low=[phase(n) for n in range(401,601) if phase(n).layer=='lower']
 r=tr.run_lower(seed,l,replay,low[:50],run,keep_replay=True,checkpoint_replay=True);training_result('step3',run,r);del l,replay;tr.cleanup()
 hr=[dict(label='H_A-S300c',mean_s=evaluate('H_A-S300c',u100,s300)['mean_s'],checkpoint=str(s300))]
 for i in range(1,6):
  p=run/f'checkpoints/A-c{i}.pt';rr=evaluate(f'H_A-c{i}',u100,p);hr.append(dict(label=f'H_A-c{i}',mean_s=rr['mean_s'],checkpoint=str(p)))
 data['H']=selected(hr);put(root/'H_selection.json',dict(candidates=hr,selected=data['H']));h=Path(data['H']['checkpoint']);persist()
 stage(f'seed_{seed}_step4');l,z=state_model(seed,u100,h,config,device);del z;run=root/'step4_attempt1'
 r=tr.run_upper(seed,l,upper_phases()[100:],run,start_n=100);check(l.ppo.counts['batches']==40,'PPO_final_40_'+str(seed));training_result('step4',run,r);del l;tr.cleanup();u200=run/'checkpoints/A-U200.pt'
 stage(f'seed_{seed}_step5');l,z=state_model(seed,u200,h,config,device)
 # Loading replay must not replace A-U200 RNG established by state_model.
 rng_before=dict(cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])
 hz=load(h,binding0());replay=hz['replay'];check(len(replay.episodes)==55,'S5_H_replay_55')
 saved_replay=dict(upper_version=replay.upper_version,flushes=replay.flushes,completed=len(replay.episodes),source=str(h))
 equal(replay.upper_version,hz['replay_metadata']['upper_version'],'S5_H_replay_label');equal(replay.flushes,hz['replay_metadata']['flushes'],'S5_H_replay_flushes')
 equal(len(replay.episodes),hz['replay_metadata']['completed'],'S5_H_replay_saved_completed')
 equal(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),rng_before,'S5_loading_H_replay_preserves_U200_RNG')
 equal(torch.get_rng_state(),z['global_cpu_rng'],'S5_RNG_from_U200_CPU');equal(torch.cuda.get_rng_state_all(),z['global_cuda_rng'],'S5_RNG_from_U200_CUDA')
 data['step5_replay_source']=saved_replay;del hz,z,rng_before;persist()
 run=root/'step5_attempt1';r=tr.run_lower(seed,l,replay,low[50:],run,keep_replay=True,checkpoint_replay=False);training_result('step5',run,r);del l,replay;tr.cleanup()
 fr=[dict(label='F_A-H',mean_s=evaluate('F_A-H',u200,h)['mean_s'],checkpoint=str(h))]
 for i in range(6,11):
  p=run/f'checkpoints/A-c{i}.pt';rr=evaluate(f'F_A-c{i}',u200,p);fr.append(dict(label=f'F_A-c{i}',mean_s=rr['mean_s'],checkpoint=str(p)))
 data['F']=selected(fr);put(root/'F_selection.json',dict(candidates=fr,selected=data['F']));persist()
 for n in range(25,201,25):
  p=root/f'step{2 if n<=100 else 4}_attempt1/checkpoints/A-U{n}.pt';evaluate(f'D-b_U{n}_cvx',p,route='cvx')
 data['hard_checks']=dict(status='PASS',SAC_counts={k:r.get('SAC_delta') for k,r in data['training_results'].items()},flushes={k:r.get('replay_flushes_delta') for k,r in data['training_results'].items()},available_completed={k:dict(min=min(r['available_completed']),max=max(r['available_completed'])) for k,r in data['training_results'].items() if r.get('available_completed')},frozen_PPO_all_lower=True,frozen_SAC_all_upper=True,PPO_batches_40=True,all_motion_steps_within20m=True,all_external_seeds_match=True,all_deterministic_stream_metrics_equal=True,contract_unchanged=True,checks=copy.deepcopy(CHECKS))
 persist();return data

def main():
 global CONTRACT_SHA
 CONTRACT_SHA=sha(CPIL/'cp_contract.json');start=time.perf_counter()
 equal((CPIL/'cp_contract.sha256').read_text(encoding='utf-8-sig').strip().split()[0],CONTRACT_SHA,'initial_contract_sidecar')
 put(CPIL/'execution_start.json',dict(time=now(),contract_sha256=CONTRACT_SHA,seed_order=[4104],workers=6,self_pid=os.getpid(),parent_pid=os.getppid()))
 data=execute_seed(4104)
 stage('final_readonly_checks');readonly=final_readonly_checks()
 data['readonly_checks']=readonly;data['total_wall_seconds']=time.perf_counter()-start;data['finished_at']=now();data['hard_checks']['checks']=copy.deepcopy(CHECKS)
 put(CPIL/'seed_4104/pipeline_data.json',data)
 stage('seed_4104_report');record=report.report_seed(4104,data);report.final_summary([record])
 put(CPIL/'timing.json',dict(total_wall_seconds=time.perf_counter()-start,finished_at=now()))
 put(CPIL/'progress.json',dict(status='complete',completed_seeds=[4104],time=now()))
 print('CURRICULUM_PILOT_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  record=failure(CURRENT_STAGE,e)
  try:report.report_failure(record,read(CPIL/'seed_4104/pipeline_data.json') if (CPIL/'seed_4104/pipeline_data.json').exists() else None)
  except BaseException as report_error:put(CPIL/'failure_report_error.json',dict(error=str(report_error),traceback=traceback.format_exc()))
  try:put(CPIL/'git_after_failure.json',git_states())
  except BaseException:pass
  raise
