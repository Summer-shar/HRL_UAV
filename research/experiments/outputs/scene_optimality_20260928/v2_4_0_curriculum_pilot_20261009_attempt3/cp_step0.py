# Attempt3; inherited final775 inputs and verified byte-identical RP reuse. Derived from prior CPIL step0. Adds previously-unreached reference/curve/timing/input audit after the original disk gate. No training/evaluation.
# Step 0 only: read-only metadata/JSON/source/snapshot inspection; no training/evaluation.
# Common guard copied from PILOT; never import any PILOT module.
from cp_common import *
import inspect,copy
from hrl_v2.runtime import configure
from hrl_v2.checkpoint import runtime_signature
import torch
import hrl_v2,hrl_uav
from hrl_v2.checkpoint import identity,runtime_record
from hrl_v2.replay import Replay
ORIG=old_run(4104)
PILOT_REF=MAIN/'outputs/scene_optimality_20260928/v2_4_0_randpre_pilot_20261005'
facts=dict(status='checking',created_at=now(),CPIL=str(CPIL),training_started=False,evaluation_started=False,checks={},not_executed=[])
protected={};inherited_expected={}
def tracked(path):
 path=Path(path);protect(path);return read(path)
def protect(path):
 path=Path(path);actual=sha(path)
 if str(path) in inherited_expected:equal(actual,inherited_expected[str(path)],'unchanged_inherited_'+str(path))
 protected[str(path)]=actual;return actual
def emit():
 facts['checks']=dict(CHECKS);facts['protected_input_sha256']=protected
 put(CPIL/'step0_facts/facts.json',facts)
 lines=['第 0 步预检记录','', '本轮未启动训练或评估。','', 'Status: '+facts['status'],'', '```json',json.dumps(facts,ensure_ascii=False,indent=2),'```']
 textfile(CPIL/'step0_facts/facts.md','\n'.join(lines)+'\n')
def file_stat(path):return dict(path=str(path),bytes=Path(path).stat().st_size)
def directory_bytes(path,exclude_checkpoints=False):
 pp=[x for x in Path(path).rglob('*') if x.is_file() and (not exclude_checkpoints or 'checkpoints' not in x.relative_to(path).parts)]
 return dict(path=str(path),files=len(pp),bytes=sum(x.stat().st_size for x in pp))
try:
 facts['git_before']=git_states();facts['imports']=assert_imports();facts['gpu']=gpu_check()
 facts['previous_failed_attempt']=str(PREVIOUS_FAILED_ATTEMPT)

 # Inherit and reverify the final 775-input contract, not the old incomplete488 facts.
 prior_work=MAIN/'outputs/scene_optimality_20260928/v2_4_0_curriculum_pilot_20261008_attempt2'
 prior_contract=tracked(prior_work/'cp_contract.json')
 prior_contract_sha=protect(prior_work/'cp_contract.json')
 equal(prior_contract_sha,(prior_work/'cp_contract.sha256').read_text(encoding='utf-8-sig').strip().split()[0],'prior_final_contract_sidecar')
 protect(prior_work/'cp_contract.sha256')
 inherited_expected=dict(prior_contract['inputs_sha256'])
 equal(len(inherited_expected),775,'prior_final_contract_775_inputs')
 for filename,expected in inherited_expected.items():
  actual=protect(filename);equal(actual,expected,'inherited_input_unchanged_'+filename)
 facts['prior_final_contract']=dict(path=str(prior_work/'cp_contract.json'),sha256=prior_contract_sha,input_count=len(inherited_expected),all_inputs_reverified=True)
 prior_failure=tracked(prior_work/'failure.json')
 facts['previous_failed_work']=str(prior_work)
 facts['previous_failure']=dict(stage=prior_failure['stage'],error=prior_failure['error'],message=prior_failure['message'])
 # Reused RP evidence is byte-identical; this is metadata verification only.
 reuse_path=CPIL/'rp_reuse_manifest.json';reuse=read(reuse_path);reuse_sha=sha(reuse_path)
 equal(reuse['status'],'PASS','RP_reuse_manifest_pass')
 equal(reuse['source_contract']['sha256'],prior_contract_sha,'RP_reuse_prior_contract')
 equal(reuse['source_contract']['protected_input_count'],775,'RP_reuse_prior_775')
 equal(reuse['source_root'],str(prior_work/'seed_4104/rp'),'RP_reuse_exact_source_root')
 for record in reuse['files']:
  source=Path(record['source']);destination=Path(record['destination'])
  check(source.is_relative_to(prior_work/'seed_4104/rp'),'RP_reuse_source_guard_'+record['relative_path'])
  check(destination.is_relative_to(CPIL/'seed_4104/rp'),'RP_reuse_destination_guard_'+record['relative_path'])
  check(source.suffix.lower() in ('.json','.jsonl'),'RP_reuse_allowed_file_'+record['relative_path'])
  equal(protect(source),record['sha256'],'RP_reuse_source_sha_'+record['relative_path'])
  equal(sha(destination),record['sha256'],'RP_reuse_copy_sha_'+record['relative_path'])
 equal(len(reuse['files']),reuse['file_count'],'RP_reuse_file_count')
 rtc=read(CPIL/'seed_4104/rp/training_checks.json')
 equal(rtc['status'],'PASS','RP_reused_training_status')
 equal(sorted(rtc['results']),['RP-L','RP-U'],'RP_reused_training_groups')
 for name,result in rtc['results'].items():equal(result['status'],'passed','RP_reused_'+name+'_status')
 rec=read(CPIL/'seed_4104/rp/RP-E/checks.json')
 equal(sorted(rec),['E5','TFOO-200'],'RP_reused_eval_groups')
 for label in ('E5','TFOO-200'):
  equal(rec[label]['status'],'PASS','RP_reused_'+label+'_status')
  result=read(CPIL/f'seed_4104/rp/RP-E/{label}/summary.json')
  equal(result['stream_consistency'],dict(passed=True,scenarios=20),'RP_reused_stream_consistency_'+label)
  for origin,reference in [('PILOT',tracked(PILOT_REF/f'seed_4104/rp/RP-E/{label}/summary.json')),('NEW',tracked(NEW/f'summaries/seed_4104/{label}.json'))]:
   equal(result['per_scenario'],reference['per_scenario'],'RP_reused_per_scenario_'+origin+'_'+label)
   equal(result['mean_s'],reference['mean_s'],'RP_reused_mean_'+origin+'_'+label)
 facts['rp_reuse']=dict(enabled=True,status='PASS',source_attempt=str(prior_work),manifest_path=str(reuse_path),manifest_sha256=reuse_sha,manifest=reuse,copied_groups=['RP-E','RP-U','RP-L'],reused_timing_keys=['RP-E_E5','RP-E_TFOO-200','RP-U','RP-L'],note='Completed PASS evidence copied byte-for-byte; no RP rerun; prior wall times are not this attempt measurements')

 device=configure();equal(str(device),'cuda:0','cuda_required');facts['runtime']=runtime_record();facts['torch_threads']=torch.get_num_threads()
 protect(PROTO);equal(sha(PROTO),PROTO_SHA,'frozen_protocol_sha')
 binding=identity(WT,CONFIG);pc=tracked(PILOT_REF/'pilot_contract.json');oc=tracked(ORIG/'contract.json');validation=tracked(WORK/'validation/validation.json')
 equal(binding,pc['identity'],'identity_PILOT');equal(binding,oc['identity'],'identity_ORIG');equal(binding,validation['identity'],'identity_validation');equal(validation['status'],'passed','validation_status')
 facts['identity']=binding
 equal(runtime_signature(runtime_record()),runtime_signature(validation['runtime']),'runtime_signature_validation')
 originals=[]
 for n in range(1,301):
  q=ORIG/f'episodes/episode-{n:04d}.json';r=tracked(q)
  check('event_sha256' in r['metrics'] and 'sac_delta' in r['train_metric'],'original_fields_'+str(n))
  originals.append(dict(global_episode=n,event_sha256=r['metrics']['event_sha256'],sac_delta=r['train_metric']['sac_delta'],truncated_delay_s=r['metrics']['truncated_delay_s']))
 manifest=tracked(ORIG/'training_manifest.json');by={r['global_episode']:r for r in manifest}
 check(all(n in by and all(k+'_seed' in by[n] for k in ('arrival','layout','load')) for n in range(1,301)),'original_external_manifest')
 checkpoint_names=sorted(q.name for q in (ORIG/'checkpoints').glob('*.pt'))
 check(not any(q.startswith('episode-0200-') for q in checkpoint_names),'no_resumable_episode200')
 facts['ORIG']=dict(episode_json_count=len(originals),verified_fields=['metrics.event_sha256','train_metric.sac_delta'],checkpoint_files=checkpoint_names,manifest_rows=len(manifest),original_episode_checks=originals)
 schemas=[]
 for n in range(25,201,25):
  q=ORIG/f'snapshots/episode-{n:04d}.pt';side=tracked(q.with_suffix('.json'));equal(protect(q),side['sha256'],'snapshot_sha_'+str(n))
  before_cpu=torch.get_rng_state().clone()
  z=torch.load(q,map_location='cpu',weights_only=False)
  check(torch.equal(before_cpu,torch.get_rng_state()),'step0_snapshot_read_CPU_RNG_'+str(n))
  equal(z['identity'],binding,'snapshot_identity_'+str(n));equal(z['kind'],'network_snapshot_not_resumable','snapshot_kind_'+str(n));equal(z['episode'],n,'snapshot_episode_'+str(n))
  names={'sac':['actor','q1','q2','target1','target2'],'ppo':['actor','value']}
  check(all(all(k in z['learner'][owner] for k in keys) for owner,keys in names.items()),'snapshot_networks_comparable_'+str(n))
  schemas.append(dict(episode=n,path=str(q),top_keys=list(z),learner_keys=list(z['learner']),second_level={k:list(v) for k,v in z['learner'].items()},comparison_networks=names,actual_contains_optimizers_counts_versions=True,resumable=False))
  del z;gc.collect()
 facts['snapshots']=dict(comparable=True,schemas=schemas,planned_check='At each25 boundary compare every state_dict tensor of all5 SAC networks and2 PPO networks; capture CPU/CUDA global RNG before and after; never instantiate Learner during checks.',note='Actual snapshot holds learner.state_dict including optimizer/count/version; it lacks replay and global RNG, remains non-resumable.')
 for name in ('run.py','archive.py','replay.py','runtime.py','collector.py','randomness.py','schedule.py'):
  protect(WT/'src/hrl_v2'/name)
 protect(PILOT_REF/'pilot_train.py');protect(PILOT_REF/'pilot_train_r2.py');protect(PILOT_REF/'pilot_common.py')
 facts['original_loop_audit']=dict(source=str(WT/'src/hrl_v2/run.py'),collection_lines=[271,315],persistence_lines=[194,220,316,325],other_actions=['source identity check','phase lookup','independent named Streams creation','adapter reset with paired external seeds','hover version set','read-only telemetry','episode/JSONL output','complete checkpoint every10','network snapshot every25','streams.state_dict read'],development_before_201=False,learner_or_replay_changes_outside_collection_SAC='only replay.set_upper_version and replay.add; no additional learning changes',global_torch_or_numpy_RNG_draws_outside_collection_SAC='none found; uuid4 uses OS entropy and does not consume these streams',pilot_path_RP1_1_to5='previous PASS record; S1 new1-to200 must independently match each event/count and snapshots',missing_training_steps_to_copy=[],storage_does_not_change_learning='checkpoint/save_network_snapshot read state only; RNG invariance tested in repository archive tests; S1 checks will independently enforce invariance')
 rr=Replay();rr.set_upper_version('hover-0');rr.episodes=[object() for _ in range(55)];before=list(rr.episodes);ff=rr.flushes;rr.upper_version='random-uniform-0'
 check(rr.episodes==before and rr.flushes==ff,'direct_label_assignment_preserves_replay_flush')
 sample=inspect.getsource(Replay.sample);add=inspect.getsource(Replay.add);setting=inspect.getsource(Replay.set_upper_version)
 check('upper_version' not in sample,'sample_no_upper_version_filter');check('len(self.episodes)>55' in add and 'pop(0)' in add,'FIFO_capacity55');check('version!=self.upper_version' in setting and 'episodes.clear()' in setting and 'flushes+=1' in setting,'set_version_clears_only_on_change')
 facts['replay']=dict(set_upper_version_only_change=True,direct_assignment_clears=False,direct_assignment_changes_flushes=False,sample_filters_upper_version=False,FIFO_capacity=55)
 s300=ORIG/'checkpoints/episode-0300-evaluated.pt';protect(s300);tracked(s300.with_suffix('.json'))
 pipeline=tracked(PILOT_REF/'seed_4104/pipeline_data.json');runmap=pipeline['runs'];net=Path(runmap['step4'])/'checkpoints/U200.pt';protect(net);tracked(net.with_suffix('.json'))
 epaths=[PILOT_REF/'seed_4104/rp/RP-E'/label for label in ('E5','TFOO-200')]+[PILOT_REF/f'seed_4104/evaluations/{label}' for label in ('D-a_TFOO_S300r','F_c9','D-b_U200_cvx')]
 egroups=[directory_bytes(q) for q in epaths];group_basis=max(v['bytes'] for v in egroups)
 training=[directory_bytes(Path(runmap[k]),exclude_checkpoints=True) for k in ('step1','step2','step3','step4','step5')]
 rp=[directory_bytes(PILOT_REF/f'seed_4104/rp/{k}/run') for k in ('RP-U','RP-L')]
 full=7*s300.stat().st_size;other=13*net.stat().st_size;evaluation=26*group_basis
 subset=full+other+evaluation;overhead=32*1024**2;estimate=subset+sum(v['bytes'] for v in training)+sum(v['bytes'] for v in rp)+overhead
 free=shutil.disk_usage(CPIL).free
 facts['disk']=dict(free_bytes=free,full_checkpoint_basis=file_stat(s300),full_checkpoint_count=7,full_checkpoint_estimated_bytes=full,other_learner_checkpoint_basis=file_stat(net),other_checkpoint_count=13,other_checkpoint_estimated_bytes=other,evaluation_group_basis=egroups,evaluation_groups=26,evaluation_estimated_bytes=evaluation,subset_estimated_bytes=subset,subset_required_double_bytes=2*subset,training_non_checkpoint_measured=training,RP_non_checkpoint_measured=rp,report_metadata_reserve_estimate_bytes=overhead,total_estimated_bytes=estimate,required_free_bytes_strictly_greater_than=2*estimate,shortfall_bytes=max(0,2*estimate-free),estimate_not_measurement=True,basis='7 full replay checkpoints +13 learner checkpoints +26 dev groups using largest measured old group; old main training output excluding checkpoints; RP-U/RP-L logs;32MiB metadata reserve',subset_already_fails=free<=2*subset)
 facts['not_executed']=['Step0 reference value/curve extraction and wall-time estimate after disk gate','contract creation','RP experiments reused from PASS evidence; not rerun','diagnostic groups','S1-S5 and H/F selections','all test evaluations']
 emit();check(free>2*estimate,'disk_free_greater_than_two_times_estimate',facts['disk'])

 # Complete step0 sections6/7, which were not reached in the disk-failed attempt.
 references={}
 tfsel=tracked(PILOT_REF/'seed_4104/TFOO_selection.json')
 equal(tfsel['selected']['label'],'TFOO-200','reuse_fixed_TFOO200_selection')
 for label in ('E5','TFOO-200'):
  old=tracked(PILOT_REF/f'seed_4104/rp/RP-E/{label}/summary.json')
  new=tracked(NEW/f'summaries/seed_4104/{label}.json')
  equal(old['mean_s'],new['mean_s'],'reference_mean_PILOT_NEW_'+label)
  equal(old['per_scenario'],new['per_scenario'],'reference_per_scenario_PILOT_NEW_'+label)
  references[label]=dict(source=str(PILOT_REF/f'seed_4104/rp/RP-E/{label}/summary.json'),NEW_source=str(NEW/f'summaries/seed_4104/{label}.json'),mean_s=old['mean_s'],per_scenario=old['per_scenario'])
 for label in ('D-b_U200_cvx','D-a_TFOO_S300r',pipeline['H']['label'],pipeline['F']['label']):
  old=tracked(PILOT_REF/f'seed_4104/evaluations/{label}/summary.json')
  references[label]=dict(source=str(PILOT_REF/f'seed_4104/evaluations/{label}/summary.json'),mean_s=old['mean_s'],per_scenario=old['per_scenario'])
 for label in ('X1','E2',*[f'S3-c{i}' for i in range(1,11)]):
  rr=tracked(NEW/f'summaries/seed_4104/{label}.json')
  references[label]=dict(source=str(NEW/f'summaries/seed_4104/{label}.json'),mean_s=rr['mean_s'],per_scenario=rr['per_scenario'])
 selection=tracked(NEW/'summaries/selection.json')['4104']
 equal(selection['Cstar']['cycle'],8,'NEW_v1_selected_cycle8')
 equal(selection['Cstar']['dev_mean_s'],references['S3-c8']['mean_s'],'NEW_v1_cycle8_mean')
 facts['references']=references;facts['TFOO_selection']=tfsel;facts['v1_dev_best']=selection['Cstar'];facts['pilot_runs']=runmap;facts['pilot_timing']=pipeline['timing']
 # Typed models are validated by the copied eval loader in RP-E; here verify immutable inputs.
 models=dict(TFOO200=old_tfoo(4104)/'checkpoints/episode-0200-anchor.pt',SD200=old_sd(4104)/'checkpoints/episode-0200.pt',ORIG_S300=s300,v1_c8=old_l(4104)/'checkpoints/cycle-08.pt',PU200=net)
 for label,q in models.items():
  protect(q);meta=tracked(q.with_suffix('.json'))
  if 'sha256' in meta:equal(meta['sha256'],protected[str(q)],'input_sidecar_sha_'+label)
 facts['input_models']={k:dict(path=str(q),sha256=protected[str(q)],bytes=q.stat().st_size) for k,q in models.items()}
 for q in (CONFIG,SCENE,PILOT_REF/'pilot_contract.sha256',PILOT_REF/'contract_correction.diff',PILOT_REF/'pilot_eval.py',PILOT_REF/'pilot_main.py',PILOT_REF/'pilot_main_r2.py',PILOT_REF/'pilot_report.py',PILOT_REF/'seed_4104/report.json'):
  protect(q)
 for n in range(1,11):protect(old_sd(4104)/f'episodes/episode-{n:04d}.json')
 for n in range(411,414):protect(old_l(4104)/f'episodes/episode-{n:04d}.json')
 for n in range(201,301):
  q=Path(runmap['step1'])/f'positions/episode-{n:04d}.json';rr=tracked(q)
  check(rr['global_episode']==n and 'random_vectors' in rr,'pilot_random_vector_input_'+str(n))
 # Curve bins retain the same episode weighting used by the trial.
 orig_file=ORIG/'train_metrics.jsonl';protect(orig_file)
 orig_rows=rows(orig_file);orig_rows=[r for r in orig_rows if 1<=r.get('global_episode',0)<=300]
 random_file=Path(runmap['step1'])/'train_metrics.jsonl';protect(random_file)
 random_rows=rows(random_file);random_rows=[r for r in random_rows if 1<=r.get('global_episode',0)<=300]
 def binned(rr):
  keyed={r['global_episode']:r for r in rr};equal(sorted(keyed),list(range(1,301)),'curve_300_unique_episodes_'+str(len(CHECKS)))
  return [dict(first=i,last=i+24,rows=25,truncated_delay_s=avg([keyed[j]['truncated_delay_s'] for j in range(i,i+25)])) for i in range(1,301,25)]
 facts['curves']=dict(original_hover=dict(source=str(orig_file),bins=binned(orig_rows)),pilot_random=dict(source=str(random_file),bins=binned(random_rows)))
 # Estimate from measured work, observed checkpoint costs and actual trial evaluation wall times.
 orig_first200=sum(r['work_seconds'] for r in orig_rows if r['global_episode']<=200)
 random_last100=sum(r['work_seconds'] for r in random_rows if r['global_episode']>=201)
 timing=pipeline['timing'];train_follow=sum(timing[k] for k in ('step2','step4','step5','step3K'))
 eval_walls=[v for k,v in timing.items() if k.startswith('dev_') or k.startswith('RP-E_')]
 ev_est=26*avg(eval_walls);rp_est=timing['RP-U']+timing['RP-L']
 # Seven complete replay saves use observed per-checkpoint times only where available.
 cp_times=[r['checkpoint_seconds'] for r in orig_rows if isinstance(r.get('checkpoint_seconds'),(int,float)) and r['checkpoint_seconds']>.01]
 cp_est=7*avg(cp_times) if cp_times else None
 estimated=sum((orig_first200,random_last100,train_follow,ev_est,rp_est,cp_est or 0))
 facts['time_estimate']=dict(estimate_not_measurement=True,original_hover_1_200_work_seconds=orig_first200,pilot_random_201_300_work_seconds=random_last100,remaining_training_from_trial_step2_step3K_step4_step5_seconds=train_follow,evaluation_groups=26,trial_eval_mean_seconds=avg(eval_walls),evaluation_seconds_estimate=ev_est,RP_U_L_seconds=rp_est,seven_full_checkpoint_seconds_estimate=cp_est,total_seconds_estimate=estimated,total_hours_estimate=estimated/3600,unmeasured=['new per-episode equality checks','new replay/RNG checkpoint read-back checks','new queue states after hover-to-random curriculum','diagnostic DG groups runtime variation','concurrent non-training system activity'],basis_sources=[str(orig_file),str(random_file),str(PILOT_REF/'seed_4104/pipeline_data.json')])
 facts['disk']['estimated_output_bytes']=facts['disk']['total_estimated_bytes']
 facts['not_executed']=['contract creation','RP experiments reused from PASS evidence; not rerun','diagnostic groups','S1-S5 and H/F selections','all test evaluations (not authorized)']
 put(CPIL/'old_input_sha256_before.json',protected)
 equal({k:protected[k] for k in inherited_expected},inherited_expected,'all775_in_final_step0_input_list')
 facts['input_protection_coverage']=dict(inherited_old_inputs=775,final_protected_count=len(protected),rp_reuse_source_files=reuse['file_count'],all_inherited_present=True)
 facts['status']='PASS';facts['finished_at']=now();emit()
 print('STEP0_PASS',json.dumps(dict(protected_files=len(protected),disk_free=facts['disk']['free_bytes'],required_free=facts['disk']['required_free_bytes_strictly_greater_than'],estimated_hours=facts['time_estimate']['total_hours_estimate'])),flush=True)

except BaseException as e:
 facts['status']='failed';facts['first_failure']=dict(error=type(e).__name__,message=str(e));emit()
 failure('step0',e);git=git_states();put(CPIL/'git_after_failure.json',git)
 after={q:sha(q) for q in protected};equal(after,protected,'read_only_inputs_unchanged_after_step0_failure')
 put(CPIL/'final_readonly_checks.json',dict(status='PASS',checked_at=now(),protected_files=len(protected),sha_before=protected,sha_after=after,frozen_protocol_unchanged=sha(PROTO)==PROTO_SHA,git=git,training_started=False,evaluation_started=False))
 lines=['# Step0 stopped','',str(e),'','No training/evaluation started; old inputs rechecked unchanged. See failure.json and step0_facts/facts.json.']
 textfile(CPIL/'summary.md','\n'.join(lines)+'\n')
 print(json.dumps(dict(status='STOPPED',stage='step0',error=type(e).__name__,disk=facts.get('disk'),protected_files_unchanged=len(protected),training_started=False,evaluation_started=False),ensure_ascii=False),flush=True)
 raise SystemExit(2)
