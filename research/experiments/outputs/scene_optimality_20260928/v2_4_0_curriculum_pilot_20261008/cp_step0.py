# Step 0 only: read-only metadata/JSON/source/snapshot inspection; no training/evaluation.
# Common guard copied from PILOT; never import any PILOT module.
from cp_common import *
import inspect,copy
import torch
import hrl_v2,hrl_uav
from hrl_v2.checkpoint import identity,runtime_record
from hrl_v2.replay import Replay
ORIG=old_run(4104)
PILOT_REF=MAIN/'outputs/scene_optimality_20260928/v2_4_0_randpre_pilot_20261005'
facts=dict(status='checking',created_at=now(),CPIL=str(CPIL),training_started=False,evaluation_started=False,checks={},not_executed=[])
protected={}
def tracked(path):
 path=Path(path);protected[str(path)]=sha(path);return read(path)
def protect(path):
 path=Path(path);protected[str(path)]=sha(path);return protected[str(path)]
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
 protect(PROTO);equal(sha(PROTO),PROTO_SHA,'frozen_protocol_sha')
 binding=identity(WT,CONFIG);pc=tracked(PILOT_REF/'pilot_contract.json');oc=tracked(ORIG/'contract.json');validation=tracked(WORK/'validation/validation.json')
 equal(binding,pc['identity'],'identity_PILOT');equal(binding,oc['identity'],'identity_ORIG');equal(binding,validation['identity'],'identity_validation');equal(validation['status'],'passed','validation_status')
 facts['identity']=binding
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
 facts['not_executed']=['Step0 reference value/curve extraction and wall-time estimate after disk gate','contract creation','RP evaluations/training','diagnostic groups','S1-S5 and H/F selections','all test evaluations']
 emit();check(free>2*estimate,'disk_free_greater_than_two_times_estimate',facts['disk'])
 facts['status']='PASS';emit()
except BaseException as e:
 facts['status']='failed';facts['first_failure']=dict(error=type(e).__name__,message=str(e));emit()
 failure('step0',e);git=git_states();put(CPIL/'git_after_failure.json',git)
 after={q:sha(q) for q in protected};equal(after,protected,'read_only_inputs_unchanged_after_step0_failure')
 put(CPIL/'final_readonly_checks.json',dict(status='PASS',checked_at=now(),protected_files=len(protected),sha_before=protected,sha_after=after,frozen_protocol_unchanged=sha(PROTO)==PROTO_SHA,git=git,training_started=False,evaluation_started=False))
 lines=['# 4104 课程式预训练试点：第 0 步停止','', '未启动任何训练或评估，未创建合同或 checkpoint。','', '第一处失败：D: 剩余空间不满足“大于估算用量的 2 倍”的硬性条件。','', '```json',json.dumps(facts.get('disk',{}),indent=2),'```','', '已读只读输入的 SHA256 前后一致；冻结协议未变；两棵 Git 工作树均干净。','', '判据 1 / 判据 2：未评估；主表无新结果。','Report evidence: step0_facts/facts.json;failure.json;final_readonly_checks.json.']
 textfile(CPIL/'summary.md','\n'.join(lines)+'\n')
 print(json.dumps(dict(status='STOPPED',stage='step0',error=type(e).__name__,disk=facts.get('disk'),protected_files_unchanged=len(protected),training_started=False,evaluation_started=False),ensure_ascii=False),flush=True)
 raise SystemExit(2)
