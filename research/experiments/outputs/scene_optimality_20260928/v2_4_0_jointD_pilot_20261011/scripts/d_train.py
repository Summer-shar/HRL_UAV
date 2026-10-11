"""Plan D output-local flow. JC collection/SAC/PPO numerical code stays unchanged."""
from j_train import *
from d_random import RandomUpperPolicy
from d_compare import exact_equal
from d_storage import save_candidate,save_full,load_resume
import hashlib

class ControlledPause(Exception):pass


def random_external(seed,n):
    return {k+'_seed':derive(seed,1,0,n,purpose=k) for k in ('arrival','layout','load')}


def replay_mix(replay):
    sources=('hover','random_upper','ppo')
    counts={s:sum(e.identity.get('behavior_source')==s for e in replay.episodes) for s in sources}
    transitions={s:sum(len(e.rewards) for e in replay.episodes if e.identity.get('behavior_source')==s) for s in sources}
    return dict(episodes=counts,transitions=transitions,episode_fraction={s:counts[s]/len(replay.episodes) if replay.episodes else 0. for s in sources},transition_fraction={s:transitions[s]/sum(transitions.values()) if sum(transitions.values()) else 0. for s in sources})


def add_complete(replay,e):
    check(e.complete,'complete_replay_episode')
    before=[(id(x),x.upper_version,x.lower_version) for x in replay.episodes]
    flush_before=replay.flushes
    replay.upper_version=e.upper_version
    replay.add(e)
    equal(replay.flushes,flush_before,'D_no_flush_after_initialization')
    equal([(id(x),x.upper_version,x.lower_version) for x in replay.episodes],(before+[(id(e),e.upper_version,e.lower_version)])[-55:],'D_true_versions_preserved')
    check(len(e.identity['lower_versions_by_transition'])==len(e.rewards),'D_true_lower_version_per_transition')


def collect_extras(learner,replay,seed,numbers,config,scene,clocks,frozen,lower,on_collected=None):
    k=(numbers[0]-201)//5+1
    contexts=[]
    for n in (400+2*k-1,400+2*k):
        before=rng_state();start=time.perf_counter();external=random_external(seed,n)
        streams=Streams(seed,1,0,n);a=PhysicsAdapter(scene,timing=clocks);a.reset(**external)
        policy=RandomUpperPolicy(learner.sac.actor,learner.ppo.actor,None,streams,mode='moving',stochastic=True,random_seed=derive(seed,1,0,n,purpose='random_upper'))
        cov=Coverage(a);upper='random-uniform-0';counts=dict(learner.sac.counts);ppo=dict(learner.ppo.counts)
        e=collect(a,policy,upper_version=upper,lower_version=lower,identity=dict(global_episode=n,behavior_source='random_upper',extra_only_SAC=True,batch=k,external=external),on_event=None,on_tick=cov.tick,reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro'])
        e.identity['lower_versions_by_transition']=[lower]*len(e.rewards)
        equal(hashes(learner),frozen,'extra_collection_frozen_parameters');equal(learner.sac.version,lower,'extra_same_batch_lower_version')
        equal(learner.sac.counts,counts,'extra_no_SAC_updates');equal(learner.ppo.counts,ppo,'extra_no_PPO_updates')
        available=len(replay.episodes);add_complete(replay,e)
        context=dict(ordinal=n,plan=dict(layer='extra_SAC_only',batch=k,alpha=None,actor_enabled=False,source_phase=dict(stage=1,cycle=0,episode=n)),episode=e,external=external,coverage=cov.result(),max_actual_step_m=float(np.linalg.norm(np.asarray([h['displacement'] for h in e.high]),axis=-1).max()),collection_seconds=time.perf_counter()-start,collection_hashes=frozen,replay_completed_before=available,replay_completed_after=len(replay.episodes),actor_relative_entropy=policy.entropy_sum/policy.entropy_count,entropy_samples=policy.entropy_count)
        contexts.append(context)
        if on_collected:on_collected(context)
        rng_assert(before,'extra_collection_no_global_RNG_changes')
        del a,policy,cov
    equal([c['ordinal'] for c in contexts],[400+2*k-1,400+2*k],'extra_IDs')
    return contexts


def collect_base(learner,replay,seed,n,config,scene,clocks,learning=None):
    before_rng=rng_state();p=source_phase(n);plan=ordinal_plan(n);external=external_seeds(seed,n)
    streams=Streams(seed,p.stage,p.cycle,p.block_episode);a=PhysicsAdapter(scene,timing=clocks);a.reset(**external)
    behavior='hover' if n%2 else 'random_upper';upper='hover-0' if n%2 else 'random-uniform-0'
    if n%2:policy=collection_policy(learner,streams,p)
    else:policy=RandomUpperPolicy(learner.sac.actor,learner.ppo.actor,None,streams,mode='moving',stochastic=True,random_seed=derive(seed,1,0,n,purpose='random_upper'))
    rng_assert(before_rng,'base_pre_episode')
    versions=[];original_lower=policy.lower
    def observed_lower(*args,**kwargs):
        versions.append(learner.sac.version)
        return original_lower(*args,**kwargs)
    policy.lower=observed_lower
    before=dict(learner.sac.counts);lower=learner.sac.version;res=[];cov=Coverage(a);start=time.perf_counter();available=len(replay.episodes)
    def event():
        guard_resources();log=learner.sac.event(replay,plan['alpha'],plan['actor_enabled'],streams)
        res.extend(v for fit in log.get('critics',[]) for v in fit['residual_mse'])
        if learning:learning(dict(kind='sac',global_episode=n,sac_ordinal=n,physical_tick=a.env.step_index,event=learner.sac.counts['triggers'],alpha=plan['alpha'],**log))
    e=collect(a,policy,upper_version=upper,lower_version=lower,identity=dict(global_episode=n,external=external,behavior_source=behavior),on_event=event,on_tick=cov.tick,reward_scale=config['objective']['reward_scale'],gamma_lower=config['objective']['gamma_micro'])
    e.identity['lower_versions_by_transition']=versions;add_complete(replay,e)
    context=dict(ordinal=n,plan=plan,episode=e,external=external,coverage=cov.result(),max_actual_step_m=float(np.linalg.norm(np.asarray([h['displacement'] for h in e.high]),axis=-1).max()),collection_seconds=time.perf_counter()-start,actor_relative_entropy=policy.entropy_sum/policy.entropy_count,entropy_samples=policy.entropy_count,replay_completed_before=available,replay_completed_after=len(replay.episodes),sac_delta=delta(learner.sac.counts,before),sac_counts=dict(learner.sac.counts),ppo_counts=dict(learner.ppo.counts),critic_residual_mse=avg(res) if res else None)
    if n==1:
        old=rows(JC/f'seed_{seed}/train/train_metrics.jsonl')[0]
        actual=dict(event_sha256=e.metrics['event_sha256'],sac=context['sac_counts'],sac_delta=context['sac_delta'],critic_residual_mse=context['critic_residual_mse'],reward_physical=float(sum(e.rewards)),reward_learning_macro_sum=float(sum(h['reward_learning'] for h in e.high)))
        expected={k:old[k] for k in actual};proof=exact_equal(actual,expected,'JC_episode1_exact',record=True)
        context['episode1_anchor']=dict(actual=actual,expected=expected,proof=proof)
    rng_assert(before_rng,'base_no_global_RNG_changes')
    return context


def replay_digest(replay):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,np.ndarray):h.update(str((x.dtype.str,x.shape)).encode());h.update(x.tobytes())
        elif torch.is_tensor(x):visit(x.detach().cpu().numpy())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):visit(str(k));visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for v in x:visit(v)
        elif hasattr(x,'__dict__'):visit(x.__dict__)
        else:h.update(repr(x).encode('utf-8'))
    visit(replay);return h.hexdigest()


def save_resume(run,learner,replay,seed,n,extra):
    save_started=time.perf_counter();before=rng_state();directory=output_path(run/'resume');directory.mkdir(exist_ok=True)
    pointer=directory/'latest.json';old=read(pointer) if pointer.exists() else None
    path=directory/f'boundary-{n:04d}-{time.time_ns()}.pt';ph=hashes(learner)
    counts=dict(sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts))
    payload=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding(),completed_episode=n,pending_evaluation=None,learner=learner.state_dict(),replay=replay,global_cpu_rng=before['cpu'],global_cuda_rng=before['cuda'],seed=seed,resumable=True,kind='jointD_full_resume',protocol_contract_sha256=sha(CPIL/'contract.json'),parameter_hashes=ph,counts=counts,extra=extra)
    save_full(path,payload)
    put(pointer,dict(path=str(path),completed_episode=n,parameter_hashes=ph,counts=counts,time=now(),write_complete=True,readback_performed=False,verification_policy='loadability and parameter/count continuity on actual resume'))
    del payload
    if old:
        oldpath=output_path(old['path']);check(oldpath.parent==directory,'delete_only_prior_resume_in_OUT')
        oldpath.unlink();oldpath.with_suffix('.json').unlink()
    rng_assert(before,'resume_output_no_global_RNG_changes')
    append(run/'resume_timings.jsonl',dict(completed_episode=n,wall_seconds=time.perf_counter()-save_started,bytes=path.stat().st_size,time=now(),readback_performed=False,sha256_performed=False,parameter_hashes_recorded=True))


def restore_resume(run,learner):
    info=read(run/'resume/latest.json');z=load_resume(info['path']);check(z['kind']=='jointD_full_resume','resume_D_full_state_only')
    permitted=contract()['lean_runtime']['accepted_previous_contract_SHA256']+[sha(CPIL/'contract.json')]
    check(z['protocol_contract_sha256'] in permitted,'resume_same_training_contract_amendment')
    n=z['completed_episode'];equal(n,info['completed_episode'],'resume_boundary_count_continuity')
    expected_sac=budget_counts(n);equal(z['learner']['sac']['counts'],expected_sac,'resume_SAC_expected_budget')
    equal(z['learner']['ppo']['counts']['batches'],max(0,(n-200)//5),'resume_PPO_batch_continuity')
    if info.get('counts'):equal(info['counts'],{k:z['learner'][k]['counts'] for k in ('sac','ppo')},'resume_recorded_counts_continuity')
    historical=[r for r in rows(run/'train_metrics.jsonl') if r['global_episode']==n]
    if historical:equal(historical[0]['sac'],expected_sac,'resume_train_row_SAC_continuity');equal(historical[0]['ppo'],z['learner']['ppo']['counts'],'resume_train_row_PPO_continuity')
    learner.load_state_dict(z['learner']);equal(learner.state_dict(),z['learner'],'actual_resume_network_optimizer_counts_equal');equal(hashes(learner),info['parameter_hashes'],'restored_parameters_equal_preinterruption_record')
    torch.set_rng_state(z['global_cpu_rng']);torch.cuda.set_rng_state_all(z['global_cuda_rng'])
    equal(rng_state(),dict(cpu=z['global_cpu_rng'],cuda=z['global_cuda_rng']),'actual_resume_RNG_equal')
    replay,extra=z['replay'],z['extra'];check(all(e.complete for e in replay.episodes),'actual_resume_replay_complete');equal(replay.flushes,0,'actual_resume_no_flush')
    put(run.parent/'recovery/restore_validation.json',dict(passed=True,time=now(),completed_episode=n,loadable=True,parameter_hashes_equal=True,counts_continuous=True,full_network_optimizer_counters_equal=True,CPU_RNG_equal=True,CUDA_RNG_equal=True,replay_loaded=True,sha256_not_recomputed=True,decision_path_changed=False))
    recovery=output_path(run/'interruptions'/str(time.time_ns()));recovery.mkdir(parents=True)
    # Preserve unsafe suffix artifacts, then continue from the verified whole-episode/batch boundary.
    offsets=z['extra']['log_offsets']
    for name,size in offsets.items():
        p=run/name
        if p.exists() and p.stat().st_size>size:
            shutil.copy2(p,recovery/name)
            with p.open('r+b') as f:f.truncate(size)
    for sub in ('episodes','positions','batches','checkpoints'):
        for p in (run/sub).glob('*'):
            if not p.is_file():continue
            nums=[int(x) for x in __import__('re').findall(r'\d+',p.stem)]
            boundary=max(nums) if nums else 0
            if sub in ('episodes','positions') and boundary>400:
                boundary=200+5*((boundary-401)//2+1)
            if boundary>n:
                dest=recovery/sub/p.name;dest.parent.mkdir(exist_ok=True);shutil.move(str(p),str(dest))
    put(recovery/'resume.json',dict(time=now(),verified_boundary=n,source=info,parameter_hashes=hashes(learner),discarded_unsafe_suffix_preserved=str(recovery)))
    return z['replay'],z['extra'],n


def run_seed(seed):
    assert_imports();source_guard();guard_resources();set_below_normal();check(seed in (4104,4103),'only_two_D_seeds')
    config=read_config(CONFIG,WT);scene=read(SCENE);clocks=ClockConfig.from_config(config)
    equal(budget_counts(200),EXPECTED200,'base_planned_budget');equal(budget_counts(),EXPECTED_FINAL,'final_planned_budget')
    run=output_path(CPIL/f'seed_{seed}/train');learner=Learner(seed,configure(),config=config);started=time.perf_counter();initial_ppo=copy.deepcopy(learner.state_dict()['ppo'])
    if run.exists():
        check((run/'resume/latest.json').exists(),'existing_run_requires_verified_resume')
        replay,extra,completed=restore_resume(run,learner);saved=extra['saved'];batches=extra['batches'];skipped=extra['skipped'];elapsed_prior=extra['elapsed_seconds'];initial_ppo=extra['initial_ppo']
    else:
        run.mkdir(parents=True);replay=Replay();saved=[];batches=[];skipped=[];completed=0;elapsed_prior=0.
        equal(replay.flushes,0,'fresh_replay_zero_flush');equal(learner.sac.counts,{k:0 for k in EXPECTED_FINAL},'fresh_no_loaded_weights_or_updates')
        put(run/'fresh_initialization.json',dict(seed=seed,hashes=hashes(learner),checkpoint_loads=0,empty_replay=True))
        manifest=[dict(ordinal_plan(n),external=external_seeds(seed,n),behavior_source='hover' if n<=200 and n%2 else 'random_upper' if n<=200 else 'ppo') for n in range(1,401)]
        manifest.extend(dict(global_episode=n,external=random_external(seed,n),behavior_source='random_upper',extra_only_SAC=True,batch=(n-401)//2+1) for n in range(401,481));put(run/'training_manifest.json',manifest)
    def progress(n,collected=None,stage='training'):
        data=dict(status='running',stage=stage,seed=seed,completed_episode=n,total_episodes=400,collected_episode=n if collected is None else collected,extra_collected=2*max(0,(n-200)//5),sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts),replay_flushes=replay.flushes,replay_mix=replay_mix(replay),elapsed_seconds=elapsed_prior+time.perf_counter()-started,time=now())
        put(run/'training_progress.json',data)
    def learning(row):append(run/'learning.jsonl',dict(seed=seed,**row))
    def collection_record(c):
        n,e=c['ordinal'],c['episode'];put(run/f'positions/episode-{n:04d}.json',dict(global_episode=n,behavior_source=e.identity['behavior_source'],coverage=c['coverage'],max_actual_step_m=c['max_actual_step_m']))
        put(run/f'episodes/episode-{n:04d}.json',dict(seed=seed,global_episode=n,identity=e.identity,metrics=dict(compact(e.metrics),event_sha256=e.metrics['event_sha256']),upper_version=e.upper_version,lower_version=e.lower_version,collection_complete=True,updates_complete=False))
        progress(completed,n,'collecting_frozen_joint_batch' if n>200 else 'mixed_base')
        if n>400:
            row=dict(global_episode=n,batch=e.identity['batch'],reward_physical=float(sum(e.rewards)),reward_learning_macro_sum=float(sum(h['reward_learning'] for h in e.high)),collection_seconds=c['collection_seconds'],behavior_source='random_upper',upper_version=e.upper_version,lower_version=e.lower_version,extra_only_SAC=True)
            append(run/'extra_metrics.jsonl',row)
    def finalize(c):
        nonlocal completed
        n,e=c['ordinal'],c['episode'];collection_record(c) if n<=200 else None
        checkpoint_seconds=0.
        if n in CANDIDATES:
            begin=time.perf_counter();saved.append(save_candidate(run/f'checkpoints/episode-{n:04d}.pt',learner,replay,seed,n));checkpoint_seconds=time.perf_counter()-begin
        row=dict(seed=seed,global_episode=n,sac_ordinal=n,behavior_source=e.identity['behavior_source'],layer=c['plan']['layer'],alpha=c['plan']['alpha'],actor_enabled=c['plan']['actor_enabled'],reward_physical=float(sum(e.rewards)),reward_learning_macro_sum=float(sum(h['reward_learning'] for h in e.high)),critic_residual_mse=c.get('critic_residual_mse'),sac_delta=c['sac_delta'],sac=c['sac_counts'],ppo=c['ppo_counts'],event_sha256=e.metrics['event_sha256'],behavior_upper_version=e.upper_version,behavior_lower_version=e.lower_version,collection_seconds=c['collection_seconds'],sac_seconds=c.get('sac_seconds',0.),ppo_seconds=c.get('ppo_seconds',0.),checkpoint_seconds=checkpoint_seconds,replay_flushes=replay.flushes,**compact(e.metrics))
        row['work_seconds']=row['collection_seconds']+row['sac_seconds']+row['ppo_seconds'];row['wall_seconds']=row['work_seconds']+checkpoint_seconds
        p=run/f'episodes/episode-{n:04d}.json';d=read(p);d.update(updates_complete=True,train_metric=row);put(p,d);append(run/'train_metrics.jsonl',row)
        if c.get('episode1_anchor'):put(run/'episode1_anchor.json',c['episode1_anchor'])
        if row['sac_delta']['skipped']:skipped.append(n)
        completed=n;progress(n,stage='mixed_base' if n<=200 else 'joint_updates');print('D_TRAIN',seed,n,flush=True)
    def boundary(n):
        offsets={name:(run/name).stat().st_size if (run/name).exists() else 0 for name in ('learning.jsonl','train_metrics.jsonl','extra_metrics.jsonl')}
        save_resume(run,learner,replay,seed,n,dict(saved=saved,batches=batches,skipped=skipped,log_offsets=offsets,elapsed_seconds=elapsed_prior+time.perf_counter()-started,initial_ppo=initial_ppo))
        if (run.parent/'pause_requested.json').exists():
            progress(n,stage='paused_at_verified_boundary');raise ControlledPause('user-authorized parallel slowdown guard')
    try:
        progress(completed)
        if completed==0:progress(0)
        for n in range(completed+1,201):
            source_guard();c=collect_base(learner,replay,seed,n,config,scene,clocks,learning);finalize(c)
            if n==200:
                equal(learner.sac.counts,EXPECTED200,'base_actual_exact_budget');equal(learner.sac.counts,contract()['lean_runtime']['JC_SAC_base_counts'][str(seed)],'base_budget_equals_JC_actual');equal(learner.state_dict()['ppo'],initial_ppo,'base_PPO_unchanged');equal(replay.flushes,0,'base_no_flush');equal(len(replay.episodes),55,'base_FIFO55')
                put(run/'base_hardchecks.json',dict(passed=True,SAC_counts=dict(learner.sac.counts),PPO_unchanged=True,replay_flushes=0))
            if n%20==0:boundary(n)
            del c
        for first in range(max(201,completed+1),401,5):
            source_guard();batch=run_joint_batch(learner,replay,seed,range(first,first+5),config,scene,clocks,on_collected=collection_record,on_learning=learning,on_ordinal=finalize)
            audit=batch['audit'];equal(audit['extra_lower_versions'],[audit['behavior_lower_version']]*2,'extra_lower_versions_hardcheck');equal(audit['normal_episodes'],list(range(first,first+5)),'PPO_normal_five_only');equal(audit['replay_flushes_delta'],0,'joint_no_flush')
            batches.append(audit);put(run/f'batches/batch-{first:04d}-{first+4:04d}.json',audit)
            if (first+4)%20==0:boundary(first+4)
            del batch
        equal(learner.sac.counts,EXPECTED_FINAL,'D_final_actual_SAC_budget');equal(learner.sac.counts,contract()['lean_runtime']['JC_SAC_final_counts'][str(seed)],'D_final_budget_equals_JC_actual');equal(delta(learner.sac.counts,EXPECTED200),EXPECTED_JOINT,'D_joint_actual_SAC_budget');equal(learner.ppo.counts['batches'],40,'D_PPO40');equal(replay.flushes,0,'D_all_flushes_zero');equal(skipped,[1],'D_only_first_skips');equal(len(saved),10,'D_ten_candidates')
        sequence=list(range(1,201))
        for k in range(1,41):sequence.extend(list(range(201+5*(k-1),206+5*(k-1)))+[400+2*k-1,400+2*k])
        equal([e.identity['global_episode'] for e in replay.episodes],sequence[-55:],'D_FIFO55_actual_collection_order')
        result=dict(status='passed',seed=seed,completed_episode=400,normal_episodes=400,extra_episodes=80,SAC_counts=dict(learner.sac.counts),joint_SAC_delta=EXPECTED_JOINT,base_SAC_counts=EXPECTED200,PPO_counts=dict(learner.ppo.counts),replay_flushes=0,final_replay=replay_summary(replay),checkpoints=saved,batch_audits=batches,hardchecks_passed=True,wall_seconds=elapsed_prior+time.perf_counter()-started,ended_at=now())
        put(run/'result.json',result);progress(400,stage='seed_training_complete_waiting_evaluation');return result
    finally:
        source_guard();del learner,replay;gc.collect();torch.cuda.empty_cache()
