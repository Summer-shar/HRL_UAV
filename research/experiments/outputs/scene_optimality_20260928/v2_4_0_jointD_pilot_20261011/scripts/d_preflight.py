from d_guard import install,ROOT
install()
from j_common import *
import py_compile,subprocess
from d_compare import exact_equal
from d_train import *
from d_setup import input_checks,write_diff


def unit_tests():
    begin=time.perf_counter();tests=[]
    equal(budget_counts(200),EXPECTED200,'unit_JC_base_budget_from_contract');equal(budget_counts(),EXPECTED_FINAL,'unit_JC_total_budget_from_contract')
    for p in (ROOT/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True,cfile=str(ROOT/'selftest'/('compile_'+p.name+'c')))
    exact_equal(np.float64(1.25),1.25,'np_float_exact');rejected=False
    try:exact_equal(np.nextafter(1.25,2.),1.25,'one_ULP_reject')
    except HardCheckError:rejected=True
    check(rejected,'one_ULP_is_not_equal');tests.append(dict(test='numeric_exact',passed=True))
    for mode in ('guard','success','worker_failure','worker_timeout'):
        start=time.perf_counter();p=subprocess.run([str(PY),'-B','-u',str(ROOT/'scripts/d_test_child.py'),mode],cwd=str(ROOT),env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        textfile(ROOT/f'selftest/{mode}.stdout.log',p.stdout);textfile(ROOT/f'selftest/{mode}.stderr.log',p.stderr)
        check((p.returncode==0)==(mode in ('guard','success')),'peripheral_child_'+mode,p.returncode)
        if mode in ('worker_failure','worker_timeout'):
            check(bool(list((ROOT/'errors').glob('*.json'))),'pool_error_traceback_saved')
        tests.append(dict(test=mode,passed=True,returncode=p.returncode,seconds=time.perf_counter()-start))
    from d_results import write_report,comparison
    reference=read(DG2/'groups/K-TFOO-4104/summary.json');reference['comparison_TFOO']=comparison(reference,reference)
    data=dict(seed=4104,ended_at='SMOKE REPORT ONLY; NOT PLAN D RESULTS',selected={},last={},groups=[reference],JC_comparison=contract()['references_D']['4104'],training=dict(SAC_counts=EXPECTED_FINAL,base_SAC_counts=EXPECTED200,joint_SAC_delta=EXPECTED_JOINT,PPO_counts=dict(batches=40),replay_flushes=0,hardchecks_passed=True),returns_below_minus300=dict(count=0,episodes=[]),PPO_update_summary=[],batch_returns_and_replay_mix=[])
    write_report(data,output_root=ROOT/'smoke/report_selftest');check((ROOT/'smoke/report_selftest/seed_4104/report.md').exists(),'report_smoke_generated');tests.append(dict(test='report_generation_fixture_not_scientific',passed=True))
    put(ROOT/'selftest/results.json',dict(passed=True,tests=tests,wall_seconds=time.perf_counter()-begin,time=now()));return tests


def smoke():
    config=read_config(CONFIG,WT);scene=read(SCENE);clocks=ClockConfig.from_config(config);started=time.perf_counter();set_below_normal();device=configure();learner=Learner(4104,device,config=config);replay=Replay();initial=copy.deepcopy(learner.state_dict()['ppo']);records=[]
    for n in (1,2):
        put(ROOT/'progress.json',dict(status='preparing',stage='smoke_S1',completed_episode=n-1,time=now()));c=collect_base(learner,replay,4104,n,config,scene,clocks)
        records.append(dict(global_episode=n,behavior=c['episode'].identity['behavior_source'],return_physical=float(sum(c['episode'].rewards)),SAC=c['sac_counts'],anchor=c.get('episode1_anchor'),replay_flushes=replay.flushes,wall_seconds=c['collection_seconds']))
        print('SMOKE_S1_EPISODE_PASS',n,flush=True);del c
    equal(learner.state_dict()['ppo'],initial,'smoke_S1_no_PPO');equal(replay.flushes,0,'smoke_S1_flush0');equal(records[1]['behavior'],'random_upper','smoke_S1_episode2_random');equal(learner.sac.counts['events'],90,'smoke_S1_events');equal(learner.sac.counts['actor_updates'],0,'smoke_S1_actor_frozen')
    # Verify full resume replacement and restoration on the same actual smoke learner.
    run=ROOT/'smoke/resume_test';run.mkdir(parents=True,exist_ok=True)
    extra=dict(saved=[],batches=[],skipped=[1],log_offsets={},elapsed_seconds=0.,initial_ppo=initial)
    save_resume(run,learner,replay,4104,2,extra);saved_hash=hashes(learner);save_resume(run,learner,replay,4104,2,extra)
    equal(len(list((run/'resume').glob('*.pt'))),1,'smoke_resume_only_latest_retained')
    restored=Learner(4104,device,config=config);rr,ex,n=restore_resume(run,restored);equal(hashes(restored),saved_hash,'smoke_resume_parameters_match');equal(replay_digest(rr),replay_digest(replay),'smoke_resume_replay_exact')
    del rr,restored,learner,replay,initial;gc.collect();torch.cuda.empty_cache()
    put(ROOT/'progress.json',dict(status='preparing',stage='smoke_S2',time=now()))
    cp=Path(contract()['smoke_S2_checkpoint']);z=load(cp,binding());check('replay' in z,'S2_fallback_has_replay');learner=Learner(4104,device,config=config);learner.load_state_dict(z['learner']);replay=z['replay'];torch.set_rng_state(z['global_cpu_rng']);torch.cuda.set_rng_state_all(z['global_cuda_rng']);del z;gc.collect()
    oldflush=replay.flushes;counts=dict(learner.sac.counts);oldppo=learner.ppo.counts['batches']
    for e in replay.episodes:e.identity['behavior_source']='hover'  # in-memory smoke-only provenance
    def smoke_collected(c):
        check(c['episode'].identity['behavior_source'] in ('ppo','random_upper'),'smoke_all_behavior_labels')
        put(ROOT/'smoke/S2_progress.json',dict(stage='collection',collected=c['ordinal'],time=now()));print('SMOKE_S2_COLLECTED',c['ordinal'],flush=True)
    def smoke_ordinal(c):
        put(ROOT/'smoke/S2_progress.json',dict(stage='SAC_updates',completed_ordinal=c['ordinal'],SAC=c['sac_counts'],time=now()));print('SMOKE_S2_ORDINAL_DONE',c['ordinal'],flush=True)
    batch=run_joint_batch(learner,replay,4104,range(201,206),config,scene,clocks,on_collected=smoke_collected,on_ordinal=smoke_ordinal)
    equal(batch['audit']['SAC_delta'],dict(triggers=450,skipped=0,events=450,critic_fits=900,actor_updates=225,target_updates=450),'smoke_S2_exact_budget')
    equal(replay.flushes,oldflush,'smoke_S2_no_flush');equal(learner.ppo.counts['batches']-oldppo,1,'smoke_S2_one_PPO');equal(batch['audit']['normal_episodes'],[201,202,203,204,205],'smoke_S2_five_normal');equal(batch['audit']['extra_episodes'],[401,402],'smoke_S2_two_extras');equal(batch['audit']['extra_lower_versions'],[batch['audit']['behavior_lower_version']]*2,'smoke_S2_same_lower_version');check(batch['audit']['extra_excluded_from_PPO'],'smoke_S2_extra_excluded')
    result=dict(passed=True,S1=records,S2=dict(source=str(cp),source_sha256=sha(cp),audit=batch['audit'],ppo_logs=batch['ppo_logs'],original_resume_replay_flushes=oldflush),resume_roundtrip_passed=True,wall_seconds=time.perf_counter()-started,time=now(),results_excluded_from_scientific_report=True)
    put(ROOT/'smoke/summary.json',result)
    del batch,learner,replay;gc.collect();torch.cuda.empty_cache()
    for p in (ROOT/'smoke').rglob('*.pt'):output_path(p).unlink();side=p.with_suffix('.json');side.unlink() if side.exists() else None
    put(ROOT/'smoke/large_files_cleanup.json',dict(time=now(),only_OUT_smoke_pt_deleted=True,remaining_pt=0));print('SMOKE_S2_PASS',flush=True);return result

if __name__=='__main__':
    try:
        input_checks();tests=unit_tests();result=smoke();
        from d_eval import group
        from d_results import archive_group
        anchor=group(dict(label='TFOO-anchor-smoke',seed=4104,upper=str(old_tfoo(4104)/'checkpoints/episode-0200-anchor.pt'),lower=None,upper_kind='TFOO',route='cvx',anchors=[str(NEW/'summaries/seed_4104/TFOO-200.json')],diagnostic_only=True),scenario_ids=[1],output_root=ROOT/'smoke/eval')
        archive_group(anchor);put(ROOT/'smoke/TFOO_anchor.json',dict(passed=True,scenarios=[1],streams=[1,2],mean_s=anchor['mean_s'],anchor_checks=anchor['anchor_checks']))
        input_checks();write_diff();put(ROOT/'preflight.json',dict(passed=True,time=now(),unit_tests=tests,smoke_summary=str(ROOT/'smoke/summary.json'),smoke_seconds=result['wall_seconds']));print('PREFLIGHT_ALL_PASS',flush=True)
    except BaseException as e:
        record=failure('preflight',e);put(ROOT/'step0/failure.json',record);raise
