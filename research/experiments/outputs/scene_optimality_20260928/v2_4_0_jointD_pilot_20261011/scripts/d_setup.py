from d_guard import install,ROOT
install()
from j_common import *
import ast,difflib,py_compile
from hrl_v2.checkpoint import identity,runtime_record
from hrl_v2.evaluate import gate_manifest
from hrl_v2.randomness import derive
from d_train import random_external


def input_checks():
    c=contract();lean=c.get('lean_runtime',{})
    if lean.get('enabled'):
        proof=ROOT/'recovery/lean_input_check.json'
        if proof.exists():
            done=read(proof);equal(done['transition_id'],lean['transition_id'],'same_lean_startup_input_check');check(done['passed'],'lean_startup_input_check_passed');return done
        changes=[]
        for p,h in lean['actual_input_SHA256'].items():
            actual=sha(p)
            if actual!=h:changes.append(dict(path=p,expected=h,actual=actual))
        check(not changes,'actual_inputs_SHA_startup_once',changes)
        actual=git_states();equal(actual,c['git_before'],'MAIN_WT_git_status_HEAD_unchanged')
        done=dict(passed=True,time=now(),transition_id=lean['transition_id'],protected_count=len(lean['actual_input_SHA256']),git=actual,check_policy='actual inputs once at lean transition')
        put(proof,done);return done
    changes=[dict(path=p,expected=h,actual=sha(p)) for p,h in c['protected_input_sha256'].items() if sha(p)!=h]
    check(not changes,'all_protected_inputs_SHA_unchanged',changes)
    equal(identity(WT,CONFIG),c['identity'],'code_config_scene_unchanged')
    actual=git_states();equal(actual,c['git_before'],'MAIN_WT_git_status_HEAD_unchanged');return dict(passed=True,protected_count=len(c['protected_input_sha256']),git=actual,time=now())


def write_diff():
    out=[];provenance=[]
    for p in sorted((ROOT/'scripts').glob('*.py')):
        baseline=JC/p.name
        old=baseline.read_text(encoding='utf-8-sig') if baseline.exists() else ''
        out.extend(difflib.unified_diff(old.splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile=str(baseline) if baseline.exists() else '/dev/null',tofile=str(p)))
        provenance.append(dict(path=str(p),sha256=sha(p),JC_source=str(baseline) if baseline.exists() else None,DG2_source=str(DG2/'scripts'/p.name) if (DG2/'scripts'/p.name).exists() else None))
    textfile(ROOT/'scripts_vs_JC.diff',''.join(out));put(ROOT/'script_provenance.json',provenance)


def prepare():
    check(not (ROOT/'execution_start.json').exists(),'prepare_before_training')
    c=read(JC/'contract.json');c.update(schema='jointD_mixed_base_plus_two_SAC_only_v1',created_at=now(),status='frozen',output=str(ROOT),seed_order=[4104,4103],initial_seeds=[4104,4103],conditional_seeds=[],conditional_expansion=False,condition=None,training_resumable=True)
    c['paths']=dict(OUT=str(ROOT),JC=str(JC),DG2=str(DG2),ATT3=str(PRIOR),PILOT=str(PILOT_REF),PY=str(PY),WT=str(WT),MAIN=str(MAIN))
    c['identity']=identity(WT,CONFIG);c['runtime']=runtime_record();c['config']=read(CONFIG);c['git_before']=git_states();c['gpu_before']=gpu_check()
    check(c['identity']==read(JC/'contract.json')['identity'],'JC_same_identity')
    protected=list(JC.glob('*.py'))+[JC/'contract.json']+list((DG2/'scripts').glob('*.py'))+[PILOT_REF/'pilot_train.py',PRIOR/'cp_train.py',PRIOR/'cp_eval.py',PRIOR/'cp_main.py']
    for seed in (4104,4103):
        protected.extend([JC/f'seed_{seed}/train/train_metrics.jsonl',JC/f'seed_{seed}/train/result.json',JC/f'seed_{seed}/{seed}report.json',old_tfoo(seed)/'checkpoints/episode-0200-anchor.pt',old_tfoo(seed)/'checkpoints/episode-0200-anchor.json',old_run(seed)/'training_manifest.json'])
    c['smoke_S2_checkpoint']=str(PRIOR/'seed_4104/step1_attempt1/checkpoints/A-S200h.pt');protected.extend([Path(c['smoke_S2_checkpoint']),Path(c['smoke_S2_checkpoint']).with_suffix('.json')])
    refs={};expect={4104:(.762376,.802633,.837859),4103:(.777382,1.023800,1.057523)};elapsed={}
    for seed in (4104,4103):
        paths=[DG2/f'groups/K-TFOO-{seed}/summary.json',DG2/f'groups/K-JC-{seed}/summary.json',DG2/f'groups/B-{seed}-400/summary.json'];protected.extend(paths)
        vals=[read(p)['mean_s'] for p in paths];equal([round(v,6) for v in vals],list(expect[seed]),'reference_6digits_'+str(seed))
        curves=[]
        for n in range(220,401,20):
            label=('A5' if n==320 else 'A6') if seed==4103 and n in (320,340) else f'B-{seed}-{n}'
            p=DG2/f'groups/{label}/summary.json';protected.append(p);curves.append(dict(episode=n,mean_s=read(p)['mean_s'],source=str(p)))
        metric=rows(JC/f'seed_{seed}/train/train_metrics.jsonl');first=metric[0]
        refs[str(seed)]=dict(TFOO=dict(mean_s=vals[0],source=str(paths[0])),JC_selected=dict(mean_s=vals[1],source=str(paths[1])),JC_upper_cvx400=dict(mean_s=vals[2],source=str(paths[2])),JC_upper_cvx_curve=curves,JC_episode1={k:first[k] for k in ('event_sha256','sac','sac_delta','critic_residual_mse','reward_physical','reward_learning_macro_sum')})
        dg=[read(p)['wall_seconds'] for p in (DG2/'groups').glob('*/summary.json') if read(p)['group']['seed']==seed]
        coll=avg([r['collection_seconds'] for r in metric[200:]])
        train=read(JC/f'seed_{seed}/train/result.json')['wall_seconds'];eval_original=sum(r.get('evaluation_seconds',0) for r in metric)
        elapsed[str(seed)]=dict(JC_train_including_candidate_eval_seconds=train,JC_candidate_eval_seconds=eval_original,JC_collection_seconds_mean=coll,training_estimated_seconds=train-eval_original+80*coll,eval_groups_min=22,eval_groups_max=23,evaluation_estimated_seconds=23*avg(dg),estimate_includes_only_JC_DG2_measured_work=True,resume_IO_and_random_environment_runtime_uncertain=True)
    dev=gate_manifest(20)['scenarios'];sealed=[{k+'_seed':derive(0,episode=n,purpose=k,split='test') for k in ('arrival','layout','load')} for n in range(1,41)]
    overlap=[];seeds401={}
    for seed in (4104,4103):
        extra={str(n):random_external(seed,n) for n in range(401,481)};seeds401[str(seed)]=extra
        for n,streams in extra.items():
            for kind in ('arrival_seed','layout_seed','load_seed'):
                if streams[kind] in {s[kind] for s in dev+sealed}:overlap.append(dict(seed=seed,episode=n,purpose=kind))
    check(not overlap,'extra_train_external_streams_disjoint_dev_test',overlap)
    c['scene_stream_isolation']=dict(passed=True,extra_streams=seeds401,dev_manifest=dev,test_namespace_only_no_test_scene_read=True,test_seed_IDs_checked=list(range(1,41)))
    c['schedule']=dict(base=dict(episodes=[1,200],odd='hover',even='random_upper',online_SAC_events_per_episode=90,upper_training=False),joint=dict(normal=[201,400],batches=40,normal_per_batch=5,extra_per_batch=2,extra_IDs=[401,480],order=['freeze','five_normal','two_random_SAC_only','PPO_original_five','SAC450'],extra_excluded_PPO=True))
    c['replay']=dict(capacity_episodes=55,sampling='original Replay.sample',clears_after_initialization=0,version_tags='true behavior_source and lower version per transition',resume='latest verified completed episode/batch boundary; parameters/RNG/optimizer/replay restored; unsafe suffix preserved')
    c['candidate_episodes']=list(range(220,401,20));c['selection']=dict(metric='truncated_delay_s',minimize=True,tie='earlier',candidates=c['candidate_episodes']);c['criteria']=dict(improvement_ge=.05,wins_ge=11,scenarios=20)
    c['diagnostics']=dict(upper_cvx_episodes=c['candidate_episodes'],TFOO_upper_new_SAC=['selected','last'],deduplicate=True,selection_participation=False)
    c['references_D']=refs;c['estimated_times']=elapsed
    c['task_timeout']=dict(task_timeout_seconds=read(DG2/'contract.json')['task_timeout']['task_timeout_seconds'],source=str(DG2/'contract.json'),method='DG2 fixed template: 3 times longest JC/ATT3 scenario')
    protected.extend([DG2/'contract.json',CONFIG,SCENE])
    baseline=Path('D:/ExperimentRecords_20261010/experiments/D/HRL_UAV_V2_speedup/outputs/lite_v2_4_0/baselines/_evidence/summary.json');protected.append(baseline)
    equal(sha(baseline),read(JC/'contract.json')['protected_input_sha256'][str(WT/'outputs/lite_v2_4_0/baselines/summary.json')],'archived_baseline_index_exact_original_SHA')
    protected.extend(baseline.parent.glob('hover_equal_*.json'))
    for seed in (4104,4103):
        anchor=NEW/f'summaries/seed_{seed}/TFOO-200.json';protected.append(anchor)
        protected.extend(Path(w['path']) for w in read(anchor)['window_files'])
    c['protected_input_sha256']={str(p):sha(p) for p in sorted(set(protected))}
    # Conservative peak includes all JC two-seed footprints plus 46 compressed DG2 groups,
    # two full resume files during replacement and one full raw group during compression.
    jcbytes=sum(p.stat().st_size for p in JC.glob('seed_*/**/*') if p.is_file())
    archive_sizes=[p.stat().st_size for p in DG2.rglob('*.zip')];archive_mean=avg(archive_sizes) if archive_sizes else 300*1024**2
    resume_size=Path(c['smoke_S2_checkpoint']).stat().st_size
    raw_peak=max(sum(Path(w['path']).stat().st_size for w in read(p)['window_files']) if all(Path(w['path']).exists() for w in read(p)['window_files']) else 3500*1024**2 for p in (DG2/'groups').glob('*/summary.json'))
    peak=jcbytes+46*archive_mean+2*resume_size+raw_peak;free=shutil.disk_usage(ROOT).free
    c['resources']=dict(free_bytes=free,estimated_peak_bytes=peak,minimum_free_bytes=max(5*1024**3,2*peak),JC_two_seed_bytes=jcbytes,DG2_archive_mean_bytes=archive_mean,raw_group_peak_bytes=raw_peak)
    check(free>=max(5*1024**3,2*peak),'disk_5GiB_and_twice_peak',c['resources'])
    c['constraints']=dict(no_source_edits=True,no_old_output_writes=True,no_test_evaluation=True,no_commit_merge=True,only_output_writable=str(ROOT),report_numbers_facts_only=True)
    c['checkpoints']=dict(candidates_include_network_optimizer=True,candidate_replay=False,resume_replay=True,resume_delete_old_only_after_new_readback=True)
    for p in (ROOT/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True,cfile=str(ROOT/'selftest'/('compile_'+p.name+'c')))
    put(ROOT/'contract.json',c);textfile(ROOT/'contract.sha256',sha(ROOT/'contract.json')+'\n');write_diff()
    put(ROOT/'step0/result.json',dict(status='passed',time=now(),input_count=len(c['protected_input_sha256']),references=refs,estimated_times=elapsed,resources=c['resources'],scene_stream_isolation_passed=True,git=c['git_before'],gpu_other_compute=0))
    put(ROOT/'start.json',dict(status='preparing',created_at=now(),OUT=str(ROOT),fresh_output=True,seed_order=[4104,4103]));put(ROOT/'progress.json',dict(status='preparing',stage='peripheral_selftests_and_smoke',time=now()))
    return c

if __name__=='__main__':
    try:prepare();print('STEP0_PASS',flush=True)
    except BaseException as e:
        record=failure('step0',e);put(ROOT/'step0/failure.json',record);raise
