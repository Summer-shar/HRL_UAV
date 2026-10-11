from d_guard import install,ROOT
install()
from j_common import *
import zipfile,hashlib
from scipy import stats
from d_eval import group
from d_setup import input_checks


def digest_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def archive_group(summary):
    if not summary.get('window_files'):return dict(raw_windows_written=False,compression_needed=False)
    folder=Path(summary['window_files'][0]['path']).parent.parent
    result=folder/'archive.json'
    if result.exists():
        data=read(result);equal(sha(data['zip_path']),data['zip_sha256'],'archive_reuse_SHA');return data
    begin=time.perf_counter();files=summary['window_files'];raw=sum(Path(w['path']).stat().st_size for w in files)
    put(folder/'window_sha256_manifest.json',files)
    target=folder/'windows.zip'
    if target.exists():
        # Preserve unfinished archive from an interruption; create a fresh complete one.
        target.rename(folder/('interrupted_windows_'+str(time.time_ns())+'.zip'))
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for w in files:
            equal(sha(w['path']),w['sha256'],'window_SHA_before_compression');z.write(w['path'],Path(w['path']).name)
    with zipfile.ZipFile(target) as z:
        for w in files:
            with z.open(Path(w['path']).name) as f:equal(digest_stream(f),w['sha256'],'window_decompressed_SHA')
    for w in files:output_path(w['path']).unlink()
    data=dict(zip_path=str(target),zip_sha256=sha(target),raw_bytes=raw,compressed_bytes=target.stat().st_size,compression_ratio=raw/target.stat().st_size,file_count=len(files),decompressed_SHA_all_passed=True,wall_seconds=time.perf_counter()-begin,time=now())
    put(result,data);return data


def run_group(g):
    input_checks();directory=ROOT/'groups'/g['label'];p=directory/'summary.json'
    if p.exists():
        result=read(p);equal(result['group'],g,'completed_group_same_specification')
        archive_group(result);return result
    if directory.exists():
        destination=ROOT/'recovery'/('partial_'+g['label']+'_'+str(time.time_ns()));directory.rename(destination)
        put(destination/'interruption_preserved.json',dict(time=now(),reason='incomplete group preserved; fresh frozen evaluation of same inputs'))
    put(ROOT/f"seed_{g['seed']}/evaluation_progress.json",dict(status='running',stage='evaluation',seed=g['seed'],group=g['label'],completed_episode=400,time=now()))
    result=group(g);archive_group(result);input_checks();return result


def comparison(a,b):
    x=np.array([a['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)],dtype=float);y=np.array([b['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)],dtype=float)
    wins=int((x<y).sum());losses=int((x>y).sum());improvement=float(1-x.mean()/y.mean())
    w=stats.wilcoxon(x,y,alternative='two-sided') if np.any(x!=y) else None
    t=stats.ttest_rel(x,y,alternative='two-sided');sg=stats.binomtest(wins,wins+losses,.5,alternative='two-sided') if wins+losses else None
    clean=lambda v:float(v) if np.isfinite(v) else None
    return dict(improvement_fraction=improvement,improvement_percent=100*improvement,wins=wins,losses=losses,ties=20-wins-losses,criterion_delay=improvement>=.05,criterion_wins=wins>=11,passed=improvement>=.05 and wins>=11,Wilcoxon_two_sided=dict(statistic=clean(w.statistic),pvalue=clean(w.pvalue)) if w else dict(statistic=0.,pvalue=1.),paired_t=dict(statistic=clean(t.statistic),pvalue=clean(t.pvalue)),sign_two_sided=dict(pvalue=clean(sg.pvalue)) if sg else dict(pvalue=1.))


def eval_seed(seed,training):
    tf=str(old_tfoo(seed)/'checkpoints/episode-0200-anchor.pt');baseg=dict(label=f'TFOO-{seed}',seed=seed,upper=tf,lower=None,upper_kind='TFOO',route='cvx',anchors=[str(NEW/f'summaries/seed_{seed}/TFOO-200.json')],diagnostic_only=True)
    baseline=run_group(baseg);results=[]
    for n,path in zip(range(220,401,20),training['checkpoints']):
        results.append(run_group(dict(label=f'pair-{seed}-{n}',seed=seed,upper=path,lower=path,upper_kind=None,route=None,anchors=[],diagnostic_only=False,episode=n)))
    selected=min(results,key=lambda r:(r['mean_s'],r['group']['episode']));last=results[-1];diags=[]
    cvx_needed={300,400,selected['group']['episode']}
    for n,path in zip(range(220,401,20),training['checkpoints']):
        if n not in cvx_needed:continue
        diags.append(run_group(dict(label=f'cvx-{seed}-{n}',seed=seed,upper=path,lower=None,upper_kind=None,route='cvx',anchors=[],diagnostic_only=True,episode=n)))
    for r in (selected,):
        n=r['group']['episode']
        if any(d['group']['label']==f'TFOO-SAC-{seed}-{n}' for d in diags):continue
        diags.append(run_group(dict(label=f'TFOO-SAC-{seed}-{n}',seed=seed,upper=tf,lower=r['group']['lower'],upper_kind='TFOO',route=None,anchors=[],diagnostic_only=True,episode=n)))
    allgroups=[baseline]+results+diags
    reference=contract()['references_D'][str(seed)];jc=read(reference['JC_selected']['source'])
    for r in allgroups:r['comparison_TFOO']=comparison(r,baseline);r['comparison_JC_selected']=comparison(r,jc)
    def entry(r):return dict(episode=r['group'].get('episode'),mean_s=r['mean_s'],checkpoint=r['group'].get('lower'),checkpoint_sha256=sha(r['group']['lower']) if r['group'].get('lower') else None,**r['comparison_TFOO'])
    metrics=rows(ROOT/f'seed_{seed}/train/train_metrics.jsonl');extras=rows(ROOT/f'seed_{seed}/train/extra_metrics.jsonl');ppo=[r for r in rows(ROOT/f'seed_{seed}/train/learning.jsonl') if r['kind']=='ppo']
    ppo_summary=[dict(batch=(r['global_episode']-200)//5,global_episode=r['global_episode'],accepted_epochs=sum(bool(e['accepted']) for e in r['ppo']),attempted_epochs=len(r['ppo']),KL_per_attempt=[dict(epoch=e['epoch'],KL=e['kl'],accepted=e['accepted']) for e in r['ppo']]) for r in ppo]
    logs=training['batch_audits'];under=[dict(global_episode=r['global_episode'],batch=(r['global_episode']-201)//5+1 if r['global_episode']<=400 and r['global_episode']>200 else r.get('batch'),behavior_source=r['behavior_source'],reward=r['reward_physical']) for r in metrics+extras if r['reward_physical'] < -300]
    report=dict(status='complete',seed=seed,ended_at=now(),contract_sha256=sha(ROOT/'contract.json'),selected=entry(selected),last=entry(last),passed=selected['comparison_TFOO']['passed'],selection_rule=contract()['selection'],criteria=contract()['criteria'],groups=allgroups,JC_comparison=reference,training=training,PPO_updates=ppo,PPO_update_summary=ppo_summary,batch_returns_and_replay_mix=logs,returns_below_minus300=dict(count=len(under),episodes=under),test_evaluation=False,diagnostic_selection_participation=False)
    directory=ROOT/f'seed_{seed}';put(directory/'report.json',report);write_report(report);return report


def write_report(r,output_root=None):
    report_root=ROOT if output_root is None else output_path(output_root)
    seed=r['seed'];lines=[f'# 方案 D seed {seed} 数值报告','',f"结束：{r['ended_at']}",f"selected：{r['selected']}",f"last：{r['last']}",'','|组|平均延迟 s|相对 TFOO 改善 %|胜场 /20|Wilcoxon p|配对 t p|符号 p|','|---|---:|---:|---:|---:|---:|---:|']
    for g in r['groups']:
        c=g['comparison_TFOO'];lines.append(f"|{g['group']['label']}|{g['mean_s']:.12f}|{c['improvement_percent']:.6f}|{c['wins']}|{c['Wilcoxon_two_sided']['pvalue']}|{c['paired_t']['pvalue']}|{c['sign_two_sided']['pvalue']}|")
    lines.extend(['','JC 参考值：',json.dumps(r['JC_comparison'],ensure_ascii=False,indent=2),'','PPO 更新 KL 与接受 epoch 数：',json.dumps(r['PPO_update_summary'],ensure_ascii=False,indent=2),'','每批正常/附加回报和经验占比：',json.dumps(r['batch_returns_and_replay_mix'],ensure_ascii=False,indent=2),'','训练硬检查：',json.dumps({k:r['training'][k] for k in ['SAC_counts','base_SAC_counts','joint_SAC_delta','PPO_counts','replay_flushes','hardchecks_passed']},indent=2),'',f"回报低于 −300：{r['returns_below_minus300']}",'','逐场景、分段驻留、未完成任务、cost、距离、队列、每批回报及 PPO 原日志详见 report.json。'])
    for g in r['groups']:
        lines.extend(['',f"## {g['group']['label']}",'',json.dumps(dict(per_scenario=g['per_scenario'],pooled=g['pooled'],observations=g['observations'],streams=g['stream_consistency']),ensure_ascii=False,indent=2)])
    textfile(report_root/f'seed_{seed}/report.md','\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    figures=report_root/'figures';figures.mkdir(exist_ok=True)
    pair=[g for g in r['groups'] if g['group']['label'].startswith('pair-')];cvx=[g for g in r['groups'] if g['group']['label'].startswith('cvx-')]
    fig,ax=plt.subplots(figsize=(8,5));ax.plot([g['group']['episode'] for g in pair],[g['mean_s'] for g in pair],'-o',label='D pair');ax.plot([g['group']['episode'] for g in cvx],[g['mean_s'] for g in cvx],'-o',label='D upper + cvx');curve=r['JC_comparison']['JC_upper_cvx_curve'];ax.plot([g['episode'] for g in curve],[g['mean_s'] for g in curve],'--',label='JC upper + cvx');ax.axhline(r['JC_comparison']['TFOO']['mean_s'],label='TFOO-200',color='black');ax.set(xlabel='Episode',ylabel='Mean truncated delay (s)');ax.legend();fig.tight_layout();fig.savefig(figures/f'seed_{seed}_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,7));b=r['batch_returns_and_replay_mix'];x=list(range(1,len(b)+1))
    axes[0].plot(x,[np.mean(a['normal_returns']) for a in b],label='5 PPO episodes');axes[0].plot(x,[np.mean(a['extra_returns']) for a in b],label='2 SAC-only episodes');axes[0].set_ylabel('Physical return');axes[0].legend()
    for behavior in ('hover','random_upper','ppo'):axes[1].plot(x,[a['replay_mix']['transition_fraction'][behavior] for a in b],label=behavior)
    axes[1].set(xlabel='Joint batch',ylabel='Replay transition fraction');axes[1].legend();fig.tight_layout();fig.savefig(figures/f'seed_{seed}_returns_replay.png',dpi=160);plt.close(fig)


def finish(reports,started):
    checks=input_checks();ended=now();summary=dict(status='complete',completed_at=ended,wall_seconds=time.time()-started,seed_order=[4104,4103],reports=[str(ROOT/f"seed_{r['seed']}/report.json") for r in reports],selected=[dict(seed=r['seed'],**r['selected']) for r in reports],last=[dict(seed=r['seed'],**r['last']) for r in reports],conditional_expansion=False,readonly_checks=checks,test_evaluation=False,implementation_corrections=read(ROOT/'implementation_corrections.json') if (ROOT/'implementation_corrections.json').exists() else [],interruptions=[read(p) for p in ROOT.glob('seed_*/train/interruptions/*/resume.json')],deviations=[dict(kind='reference_file_archival',decision_path_changed=False),dict(kind='user_authorized_lean_runtime',decision_path_changed=False,details=read(ROOT/'contract.json')['lean_runtime'])],preflight=read(ROOT/'preflight.json'))
    put(ROOT/'summary.json',summary);textfile(ROOT/'summary.md','# 方案 D 试点\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    models=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.glob('seed_*/train/**/*.pt')]
    c=contract();record={'schema':'experiment_record_step1_v1','记录时间_北京时间':ended,'身份':{'原实验路径':str(ROOT),'用途':'方案 D 试点','seed':[4104,4103],'训练集':'train','验证集':'dev gate20 流1/2','测试集':'未访问'},'代码':{'git状态与HEAD':c['git_before'],'冻结源身份':c['identity'],'脚本差异':str(ROOT/'scripts_vs_JC.diff')},'配置':{'配置文件全文':c['config'],'关键参数':dict(alpha=c['alpha_schedule'],actor_frozen_ordinals=c['actor_frozen_ordinals'],replay_capacity=55)},'运行情况':dict(开始时间=read(ROOT/'execution_start.json')['time'],结束时间=ended,耗时=summary['wall_seconds'],状态='complete',硬检查=checks),'结果':summary,'模型清单':models,'原样复制文件':read(ROOT/'script_provenance.json'),'信息读取问题':[],'未知字段':[],'清理边界':{'唯一写入':str(ROOT),'已有输出修改':False}}
    put(ROOT/'record.json',record);textfile(ROOT/'record.md','# 实验记录\n\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n');put(ROOT/'progress.json',dict(status='complete',stage='all_reports_and_readonly_checks_complete',time=ended,completed_seeds=[4104,4103]));return summary
