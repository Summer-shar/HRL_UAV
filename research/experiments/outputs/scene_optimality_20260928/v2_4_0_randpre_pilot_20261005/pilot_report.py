# Reporting derived from NEW/report_diag.py/logs_diag.py; new P1-P6 selection and criteria only.
from pilot_common import *
from scipy.stats import wilcoxon,ttest_rel,binomtest

def comparison(cand,ref):
 a=np.array([cand['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)]);b=np.array([ref['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)]);d=a-b;wins=int(np.count_nonzero(d<0));losses=int(np.count_nonzero(d>0));n=wins+losses
 return dict(candidate_mean_s=float(a.mean()),reference_mean_s=float(b.mean()),relative_difference=float(a.mean()/b.mean()-1),wins=wins,losses=losses,ties=20-n,wilcoxon_two_sided_p=float(wilcoxon(a,b,zero_method='wilcox',alternative='two-sided').pvalue) if n else 1.,paired_t_two_sided_p=float(ttest_rel(a,b).pvalue) if n else 1.,sign_two_sided_p=float(binomtest(wins,n,.5,alternative='two-sided').pvalue) if n else 1.)

def curve(values,labels):return dict(labels=labels,values=values,tail=tail_label(values),max_adjacent_relative_fluctuation=max((abs(values[i]/values[i-1]-1) for i in range(1,len(values))),default=None),max_adjacent_absolute_fluctuation_s=max((abs(values[i]-values[i-1]) for i in range(1,len(values))),default=None))

def training_bins(run,width):
 rr=rows(Path(run)/'train_metrics.jsonl');return [dict(first=ss[0]['global_episode'],last=ss[-1]['global_episode'],mean_s=avg([r['truncated_delay_s'] for r in ss]),count=len(ss)) for i in range(0,len(rr),width) for ss in [rr[i:i+width]]]

def render_table(header,rows_):
 return ['|'+'|'.join(header)+'|','|'+'|'.join(['---']*len(header))+'|',*['|'+'|'.join(str(v) for v in r)+'|' for r in rows_]]

def report_seed(seed,data):
 root=PILOT/f'seed_{seed}';groups=data['groups'];F=groups[data['F']['label']];tf=groups[data['TFOO_dev']['label']];e5=groups['E5'];tf200=groups['TFOO-200'];uv=groups['D-b_U200_cvx'];criterion1=uv['mean_s']<e5['mean_s'];gap=(e5['mean_s']-uv['mean_s'])/(e5['mean_s']-tf200['mean_s']);criterion2=F['mean_s']/tf['mean_s']-1<=-.05 and comparison(F,tf)['wins']>=11
 cs=read(NEW/'summaries/selection.json')[str(seed)]['Cstar'];oldbest=hist(seed,'E2' if cs['cycle']==0 else f'S3-c{cs["cycle"]}');refs={'TFOO-dev':tf,'TFOO-200':tf200,'E5':e5,'v1-dev-best':oldbest,'v1-c10':hist(seed,'S3-c10')};comps={k:comparison(F,v) for k,v in refs.items()}
 facts=read(PILOT/'step0_facts/facts.json')['seeds'][str(seed)];original=facts['curves']['stage1']['bins'];new=training_bins(data['runs']['step1'],25)
 c1=[groups[f'H_c{i}']['mean_s'] for i in range(1,6)];c2=[groups[f'F_c{i}']['mean_s'] for i in range(6,11)];kk=[groups[f'K{i}']['mean_s'] for i in range(1,6)];initial=groups['H_S300r']['mean_s'];oldc=[hist(seed,f'S3-c{i}')['mean_s'] for i in range(1,11)]
 curves=dict(new_stage1=curve([r['mean_s'] for r in new],[r['last'] for r in new]),original_stage1=curve([r['truncated_delay_s'] for r in original],[r['last'] for r in original]),D_a=curve([groups[f'D-a_TFOO_S{n}r']['mean_s'] for n in (100,200,300)],['S100r','S200r','S300r']),U_cvx=curve([groups[f'D-b_U{n}_cvx']['mean_s'] for n in range(25,201,25)],[f'U{n}' for n in range(25,201,25)]),TFOO_cvx=curve([groups[f'TFOO-{n}']['mean_s'] for n in sorted(data['TFOO_checkpoints_evaluated'])],sorted(data['TFOO_checkpoints_evaluated'])),c1_c5=curve(c1,[f'c{i}' for i in range(1,6)]),c6_c10=curve(c2,[f'c{i}' for i in range(6,11)]),v1_c1_c10=curve(oldc,[f'v1-c{i}' for i in range(1,11)]),K1_K5=curve(kk,[f'K{i}' for i in range(1,6)]))
 replay=dict(K_best=min(kk),K_mean=avg(kk),c_best=min(c1),c_mean=avg(c1),initial_U100_S300r=initial,K1_relative_change=kk[0]/initial-1,c1_relative_change=c1[0]/initial-1)
 record=dict(seed=seed,contract_sha256=sha(PILOT/'pilot_contract.json'),H=data['H'],F=data['F'],TFOO_dev=data['TFOO_dev'],criterion1=dict(passed=criterion1,U200_cvx_mean_s=uv['mean_s'],E5_mean_s=e5['mean_s'],gap_fraction=gap),criterion2=dict(passed=criterion2,relative_difference=F['mean_s']/tf['mean_s']-1,wins=comps['TFOO-dev']['wins']),comparisons=comps,curves=curves,replay_3K=replay,timing=data['timing'],hard_checks={k:v for k,v in data['hard_checks'].items() if k!='checks'},raw_hard_check_count=len(data['hard_checks'].get('checks',{})),RP=data['RP'],groups=groups,runs=data['runs'],v1_dev_best=cs,TFOO_skipped=data['TFOO_skipped'],all_original_first_stage_bins=original,new_first_stage_bins=new,diagnostics_only=['D-a','D-b','K1-K5'],implementation_corrections=read(PILOT/'implementation_corrections.json'))
 put(root/'report.json',record)
 lines=[f'# seed {seed} 试点记录','',f'合同SHA256：`{record["contract_sha256"]}`。仅dev场景；test评估次数=0。','',f'H={data["H"]["label"]}；F={data["F"]["label"]}；TFOO-dev={data["TFOO_dev"]["label"]}。','', '## 判据','']
 lines+=render_table(['判据','通过','数值'],[['1',criterion1,f'U200cvx={uv["mean_s"]:.12f}; E5={e5["mean_s"]:.12f}; gap fraction={gap:.12f}'],['2',criterion2,f'F/TFOOdev-1={record["criterion2"]["relative_difference"]:.12f}; wins={record["criterion2"]["wins"]}/20']])
 lines+=['','## 全部dev均值','']+render_table(['组合','20场景均值(s)','high unfinished/window','low unfinished/window','cost/s'],[[k,f'{r["mean_s"]:.12f}',r['unfinished_high_per_window'],r['unfinished_low_per_window'],f'{r["cost_per_s"]:.12f}'] for k,r in groups.items()])
 lines+=['','## F配对比较','']+render_table(['参考','cand/ref-1','胜场','Wilcoxon双侧p','配对t双侧p','符号双侧p'],[[k,f'{c["relative_difference"]:.12f}',c['wins'],f'{c["wilcoxon_two_sided_p"]:.12g}',f'{c["paired_t_two_sided_p"]:.12g}',f'{c["sign_two_sided_p"]:.12g}'] for k,c in comps.items()])
 lines+=['','## 曲线与末段标签','']
 for name,c in curves.items():
  lines+=['',f'### {name}','']+render_table(['节点','truncated_delay_s'],[[k,f'{v:.12f}'] for k,v in zip(c['labels'],c['values'])]);lines+=['',f'末段改进={c["tail"]["improvement"]:.12f}；{c["tail"]["label"]}。']
  if name.startswith('c') or name.startswith('v1_c'):lines+=[f'相邻块最大相对波动={c["max_adjacent_relative_fluctuation"]:.12f}；最大绝对波动={c["max_adjacent_absolute_fluctuation_s"]:.12f} s。']
 lines+=['','## 经验池/计数/耗时','']+render_table(['步骤','可用完整轮数min/max','flush增量','skipped触发','wall_seconds'],[[k,f'{min(v.get("available_completed",[0]))}/{max(v.get("available_completed",[0]))}' if 'available_completed' in v else '不使用Replay',v.get('replay_flushes_delta','不使用Replay'),v.get('SAC_delta',{}).get('skipped',0),f'{v["wall_seconds"]:.6f}'] for k,v in data['training_results'].items()])
 lines+=['','3K对照：','', '```json',json.dumps(replay,ensure_ascii=False,indent=2),'```','','## 随机动作和位置覆盖（25轮汇总）','']
 for name,run in data['runs'].items():
  p=Path(run)/'position_25_bins.json'
  if p.exists():lines+=['',f'### {name}','', '```json',json.dumps(read(p),ensure_ascii=False,indent=2),'```']
 lines+=['','## 全部逐场景时延(s)','', '场景顺序为01–20；每个场景均为两个确定性stream的PooledMetrics结果。','']
 for k,r in groups.items():lines+=[f'- {k}: '+', '.join(f'{r["per_scenario"][str(i)]["truncated_delay_s"]:.12f}' for i in range(1,21))]
 lines+=['','## RP和硬检查','', '```json',json.dumps(dict(RP=data['RP'],hard_checks={k:v for k,v in data['hard_checks'].items() if k!='checks'},raw_hard_check_count=len(data['hard_checks'].get('checks',{}))),ensure_ascii=False,indent=2),'```','','## 每步墙钟(s)','', '```json',json.dumps(data['timing'],ensure_ascii=False,indent=2),'```','','## 新脚本实现更正','', '```json',json.dumps(record['implementation_corrections'],ensure_ascii=False,indent=2),'```']
 textfile(root/'report.md','\n'.join(lines)+'\n');plot(seed,curves);return record

def plot(seed,curves):
 os.environ['MPLCONFIGDIR']=str(PILOT/'mplconfig')
 try:
  import matplotlib
  matplotlib.use('Agg');import matplotlib.pyplot as plt
 except ImportError as e:put(PILOT/f'seed_{seed}/plot_status.json',dict(status='skipped',error=str(e)));return
 fig,ax=plt.subplots(2,2,figsize=(12,9))
 for name in ('new_stage1','original_stage1'):
  c=curves[name];ax[0,0].plot(c['labels'],c['values'],'o-',label=name)
 for name in ('U_cvx','TFOO_cvx'):
  c=curves[name];x=[int(str(v).lstrip('U')) for v in c['labels']];ax[0,1].plot(x,c['values'],'o-',label=name)
 for name in ('c1_c5','c6_c10','v1_c1_c10','K1_K5'):
  c=curves[name];x=[int(''.join(ch for ch in str(v).split('-')[-1] if ch.isdigit())) for v in c['labels']];ax[1,0].plot(x,c['values'],'o-',label=name)
 c=curves['D_a'];ax[1,1].plot([100,200,300],c['values'],'o-',label='D-a TFOO+Sr')
 for a in ax.flat:a.set_ylabel('truncated delay / s');a.grid(alpha=.2);a.legend()
 fig.suptitle(f'seed {seed} dev/training diagnostics');fig.tight_layout();fig.savefig(output_path(PILOT/f'seed_{seed}/curves.png'),dpi=150);plt.close(fig);put(PILOT/f'seed_{seed}/plot_status.json',dict(status='passed'))

def final_summary(records):
 rr=records;merged={}
 for role in ('F','TFOO_dev'):
  per={str(i):dict(truncated_delay_s=avg([r['groups'][r[role]['label']]['per_scenario'][str(i)]['truncated_delay_s'] for r in rr])) for i in range(1,21)};merged[role]=dict(per_scenario=per)
 cmp=comparison(merged['F'],merged['TFOO_dev']);out=dict(seeds=[r['seed'] for r in rr],merged_scenario_mean_comparison=cmp,all_criterion2_passed=all(r['criterion2']['passed'] for r in rr),contract_sha256=sha(PILOT/'pilot_contract.json'),git_after=git_states(),frozen_protocol_sha256_after=sha(PROTO),records=[str(PILOT/f'seed_{r["seed"]}/report.json') for r in rr]);check(sha(PROTO)==PROTO_SHA,'final_protocol_unchanged');put(PILOT/'summary.json',out)
 lines=['# 随机位置预训练 + 粗交替试点汇总','',f'合同SHA256：`{out["contract_sha256"]}`。','', 'seed 4104全部完成后才执行4105；3K分支已执行；test评估次数=0。','']
 lines+=render_table(['seed','H','F','TFOO-dev','判据1','gap fraction','判据2','F/TFOOdev-1','胜场'],[[r['seed'],r['H']['label'],r['F']['label'],r['TFOO_dev']['label'],r['criterion1']['passed'],f'{r["criterion1"]["gap_fraction"]:.12f}',r['criterion2']['passed'],f'{r["criterion2"]["relative_difference"]:.12f}',r['criterion2']['wins']] for r in rr])
 lines+=['','## 曲线末段','']+render_table(['曲线',str(rr[0]['seed']),str(rr[1]['seed'])],[[name,*[f'{r["curves"][name]["tail"]["improvement"]:.12f}; {r["curves"][name]["tail"]["label"]}' for r in rr]] for name in rr[0]['curves']])
 lines+=['','## 3K经验池对照','']+render_table(['seed','K最好','c1–c5最好','K均值','c1–c5均值','K1/start-1','c1/start-1'],[[r['seed'],*[f'{r["replay_3K"][k]:.12f}' for k in ('K_best','c_best','K_mean','c_mean','K1_relative_change','c1_relative_change')]] for r in rr])
 lines+=['','## 两seed按场景先均值后配对，n=20','', '```json',json.dumps(cmp,ensure_ascii=False,indent=2),'```','','## 每步墙钟(s)','']
 for r in rr:lines+=[f'seed {r["seed"]}:','', '```json',json.dumps(r['timing'],ensure_ascii=False,indent=2),'```','']
 lines+=['## RP和全部硬检查','']
 for r in rr:lines+=[f'### seed {r["seed"]}','', '```json',json.dumps(dict(RP=r['RP'],hard_checks={k:v for k,v in r['hard_checks'].items() if k!='checks'},raw_hard_check_count=r.get('raw_hard_check_count',0)),ensure_ascii=False,indent=2),'```','']
 lines+=['## 最终Git与协议','', '```json',json.dumps(out['git_after'],ensure_ascii=False,indent=2),'```','',f'冻结协议SHA256：{out["frozen_protocol_sha256_after"]}。','',f'两个seed判据2都通过={out["all_criterion2_passed"]}；未自动扩展。','', '## 脚本实现更正','', '```json',json.dumps(read(PILOT/'implementation_corrections.json'),ensure_ascii=False,indent=2),'```']
 textfile(PILOT/'summary.md','\n'.join(lines)+'\n');return out