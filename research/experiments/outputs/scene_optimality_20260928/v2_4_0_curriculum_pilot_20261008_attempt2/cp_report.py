# Derived from PILOT/pilot_report.py; curriculum 4104 labels, sources and no-reset diagnostics.
# Only CPIL outputs are writable; historical PILOT modules are never imported.
from cp_common import *
from scipy.stats import wilcoxon, ttest_rel, binomtest

REPORT_ROOT=Path(__file__).resolve().parent
PILOT_REFERENCE=MAIN/'outputs/scene_optimality_20260928/v2_4_0_randpre_pilot_20261005'

def comparison(cand,ref):
 a=np.array([cand['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)])
 b=np.array([ref['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)])
 d=a-b;wins=int(np.count_nonzero(d<0));losses=int(np.count_nonzero(d>0));n=wins+losses
 return dict(candidate_mean_s=float(a.mean()),reference_mean_s=float(b.mean()),relative_difference=float(a.mean()/b.mean()-1),wins=wins,losses=losses,ties=20-n,wilcoxon_two_sided_p=float(wilcoxon(a,b,zero_method='wilcox',alternative='two-sided').pvalue) if n else 1.,paired_t_two_sided_p=float(ttest_rel(a,b).pvalue) if n else 1.,sign_two_sided_p=float(binomtest(wins,n,.5,alternative='two-sided').pvalue) if n else 1.)

def table(head,rs):
 return ['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|',*['|'+'|'.join(str(v).replace('|',' / ') for v in row)+'|' for row in rs]]
def number(x,ndigits=12):
 return '缺失' if x is None else (str(x) if isinstance(x,(str,bool)) else f'{x:.{ndigits}f}')
def json_block(obj):
 fence=chr(96)*3
 return [fence+'json',json.dumps(obj,ensure_ascii=False,indent=2),fence]
def source_label(path,key=None):
 return str(path)+(('#'+key) if key else '')
def training_bins(run,width=25,first=None,last=None):
 path=Path(run)/'train_metrics.jsonl'
 rr=rows(path)
 rr=[r for r in rr if (first is None or r['global_episode']>=first) and (last is None or r['global_episode']<=last)]
 out=[]
 for j in range(0,len(rr),width):
  ss=rr[j:j+width]
  out.append(dict(first=ss[0]['global_episode'],last=ss[-1]['global_episode'],mean_s=avg([r['truncated_delay_s'] for r in ss]),count=len(ss),source=str(path),field='truncated_delay_s'))
 return out
def simple_curve(bins):
 vals=[r['mean_s'] for r in bins]
 return dict(bins=bins,tail=tail_label(vals) if len(vals)>=2 else None)
def corrections():
 p=REPORT_ROOT/'implementation_corrections.json'
 return read(p) if p.exists() else []
def group_source(g,fallback):
 files=g.get('window_files',[])
 return str(Path(files[0]['path']).parent.parent/'summary.json') if files else str(fallback)
def hardcheck_items(obj,prefix=''):
 result=[]
 for key,value in obj.items():
  path=prefix+str(key)
  if isinstance(value,dict) and 'status' in value:
   result.append(dict(name=path,status=value['status'],details=value.get('details')))
  elif isinstance(value,dict):
   result.extend(hardcheck_items(value,path+'.'))
 return result

def report_seed(seed,data):
 if seed!=4104:raise RuntimeError('Only seed 4104 is authorized.')
 root=REPORT_ROOT/f'seed_{seed}'
 source=root/'pipeline_data.json'
 groups=data['groups'];tf=groups['TFOO-200'];e5=groups['E5']
 F=groups[data['F']['label']];H=groups[data['H']['label']]
 uv=groups['D-b_U200_cvx']
 old_source=PILOT_REFERENCE/f'seed_{seed}/pipeline_data.json';old=read(old_source);og=old['groups']
 contract=read(REPORT_ROOT/'cp_contract.json')
 c1=bool(uv['mean_s']<e5['mean_s'])
 gap=(e5['mean_s']-uv['mean_s'])/(e5['mean_s']-tf['mean_s'])
 cf=comparison(F,tf);c2=bool(F['mean_s']/tf['mean_s']-1<=-.05 and cf['wins']>=11)
 crit1=dict(passed=c1,U200_cvx_mean_s=uv['mean_s'],E5_mean_s=e5['mean_s'],TFOO200_mean_s=tf['mean_s'],gap_fraction=gap,formula='(E5-A-U200cvx)/(E5-TFOO200)',pilot_U200_cvx_mean_s=og['D-b_U200_cvx']['mean_s'],relative_to_pilot_U200_cvx=uv['mean_s']/og['D-b_U200_cvx']['mean_s']-1)
 crit2=dict(passed=c2,relative_difference=F['mean_s']/tf['mean_s']-1,wins=cf['wins'],required_relative_difference=-.05,required_wins=11)
 pairings=[('TFOO-200','TFOO-200','TFOO-200'),('E5','E5','E5'),('A-F',data['F']['label'],old['F']['label']),('A-H',data['H']['label'],old['H']['label']),('A-U200+cvx','D-b_U200_cvx','D-b_U200_cvx'),('A-U100+cvx','D-b_U100_cvx','D-b_U100_cvx'),('DG-SAC300','DG-SAC300',None),('DG-v1c8','DG-v1c8',None),('TFOO-200+A-S200h','D-a_TFOO_A-S200h','D-a_TFOO_S200r'),('TFOO-200+A-S300c','D-a_TFOO_A-S300c','D-a_TFOO_S300r')]
 main=[]
 for label,key,oldkey in pairings:
  g=groups[key];cmp=comparison(g,tf)
  oldcmp=comparison(og[oldkey],og['TFOO-200']) if oldkey else None
  main.append(dict(label=label,group_key=key,mean_s=g['mean_s'],comparison=cmp,source=group_source(g,source),pilot_group=oldkey,pilot_comparison=oldcmp,pilot_source=source_label(old_source,'groups.'+oldkey) if oldkey else None,pilot_note='旧S200r为随机200轮，不等同本次悬停200轮' if key=='D-a_TFOO_A-S200h' else None))
 new_bins=training_bins(data['runs']['step1'],25,1,300)
 original_bins=training_bins(old_run(seed),25,1,300)
 random_bins=training_bins(old['runs']['step1'],25,1,300)
 curves=dict(curriculum_stage1=simple_curve(new_bins),original_hover_stage1=simple_curve(original_bins),pilot_random_stage1=simple_curve(random_bins))
 curves['upper_cvx']=dict(labels=list(range(25,201,25)),values=[groups[f'D-b_U{n}_cvx']['mean_s'] for n in range(25,201,25)],source=str(source))
 block_rows=[]
 for step,start,prefix,initial_key in [('step3',1,'H_A-c','H_A-S300c'),('step5',6,'F_A-c','F_A-H')]:
  bins=training_bins(data['runs'][step],10)
  prev_train=None;prev_dev=groups[initial_key]['mean_s']
  for k,b in enumerate(bins,start):
   current=groups[f'{prefix}{k}']['mean_s']
   train_change=None if prev_train is None else b['mean_s']/prev_train-1
   dev_change=current/prev_dev-1
   block_rows.append(dict(block=f'A-c{k}',step=step,first=b['first'],last=b['last'],episodes=b['count'],training_mean_s=b['mean_s'],dev_mean_s=current,training_relative_to_previous_block=train_change,dev_relative_to_previous_candidate=dev_change,training_worsened_over50pct=None if train_change is None else train_change>.5,dev_worsened_over50pct=dev_change>.5,training_source=b['source'],dev_source=group_source(groups[f'{prefix}{k}'],source),previous_dev_group=initial_key if k==start else f'{prefix}{k-1}'))
   prev_train=b['mean_s'];prev_dev=current
  # Do not compare train bins across the intervening upper training stage.
 curves['lower_blocks']=block_rows
 selection=read(NEW/'summaries/selection.json')[str(seed)]['Cstar']
 reference_groups={'X1':hist(seed,'X1'),'v1-dev-selected':hist(seed,'E2' if selection['cycle']==0 else f'S3-c{selection["cycle"]}')}
 histrefs=[]
 for label,g in reference_groups.items():
  key='X1' if label=='X1' else ('E2' if selection['cycle']==0 else f'S3-c{selection["cycle"]}')
  histrefs.append(dict(label=label,mean_s=g['mean_s'],comparison=comparison(g,tf),source=str(NEW/f'summaries/seed_{seed}/{key}.json')))
 scenario_rows=[]
 for i in range(1,21):
  scenario_rows.append(dict(scenario=i,pilot_unstable_marker=i in (1,3,4,8,19),TFOO200=tf['per_scenario'][str(i)]['truncated_delay_s'],A_F=F['per_scenario'][str(i)]['truncated_delay_s'],A_H=H['per_scenario'][str(i)]['truncated_delay_s'],A_U200_cvx=uv['per_scenario'][str(i)]['truncated_delay_s']))
 training_results=data['training_results']
 count_rows=[]
 for step,result in training_results.items():
  available=result.get('available_completed')
  count_rows.append(dict(step=step,SAC_delta=result.get('SAC_delta'),replay_flushes_delta=result.get('replay_flushes_delta'),skipped_episodes=result.get('skipped_episodes'),available_completed_min=min(available) if available else None,available_completed_max=max(available) if available else None,PPO_batches_delta=result.get('PPO_batches_delta'),PPO_batches_total=result.get('PPO_batches_total'),PPO_complete_unchanged=result.get('PPO_complete_unchanged'),SAC_complete_unchanged=result.get('SAC_complete_unchanged'),wall_seconds=result.get('wall_seconds'),source=str(Path(data['runs'][step])/'result.json')))
 checks=hardcheck_items(data.get('hard_checks',{}))
 if not checks:
  checks=hardcheck_items(dict(CHECKS))
 all_checks_status=data.get('hard_checks',{}).get('status','未提供最终汇总状态')
 deviation=contract.get('deviations',contract.get('deviation_list','合同未提供偏离清单'))
 record=dict(status='completed',seed=seed,ended_at=data.get('finished_at',now()),contract_sha256=sha(REPORT_ROOT/'cp_contract.json'),criterion1=crit1,criterion2=crit2,both_criteria_passed=c1 and c2,decision_rule=contract.get('criteria'),main_table=main,H=data['H'],F=data['F'],TFOO_dev=data['TFOO_dev'],historical_reference_rows=histrefs,curves=curves,per_scenario_table=scenario_rows,counts_and_flushes=count_rows,RP=data['RP'],hard_checks=data.get('hard_checks',{}),hard_check_items=checks,hard_check_status=all_checks_status,implementation_corrections=corrections(),deviations=deviation,timing=data['timing'],stage_timing_sum_seconds=float(sum(data['timing'].values())),total_wall_seconds=data.get('total_wall_seconds'),groups=groups,runs=data['runs'],sources=dict(current_pipeline=str(source),pilot_pipeline=str(old_source),contract=str(REPORT_ROOT/'cp_contract.json'),step0=str(REPORT_ROOT/'step0_facts/facts.json'),training='Per-bin source paths above; bins are unweighted episode means',selection=str(NEW/'summaries/selection.json')))
 put(root/'report.json',record)
 lines=['# seed 4104 课程式预训练试点报告','',f'结束时间：{record["ended_at"]}。合同 SHA256：{record["contract_sha256"]}。','', '仅验证场景 1–20、stream 1/2、确定性策略；未执行 test 评估。每场景先按两个窗口汇总，再对20个场景等权取均值。统计检验的独立配对单位为20个场景，两个确定性stream不当作40个独立样本。','',f'A-H={data["H"]["label"]}；A-F={data["F"]["label"]}。总判定为两项判据同时通过：{c1 and c2}。','', '## 判据','']
 lines+=table(['判据','通过','实测数值'],[['1',c1,f'A-U200cvx={number(uv["mean_s"])}；E5={number(e5["mean_s"])}；差距补上比例={number(gap)}'],['2',c2,f'A-F/TFOO200-1={number(crit2["relative_difference"])}；wins={crit2["wins"]}/20；阈值<=-0.05且>=11']])
 lines+=['',f'来源：{source} groups；判据和检验设置：{REPORT_ROOT/"cp_contract.json"} criteria。','',f'差距补上比例=(E5-A-U200cvx)/(E5-TFOO200)。旧试点U200+cvx={number(crit1["pilot_U200_cvx_mean_s"])}，本次相对旧试点={number(crit1["relative_to_pilot_U200_cvx"])}。','', '## 主表','', '相对差均为 candidate/TFOO-200-1；负值较好。p值只报告，不参与模型选择或判据。']
 lines+=table(['组','验证均值s','相对TFOO差','胜场/20','Wilcoxon双侧p','配对t双侧p','符号双侧p','旧试点对应组','旧均值s','旧相对TFOO差','旧胜场'],[[r['label'],number(r['mean_s']),number(r['comparison']['relative_difference']),r['comparison']['wins'],number(r['comparison']['wilcoxon_two_sided_p']),number(r['comparison']['paired_t_two_sided_p']),number(r['comparison']['sign_two_sided_p']),r['pilot_group'] or '无对应组',number(r['pilot_comparison']['candidate_mean_s'] if r['pilot_comparison'] else None),number(r['pilot_comparison']['relative_difference'] if r['pilot_comparison'] else None),r['pilot_comparison']['wins'] if r['pilot_comparison'] else '未评估'] for r in main])
 lines+=['','旧S200r是随机上层200轮，作为历史对应节点并列，不能写成同一策略的重复。DG-*仅诊断，不参与选择。','']
 for r in main:lines+=[f'- {r["label"]}：本次 {r["source"]}；旧试点 {r["pilot_source"] or "无对应评估"}。']
 lines+=['','## 历史参考与D-a对照','']+table(['历史组','均值s','相对TFOO差','胜场','来源'],[[r['label'],number(r['mean_s']),number(r['comparison']['relative_difference']),r['comparison']['wins'],r['source']] for r in histrefs])
 da=[]
 for key in ('D-a_TFOO_A-S200h','D-a_TFOO_A-S300c'):
  for ref_label,ref_g in [('X1',reference_groups['X1']),('pilot_TFOO_S300r',og['D-a_TFOO_S300r'])]:
   cmp=comparison(groups[key],ref_g)
   da.append([key,ref_label,number(cmp['candidate_mean_s']),number(cmp['reference_mean_s']),number(cmp['relative_difference']),cmp['wins']])
 lines+=['']+table(['本次D-a','参考','本次均值s','参考均值s','相对差','胜场'],da)
 lines+=['',f'旧试点TFOO+S300r来源：{old_source} groups.D-a_TFOO_S300r。','', '## 第一阶段每25轮训练曲线','', '每个格为该25轮训练截断时延的算术均值。训练场景与验证场景不相同，训练曲线不用于选择A-H/A-F。']
 lines+=table(['训练轮','本次S1秒','原悬停秒','旧试点随机秒'],[[f'{a["first"]}–{a["last"]}',number(a['mean_s']),number(b['mean_s']),number(c['mean_s'])] for a,b,c in zip(new_bins,original_bins,random_bins)])
 lines+=['',f'本次来源：{Path(data["runs"]["step1"])/"train_metrics.jsonl"}；原悬停：{old_run(seed)/"train_metrics.jsonl"}；旧随机：{Path(old["runs"]["step1"])/"train_metrics.jsonl"}。','', '201–300轮改用随机上层，不清经验池；逐轮回放构成以step1记录为准。三条曲线及重点窗口见 curves.png。','', '## 下层块训练与验证','', '超过50%变差只做标记，不触发加训、换模型或额外选择。验证c1/c6各与其阶段起点相比；训练c1/c6没有上一块，不跨中间上层训练阶段强行配对。']
 lines+=table(['块','原global轮','训练均值s','验证均值s','训练相对前块','验证相对前候选','训练恶化>50%','验证恶化>50%'],[[r['block'],f'{r["first"]}–{r["last"]}',number(r['training_mean_s']),number(r['dev_mean_s']),number(r['training_relative_to_previous_block']),number(r['dev_relative_to_previous_candidate']),r['training_worsened_over50pct'],r['dev_worsened_over50pct']] for r in block_rows])
 for step in ('step3','step5'):lines+=[f'- {step}训练来源：{Path(data["runs"][step])/"train_metrics.jsonl"}；验证来源：{source} groups。']
 lines+=['','## 逐场景截断时延(s)','', '*为旧试点标出的场景1、3、4、8、19，仅保留标记。']
 lines+=table(['场景','TFOO-200','A-F','A-H','A-U200+cvx'],[[str(r['scenario'])+('*' if r['pilot_unstable_marker'] else ''),*[number(r[k]) for k in ('TFOO200','A_F','A_H','A_U200_cvx')]] for r in scenario_rows])
 lines+=['',f'来源：{source} 各组 per_scenario；不混用pooled总体时延。','', '## 全部验证组','']
 lines+=table(['组','均值s','高优未完成/窗口','低优未完成/窗口','cost/s'],[[k,number(g['mean_s']),number(g.get('unfinished_high_per_window')),number(g.get('unfinished_low_per_window')),number(g.get('cost_per_s'))] for k,g in groups.items()])

 lines+=['','## 硬检查、计数与经验池','',f'最终硬检查汇总：{all_checks_status}。原始核对明细保存在pipeline_data.json/hard_checks及各运行result.json。','']
 lines+=table(['步骤','SAC triggers/skipped/events/critic/target/actor','flush增量','skipped轮','可用完整轮min/max','PPO批次增量/累计','wall秒'],[[r['step'],' / '.join(str(r['SAC_delta'][k]) for k in ('triggers','skipped','events','critic_fits','target_updates','actor_updates')) if r['SAC_delta'] else '冻结SAC，无更新',r['replay_flushes_delta'] if r['replay_flushes_delta'] is not None else '不使用Replay',str(r['skipped_episodes']) if r['skipped_episodes'] is not None else '不使用Replay',f'{r["available_completed_min"]}/{r["available_completed_max"]}' if r['available_completed_min'] is not None else '不使用Replay',f'{r["PPO_batches_delta"]}/{r["PPO_batches_total"]}' if r['PPO_batches_delta'] is not None else '冻结PPO',number(r['wall_seconds'],6)] for r in count_rows])
 lines+=['','各计数直接来自训练result.json/episode JSON及checkpoint内的状态核对；不读取paused_G1 progress.json中的SAC计数。']
 for r in count_rows:lines+=[f'- {r["step"]}：{r["source"]}。']
 lines+=['','### 逐项硬检查','']+table(['检查','状态','详细证据'],[[r['name'],r['status'],str(r['details'])] for r in checks])
 lines+=['','### RP汇总','']+json_block(data['RP'])
 stage1_result=training_results.get('step1',{})
 extra1={k:v for k,v in stage1_result.items() if k not in ('position_bins','available_completed','checkpoints','wall_seconds','status','episodes','PPO_complete_unchanged','SAC_delta','replay_flushes_delta','skipped_episodes')}
 lines+=['','### S1分段、悬停复现和回放组成记录','']+json_block(extra1)
 lines+=['','### 实现修正记录','']+json_block(record['implementation_corrections'])
 lines+=['','## 相对冻结协议/旧试点的偏离','']+json_block(deviation)
 lines+=['','## 墙钟时间','']+table(['步骤','实测wall秒'],[[k,number(v,6)] for k,v in data['timing'].items()])
 lines+=['',f'阶段计时字段之和={number(record["stage_timing_sum_seconds"],6)} s；完整流水线wall={number(record["total_wall_seconds"],6)} s。两者口径不同，缺少完整wall字段时不把阶段之和冒充总墙钟。',f'来源：{source} timing/total_wall_seconds。']
 final_checks=REPORT_ROOT/'final_readonly_checks.json'
 lines+=['','## 只读复核','']+(json_block(read(final_checks)) if final_checks.exists() else ['最终复核文件尚未生成；不得据此报告输入未变。'])
 lines+=['','## 数据来源与范围','']+json_block(record['sources'])
 lines+=['','所有统计只使用已有JSON/JSONL，不重算物理轨迹；图像只渲染上述数值。检验不参与选择，报告不提出后续方案。']
 textfile(root/'report.md','\n'.join(lines)+'\n')
 plot(seed,curves)
 return record

def plot(seed,curves):
 os.environ['MPLCONFIGDIR']=str(REPORT_ROOT/'mplconfig')
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError as e:
  put(REPORT_ROOT/f'seed_{seed}/plot_status.json',dict(status='skipped',error=str(e)))
  return
 fig,ax=plt.subplots(2,2,figsize=(13,9))
 for name in ('curriculum_stage1','original_hover_stage1','pilot_random_stage1'):
  bins=curves[name]['bins'];xs=[r['last'] for r in bins];ys=[r['mean_s'] for r in bins]
  ax[0,0].plot(xs,ys,'o-',label=name)
  selected_bins=[r for r in bins if r['first']>=201]
  ax[0,1].plot([r['last'] for r in selected_bins],[r['mean_s'] for r in selected_bins],'o-',label=name)
 for step in ('step3','step5'):
  rr=[r for r in curves['lower_blocks'] if r['step']==step]
  x=[int(r['block'].replace('A-c','')) for r in rr]
  ax[1,0].plot(x,[r['training_mean_s'] for r in rr],'o-',label=step+'_training')
  ax[1,0].plot(x,[r['dev_mean_s'] for r in rr],'s--',label=step+'_dev')
 ax[1,1].plot(curves['upper_cvx']['labels'],curves['upper_cvx']['values'],'o-',label='A-U + cvx dev')
 for axis in ax.flat:
  axis.set_ylabel('truncated delay / s');axis.grid(alpha=.2);axis.legend(fontsize=8)
 ax[0,0].set_xlabel('S1 episode (25-episode mean)');ax[0,1].set_xlabel('S1 episodes 201-300')
 ax[1,0].set_xlabel('lower block');ax[1,1].set_xlabel('upper training episode')
 fig.suptitle(f'seed {seed} curriculum pilot (dev only)');fig.tight_layout()
 fig.savefig(output_path(REPORT_ROOT/f'seed_{seed}/curves.png'),dpi=150);plt.close(fig)
 put(REPORT_ROOT/f'seed_{seed}/plot_status.json',dict(status='passed',source=str(REPORT_ROOT/f'seed_{seed}/report.json')))

def final_summary(records):
 if len(records)!=1 or records[0]['seed']!=4104:raise RuntimeError('Summary is restricted to exactly seed 4104.')
 r=records[0]
 post=REPORT_ROOT/'final_readonly_checks.json';git=REPORT_ROOT/'git_after.json'
 out=dict(status=r['status'],seed=4104,ended_at=r['ended_at'],criterion1=r['criterion1'],criterion2=r['criterion2'],both_criteria_passed=r['both_criteria_passed'],H=r['H'],F=r['F'],main_table=r['main_table'],contract_sha256=r['contract_sha256'],hard_check_status=r['hard_check_status'],implementation_corrections=r['implementation_corrections'],total_wall_seconds=r['total_wall_seconds'],stage_timing_sum_seconds=r['stage_timing_sum_seconds'],final_readonly_checks=read(post) if post.exists() else None,git_after=read(git) if git.exists() else None,report=str(REPORT_ROOT/'seed_4104/report.md'),report_json=str(REPORT_ROOT/'seed_4104/report.json'))
 put(REPORT_ROOT/'summary.json',out)
 lines=['# seed 4104 课程式预训练试点汇总','',f'结束时间：{r["ended_at"]}；合同SHA256：{r["contract_sha256"]}。','',f'判据1通过={r["criterion1"]["passed"]}；判据2通过={r["criterion2"]["passed"]}；两项同时通过={r["both_criteria_passed"]}。','',f'硬检查={r["hard_check_status"]}；实现修正记录如下。','']
 lines+=table(['组','验证均值s','相对TFOO-200差','胜场/20','Wilcoxon双侧p','配对t双侧p','符号双侧p'],[[x['label'],number(x['mean_s']),number(x['comparison']['relative_difference']),x['comparison']['wins'],number(x['comparison']['wilcoxon_two_sided_p']),number(x['comparison']['paired_t_two_sided_p']),number(x['comparison']['sign_two_sided_p'])] for x in r['main_table']])
 lines+=['',f'各数字来源、逐场景表、三条训练曲线、检查与计数见：{out["report"]}；机器可读记录：{out["report_json"]}。','',f'总墙钟={number(r["total_wall_seconds"],6)} s；阶段计时之和={number(r["stage_timing_sum_seconds"],6)} s。','',f'A-H={r["H"]["label"]}，A-F={r["F"]["label"]}。诊断组不参与选模型；未自动扩展到4105或其他阶段。','', '## 实现修正','']+json_block(r['implementation_corrections'])
 lines+=['','## 最终只读和Git复核','']+json_block(dict(final_readonly_checks=out['final_readonly_checks'],git_after=out['git_after']))
 textfile(REPORT_ROOT/'summary.md','\n'.join(lines)+'\n')
 return out

def report_failure(failure_record,data=None):
 # No new evaluations; missing criteria/results stay explicitly unreported.
 p=REPORT_ROOT/'seed_4104/pipeline_data.json'
 partial=data if data is not None else (read(p) if p.exists() else {})
 completed_groups=partial.get('groups',{})
 record=dict(status='failed',seed=4104,time=now(),first_failure=failure_record,criterion1='未完成，不判定',criterion2='未完成，不判定',completed_group_means={k:v['mean_s'] for k,v in completed_groups.items()},runs=partial.get('runs',{}),timing=partial.get('timing',{}),implementation_corrections=corrections(),source=str(p) if p.exists() else '未生成pipeline_data.json')
 put(REPORT_ROOT/'seed_4104/report.json',record)
 lines=['# seed 4104 课程式预训练试点：失败记录','',f'记录时间：{record["time"]}。判据1、判据2未完成，不判定。','', '## 第一处失败','']+json_block(failure_record)
 lines+=['','## 已完成验证组','']+table(['组','验证均值s'],[[k,number(v)] for k,v in record['completed_group_means'].items()])
 lines+=['',f'来源：{record["source"]}。未完成的训练、评估和比较不填数值。','', '## 实现修正','']+json_block(record['implementation_corrections'])
 textfile(REPORT_ROOT/'seed_4104/report.md','\n'.join(lines)+'\n')
 textfile(REPORT_ROOT/'summary.md','\n'.join(lines)+'\n')
 return record
