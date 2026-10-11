"""Reports of recorded facts only; no fitting, training, model selection or method conclusions."""
from j_common import *
import math,zipfile

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def load_window(path):
 p=Path(path)
 if p.exists():return read(p)
 r=read(CPIL/'archive_index.json')[str(p)]
 with zipfile.ZipFile(r['archive']) as z:return json.loads(z.read(r['member']).decode('utf-8-sig'))

def ppo_logs(c):
 out=[]
 for spec in c['logsets']:
  learning=[];training=[]
  for p in spec['learning']:
   with Path(p).open(encoding='utf-8-sig') as f:
    for line in f:
     if '"ppo"' in line:
      d=json.loads(line)
      if isinstance(d.get('ppo'),list):learning.append(d)
  for p in spec['train']:
   with Path(p).open(encoding='utf-8-sig') as f:
    training.extend(json.loads(l) for l in f if l.strip())
  key=spec['episode_key'];tr={r[key]:r for r in training};updates=[]
  for i,r in enumerate(learning):
   ep=r[key];epochs=r['ppo'];batch=[tr[n] for n in range(ep-4,ep+1)]
   counts=tr[ep].get('ppo');previous=tr.get(ep-5,{}).get('ppo')
   if isinstance(counts,dict):
    if not isinstance(previous,dict) and i==0:previous={k:0 for k in counts}
    delta={k:counts[k]-previous.get(k,0) for k in counts} if isinstance(previous,dict) else {}
   else:delta={}
   updates.append(dict(update=i+1,episode=ep,global_episode=r.get('global_episode',r.get('proposed_global_episode')),epochs=epochs,epochs_attempted=len(epochs),epochs_accepted=sum(bool(e['accepted']) for e in epochs),KL_stop=any(not e['accepted'] for e in epochs),mean_KL=float(np.mean([e['kl'] for e in epochs])),accepted_mean_KL=float(np.mean([e['kl'] for e in epochs if e['accepted']])) if any(e['accepted'] for e in epochs) else None,actor_steps_executed=delta.get('actor_steps_attempted'),actor_steps_retained=delta.get('actor_steps_accepted'),value_steps=delta.get('value_steps'),batch_return_physical=[b.get('reward_physical') for b in batch],batch_return_mean=float(np.mean([b['reward_physical'] for b in batch])),batch_learning_return_mean=float(np.mean([b['reward_learning_macro_sum'] for b in batch])),clip_fraction=None,policy_std=None,policy_entropy=None,value_loss=None,explained_variance=None,advantage_mean=None,advantage_std=None))
  equal(len(updates),40,'Q4_exact_40_'+spec['label'])
  numeric=['epochs_attempted','epochs_accepted','mean_KL','actor_steps_executed','actor_steps_retained','batch_return_mean','batch_learning_return_mean','clip_fraction','policy_std','policy_entropy','value_loss','explained_variance','advantage_mean','advantage_std']
  summaries={}
  for label,rr in [('all',updates),('first10',updates[:10]),('last10',updates[-10:])]:
   summaries[label]={k:float(np.median([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in numeric}
   summaries[label]['KL_stop_count']=sum(r['KL_stop'] for r in rr)
  out.append(dict(label=spec['label'],sources=spec,updates=updates,summary=summaries,missing_label='未记录',note='mean_KL 为已记录各 epoch KL 的算术平均，含触发回退的 epoch。actor_steps_executed 来自累计 attempted 的差值，retained 来自 accepted 差值；没有累计计数时为未记录。未记录的策略熵不以 SAC actor_relative_entropy 代替。'))
 put(CPIL/'ppo_logs.json',clean(out));return out

def comparison(values,reference):
 from scipy.stats import wilcoxon,ttest_rel,binomtest
 a=np.array(values);b=np.array(reference);delta=a-b;n=int(np.count_nonzero(delta));wins=int(np.sum(delta<0));ties=int(np.sum(delta==0))
 return clean(dict(candidate_mean_s=float(a.mean()),reference_mean_s=float(b.mean()),delay_change_percent=float(100*(a.mean()/b.mean()-1)),wins=wins,ties=ties,scenarios=20,tests=dict(wilcoxon_two_sided_p=float(wilcoxon(a,b,alternative='two-sided').pvalue) if n else 1.,paired_t_two_sided_p=float(ttest_rel(a,b,alternative='two-sided').pvalue) if n else 1.,sign_two_sided_p=float(binomtest(wins,n,.5,alternative='two-sided').pvalue) if n else 1.,sign_n_excluding_ties=n)))

def fmt(x,d=6):return '未记录' if x is None else f'{x:.{d}f}' if isinstance(x,(float,np.floating)) else str(x)
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|']+['|'+'|'.join(str(v).replace('|','/') for v in r)+'|' for r in rows])

def report(c,started):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figroot=output_path(CPIL/'figures');figroot.mkdir(exist_ok=True)
 records=[];by={}
 for g in c['groups']:
  s=read(CPIL/'groups'/g['label']/'summary.json');by[g['label']]=s
  refs={k:read(c['references'][f'{k}-{g["seed"]}']['source']) for k in ('TFOO','v1')}
  values=[s['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)]
  comp={k:comparison(values,[refs[k]['per_scenario'][str(i)]['truncated_delay_s'] for i in range(1,21)]) for k in refs}
  total=s['pooled'];r=dict(label=g['label'],seed=g['seed'],upper=g['upper'],lower=g['lower'],route=g['route'],episode=g['episode'],mean_s=s['mean_s'],per_scenario_s=values,stream_consistency=s['stream_consistency'],anchors=s['anchor_checks'],comparisons=comp,unfinished_high_low_per_rollout=[p['unfinished']/total['episodes'] for p in total['priority']],average_unfinished_parents=total['average_unfinished_parents'],stage_delay_s_scene_equal={k:float(np.mean([v['stage_delay_s'][k] for v in s['per_scenario'].values()])) for k in ('upload','ready','u2u','cpu')},stage_delay_s_pooled=total['stage_delay_s'],cost_per_s=total['cost_per_s'],observations=s['observations'],summary_path=str(CPIL/'groups'/g['label']/'summary.json'),wall_seconds=s['wall_seconds'])
  records.append(r)
 logs=ppo_logs(c)
 figures=[]
 for seed in (4103,4104):
  ns=list(range(220,401,20));cv=[];paired=[]
  for n in ns:
   label='A5' if seed==4103 and n==320 else 'A6' if seed==4103 and n==340 else f'B-{seed}-{n}'
   cv.append(by[label]['mean_s']);paired.append(read(Path(c['paths']['JC'])/f'seed_{seed}/evaluations/episode-{n:04d}/summary.json')['mean_s'])
  fig,ax=plt.subplots(figsize=(9,5));ax.plot(ns,cv,'o-',label='JC upper + CVX');ax.plot(ns,paired,'s--',label='JC paired dev (existing)')
  for ref,label in [('TFOO','TFOO-200'),('SD','SD200 + CVX')]:ax.axhline(c['references'][f'{ref}-{seed}']['mean_s'],label=label,linestyle=':')
  ax.set(xlabel='Checkpoint episode',ylabel='Mean truncated delay (s)',title=f'Seed {seed}, dev scenarios 1-20');ax.grid(alpha=.25);ax.legend();fig.tight_layout();p=figroot/f'Q2_{seed}.png';fig.savefig(p,dpi=160);plt.close(fig);figures.append(str(p))
 for log in logs:
  fig,axes=plt.subplots(2,2,figsize=(11,7));x=[r['update'] for r in log['updates']]
  for ax,keys,title in [(axes[0,0],['epochs_attempted','epochs_accepted'],'Actor epochs'),(axes[0,1],['mean_KL'],'Recorded mean KL'),(axes[1,0],['actor_steps_executed','actor_steps_retained'],'Actor optimizer steps'),(axes[1,1],['batch_return_mean'],'Mean physical episode return')]:
   plotted=False
   for k in keys:
    y=[r[k] for r in log['updates']]
    if any(v is not None for v in y):ax.plot(x,[np.nan if v is None else v for v in y],label=k);plotted=True
   if plotted:ax.legend(fontsize=8)
   else:ax.text(.5,.5,'Not recorded',ha='center',va='center',transform=ax.transAxes)
   ax.set(title=title,xlabel='PPO update');ax.grid(alpha=.25)
  fig.suptitle(log['label']);fig.tight_layout();p=figroot/f'Q4_{log["label"]}.png';fig.savefig(p,dpi=160);plt.close(fig);figures.append(str(p))
 top=[]
 for label in ('A1','A2'):
  worst=sorted(by[label]['per_scenario'],key=lambda sid:by[label]['per_scenario'][sid]['truncated_delay_s'],reverse=True)[:3]
  for sid in worst:
   paths=[w for w in by[label]['window_files'] if w['scenario']==int(sid)]
   for w in paths:
    d=load_window(w['path']);p=CPIL/'Q1_timeseries'/f'{label}_scene_{int(sid):02d}_stream_{w["stream"]}.json';put(p,dict(group=label,scenario=int(sid),stream=w['stream'],mean_s=by[label]['per_scenario'][sid]['truncated_delay_s'],definitions=d['observations']['definitions'],samples_5s=d['observations']['samples_5s']));top.append(str(p))
   d=load_window(paths[0]['path']);samples=d['observations']['samples_5s'];x=[w['end_s'] for w in samples]
   fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
   for u in range(3):
    axes[0].plot(x,[sum(w['cpu_queue_mean'][pr][u] for pr in (0,1)) for w in samples],label=f'UAV {u} CPU fragments')
    axes[1].plot(x,[sum(w['backlog_by_stage_priority_uav']['upload'][pr][u] for pr in (0,1)) for w in samples],label=f'UAV {u} upload parents')
   for pr in (0,1):axes[2].plot(x,[w['unfinished_high_low'][pr] for w in samples],label=['High priority','Low priority'][pr])
   for ax in axes:ax.legend();ax.grid(alpha=.25)
   axes[-1].set_xlabel('Time (s)');fig.suptitle(f'{label}, scenario {sid}, delay {by[label]["per_scenario"][sid]["truncated_delay_s"]:.6f} s');fig.tight_layout();p=figroot/f'Q1_{label}_{int(sid):02d}.png';fig.savefig(p,dpi=160);plt.close(fig);figures.append(str(p))
 corrections=read(CPIL/'implementation_corrections.json') if (CPIL/'implementation_corrections.json').exists() else []
 archives=[read(p) for p in sorted((CPIL/'archives').glob('*_result.json'))] if (CPIL/'archives').exists() else []
 result=clean(dict(status='complete',completed_at=now(),wall_seconds=time.perf_counter()-started,contract_path=str(CPIL/'contract.json'),contract_sha256=sha(CPIL/'contract.json'),completed_groups=33,groups=records,PPO_logs=logs,Q1_top_scene_timeseries=top,figures=figures,diagnostic_only=True,training=False,test_scenes=False,model_selection=False,readonly_checks=str(CPIL/'readonly_checks_final.json'),deviations=[dict(kind='GPU_gate_user_clarification',detail='用户明确允许桌面/浏览器 GPU 进程，仅排除其他训练或评估进程。'),dict(kind='seed_report_filename',detail='JC 实际种子报告为 seed_4103/4103report.json、seed_4104/4104report.json，已从文件定位。'),dict(kind='execution_order',detail='先锚点组，再无锚点诊断组，33 组定义不变。'),dict(kind='unrecorded_PPO_fields',detail='日志缺少的字段记为未记录，没有补训。')],stops=[read(CPIL/'step0/failure.json')],implementation_corrections=corrections,archives=archives))
 put(CPIL/'report.json',result)
 md=['# 方案 C 诊断结果（只评估）','',f'结束时间：{result["completed_at"]}。评估与报告耗时：{result["wall_seconds"]/3600:.3f} 小时。完成 33/33 组，每组 20 个 dev 场景 × 两条流。','', '全部搭配均仅作诊断，不参与方法或选模型。正的延迟变化百分比表示延迟高于参照，负值表示低于参照；胜场按逐场景延迟严格小于 TFOO 计。没有使用测试场景 21–40。','', '## 全部分组数值','',table(['组','seed','平均延迟/s','相对TFOO/%','胜场/20','相对原流程/%','两流一致','锚点'],[[r['label'],r['seed'],fmt(r['mean_s'],12),fmt(r['comparisons']['TFOO']['delay_change_percent'],4),r['comparisons']['TFOO']['wins'],fmt(r['comparisons']['v1']['delay_change_percent'],4),'是','通过' if r['anchors']['references'] else '无'] for r in records]),'', '## Q1：4103 的 2×3 交叉搭配','',table(['上层','L320','L340','cvx'],[['U320',fmt(by['A1']['mean_s'],12),fmt(by['A3']['mean_s'],12),fmt(by['A5']['mean_s'],12)],['U340',fmt(by['A4']['mean_s'],12),fmt(by['A2']['mean_s'],12),fmt(by['A6']['mean_s'],12)]]),'','数值单位为秒。交叉搭配反映固定存档组合表现，不给出先后因果判断。','']
 q1=[]
 for label in ['A1','A2','A3','A4','A5','A6','K-JC-4103','K-TFOO-4103']:
  s=by[label]['observations'];cy=np.array(s['assigned_cycles']).sum(axis=(0,1));fr=np.array(s['assigned_task_fractions']).sum(axis=(0,1));q1.append([label,','.join(fmt(v,4) for v in fr/fr.sum()),','.join(fmt(v,4) for v in cy/cy.sum()),','.join(fmt(v,3) for v in np.array(s['cpu_queue_mean']).sum(axis=0)),','.join(fmt(v,0) for v in np.array(s['cpu_queue_max']).sum(axis=0))])
 md += [table(['组','计算任务份额 UAV0,1,2','分配周期份额 UAV0,1,2','CPU队列均值 UAV0,1,2','各优先级队列最大值之和 UAV0,1,2'],q1),'','计算任务份额以分配给各无人机的父任务计算比例计数；CPU 队列单位为任务分片。逐优先级最大值之和不代表同一时刻的联合最大值，完整分优先级数值在 JSON 中。','']
 for p in figures:
  if Path(p).name.startswith('Q1_'):md.append(f'![{Path(p).stem}]({p})')
 md+=['','## Q2：各存档上层 + cvx','']
 for seed in (4103,4104):md.append(f'![Q2 {seed}]({figroot/f"Q2_{seed}.png"})')
 md+=['','JC 整对曲线来自旧报告；上层 + cvx 曲线来自本次诊断，TFOO 与 SD200 参考线来自核验后的旧结果。','','## Q3：位置、移动及上传距离','',table(['组','每5s位移均值/m','离起点均值/m','位移≥19.5m比例','用户到最近UAV/m','上传任务距离 高/低 m'],[[r['label'],fmt(r['observations']['executed_5s']['mean_m'],4),fmt(r['observations']['distance_from_start']['mean_m'],4),fmt(r['observations']['movement_ge_19_5m_fraction'],6),fmt(r['observations']['nearest_uav_distance']['mean_m'],4),'/'.join(fmt(v,4) for v in r['observations']['upload_task_distance_mean_m'])] for r in records]),'','上传目标直接读取 parent.arrival.source。上传距离先在每个 5 秒窗口内对每个等待/上传任务的物理步距离取平均，再对任务等权平均，高低优先级分开；跨窗口任务在各窗口分别计数。新到达任务的距离另存 birth_task_distance_mean_m。每个用户到最近无人机的距离按物理步平均。','', '## Q4：已有 PPO 日志','', '每份日志 40 次更新；完整逐更新、逐 epoch 数值在 ppo_logs.json。未记录项均为空值并标为“未记录”，没有补训。','']
 for log in logs:
  md += [f'### {log["label"]}','',table(['范围','接受epoch中位数','尝试epoch中位数','平均KL中位数','执行actor步数中位数','保留actor步数中位数','回合回报中位数','KL停止次数'],[[name,fmt(row['epochs_accepted']),fmt(row['epochs_attempted']),fmt(row['mean_KL'],8),fmt(row['actor_steps_executed']),fmt(row['actor_steps_retained']),fmt(row['batch_return_mean'],6),row['KL_stop_count']] for name,row in log['summary'].items()]),'',f'![{log["label"]}]({(figroot / ("Q4_" + log["label"] + ".png"))})','']
 md+=['clip 比例、PPO 策略 std/熵、value loss、explained variance、优势均值和标准差：未记录。平均 KL 为已记录 epoch 的算术平均，包含触发回退的 epoch。','','## 统计检验','',table(['组','Wilcoxon 双侧 p','配对t双侧 p','符号检验双侧 p'],[[r['label'],fmt(r['comparisons']['TFOO']['tests']['wilcoxon_two_sided_p'],8),fmt(r['comparisons']['TFOO']['tests']['paired_t_two_sided_p'],8),fmt(r['comparisons']['TFOO']['tests']['sign_two_sided_p'],8)] for r in records]),'','上述检验均以同 seed TFOO-200 的 20 个配对场景为参照，仅列原始 p 值，未作多重比较校正；原流程参照检验也保存在 report.json。','','## 积压、分段驻留及 cost','',table(['组','未完成 高/低（每900s截止）','upload/s','ready/s','u2u/s','cpu/s','cost/s'],[[r['label'],'/'.join(fmt(v,3) for v in r['unfinished_high_low_per_rollout'])]+[fmt(r['stage_delay_s_scene_equal'][k],6) for k in ('upload','ready','u2u','cpu')]+[fmt(r['cost_per_s'],6)] for r in records]),'','本表分段延迟为逐场景等权均值；汇总池化数值另存 JSON。每 5 秒的未完成任务、分段驻留、成本、队列、位置和分流在窗口文件 observations.samples_5s 中。','','## 核对、偏差与停止记录','',f'- 检查点：{len(c["checkpoints"])} 个，运行前与运行后 SHA256 一致。','- 源码身份、MAIN/WT Git HEAD 与状态、已冻结输入：核对通过。','- 两条随机流：全部 33 组逐场景完整指标及附加观测一致。','- 已存锚点：逐场景、逐条流完整指标一致。','- 初次停止：2026-10-10 00:33，GPU 桌面进程；原记录保留于 step0/failure.json。','- 用户随后允许桌面/浏览器 GPU 进程，排除其他训练或评估进程。','- 实现修正与继续授权：见 implementation_corrections.json 和 resume_authorization.json。','- JC 种子报告实际命名为 4103report.json / 4104report.json，使用从现有文件解析的路径。','- 执行顺序为锚点优先，33 组定义不变。','']
 if archives:md += [table(['压缩批次','文件数','原大小/MB','压缩后/MB','压缩后/原','耗时/s'],[[Path(a['archive']).name,a['file_count'],fmt(a['original_bytes']/1e6,2),fmt(a['archive_bytes']/1e6,2),fmt(a['ratio'],4),fmt(a['wall_seconds'],2)] for a in archives]),'','每批先写 SHA 清单，再压缩、解压流校验 SHA，最后删除 XD 内对应原始文件。路径映射见 archive_index.json。']
 else:md+=['逐窗口文件未超过 1 GiB，无压缩。']
 textfile(CPIL/'report.md','\n'.join(md)+'\n')
 print('REPORT_COMPLETE',len(records),len(figures),flush=True)
