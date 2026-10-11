# Factual Chinese report and exact table extract requested for the chat reply.
from diag_shared import *

def report(pre,replays,fresh,logs,selection,summaries,cross,posthoc):
 lines=['# v2.4.0 多 seed dev 诊断','',f'输出目录：{NEW}','', '本任务仅评估 dev，未训练、未更新参数、未执行任何 test 窗口。C* 为事后在同一 dev 上选择，所列 C* dev 值偏乐观；所有交叉比较为 dev，非确认性。',''];tables=[]
 def table(headers,rows):
  fmt=lambda v:'缺失' if v is None else f'{v:.12g}' if isinstance(v,float) else str(v)
  text=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(fmt(v) for v in r)+' |' for r in rows]+['']
  lines.extend(text);tables.extend(text)
 def section(title):lines.extend(['## '+title,'']);tables.extend(['## '+title,''])
 section('1 检查结果')
 table(['检查','结果','关键数值'],[[k,v.get('status','PASS'),json.dumps({kk:vv for kk,vv in v.items() if kk not in ('status','disjoint_checks')},ensure_ascii=False)] for k,v in CHECKS.items()])
 table(['运行环境','值'],[['解释器',pre['runtime']['executable']],['sys.prefix',pre['sys_prefix']],['torch',pre['runtime']['torch']],['CUDA',pre['runtime']['cuda']],['设备',pre['device']],['torch 线程',pre['torch_threads']],['协议 SHA256',pre['protocol_sha256']]])
 table(['回放组','完整 metrics 逐位一致','actor 哈希不变','墙钟秒'],[[r['group'],r['status'],r['actor_hash_unchanged'],r['wall_seconds']] for r in replays])
 table(['4101 复用组','per_scenario 全字段逐位一致','均值差秒'],[[r['group'],r['per_scenario_equal'],r['mean_difference_s']] for r in read(NEW/'reuse_4101_checks.json')])
 section('2 Q1：c0–c10 dev 表现')
 lines.extend(['截断时延为 20 场景等权均值；每场景 PooledMetrics 合并 stream 1、2。训练块均值按该块 10 个下层轮等权计算。',''])
 table(['seed','块','dev 截断时延秒','该块训练均值秒'],[[s,r['cycle'],r['dev_mean_s'],r['training_block_mean_s'] if r['cycle'] else ''] for s in SEEDS for r in selection[str(s)]['candidates']])
 table(['seed','E5（SD200+cvx）秒','TFOO-200 秒','C*','C* 秒','事后排名（好→差）'],[[s,summaries[(s,'E5')]['mean_s'],summaries[(s,'TFOO-200')]['mean_s'],selection[str(s)]['Cstar']['cycle'],selection[str(s)]['Cstar']['dev_mean_s'],','.join('c'+str(r['cycle']) for r in selection[str(s)]['ranking'])] for s in SEEDS])
 lines.extend(['以下四段时延与 cost/s 为 40 窗口池化值，未完成数除以 40 窗口；它们与场景等权截断时延采用各自明确口径。',''])
 table(['seed','组','dev 截断s','高优未完成/窗口','低优未完成/窗口','cost/s','upload s','ready s','u2u s','cpu s'],[[s,g,summaries[(s,g)]['mean_s'],summaries[(s,g)]['unfinished_high_per_window'],summaries[(s,g)]['unfinished_low_per_window'],summaries[(s,g)]['cost_per_s'],*[summaries[(s,g)]['stage_delay_s'][k] for k in ('upload','ready','u2u','cpu')]] for s in SEEDS for g in BASEGROUPS+['X3']])
 table(['seed','CVX组','solver_calls','fallback_calls'],[[s,g,summaries[(s,g)]['diagnostics'].get('solver_calls'),summaries[(s,g)]['diagnostics'].get('fallback_calls')] for s in SEEDS for g in ('E5','TFOO-200')])
 plot_dev(selection,summaries)
 for s in SEEDS:lines.extend([f'![seed {s} c0–c10](dev_{s}.png)',''])
 section('3 Q2：已有训练日志')
 lines.extend(['每轮事件/fit 的数值先按轮取均值和最大值；跨轮/跨块统计再等权取均值。critic 两个网络分量在 CSV 中分别保留；下表 critic 残差列来自 train_metrics.critic_residual_mse。缺失不填零。完整字段名和 attempt 选择记录见各阶段 metadata.json。','', '### a) 4103 c9/c10 与事后阈值',''])
 on=logs['onset4103'];table(['事后定义','c1–c9逐轮均值s','两倍阈值s','第一个超过阈值的global','该轮时延s'],[[on['posthoc_definition'],on['c1_c9_mean_s'],on['threshold_s'],on['first_global_episode'],on['first_delay_s']]])
 cols=['truncated_delay_s','unfinished_parents','oldest_unfinished_age_s','cost_per_s','actor_relative_entropy','critic_residual_mse','critic.target_mean.mean','actor.gradient_norm.mean','learning_event_rows','learning_skipped_rows','learning_actor_updates']
 table(['global','时延s','未完成','最老年龄s','cost/s','相对熵','critic残差','target_mean','actor梯度','事件行','跳过','actor更新'],[[r['global_episode'],*[r.get(k) for k in cols]] for r in on['c9_c10_rows']])
 table(['事后阈值前5轮','时延s','未完成','相对熵','critic残差','target_mean','actor梯度'],[[r['global_episode'],*[r.get(k) for k in ['truncated_delay_s','unfinished_parents','actor_relative_entropy','critic_residual_mse','critic.target_mean.mean','actor.gradient_norm.mean']]] for r in on['preceding_five']])
 lines.extend(['此前 5 轮的全部请求字段（能耗、分段时延、优先级与 critic 两个分量的均值/最大值）原样保留于 logs_analysis/summary.json.onset4103.preceding_five 和 seed_4103/stage3.csv。','', '### b) 第三阶段每块均值',''])
 table(['seed','块','global范围','时延s','相对熵','critic残差','target_mean','actor梯度','alpha'],[[s,b['cycle'],f"{b['global_first']}–{b['global_last']}",b['delay_s'],b['entropy'],b['critic_residual'],b['target_mean'],b['actor_gradient'],b['alpha']] for s in SEEDS for b in logs['stage3_blocks'][str(s)]])
 table(['seed','块范围','时延s','相对熵','critic残差','target_mean','actor梯度','记录的alpha'],[[s,name,b['delay_s'],b['entropy'],b['critic_residual'],b['target_mean'],b['actor_gradient'],b['alpha_values']] for s in SEEDS for name,b in logs['alpha_halves'][str(s)].items()])
 lines.extend(['### c) 第一阶段每25轮比较',''])
 for key,label in [('truncated_delay_s','截断时延s'),('actor_relative_entropy','相对熵'),('critic_residual_mse','critic残差')]:
  lines.extend([label+'；各 seed 同期训练轮，未视为同一外生场景。',''])
  table(['轮次',*[str(s) for s in SEEDS]],[[f'{i*25+1}–{(i+1)*25}',*[logs['stage1_25_bins'][str(s)][i].get(key) for s in SEEDS]] for i in range(12)])
 table(['seed','G1 passed','candidate均值s','equal均值s','equal胜场','local均值s','local胜场'],[[s,g.get('passed'),g.get('equal',{}).get('candidate_mean_s'),g.get('equal',{}).get('reference_mean_s'),g.get('equal',{}).get('wins'),g.get('local',{}).get('reference_mean_s'),g.get('local',{}).get('wins')] for s in SEEDS for g in [logs['G1'][str(s)]]])
 lines.extend(['4102 diagnosis_required.json 原文（文件不存在时写缺失；不删 pairs）：','```json',json.dumps(logs['diagnosis_required']['4102'],ensure_ascii=False,indent=2),'```',''])
 lines.extend(['### d) SD200 / TFOO 每10轮训练均值与 PPO 更新',''])
 table(['上层轮次',*[f'{s} SD200 s / TFOO s' for s in SEEDS]],[[f'{i*10+1}–{(i+1)*10}',*[f"{logs['upper_10_bins'][str(s)]['stage2'][i].get('truncated_delay_s'):.12g} / {logs['upper_10_bins'][str(s)]['TFOO'][i].get('truncated_delay_s'):.12g}" for s in SEEDS]] for i in range(20)])
 pporows=[]
 for s in SEEDS:
  for st in ('stage2','TFOO'):
   m=logs['metadata'][str(s)][st];c=m['final_ppo_counts'] or {};rr=logs['data'][str(s)][st]
   pporows.append([s,st,m['ppo_batches'],*[c.get(k) for k in ('batches','actor_steps_attempted','actor_steps_accepted','value_steps','rollbacks')],avg([r.get('ppo.kl.mean') for r in rr])])
 table(['seed','运行','日志批次','batches','actor attempted','actor accepted','value steps','rollbacks','epoch KL均值'],pporows)
 table(['seed','阶段','多attempt的global数量','PPO实际数值字段'],[[s,st,len(m['multiple_attempts']),','.join(k for k in m['recorded_learning_numeric_fields'] if k.startswith('ppo.')) or '未记录'] for s in SEEDS for st,m in logs['metadata'][str(s)].items() if st in ('stage2','TFOO')])
 lines.extend(['### e) 基线 checkpoint 与快照清单','', '全部文件名及字节大小见 checkpoint_inventory.md 和 logs_analysis/checkpoint_inventory.csv，均直接取目录项；未为该清单计算 SHA256。',''])
 table(['seed','基线','checkpoint文件数','snapshot文件数','总字节'],[[s,b,sum(r['folder']=='checkpoints' for r in rr),sum(r['folder']=='snapshots' for r in rr),sum(r['bytes'] for r in rr)] for s in SEEDS for b in ('NC','TOO','TFOO') for rr in [[r for r in logs['checkpoint_inventory'] if r['seed']==s and r['baseline']==b]]])
 inventory=['# 全部基线 checkpoint / snapshot 清单','','|seed|基线|目录|文件名|字节|','|---|---|---|---|---|']+['| '+' | '.join(map(str,[r['seed'],r['baseline'],r['folder'],r['name'],r['bytes']]))+' |' for r in logs['checkpoint_inventory']]
 (NEW/'checkpoint_inventory.md').write_text('\n'.join(inventory)+'\n',encoding='utf-8')
 section('4 Q3：上下层交叉（dev，非确认性）')
 for name,t in cross['tables'].items():
  lines.extend([f'### {name}','']);tables.extend([f'### {name}',''])
  table(['上层','cvx','SAC300','C10','C*'],[[upper,*[t[upper+'/'+col] for col in ('cvx','SAC300','C10','Cstar')]] for upper in ('SD200','TFOO')])
 lines.extend(['5 seed 汇总先对同一场景取 5 seed 均值，再对 20 场景等权汇总。相对差=(candidate/reference−1)×100%，负数表示候选时延更低。Wilcoxon 双侧、zero_method=wilcox；t 检验双侧；符号检验只计非零差。',''])
 table(['seed/汇总','候选','参考','候选均值s','参考均值s','相对差%','胜场/20','Wilcoxon p','配对t p','符号p'],[[r['context'],r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['wilcoxon_p'],r['t_p'],r['sign_p']] for r in cross['comparisons']])
 section('5 dev 与已有 test 一致性（事后）')
 table(['seed','组','dev均值s','已有test均值s'],[[r['seed'],r['group'],r['dev_mean_s'],r['test_mean_s']] for r in posthoc['rows']])
 table(['seed','dev排序（好→差）','test排序（好→差）'],[[r['seed'],' > '.join(r['dev_ranking']),' > '.join(r['test_ranking'])] for r in posthoc['rankings']])
 table(['格子数','Spearman rho','双侧p'],[[posthoc['spearman']['cells'],posthoc['spearman']['rho'],posthoc['spearman']['pvalue']]])
 table(['单列重点','dev均值s','已有test均值s'],[[str(r['seed'])+' '+r['group'],r['dev_mean_s'],r['test_mean_s']] for r in posthoc['special']])
 section('6 Q4：新 test 清单（仅种子，不运行）')
 table(['项','值'],[['schema','hrl_v2_fresh_test_manifest_candidate_1_0'],['场景', '21–40，共20'],['policy_streams',fresh['policy_streams']],['外部种子数/唯一数',f"{fresh['external_seeds']}/{fresh['unique_external_seeds']}"],['1–20公式与manifest(test)完全一致',fresh['formula_1_20_equal']],['构造新test环境数',0],['本任务test窗口数',0],['json_digest',fresh['json_digest']],['文件SHA256',fresh['sha256']]])
 table(['禁止重合集合','种子数','交集数','检查'],[[name,r['seed_count'],len(r['intersection']),'PASS' if r['passed'] else 'FAIL'] for name,r in fresh['disjoint_checks'].items()])
 lines.extend(['## 7 偏差、修正和实测墙钟','', '无源码/config/scene/协议/旧结果修改；没有训练或参数更新。实现性修正次数=0。复用 4101 窗口均记录原文件 SHA256；X3 等于 X1/X2 时直接复用，未重复执行。已有 test 仅从汇总读取。全量数值保存为 JSON/CSV，显示表保留12位有效数字。',''])
 lines.extend(['|步骤|实测墙钟秒|','|---|---|']+['| '+k+' | '+str(v)+' |' for k,v in TIMING.items()]+[''])
 lines.extend(['完整实际字段清单：','```json',json.dumps({str(s):{st:dict(csv_fields=m['recorded_csv_fields'],learning_numeric_fields=m['recorded_learning_numeric_fields'],multiple_attempts=m['multiple_attempts']) for st,m in logs['metadata'][str(s)].items()} for s in SEEDS},ensure_ascii=False,indent=2),'```',''])
 (NEW/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(NEW/'tables_1_6.md').write_text('\n'.join(tables)+'\n',encoding='utf-8')

def plot_dev(selection,summaries):
 os.environ['MPLCONFIGDIR']=str(NEW/'mplconfig')
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError:return
 for seed in SEEDS:
  r=selection[str(seed)]['candidates'];fig,ax=plt.subplots(figsize=(9,4));ax.plot([x['cycle'] for x in r],[x['dev_mean_s'] for x in r],'o-',label='c0-c10')
  for g in ('E5','TFOO-200'):ax.axhline(summaries[(seed,g)]['mean_s'],linestyle='--',label=g)
  ax.set_xticks(range(11));ax.set_ylabel('dev truncated delay (s)');ax.set_xlabel('cycle (c0=E2)');ax.set_title('seed '+str(seed));ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(NEW/f'dev_{seed}.png',dpi=150);plt.close(fig)
