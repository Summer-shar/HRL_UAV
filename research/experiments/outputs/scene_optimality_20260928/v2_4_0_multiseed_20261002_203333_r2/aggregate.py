# Aggregate frozen five-seed test results; statistics copied from evaluate_test.py.
import sys
sys.dont_write_bytecode=True
from common import *
assert_imports()
import numpy as np
import scipy.stats as stats
from hrl_v2.gates import paired,unfinished_per_episode
OUT=OUTM/'aggregate'

def compare(summary):
 comparisons=[]
 def add(cname,rname,category):
  c=summary['per_scenario'][cname];r=summary['per_scenario'][rname];p=paired(c,r)
  keys=sorted(c,key=int);a=np.array([c[k]['truncated_delay_s'] for k in keys]);b=np.array([r[k]['truncated_delay_s'] for k in keys]);d=a-b;nonzero=d!=0
  t=stats.ttest_rel(a,b,alternative='two-sided');w=stats.wilcoxon(d,alternative='two-sided',zero_method='wilcox');ranks=stats.rankdata(np.abs(d[nonzero]));wplus=float(ranks[d[nonzero]>0].sum());sign=stats.binomtest(p['wins'],int(nonzero.sum()),0.5,alternative='two-sided')
  row=dict(candidate=cname,reference=rname,category=category,primary_endpoint=cname=='Proposed' and rname=='TFOO',scenarios=p['scenarios'],candidate_mean_s=p['candidate_mean_s'],reference_mean_s=p['reference_mean_s'],relative_difference_percent=100*(p['candidate_mean_s']/p['reference_mean_s']-1),wins=p['wins'],pairs=p['pairs'],candidate_unfinished_per_episode=unfinished_per_episode(c),reference_unfinished_per_episode=unfinished_per_episode(r),statistics=dict(mean_d=float(d.mean()),sd_d=float(d.std(ddof=1)),nonzero_differences=int(nonzero.sum()),ttest_rel=dict(statistic=float(t.statistic),pvalue=float(t.pvalue),alternative='two-sided'),wilcoxon=dict(statistic=float(w.statistic),W_plus=wplus,pvalue=float(w.pvalue),alternative='two-sided',zero_method='wilcox'),binomtest=dict(wins=p['wins'],n=int(nonzero.sum()),pvalue=float(sign.pvalue),alternative='two-sided')))
  if category=='main':
   ca=np.array([c[k]['cost_per_s'] for k in keys]);rb=np.array([r[k]['cost_per_s'] for k in keys]);row['cost_per_s']=dict(candidate_mean=float(ca.mean()),reference_mean=float(rb.mean()),relative_difference_percent=float(100*(ca.mean()/rb.mean()-1)),wins=int((ca<rb).sum()))
  comparisons.append(row)
 for ref in ('TFOO','NC','CA','TOO'):add('Proposed',ref,'main')
 for ref in ('Abl_TwoStage','Abl_HoverSAC','Abl_UpperCVX'):add('Proposed',ref,'ablation')
 add('Abl_TwoStage','TFOO','supplementary');return comparisons


def main():
 protocol_check();check(not OUT.exists(),'new_aggregate_directory');OUT.mkdir()
 source=read(T4101/'test_summary.json');reference=read(T4101/'test_comparisons.json')[0]
 recomputed=compare(source)[0];equal(recomputed,reference,'4101_primary_exact_recomputation')
 groups=['Proposed','NC','CA','TOO','TFOO','Abl_TwoStage','Abl_HoverSAC','Abl_UpperCVX']
 allsummary={4101:source,**{s:read(OUTM/f'seed_{s}/test/test_summary.json') for s in (4102,4103,4104,4105)}}
 for s,summary in allsummary.items():
  check(summary['protocol_sha256']==PROTO_SHA and summary['manifest']==source['manifest'],'aggregate_protocol_manifest',s)
 average={}
 for group in groups:
  average[group]={}
  for sid in map(str,range(1,21)):
   # gates.paired also reads unfinished and windows; average every recorded scalar
   # used by the comparisons, preserving per-window unfinished normalization.
   original=source['per_scenario'][group][sid];row=dict(original)
   for key in ('truncated_delay_s','cost_per_s','unfinished_parents','windows'):
    row[key]=float(np.mean([allsummary[s]['per_scenario'][group][sid][key] for s in allsummary]))
   row['priority']=[dict(p) for p in original['priority']]
   for p in (0,1):
    row['priority'][p]['unfinished']=float(np.mean([allsummary[s]['per_scenario'][group][sid]['priority'][p]['unfinished'] for s in allsummary]))
   average[group][sid]=row
 aggregate=compare(dict(per_scenario=average))
 perseeds={s:compare(summary)[0] for s,summary in allsummary.items()}
 across={g:dict(mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)),per_seed={str(s):v for s,v in zip(allsummary,vals)}) for g in groups for vals in [[float(np.mean([r['truncated_delay_s'] for r in sm['per_scenario'][g].values()])) for sm in allsummary.values()]]}
 gates={4101:read(P4101/'gates/episode-0300.json'),**{s:read(WORK/f'runs/Proposed/seed_{s}/gates/episode-0300.json') for s in (4102,4103,4104,4105)}}
 cycles={4101:read(S34101/'training_summary.json'),**{s:read(OUTM/f'seed_{s}/stage3/training_summary.json') for s in (4102,4103,4104,4105)}}
 # 4101 driver predates cycle summary: derive only from its recorded train_metrics.
 for s,r in cycles.items():
  if 'cycle_mean_truncated_delay_s' not in r:
   from hrl_v2.schedule import phase
   rr=[json.loads(line) for line in (S34101/'runs/stage3/train_metrics.jsonl').read_text().splitlines() if line.strip()]
   r['cycle_mean_truncated_delay_s']={str(c):float(np.mean([x['truncated_delay_s'] for x in rr if phase(x['global_episode']).cycle//2==c])) for c in range(1,11)}
 def occupied(paths):return sum(f.stat().st_size for p in paths for f in Path(p).rglob('*') if f.is_file())
 disks={str(s):occupied([WORK/f'references/seed_{s}',WORK/f'runs/Proposed/seed_{s}',*[WORK/f'runs/{n}/seed_{s}' for n in ('NC','TOO','TFOO')],OUTM/f'seed_{s}']) for s in (4102,4103,4104,4105)}
 disks['4101']=occupied([P4101,B4101,R24101,S34101,T4101])
 timing=read(WORK/'queue_status.json')['completed_steps']
 result=dict(status='passed',protocol_sha256=PROTO_SHA,seed4101_recompute_exact=True,primary=aggregate[0],comparisons=aggregate,per_seed={str(s):r for s,r in perseeds.items()},proposed_mean_better_seeds=sum(r['candidate_mean_s']<r['reference_mean_s'] for r in perseeds.values()),total_seeds=5,across_seed=across,G1={str(s):r for s,r in gates.items()},stage3={str(s):r['cycle_mean_truncated_delay_s'] for s,r in cycles.items()},step_timings=timing,disk_bytes=disks,only4101_Alternating=dict(summary=source['by_mode']['Abl_Alternating'],comparisons=[r for r in read(T4101/'test_comparisons.json') if r['reference']=='Abl_Alternating']))
 write(OUT/'aggregate.json',result);protocol_check()
 lines=['# v2.4.0 五 seed 汇总','','4101 主终点完整对象复算逐位一致；协议SHA256：'+PROTO_SHA,'']
 def table(headers,rows):
  lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
  lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows);lines.append('')
 def comp(rs):return [[r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue'],r['statistics']['ttest_rel']['pvalue'],r['statistics']['binomtest']['pvalue']] for r in rs]
 lines+=['## 主终点与次要比较','','20场景；先对5 seed的场景池化值取算术均值，场景等权；Wilcoxon双侧，zero_method=wilcox；次要比较未作多重校正。','']
 table(['候选','参考','候选均值s','参考均值s','相对差%','胜场','Wilcoxon p','配对t p','符号p'],comp(aggregate))
 lines+=['## 主终点逐场景',''];table(['场景','Proposed s','TFOO s','差s','胜'],[[r['scenario'],r['candidate_s'],r['reference_s'],r['candidate_s']-r['reference_s'],r['win']] for r in aggregate[0]['pairs']])
 lines+=['## 每seed主终点',''];table(['seed','Proposed s','TFOO s','相对差%','胜场','Wilcoxon p'],[[s,r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue']] for s,r in perseeds.items()])
 lines+=[f"Proposed场景等权均值更低的seed数：{result['proposed_mean_better_seeds']}/5。",'','## 主比较cost/s',''];table(['参考','候选cost/s','参考cost/s','相对差%','胜场'],[[r['reference'],r['cost_per_s']['candidate_mean'],r['cost_per_s']['reference_mean'],r['cost_per_s']['relative_difference_percent'],r['cost_per_s']['wins']] for r in aggregate if r['category']=='main'])
 lines+=['## 跨seed均值及样本标准差',''];table(['组','均值s','样本标准差s'],[[g,r['mean'],r['std']] for g,r in across.items()])
 lines+=['## 仅4101的Alternating','',json.dumps(result['only4101_Alternating']['comparisons'],ensure_ascii=False,indent=2),'','## G1（原始检查结果）','']
 for s,g in gates.items():lines+=[f'### seed {s}','','```json',json.dumps(g,ensure_ascii=False,indent=2),'```','']
 lines+=['## 第三阶段每cycle训练截断时延均值',''];table(['seed']+[str(c) for c in range(1,11)],[[s]+[r['cycle_mean_truncated_delay_s'][str(c)] for c in range(1,11)] for s,r in cycles.items()])
 lines+=['## 每步实测墙钟',''];table(['seed','步骤','秒'],[[r.get('seed'),r['step'],r['wall_seconds']] for r in timing])
 lines+=['## 磁盘占用（文件字节数之和）',''];table(['seed','字节'],[[s,v] for s,v in disks.items()])
 text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text,encoding='utf-8')
 final=['# v2.4.0 seeds 4102–4105 全流程与五 seed 汇总','','## 检查结果','','启动检查：全部通过。validation：passed。','']
 for step in ('RP-W','RP-S2','RP-S3','RP-T','RP-REF'):
  r=read(OUTM/f'repro_4101/{step}/checks.json');final.append(f"- {step}: {r['status']}；墙钟 {r['wall_seconds']} s。")
 final+=['','## 每seed硬检查','']
 for s in (4102,4103,4104,4105):
  for step,path in (('reference',WORK/f'references/seed_{s}/checks.json'),('stage1',WORK/f'runs/Proposed/seed_{s}/checks.json'),('stage2',OUTM/f'seed_{s}/stage2/checks.json'),('stage3',OUTM/f'seed_{s}/stage3/checks.json'),('baselines',OUTM/f'seed_{s}/baseline_checks.json'),('test',OUTM/f'seed_{s}/test/checks.json')):
   final.append(f"- seed {s} {step}: {read(path)['status']}。")
 final+=['','## 驱动差异逐hunk对照','']
 for name,record in read(OUTM/'driver_diffs/hunks.json').items():
  final+=[f'### {name}','','来源：'+record['source'],'']
  for h in record['hunks']:
   final+=[h['hunk'],'']+[f'- {k}: {v}' for k,v in h['allowed_categories'].items()]+['','```diff']+h['changes']+['```','']
 final+=[text,'','## 脚本错误与修正','',json.dumps(read(OUTM/'implementation_correction.json'),ensure_ascii=False,indent=2),'','## 最终Git状态','','```json',json.dumps(git_states(),ensure_ascii=False,indent=2),'```','']
 (OUTM/'report.md').write_text('\n'.join(final),encoding='utf-8')
if __name__=='__main__':main()
