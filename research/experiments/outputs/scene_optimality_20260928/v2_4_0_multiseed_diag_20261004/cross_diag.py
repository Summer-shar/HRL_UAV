from diag_shared import *
from hrl_v2.gates import paired
import scipy.stats as stats

def group_for_cstar(seed,selection):
 n=selection[str(seed)]['Cstar']['cycle'];return 'E2' if n==0 else f'S3-c{n}'
def cross_analysis(summaries,selection):
 contexts={}
 for seed in SEEDS:
  g=group_for_cstar(seed,selection);mapping={'SD200/cvx':'E5','SD200/SAC300':'E2','SD200/C10':'S3-c10','SD200/Cstar':g,'TFOO/cvx':'TFOO-200','TFOO/SAC300':'X1','TFOO/C10':'X2','TFOO/Cstar':'X3'}
  contexts[str(seed)]={k:summaries[(seed,v)]['per_scenario'] for k,v in mapping.items()}
 contexts['5-seed']={cell:{str(sid):dict(truncated_delay_s=avg([contexts[str(s)][cell][str(sid)]['truncated_delay_s'] for s in SEEDS]),unfinished_parents=avg([contexts[str(s)][cell][str(sid)]['unfinished_parents'] for s in SEEDS])) for sid in range(1,21)} for cell in contexts['4101']}
 tables={name:{cell:avg([r['truncated_delay_s'] for r in rows.values()]) for cell,rows in cells.items()} for name,cells in contexts.items()};comparisons=[]
 for context,cells in contexts.items():
  specs=[]
  for candidate,cell in [('S3-Cstar','SD200/Cstar'),('S3-c10','SD200/C10'),('X1','TFOO/SAC300'),('X2','TFOO/C10'),('X3','TFOO/Cstar')]:specs.append((candidate,'TFOO-200',cell,'TFOO/cvx','against_TFOO'))
  for col in ('cvx','SAC300','C10','Cstar'):specs.append(('TFOO/'+col,'SD200/'+col,'TFOO/'+col,'SD200/'+col,'same_lower'))
  for upper in ('SD200','TFOO'):
   for col in ('SAC300','C10','Cstar'):specs.append((upper+'/'+col,upper+'/cvx',upper+'/'+col,upper+'/cvx','same_upper'))
  for candidate,reference,cc,rc,kind in specs:
   c=cells[cc];r=cells[rc];p=paired(c,r);a=np.array([c[str(i)]['truncated_delay_s'] for i in range(1,21)]);b=np.array([r[str(i)]['truncated_delay_s'] for i in range(1,21)]);d=a-b;nz=d!=0
   if nz.any():
    w=stats.wilcoxon(d,alternative='two-sided',zero_method='wilcox');t=stats.ttest_rel(a,b,alternative='two-sided');sign=stats.binomtest(p['wins'],int(nz.sum()),.5,alternative='two-sided')
    stat=dict(wilcoxon_p=float(w.pvalue),wilcoxon_statistic=float(w.statistic),t_p=float(t.pvalue) if np.isfinite(t.pvalue) else None,t_statistic=float(t.statistic) if np.isfinite(t.statistic) else str(t.statistic),sign_p=float(sign.pvalue))
   else:stat=dict(wilcoxon_p=1.,wilcoxon_statistic=0.,t_p=1.,t_statistic=0.,sign_p=1.,all_differences_zero=True)
   comparisons.append(dict(context=context,candidate=candidate,reference=reference,category=kind,label='dev，非确认性',candidate_mean_s=p['candidate_mean_s'],reference_mean_s=p['reference_mean_s'],relative_difference_percent=100*(p['candidate_mean_s']/p['reference_mean_s']-1),wins=p['wins'],nonzero_differences=int(nz.sum()),pairs=p['pairs'],**stat))
 result=dict(label='dev，非确认性',tables=tables,comparisons=comparisons,aggregate_per_scenario=contexts['5-seed']);put(NEW/'cross/summary.json',result);save_csv(NEW/'cross/comparisons.csv',[{k:v for k,v in r.items() if k!='pairs'} for r in comparisons]);return result

def dev_test_analysis(summaries):
 rows=[];ranks=[];source=[]
 mapping={'Proposed':'S3-c10','TwoStage':'E2','UpperCVX':'E5','TFOO-200':'TFOO-200'};oldnames={'Proposed':'Proposed','TwoStage':'Abl_TwoStage','UpperCVX':'Abl_UpperCVX','TFOO-200':'TFOO'}
 for seed in SEEDS:
  p=(T4101 if seed==4101 else OUTM/f'seed_{seed}/test')/'test_summary.json';old=read(p);source.append(dict(seed=seed,path=str(p),sha256=sha(p)))
  for name,group in mapping.items():
   testmean=avg([r['truncated_delay_s'] for r in old['per_scenario'][oldnames[name]].values()]);rows.append(dict(seed=seed,group=name,dev_mean_s=summaries[(seed,group)]['mean_s'],test_mean_s=testmean,label='事后'))
  ss=[r for r in rows if r['seed']==seed];ranks.append(dict(seed=seed,dev_ranking=[r['group'] for r in sorted(ss,key=lambda r:r['dev_mean_s'])],test_ranking=[r['group'] for r in sorted(ss,key=lambda r:r['test_mean_s'])]))
 r=stats.spearmanr([r['dev_mean_s'] for r in rows],[r['test_mean_s'] for r in rows]);result=dict(label='事后，仅从已有 test 汇总读取；本任务 test 窗口数=0',rows=rows,rankings=ranks,spearman=dict(cells=20,rho=float(r.statistic),pvalue=float(r.pvalue)),test_sources=source,special=[r for r in rows if (r['seed'],r['group']) in ((4103,'Proposed'),(4102,'TwoStage'))])
 put(NEW/'cross/dev_test_posthoc.json',result);save_csv(NEW/'cross/dev_test_posthoc.csv',rows);return result
