# Read-only log aggregation; statistics use recorded fields and the final attempt.
from diag_shared import *
from collections import defaultdict

def flatnumeric(value,prefix=''):
 out={}
 if isinstance(value,dict):
  for k,v in value.items():out.update(flatnumeric(v,(prefix+'.' if prefix else '')+str(k)))
 elif isinstance(value,list):
  for i,v in enumerate(value):out.update(flatnumeric(v,(prefix+'.' if prefix else '')+str(i)))
 elif isinstance(value,(int,float,bool)) and np.isfinite(value):out[prefix]=float(value)
 return out

def stage_logs(seed,stage,root):
 trains=jsonrows(root/'train_metrics.jsonl');learns=jsonrows(root/'learning.jsonl')
 if stage=='stage1':trains=[r for r in trains if r.get('global_episode',r.get('episode',0))<=300];learns=[r for r in learns if r.get('global_episode',r.get('episode',0))<=300]
 episodes=defaultdict(list);orders=defaultdict(list);bylearn=defaultdict(list)
 for r in trains+learns:
  g=r.get('global_episode',r.get('episode'));attempt=r.get('attempt_id','<未记录>')
  if attempt not in orders[g]:orders[g].append(attempt)
 for r in trains:episodes[r.get('global_episode',r.get('episode'))].append(r)
 for r in learns:bylearn[r.get('global_episode',r.get('episode'))].append(r)
 table=[];fields=set();learningfields=set();ppo_batches=[]
 for g in sorted(episodes):
  attempt=orders[g][-1];trs=[r for r in episodes[g] if r.get('attempt_id','<未记录>')==attempt];r=trs[-1] if trs else episodes[g][-1]
  row=dict(seed=seed,stage=stage,episode=r.get('episode'),global_episode=g,attempt_id=attempt)
  for k in ['truncated_delay_s','weighted_completed_delay_s','unfinished_parents','oldest_unfinished_age_s','cost_per_s','energy_j','stage_delay_s','alpha','actor_relative_entropy','critic_residual_mse','priority','sac_delta','work_seconds','checkpoint_seconds','evaluation_seconds','wall_seconds']:
   if k in r:row.update(flatnumeric(r[k],k))
  sums=defaultdict(list);ppo=defaultdict(list);eventrows=skipped=actorupdates=0
  for l in bylearn[g]:
   if l.get('attempt_id','<未记录>')!=attempt:continue
   learningfields.update(flatnumeric(l).keys())
   if 'critics' in l:
    for c in l['critics']:
     for k,v in flatnumeric(c,'critic').items():sums[k].append(v)
   if isinstance(l.get('actor'),dict):
    actorupdates+=1
    for k,v in flatnumeric(l['actor'],'actor').items():sums[k].append(v)
   if 'skipped' in l:eventrows+=1;skipped+=int(l['skipped'])
   if 'ppo' in l:
    ppo_batches.append(dict(global_episode=g,episode=l.get('episode'),attempt_id=attempt,epochs=len(l['ppo'])))
    for p in l['ppo']:
     for k,v in flatnumeric(p,'ppo').items():ppo[k].append(v)
  row.update(learning_event_rows=eventrows,learning_skipped_rows=skipped,learning_actor_updates=actorupdates)
  for k,v in sums.items():row[k+'.mean']=avg(v);row[k+'.max']=float(max(v))
  for k,v in ppo.items():row[k+'.mean']=avg(v)
  fields.update(row);table.append(row)
 save_csv(NEW/f'logs_analysis/seed_{seed}/{stage}.csv',table)
 duplicates=[dict(global_episode=g,attempt_ids=aa,selected_attempt=aa[-1]) for g,aa in sorted(orders.items()) if len(aa)>1]
 counts=None;last=root/'episodes/episode-0200.json'
 if stage in ('stage2','TFOO') and last.exists():counts=read(last).get('ppo')
 meta=dict(seed=seed,stage=stage,run=str(root),train_metrics_rows=len(trains),learning_rows=len(learns),statistics_rows=len(table),recorded_csv_fields=sorted(fields),recorded_learning_numeric_fields=sorted(learningfields),multiple_attempts=duplicates,selection='按文件顺序最后一个 attempt；该 attempt 的 train_metrics 最后一行',ppo_batches=len(ppo_batches),ppo_batch_records=ppo_batches,final_ppo_counts=counts)
 put(NEW/f'logs_analysis/seed_{seed}/{stage}_metadata.json',meta)
 return table,meta

def blocks(table,width):
 result=[]
 for i in range(0,len(table),width):
  rr=table[i:i+width];keys=set(k for r in rr for k,v in r.items() if isinstance(v,(float,int)) and k not in ('seed','global_episode','episode'))
  row={k:avg([r.get(k) for r in rr]) for k in sorted(keys)};row.update(first_global=rr[0]['global_episode'],last_global=rr[-1]['global_episode'],rows=len(rr));result.append(row)
 return result

def log_analysis():
 data={};metadata={};stage3={};stage1bins={};upperbins={};inventory=[];diagnoses={};g1={}
 for seed in SEEDS:
  data[str(seed)]={};metadata[str(seed)]={}
  for stage,root in roots(seed).items():
   table,meta=stage_logs(seed,stage,root);data[str(seed)][stage]=table;metadata[str(seed)][stage]=meta;print('LOGS_READ',seed,stage,len(table),flush=True)
  table=data[str(seed)]['stage3'];bs=blocks(table,10);check(len(bs)==10,'stage3_ten_blocks',dict(seed=seed,blocks=len(bs)))
  stage3[str(seed)]=[dict(cycle=i+1,global_first=b['first_global'],global_last=b['last_global'],delay_s=b.get('truncated_delay_s'),entropy=b.get('actor_relative_entropy'),critic_residual=b.get('critic_residual_mse'),critic_mse_q1=b.get('critic.residual_mse.0.mean'),critic_mse_q2=b.get('critic.residual_mse.1.mean'),target_mean=b.get('critic.target_mean.mean'),actor_gradient=b.get('actor.gradient_norm.mean'),alpha=b.get('alpha')) for i,b in enumerate(bs)]
  stage1bins[str(seed)]=blocks(data[str(seed)]['stage1'],25)
  upperbins[str(seed)]={stage:blocks(data[str(seed)][stage],10) for stage in ('stage2','TFOO')}
  run=roots(seed)['stage1'];dp=run/'diagnosis_required.json';diagnoses[str(seed)]=read(dp) if dp.exists() else '缺失';gp=run/'gates/episode-0300.json';g1[str(seed)]=read(gp) if gp.exists() else '缺失'
  base=B4101 if seed==4101 else WORK
  for label in ('NC','TOO','TFOO'):
   for folder in ('checkpoints','snapshots'):
    root=base/f'runs/{label}/seed_{seed}/{folder}'
    if root.exists():
     for p in sorted(root.iterdir()):
      if p.is_file():inventory.append(dict(seed=seed,baseline=label,folder=folder,name=p.name,bytes=p.stat().st_size,path=str(p)))
 c3=data['4103']['stage3'];threshold=2*avg([r.get('truncated_delay_s') for r in c3[:90]]);first=next((i for i,r in enumerate(c3) if r['truncated_delay_s']>threshold),None)
 onset=dict(posthoc_definition='第一个时延超过 c1–c9 逐轮时延均值两倍的轮次',c1_c9_mean_s=threshold/2,threshold_s=threshold,first_global_episode=c3[first]['global_episode'] if first is not None else None,first_delay_s=c3[first]['truncated_delay_s'] if first is not None else None,preceding_five=c3[max(0,first-5):first] if first is not None else [],onset_row=c3[first] if first is not None else None,c9_c10_rows=c3[80:100])
 alpha_halves={str(seed):{name:dict(delay_s=avg([r['truncated_delay_s'] for r in rr]),entropy=avg([r.get('actor_relative_entropy') for r in rr]),critic_residual=avg([r.get('critic_residual_mse') for r in rr]),target_mean=avg([r.get('critic.target_mean.mean') for r in rr]),actor_gradient=avg([r.get('actor.gradient_norm.mean') for r in rr]),alpha_values=sorted(set(r['alpha'] for r in rr))) for name,rr in [('c1-c5',data[str(seed)]['stage3'][:50]),('c6-c10',data[str(seed)]['stage3'][50:])]} for seed in SEEDS}
 result=dict(data=data,metadata=metadata,stage3_blocks=stage3,stage1_25_bins=stage1bins,upper_10_bins=upperbins,checkpoint_inventory=inventory,diagnosis_required=diagnoses,G1=g1,onset4103=onset,alpha_halves=alpha_halves)
 put(NEW/'logs_analysis/summary.json',result);save_csv(NEW/'logs_analysis/checkpoint_inventory.csv',inventory)
 CHECKS['logs']=dict(status='PASS',stage_CSVs=20,multiple_attempt_episodes=sum(len(meta['multiple_attempts']) for seed in metadata.values() for meta in seed.values()))
 plot_logs(result);return result

def plot_logs(logs):
 os.environ['MPLCONFIGDIR']=str(NEW/'mplconfig')
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError as e:
  put(NEW/'plot_status.json',dict(status='skipped',reason=str(e)));return
 for seed in SEEDS:
  r=logs['data'][str(seed)]['stage3'];fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
  for ax,k in zip(axes,['truncated_delay_s','actor_relative_entropy','critic_residual_mse','actor.gradient_norm.mean']):
   ax.plot([x['global_episode'] for x in r],[x.get(k,np.nan) for x in r]);ax.set_ylabel(k)
   for block in logs['stage3_blocks'][str(seed)]:ax.axvline(block['global_first'],color='gray',alpha=.2)
   ax.axvline(511,color='red',linestyle='--',label='alpha=0 at global 511');ax.grid(alpha=.2)
  axes[-1].set_xlabel('global episode');fig.tight_layout();fig.savefig(NEW/f'logs_analysis/stage3_{seed}.png',dpi=150);plt.close(fig)
 put(NEW/'plot_status.json',dict(status='PASS',stage3_figures=5))
