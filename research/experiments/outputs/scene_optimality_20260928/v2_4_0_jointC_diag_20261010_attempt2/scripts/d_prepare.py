from d_guard import install
install()
from j_common import *
from hrl_v2.checkpoint import identity
from hrl_v2.evaluate import gate_manifest

JC=WT/'outputs/lite_v2_4_0/joint_sync_pilot_20261009_131658'
R=MAIN/'outputs/scene_optimality_20260928'
P=R/'v2_4_0_randpre_pilot_20261005'
protected={}; inventory={}; references={}; groups=[]

def protect(p):
 p=Path(p).resolve(); check(p.is_file(),'input_exists',str(p)); protected[str(p)]=sha(p); return read(p)

def checkpoint(key,p,source,field,expected=None):
 p=Path(p).resolve(); actual=sha(p)
 if expected: equal(actual,expected,'checkpoint_historical_sha_'+key)
 inventory[key]=dict(path=str(p),sha256=actual,bytes=p.stat().st_size,source=str(source),source_field=field)
 protected[str(p)]=actual
 return str(p)

def anchor(p):
 d=protect(p)
 for w in d['window_files']:
  wp=Path(w['path']);check(wp.is_file(),'anchor_window_exists',str(wp))
  equal(sha(wp),w['sha256'],'anchor_window_sha_'+str(wp));protected[str(wp.resolve())]=w['sha256']
 return str(p)

def ref(key,p,value,digits):
 d=protect(p);equal(round(d['mean_s'],digits),round(value,digits),'reference_'+key)
 references[key]=dict(mean_s=d['mean_s'],source=str(p),expected=value,digits=digits)
 return d

def add(label,seed,upper,lower=None,route=None,upper_kind=None,anchors=(),episode=None):
 groups.append(dict(label=label,seed=seed,upper=inventory[upper]['path'],lower=inventory[lower]['path'] if lower else None,route=route,upper_kind=upper_kind,anchors=[anchor(Path(x)) for x in anchors],episode=episode,diagnostic_only=True,eligible_for_selection=False))

def main():
 start=time.perf_counter()
 creation=protect(CPIL/'creation_record.json');check(creation['attempt2_absent_before_creation'],'attempt2_initially_absent')
 old_manifest=protect(CPIL/'old_attempt_frozen_manifest.json')
 for p,h in old_manifest['files'].items():
  equal(sha(p),h,'old_attempt_unchanged_before_'+p);protected[p]=h
 git=git_states();gpu=gpu_check();assert_imports()
 jc=protect(JC/'contract.json');equal(identity(WT,CONFIG),jc['identity'],'current_source_identity')
 old=protect(NEW/'preflight.json')
 reports={};logsets=[];times=[]
 for seed in (4103,4104):
  rp=JC/f'seed_{seed}/{seed}report.json';reports[seed]=protect(rp)
  for row in reports[seed]['main_table']:
   n=row['episode'];checkpoint(f'JC-{seed}-{n}',row['checkpoint_path'],rp,f'main_table.episode={n}.checkpoint_path',row['checkpoint_sha256'])
   sm=protect(row['summary_path']);times.append(sm['wall_seconds'])
  for label in ('TFOO','SD200',f'C{6 if seed==4103 else 8}'):
   entry=old['checkpoints'][str(seed)][label];checkpoint(f'{label}-{seed}',entry['path'],NEW/'preflight.json',f'checkpoints.{seed}.{label}.path',entry['sha256'])
  base=NEW/f'summaries/seed_{seed}'
  ref(f'TFOO-{seed}',base/'TFOO-200.json',.777382062196 if seed==4103 else .762375885110,12)
  ref(f'v1-{seed}',base/f'S3-c{6 if seed==4103 else 8}.json',.780017445356 if seed==4103 else .767044685586,12)
  ref(f'SD-{seed}',base/'E5.json',.824925236737 if seed==4103 else .893591794720,12)
  ref(f'JC-selected-{seed}',Path(reports[seed]['selected']['summary_path']),1.024 if seed==4103 else .803,3)
  cvxp=Path(reports[seed]['selected']['summary_path']).parent/'cvx/summary.json'
  cvx=protect(cvxp);references[f'JC-selected-cvx-{seed}']=dict(mean_s=cvx['mean_s'],source=str(cvxp))
  logsets.append(dict(label=f'JC-{seed}',learning=[reports[seed]['training_result']['learning_path']],train=[reports[seed]['training_result']['train_metrics_path']],episode_key='global_episode',min_episode=201))
  tr=Path(inventory[f'TFOO-{seed}']['path']).parent.parent
  logsets.append(dict(label=f'TFOO-{seed}',learning=[str(tr/'learning.jsonl')],train=[str(tr/'train_metrics.jsonl')],episode_key='episode',min_episode=1))
 pp=P/'seed_4104/step4_attempt1/result.json';pu=protect(pp)
 u200=next(p for p in pu['checkpoints'] if Path(p).name=='U200.pt')
 checkpoint('PU-4104',u200,pp,'checkpoints.U200.pt')
 pu_anchor=P/'seed_4104/evaluations/D-b_U200_cvx/summary.json';ref('PU-4104',pu_anchor,.757051,6)
 for n,v in ((320,1.129),(340,5.892)):ref(f'JC-4103-{n}',JC/f'seed_4103/evaluations/episode-{n:04d}/summary.json',v,3)
 logsets.append(dict(label='PU-4104',learning=[str(P/f'seed_4104/{st}/learning.jsonl') for st in ('step2_attempt1','step4_attempt1')],train=[str(P/f'seed_4104/{st}/train_metrics.jsonl') for st in ('step2_attempt1','step4_attempt1')],episode_key='episode',min_episode=1))
 for item in logsets:
  po=[];fields=set();epfields=set();trainfields=set()
  for p in item['learning']:
   p=Path(p);protected[str(p)]=sha(p)
   with p.open(encoding='utf-8-sig') as f:
    for line in f:
     if '"ppo"' not in line:continue
     d=json.loads(line)
     if not isinstance(d.get('ppo'),list):continue
     po.append(d);fields.update(d);epfields.update(k for e in d['ppo'] for k in e)
  for p in item['train']:
   p=Path(p);protected[str(p)]=sha(p)
   with p.open(encoding='utf-8-sig') as f:
    for line in f:
     if line.strip():trainfields.update(json.loads(line))
  check(len(po)==40,'PPO_update_count_'+item['label'],len(po))
  item.update(update_count=len(po),learning_fields=sorted(fields),epoch_fields=sorted(epfields),train_fields=sorted(trainfields),unrecorded_fields=['clip_fraction','policy_std','policy_entropy','value_loss','explained_variance','advantage_mean','advantage_std'],actor_steps='from recorded cumulative actor counters if available; otherwise 未记录')
 a=lambda n:JC/f'seed_4103/evaluations/episode-{n:04d}/summary.json'
 for label,u,l,route,anc in [('A1',320,320,None,[a(320)]),('A2',340,340,None,[a(340)]),('A3',320,340,None,[]),('A4',340,320,None,[]),('A5',320,None,'cvx',[]),('A6',340,None,'cvx',[])]:add(label,4103,f'JC-4103-{u}',f'JC-4103-{l}' if l else None,route,anchors=anc,episode=u)
 for seed in (4104,4103):
  for n in range(220,401,20):
   if seed==4103 and n in (320,340):continue
   anc=[JC/f'seed_{seed}/evaluations/episode-{n:04d}/cvx/summary.json'] if n==reports[seed]['selected']['episode'] else []
   add(f'B-{seed}-{n}',seed,f'JC-{seed}-{n}',route='cvx',anchors=anc,episode=n)
 for seed in (4103,4104):
  base=NEW/f'summaries/seed_{seed}';cy=6 if seed==4103 else 8
  add(f'K-TFOO-{seed}',seed,f'TFOO-{seed}',route='cvx',upper_kind='TFOO',anchors=[base/'TFOO-200.json'])
  add(f'K-v1-{seed}',seed,f'C{cy}-{seed}',f'C{cy}-{seed}',anchors=[base/f'S3-c{cy}.json'])
  add(f'K-SD-{seed}',seed,f'SD200-{seed}',route='cvx',anchors=[base/'E5.json'])
  n=reports[seed]['selected']['episode'];add(f'K-JC-{seed}',seed,f'JC-{seed}-{n}',f'JC-{seed}-{n}',anchors=[reports[seed]['selected']['summary_path']],episode=n)
 add('K-PU-4104',4104,'PU-4104',route='cvx',anchors=[pu_anchor])
 equal(len(groups),33,'group_count')
 disk=shutil.disk_usage(CPIL).free
 # 1,320 rollouts x <=2 MiB compressed-unneeded scalar/window JSON + 1 GiB figures, logs, staging; round to 4 GiB peak.
 peak=4*1024**3;check(disk>=max(5*1024**3,2*peak),'disk_free_at_least_5GiB_and_2x_peak',dict(free=disk,peak=peak))
 estimated_base=33*float(np.mean(times));estimate=estimated_base*2.5+7200
 check(estimate<=12*3600,'estimated_total_at_most_12h',estimate)
 for p in (CONFIG,SCENE,JC/'j_eval.py',JC/'j_common.py',PRIOR/'cp_eval.py'):protected[str(p.resolve())]=sha(p)
 c=dict(schema='jointC_diagnostic_attempt2',created_at=now(),status='frozen',identity=jc['identity'],paths=dict(JC=str(JC),XD=str(CPIL),R=str(R),WT=str(WT),MAIN=str(MAIN),NEW=str(NEW),PILOT=str(P),ATT3=str(PRIOR),WORK=str(WORK),OUTM=str(OUTM)),git_before=git,gpu_gate='query-compute-apps plus driver process types; graphics recorded, other compute blocks',gpu_check=gpu,checkpoints=inventory,references=references,groups=groups,seed_order=[4103,4104],evaluation=dict(manifest=gate_manifest(20),streams=[1,2],deterministic=True,metric='truncated_delay_s',workers=6,device='cuda:0',anchor_comparison='full per-window metrics exact; both streams exact; no test scenes'),logsets=logsets,protected_inputs=protected,resources=dict(free_bytes=disk,estimated_peak_bytes=peak,historical_group_seconds=times,estimated_base_seconds=estimated_base,instrumentation_multiplier_allowance=2.5,preparation_reporting_allowance_seconds=7200,estimated_total_seconds=estimate),constraints=dict(train=False,source_edits=False,checkpoint_edits=False,diagnostic_only=True,selection=False,test_scenes=False,commit=False,merge=False,write_root=str(CPIL)),script_provenance=dict(j_eval=str(JC/'j_eval.py'),j_common=str(JC/'j_common.py'),common_only_change='CPIL points to XD parent of scripts; extended Windows path guard preserved'),prior_stop_preserved=True,implementation_corrections_limit=None,same_error_max_failed_repairs=3,task_timeout=protect(CPIL/'historical_timing.json'))
 check(not (CPIL/'contract.json').exists(),'new_contract')
 put(CPIL/'contract.json',c);textfile(CPIL/'contract.sha256',sha(CPIL/'contract.json')+'\n')
 put(CPIL/'step0/passed.json',dict(status='passed',time=now(),wall_seconds=time.perf_counter()-start,checkpoint_count=len(inventory),protected_input_count=len(protected),group_count=len(groups),estimated_hours=estimate/3600,checks=CHECKS))
 put(CPIL/'progress.json',dict(status='preparing',stage='step0_passed',completed_groups=0,total_groups=33,time=now()))
 print(json.dumps(dict(status='passed',checkpoints=len(inventory),groups=len(groups),estimated_hours=estimate/3600,free_gib=disk/1024**3),ensure_ascii=False))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure('step0',e)
  put(CPIL/'step0/failure.json',dict(time=now(),error=type(e).__name__,message=str(e)))
  raise
