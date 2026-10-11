# v2.4.0 seeds 4102–4105 全流程与五 seed 汇总

## 检查结果

启动检查：全部通过。validation：passed。

- RP-W: passed；墙钟 1.5334200999999998 s。
- RP-S2: passed；墙钟 193.43507 s。
- RP-S3: passed；墙钟 104.05985270000001 s。
- RP-T: passed；墙钟 162.843553 s。
- RP-REF: passed；墙钟 28.2517246 s。

## 每seed硬检查

- seed 4102 reference: passed。
- seed 4102 stage1: passed。
- seed 4102 stage2: passed。
- seed 4102 stage3: passed。
- seed 4102 baselines: passed。
- seed 4102 test: passed。
- seed 4103 reference: passed。
- seed 4103 stage1: passed。
- seed 4103 stage2: passed。
- seed 4103 stage3: passed。
- seed 4103 baselines: passed。
- seed 4103 test: passed。
- seed 4104 reference: passed。
- seed 4104 stage1: passed。
- seed 4104 stage2: passed。
- seed 4104 stage3: passed。
- seed 4104 baselines: passed。
- seed 4104 test: passed。
- seed 4105 reference: passed。
- seed 4105 stage1: passed。
- seed 4105 stage2: passed。
- seed 4105 stage3: passed。
- seed 4105 baselines: passed。
- seed 4105 test: passed。

## 驱动差异逐hunk对照

### stage2_seed.py

来源：D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_sacdet_r2\sacdet_4101.py

@@ -1,9 +1,12 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D3: 删除随机SAC对照及比较分支；复现由RP-S2执行
- D4: 删除dev评估；保留训练与冻结检查
- D9: 所有子进程导入检查、完整祖先链GPU排除、后检记录

```diff
+# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_sacdet_r2\sacdet_4101.py; changes restricted to seed/path and approved D1-D9.
-import torch,hrl_v2
+import torch,hrl_v2,hrl_uav
+from common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports
+assert_imports()
```

@@ -16,10 +19,10 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D3: 删除随机SAC对照及比较分支；复现由RP-S2执行
- D4: 删除dev评估；保留训练与冻结检查
- D9: 所有子进程导入检查、完整祖先链GPU排除、后检记录

```diff
-OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'
+OUT=envpath('OUT',stage_path('stage2'))
-P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
-B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'
+P=envpath('P',run_path())
+B=envpath('VALIDATION_ROOT',WORK)
```

@@ -35,7 +38,7 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D3: 删除随机SAC对照及比较分支；复现由RP-S2执行
- D4: 删除dev评估；保留训练与冻结检查
- D9: 所有子进程导入检查、完整祖先链GPU排除、后检记录

```diff
- STAGE='preflight';record=dict(status='checking',previous_failed_attempt=str(PREVIOUS),previous_failure_reason='GPU 检查未排除自身进程')
+ STAGE='preflight';record=dict(status='checking')
```

@@ -43,70 +46,43 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D3: 删除随机SAC对照及比较分支；复现由RP-S2执行
- D4: 删除dev评估；保留训练与冻结检查
- D9: 所有子进程导入检查、完整祖先链GPU排除、后检记录

```diff
- command="Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4"
- raw=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,encoding='utf-8',errors='replace')
- procs=json.loads(raw) if raw.strip() else [];procs=procs if isinstance(procs,list) else [procs]
- gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
- gpu_rows=list(csv.DictReader(gpu.splitlines()));gpu_pids={int(next(iter(row.values())).strip()) for row in gpu_rows}
- excluded={os.getpid(),os.getppid()};keywords=('hrl_v2','train','launch','diag','evaluate','paper')
- other=[]
- for proc in procs:
-  if int(proc['ProcessId']) in excluded:continue
-  cmd=(proc.get('CommandLine') or '').lower()
-  if int(proc['ProcessId']) in gpu_pids or any(k in cmd for k in keywords):other.append(proc)
- record.update(cim_raw=raw,python_processes=procs,nvidia_smi_raw=gpu,nvidia_smi_rows=gpu_rows,excluded_pids=sorted(excluded),other_training_processes=other)
- atomic_json(OUT/'preflight.json',record)
- require(not other,'other_training_processes',other)
+ record.update(**gpu_check());atomic_json(OUT/'preflight.json',record)
-  saved=load(P/'checkpoints'/filename,binding);learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner']);hs[label]=hashes(learner)
+  saved=load(P/'checkpoints'/filename,binding);learner=Learner(SEED,device,config=config);learner.load_state_dict(saved['learner']);hs[label]=hashes(learner)
- atomic_json(OUT/'contract.json',dict(previous_failed_attempt=str(PREVIOUS),previous_failure_reason='GPU 检查未排除自身进程',identity=binding,config_path=str(CONFIG),seed=4101,source_checkpoint=str(P/'checkpoints/episode-0300-evaluated.pt'),source_checkpoint_sha256=digest(P/'checkpoints/episode-0300-evaluated.pt'),runtime=record['runtime'],created_at=time.strftime('%Y-%m-%d %H:%M:%S'),initial_hashes=hs['evaluated'],upper_phases=record['upper_phases'],control_episodes=25,sacdet_episodes=200,ppo_batches=40,only_collection_protocol_difference='SAC lower deterministic decoding; PPO upper remains stochastic',implementation_repair_limit=1,auto_resume=False))
+ atomic_json(OUT/'contract.json',dict(identity=binding,config_path=str(CONFIG),seed=SEED,source_checkpoint=str(P/'checkpoints/episode-0300-evaluated.pt'),source_checkpoint_sha256=digest(P/'checkpoints/episode-0300-evaluated.pt'),runtime=record['runtime'],created_at=time.strftime('%Y-%m-%d %H:%M:%S'),initial_hashes=hs['evaluated'],upper_phases=record['upper_phases'],control_episodes=0,sacdet_episodes=200,ppo_batches=40,only_collection_protocol_difference='SAC lower deterministic decoding; PPO upper remains stochastic',implementation_repair_limit=1,auto_resume=False))
- state=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding,completed_episode=n,pending_evaluation=None,episode=n,learner=learner.state_dict(),parameter_hashes=hs,resumable=False,kind=kind,seed=4101)
+ state=dict(schema=SCHEMA,safe_episode_boundary=True,identity=binding,completed_episode=n,pending_evaluation=None,episode=n,learner=learner.state_dict(),parameter_hashes=hs,resumable=False,kind=kind,seed=SEED)
-
-def original_ppo():
- rows={}
- for line in (P/'learning.jsonl').read_text(encoding='utf-8').splitlines():
-  row=json.loads(line)
-  if 'ppo' in row:rows.setdefault(row['global_episode'],[]).append(row)
- return rows
- learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner'])
+ learner=Learner(SEED,device,config=config);learner.load_state_dict(saved['learner'])
- phases=upper_phases()[:total];batch=[];comparisons=[];oldppo=original_ppo() if lower_stochastic else {};first_difference=None;updates_count=0
- def compare(episode,key,new,old):
-  nonlocal first_difference
-  equal=new==old;row=dict(episode=episode,field=key,equal=equal)
-  if not equal:
-   row.update(candidate=new,reference=old)
-   if first_difference is None:first_difference=row.copy()
-  comparisons.append(row)
+ phases=upper_phases()[:total];batch=[];updates_count=0
-  started=time.perf_counter();streams=Streams(4101,p.stage,p.cycle,p.block_episode);adapter=PhysicsAdapter(scene,timing=clocks)
-  external={k+'_seed':derive(4101,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};adapter.reset(**external)
+  started=time.perf_counter();streams=Streams(SEED,p.stage,p.cycle,p.block_episode);adapter=PhysicsAdapter(scene,timing=clocks)
+  external={k+'_seed':derive(SEED,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};adapter.reset(**external)
```

@@ -117,13 +93,6 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D3: 删除随机SAC对照及比较分支；复现由RP-S2执行
- D4: 删除dev评估；保留训练与冻结检查
- D9: 所有子进程导入检查、完整祖先链GPU排除、后检记录

```diff
-  if lower_stochastic:
-   old=read(P/'episodes'/f'episode-{p.global_episode:04d}.json')
-   for key in ('event_sha256','truncated_delay_s'):compare(n,key,metrics[key],old['metrics'][key])
-   compare(n,'reward_physical',reward,old['train_metric']['reward_physical'])
-   if logs is not None:
-    expected=oldppo.get(p.global_episode,[]);compare(n,'ppo_log_row_count',len(expected),1)
-    if len(expected)==1:compare(n,'ppo_update_log',logs,expected[0]['ppo'])
```

@@ -132,30 +101,17 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D3: 删除随机SAC对照及比较分支；复现由RP-S2执行
- D4: 删除dev评估；保留训练与冻结检查
- D9: 所有子进程导入检查、完整祖先链GPU排除、后检记录

```diff
- if lower_stochastic:
-  path=P/'snapshots/episode-0325.pt';side=read(path.with_suffix('.json'));require(side['sha256']==digest(path),'control_reference_snapshot_sha')
-  old=torch.load(path,map_location='cpu',weights_only=False);reference=Learner(4101,device,config=config);reference.load_state_dict(old['learner']);rh=hashes(reference)
-  compare(25,'ppo_actor_hash',hs['ppo_actor'],rh['ppo_actor']);compare(25,'ppo_value_hash',hs['ppo_value'],rh['ppo_value']);del old,reference
-  equiv=dict(status='PASS' if first_difference is None else 'FAIL',first_difference=first_difference,comparisons=comparisons,metadata_note='PPO update entries compared in full; original attempt_id is an old run marker, not a PPO output')
-  atomic_json(OUT/'loop_equiv.json',equiv);CHECKS['LOOP_EQUIV']=equiv['status'];print('LOOP_EQUIV',equiv['status'],flush=True)
- result=dict(status='complete',episodes=total,ppo_updates=updates_count,sac_actor_hash_unchanged=True,sac_counts_unchanged=True,initial_hashes=initial_hashes,final_hashes=hs,wall_seconds=time.perf_counter()-start)
+ result=dict(status='complete',episodes=total,ppo_updates=updates_count,sac_actor_hash_unchanged=True,sac_counts_unchanged=True,initial_hashes=initial_hashes,final_hashes=hs,initial_sac_counts=initial_counts,final_sac_counts=dict(learner.sac.counts),final_ppo_counts=dict(learner.ppo.counts),wall_seconds=time.perf_counter()-start)
- parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('preflight','train','all'),default='all');args=parser.parse_args()
- if args.stage=='preflight':return
- control=train_run('control_stoch',True,25,config,binding,device)
- atomic_json(OUT/'training_summary.json',dict(control=control,sacdet=mainrun))
- if args.stage=='all':
-  global STAGE
-  STAGE='evaluation_and_analysis'
-  from evaluate_sacdet import finish
-  finish(config,binding,device)
- git=git_states();atomic_json(OUT/'git_after.json',git)
- print('SACDET_DIAGNOSTIC_COMPLETE',flush=True)
+ for n in (100,200):require((OUT/f'runs/sacdet/checkpoints/episode-{n:04d}.pt').is_file(),'stage2_checkpoint',n)
+ atomic_json(OUT/'training_summary.json',dict(sacdet=mainrun))
+ atomic_json(OUT/'checks.json',dict(status='passed',preflight=CHECKS,sacdet=mainrun))
+ atomic_json(OUT/'git_after.json',git_states())
-  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS));traceback.print_exc();sys.exit(1)
+  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));raise
```

### stage3_seed.py

来源：D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_stage3\stage3_4101.py

@@ -1,9 +1,12 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
+# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_stage3\stage3_4101.py; changes restricted to seed/path and approved D1-D9.
+from common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports
+assert_imports()
```

@@ -15,14 +18,13 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
-OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_stage3'
-R2=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'
-P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
-B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'
+OUT=envpath('OUT',stage_path('stage3'))
+R2=envpath('R2',stage_path('stage2'))
+P=envpath('P',run_path())
+B=envpath('VALIDATION_ROOT',WORK)
-C590=P/'checkpoints/episode-0590-complete.pt'
```

@@ -54,7 +56,7 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
- saved=load(path,binding);learner=Learner(4101,device,config=config);learner.load_state_dict(saved['learner']);return learner,saved
+ saved=load(path,binding);learner=Learner(SEED,device,config=config);learner.load_state_dict(saved['learner']);return learner,saved
```

@@ -67,17 +69,7 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
- cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4"
- raw=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True,encoding='utf-8',errors='replace')
- procs=json.loads(raw) if raw.strip() else [];procs=procs if isinstance(procs,list) else [procs]
- gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
- gpuro=list(csv.DictReader(gpu.splitlines()));gpids={int(next(iter(r.values())).strip()) for r in gpuro}
- excluded={os.getpid(),os.getppid()};other=[]
- for proc in procs:
-  pid=int(proc['ProcessId']);cmd=(proc.get('CommandLine') or '').lower()
-  if pid not in excluded and (pid in gpids or any(k in cmd for k in ('hrl_v2','train','launch','diag','evaluate','paper'))):other.append(proc)
- record.update(cim_raw=raw,python_processes=procs,nvidia_smi_raw=gpu,nvidia_smi_rows=gpuro,excluded_pids=sorted(excluded),other_training_processes=other)
- atomic_json(OUT/'preflight.json',record);require(not other,'other_training_processes',other)
+ record.update(**gpu_check());atomic_json(OUT/'preflight.json',record)
```

@@ -95,7 +87,7 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
- atomic_json(OUT/'contract.json',dict(identity=binding,source_checkpoints={label:dict(path=str(path),sha256=digest(path)) for label,path in (('SACDET-200',SD),('S300',S300),('P590',C590))},LOWER=table,expected_sac_delta=EXPECTED,seed=4101,criterion=dict(candidate='S3-c10',reference='E2',scenario_count=20,streams=[1,2],mean_less_than_s=0.912816585,minimum_wins=11,verdict_if_both='STAGE3_HELPS',otherwise='STAGE3_NOT_HELPFUL',intermediate_models='S3-c1..S3-c9 report only; no selection'),only_protocol_difference='上层冻结为 SACDET-200，因此经验池只在第一块开头因上层版本变化清空一次；其余规则与原第三阶段下层块相同。',created_at=time.strftime('%Y-%m-%d %H:%M:%S')))
+ atomic_json(OUT/'contract.json',dict(identity=binding,source_checkpoints={label:dict(path=str(path),sha256=digest(path)) for label,path in (('SACDET-200',SD),('S300',S300))},LOWER=table,expected_sac_delta=EXPECTED,seed=SEED,intermediate_models="S3-c1..S3-c9 report only; no selection",only_protocol_difference='上层冻结为 SACDET-200，因此经验池只在第一块开头因上层版本变化清空一次；其余规则与原第三阶段下层块相同。',created_at=time.strftime('%Y-%m-%d %H:%M:%S')))
```

@@ -111,8 +103,8 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
-  streams=Streams(4101,p.stage,p.cycle,p.block_episode);adapter=PhysicsAdapter(scene,timing=clocks)
-  external={k+'_seed':derive(4101,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};adapter.reset(**external)
+  streams=Streams(SEED,p.stage,p.cycle,p.block_episode);adapter=PhysicsAdapter(scene,timing=clocks)
+  external={k+'_seed':derive(SEED,p.stage,p.cycle,p.block_episode,purpose=k) for k in ('arrival','layout','load')};adapter.reset(**external)
```

@@ -132,51 +124,31 @@

- seed/path: SEED及本seed输入输出、WORK validation
- D5: 删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查
- D9: 导入检查、祖先链GPU排除及后检记录

```diff
-def loop_equiv(config,binding,device):
- global STAGE
- STAGE='LOOP_EQUIV';learner,saved=new_learner(C590,binding,config,device);replay=saved['replay'];restore_rng(saved);del saved
- timing=run_lower(learner,replay,[phase(n) for n in (591,592,593)],OUT/'runs/loop_equiv');STAGE='LOOP_EQUIV_comparison'
- checks=[];newmetrics={r['global_episode']:r for r in rows(OUT/'runs/loop_equiv/train_metrics.jsonl')};original={r['global_episode']:r for r in rows(P/'train_metrics.jsonl')}
- expected={591:dict(triggers=90,skipped=90,events=0,critic_fits=0,actor_updates=0,target_updates=0),592:dict(triggers=90,skipped=0,events=90,critic_fits=180,actor_updates=0,target_updates=90),593:dict(triggers=90,skipped=0,events=90,critic_fits=180,actor_updates=45,target_updates=90)}
- def compare(label,a,b):
-  d=firstdiff(normalize(a),normalize(b));checks.append(dict(check=label,equal=d is None))
-  if d:
-   atomic_json(OUT/'loop_equiv.json',dict(status='FAIL',first_difference=d,failed_check=label,checks=checks,**timing));require(False,'LOOP_EQUIV',dict(check=label,first_difference=d))
- for n in (591,592,593):
-  candidate=read(OUT/f'runs/loop_equiv/episodes/episode-{n:04d}.json');reference=read(P/f'episodes/episode-{n:04d}.json')
-  for k in ('event_sha256','truncated_delay_s'):compare(f'{n}.metrics.{k}',candidate['metrics'][k],reference['metrics'][k])
-  for k in ('reward_physical','sac_delta'):compare(f'{n}.train_metric.{k}',newmetrics[n][k],original[n][k])
-  compare(f'{n}.expected_sac_delta',newmetrics[n]['sac_delta'],expected[n])
- newlogs=rows(OUT/'runs/loop_equiv/learning.jsonl');oldlogs=[r for r in rows(P/'learning.jsonl') if r['global_episode'] in (591,592,593)]
- for n in (591,592,593):
-  aa=[{k:v for k,v in r.items() if k!='attempt_id'} for r in newlogs if r['global_episode']==n];bb=[{k:v for k,v in r.items() if k!='attempt_id'} for r in oldlogs if r['global_episode']==n]
-  compare(f'{n}.learning_row_count',len(aa),90);compare(f'{n}.learning',aa,bb)
- atomic_json(OUT/'loop_equiv.json',dict(status='PASS',first_difference=None,checks=checks,learning_rows=270,**timing));CHECKS['LOOP_EQUIV']='PASS';print('LOOP_EQUIV_PASS',flush=True)
- del learner,replay;gc.collect();torch.cuda.empty_cache();return timing
-
-def formal(config,binding,device,lower):
+def formal(config,binding,device,lower,phases_limit=None):
- timing=run_lower(learner,replay,lower,OUT/'runs/stage3');STAGE='formal_checks';after=hashes(learner);delta={k:v-sac[k] for k,v in learner.sac.counts.items()}
+ selected=lower if phases_limit is None else lower[:phases_limit]
+ timing=run_lower(learner,replay,selected,OUT/'runs/stage3');STAGE='formal_checks';after=hashes(learner);delta={k:v-sac[k] for k,v in learner.sac.counts.items()}
- require(delta==EXPECTED and result['skipped_global_episodes']==[411],'SAC_counts',result);CHECKS['SAC_counts']='PASS'
+ if phases_limit is None:
+  require(delta==EXPECTED and result['skipped_global_episodes']==[411],'SAC_counts',result);CHECKS['SAC_counts']='PASS'
+  side=read(OUT/'runs/stage3/checkpoints/cycle-10.json');require(side['completed_episode']==600 and side['pending_evaluation'] is None,'cycle10_sidecar')
+ metrics=rows(OUT/'runs/stage3/train_metrics.jsonl')
+ result['cycle_mean_truncated_delay_s']={str(c):sum(r['truncated_delay_s'] for r in metrics if phase(r['global_episode']).cycle//2==c)/sum(phase(r['global_episode']).cycle//2==c for r in metrics) for c in sorted({phase(r['global_episode']).cycle//2 for r in metrics})}
+ atomic_json(OUT/'training_summary.json',result)
- config,binding,device,lower=preflight();loop_equiv(config,binding,device);formal(config,binding,device,lower)
- global STAGE
- STAGE='evaluation'
- from evaluate_stage3 import finish
- finish(config,binding,device,total_start=TOTAL_START)
- print('STAGE3_COMPLETE',flush=True)
+ config,binding,device,lower=preflight();result=formal(config,binding,device,lower)
+ atomic_json(OUT/'checks.json',dict(status='passed',checks=CHECKS,result=result))
+ atomic_json(OUT/'git_after.json',git_states())
-  failure=dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError))
-  atomic_json(OUT/'failure.json',failure);traceback.print_exc();sys.exit(1)
+  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));raise
```

### reference_seed.py

来源：D:\HRL_UAV_V2_speedup\tools\run_v2_4_0_reference.py

@@ -1,3 +1,4 @@

- seed/path: master=SEED，OUTPUT=本seed reference；导入路径检查；160窗口后检和与4101只读比较
- D2: 每seed冻结自身参考，原check及聚合不变

```diff
+# Derived from D:\HRL_UAV_V2_speedup\tools\run_v2_4_0_reference.py; changes restricted to seed/path and approved D1-D9.
```

@@ -6,6 +7,10 @@

- seed/path: master=SEED，OUTPUT=本seed reference；导入路径检查；160窗口后检和与4101只读比较
- D2: 每seed冻结自身参考，原check及聚合不变

```diff
+import sys
+sys.dont_write_bytecode=True
+from common import SEED,WORK,envpath,assert_imports,REF4101,read
+assert_imports()
```

@@ -21,7 +26,7 @@

- seed/path: master=SEED，OUTPUT=本seed reference；导入路径检查；160窗口后检和与4101只读比较
- D2: 每seed冻结自身参考，原check及聚合不变

```diff
-OUTPUT=ROOT/'outputs/lite_v2_4_0'
+OUTPUT=envpath('OUT',WORK/f'references/seed_{SEED}')
```

@@ -32,7 +37,7 @@

- seed/path: master=SEED，OUTPUT=本seed reference；导入路径检查；160窗口后检和与4101只读比较
- D2: 每seed冻结自身参考，原check及聚合不变

```diff
-                        label,*COMBINATIONS[label],4101)
+                        label,*COMBINATIONS[label],SEED)
```

@@ -72,8 +77,14 @@

- seed/path: master=SEED，OUTPUT=本seed reference；导入路径检查；160窗口后检和与4101只读比较
- D2: 每seed冻结自身参考，原check及聚合不变

```diff
-    summary=freeze(CONFIG,OUTPUT/'baselines',20,12,include_greedy=True)
+    summary=freeze(CONFIG,OUTPUT/'baselines',20,12,master=SEED,include_greedy=True)
+    if not (summary['identity']==binding and summary['master']==SEED and summary['manifest']==gate_manifest(20) and summary['required_wins']==12 and checked['passed']):raise AssertionError('Reference hard checks failed')
+    same=0
+    for scenario,stream,label in jobs:
+        name=f'{label}_{scenario["scenario"]:02d}_{stream}.json'
+        same+=int(read(OUTPUT/'baselines'/name)['metrics']==read(REF4101/name)['metrics'])
+    atomic_json(OUTPUT/'checks.json',dict(status='passed',check=checked,matching_REF4101_full_metrics=same,total=160))
```

### test_seed.py

来源：D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_test\test_4101.py

@@ -1,3 +1,4 @@

- seed/path: SEED及输入/输出、身份链WORK validation、test manifest对4101检查
- D6: 8组保留原构造；去掉Alternating及600检查；只读已有PROTO；去掉dev replay及Panel A SHA
- D9: 导入及祖先链GPU检查、记录硬检查

```diff
+# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_test\test_4101.py; changes restricted to seed/path and approved D1-D9.
```

@@ -5,22 +6,24 @@

- seed/path: SEED及输入/输出、身份链WORK validation、test manifest对4101检查
- D6: 8组保留原构造；去掉Alternating及600检查；只读已有PROTO；去掉dev replay及Panel A SHA
- D9: 导入及祖先链GPU检查、记录硬检查

```diff
+from common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports,PROTO,PROTO_SHA,protocol_check,P4101
+assert_imports()
-P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
-B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900';A=B/'paper_dev_A_P600_B200'
-R2=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'
-S3=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_stage3'
-OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_test'
+P=envpath('P',run_path())
+B=envpath('B',WORK)
+R2=envpath('R2',stage_path('stage2'))
+S3=envpath('S3',stage_path('stage3'))
+OUT=envpath('OUT',stage_path('test'))
-PATHS=dict(C10=S3/'runs/stage3/checkpoints/cycle-10.pt',SD200=R2/'runs/sacdet/checkpoints/episode-0200.pt',S300=P/'checkpoints/episode-0300-evaluated.pt',P600=P/'checkpoints/episode-0600-evaluated.pt',P600_pending=P/'checkpoints/episode-0600-pending.pt',**{name:B/f'runs/{name}/seed_4101/checkpoints/episode-0200-anchor.pt' for name in ('NC','TOO','TFOO')})
-GROUPS=['Proposed','NC','CA','TOO','TFOO','Abl_TwoStage','Abl_Alternating','Abl_HoverSAC','Abl_UpperCVX']
-CONSTRUCTIONS=dict(Proposed="Policy(C10.sac.actor, C10.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",NC="comparison_policy(NC learner, scene, streams, 'moving_local', stochastic=False)",CA="comparison_policy(C10 learner, scene, streams, 'hover_equal', stochastic=False)",TOO="comparison_policy(TOO learner, scene, streams, 'moving_equal', stochastic=False)",TFOO="comparison_policy(TFOO learner, scene, streams, 'moving_cvx', stochastic=False)",Abl_TwoStage="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_Alternating="comparison_policy(P600 learner, scene, streams, 'moving', stochastic=False)",Abl_HoverSAC="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='hover', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_UpperCVX="fixed_upper_policy(SD200.ppo, scene, streams, 'cvx', stochastic=False, mode='moving')")
-OWNERS=dict(Proposed=['C10'],NC=['NC'],CA=['C10'],TOO=['TOO'],TFOO=['TFOO'],Abl_TwoStage=['S300','SD200'],Abl_Alternating=['P600'],Abl_HoverSAC=['S300','SD200'],Abl_UpperCVX=['SD200'])
+PATHS=dict(C10=S3/'runs/stage3/checkpoints/cycle-10.pt',SD200=R2/'runs/sacdet/checkpoints/episode-0200.pt',S300=P/'checkpoints/episode-0300-evaluated.pt',**{name:B/f'runs/{name}/seed_{SEED}/checkpoints/episode-0200-anchor.pt' for name in ('NC','TOO','TFOO')})
+GROUPS=['Proposed','NC','CA','TOO','TFOO','Abl_TwoStage','Abl_HoverSAC','Abl_UpperCVX']
+CONSTRUCTIONS=dict(Proposed="Policy(C10.sac.actor, C10.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",NC="comparison_policy(NC learner, scene, streams, 'moving_local', stochastic=False)",CA="comparison_policy(C10 learner, scene, streams, 'hover_equal', stochastic=False)",TOO="comparison_policy(TOO learner, scene, streams, 'moving_equal', stochastic=False)",TFOO="comparison_policy(TFOO learner, scene, streams, 'moving_cvx', stochastic=False)",Abl_TwoStage="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='moving', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_HoverSAC="Policy(S300.sac.actor, SD200.ppo.actor, None, streams, mode='hover', stochastic=False, upper_stochastic=False, deployment=scene['deployment'])",Abl_UpperCVX="fixed_upper_policy(SD200.ppo, scene, streams, 'cvx', stochastic=False, mode='moving')")
+OWNERS=dict(Proposed=['C10'],NC=['NC'],CA=['C10'],TOO=['TOO'],TFOO=['TFOO'],Abl_TwoStage=['S300','SD200'],Abl_HoverSAC=['S300','SD200'],Abl_UpperCVX=['SD200'])
```

@@ -60,33 +63,24 @@

- seed/path: SEED及输入/输出、身份链WORK validation、test manifest对4101检查
- D6: 8组保留原构造；去掉Alternating及600检查；只读已有PROTO；去掉dev replay及Panel A SHA
- D9: 导入及祖先链GPU检查、记录硬检查

```diff
- config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);validation=read(B/'validation/validation.json')
+ config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);validation=read(envpath('VALIDATION_ROOT',WORK)/'validation/validation.json')
- cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4"
- raw=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True,encoding='utf-8',errors='replace')
- procs=json.loads(raw) if raw.strip() else [];procs=procs if isinstance(procs,list) else [procs]
- gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,encoding='utf-8',errors='replace')
- gpuro=list(csv.DictReader(gpu.splitlines()));gpids={int(next(iter(r.values())).strip()) for r in gpuro};excluded={os.getpid(),os.getppid()};other=[]
- for proc in procs:
-  pid=int(proc['ProcessId']);command=(proc.get('CommandLine') or '').lower()
-  if pid not in excluded and (pid in gpids or any(k in command for k in ('hrl_v2','train','launch','diag','evaluate','paper'))):other.append(proc)
- record.update(cim_raw=raw,python_processes=procs,nvidia_smi_raw=gpu,nvidia_smi_rows=gpuro,excluded_pids=sorted(excluded),other_training_processes=other);atomic_json(OUT/'preflight.json',record)
- require(not other,'other_training_or_evaluation_processes',other);CHECKS['P4']='PASS'
- m=read(P/'test_manifest.json');require(m==manifest('test'),'test_manifest_exact')
+ record.update(**gpu_check());atomic_json(OUT/'preflight.json',record)
+ m=read(P/'test_manifest.json');require(m==manifest('test')==read(P4101/'test_manifest.json'),'test_manifest_exact')
- metadata={};panel=read(A/'summary_deterministic.json')
+ metadata={}
-  kind=name if name in ('NC','TOO','TFOO') else 'Proposed' if name in ('P600','P600_pending') else None
+  kind=name if name in ('NC','TOO','TFOO') else None
-  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(4101,device,config)
+  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(SEED,device,config)
```

@@ -94,53 +88,26 @@

- seed/path: SEED及输入/输出、身份链WORK validation、test manifest对4101检查
- D6: 8组保留原构造；去掉Alternating及600检查；只读已有PROTO；去掉dev replay及Panel A SHA
- D9: 导入及祖先链GPU检查、记录硬检查

```diff
-   require(sha==panel['identity']['checkpoints'][name],'baseline_A_checkpoint_sha',name)
- require(metadata['P600']['parameter_hashes']==metadata['P600_pending']['parameter_hashes'],'P600_pending_evaluated_hashes')
-def freeze_protocol(config,binding,m,metadata):
+def main():
+ sys.modules['test_seed']=sys.modules[__name__]
+ config,binding,device,m,metadata=preflight();freeze=dict(path=str(PROTO),sha256=protocol_check());CHECKS['protocol_hash']='PASS'
+ from evaluate_test_seed import test_panel,summarize,compare,report
- STAGE='protocol_freeze';start=time.perf_counter();require(not (OUT/'windows').exists(),'freeze_before_windows')
- stamp=now();cp=lambda name:{k:metadata[name][k] for k in ('path','sha256','parameter_hashes')}
- s3contract=read(S3/'contract.json');r2contract=read(R2/'contract.json')
- protocol=dict(schema='hrl_v2_4_0_three_stage_protocol_1',frozen_at=stamp,frozen_before_any_test_window=True,
-  code=dict(worktree=str(WT),branch='codex/v2-lite-speedup',HEAD=subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip(),identity=binding,CONFIG=dict(path=str(CONFIG),sha256=digest(CONFIG)),SCENE=dict(path=str(SCENE),sha256=digest(SCENE))),
-  history=dict(original_preregistration='600 轮：1–300 悬停训练 SAC；301–400 固定 SAC 训练 PPO；401–600 上下层每 10 轮交替',deviations=['第二阶段：上层 200 轮在 upper_phases() 身份上连续训练，SAC 冻结且确定性解码，PPO 随机采集','第三阶段：原 401–600 中 100 个下层轮次连续训练，PPO 冻结；经验池只在第一块开头清空一次','最终模型取第三阶段结束时的 learner，不再用 dev_best'],decided_after=['SACDET r2 的 dev 结果','第三阶段修复试验的 dev 结果'],note='本协议在看过 4101 dev 结果后制定；4101 test 为 test 划分的首次使用'),
-  stages=dict(stage1=dict(name='SAC 悬停训练',global_episodes='1–300',procedure='hrl_v2.run 原第一阶段，语义不变',seed_4101_end_checkpoint=cp('S300')),
-   stage2=dict(name='上层 PPO 训练（下层冻结、确定性）',episode_identities=[p.global_episode for p in upper_phases()],init='第一阶段末 learner，以及该 checkpoint 的 global CPU/CUDA RNG 状态',lower='SAC 冻结，stochastic=False',upper_collection='upper_stochastic=True',ppo_update='每 5 轮一次，共 40 批',driver=dict(path=str(R2/'sacdet_4101.py'),sha256=digest(R2/'sacdet_4101.py'),entry="train_run('sacdet', lower_stochastic=False, total=200)"),seed_4101_output=cp('SD200')),
-   stage3=dict(name='下层 SAC 微调（上层冻结）',episode_identities=s3contract['LOWER'],rules=dict(only_protocol_difference=s3contract['only_protocol_difference'],clarification='经验池按原规则在上层版本变化时清空；因上层冻结，只在第一块（global 411）开头清空一次，之后累积'),driver=dict(path=str(S3/'stage3_4101.py'),sha256=digest(S3/'stage3_4101.py')),seed_4101_output=cp('C10'))),
-  final_model=dict(rule='第三阶段 100 个下层轮次全部结束后的 learner（cycle-10）；不按 dev 挑选；cycle-1..9 只作报告曲线',seed_4101={k:metadata['C10'][k] for k in ('path','sha256')}),
-  evaluation=dict(policy='上下层均为确定性',proposed_construction=CONSTRUCTIONS['Proposed'],streams="Streams(master, episode=scenario, split='test', policy_stream=stream)",test_manifest=dict(path=str(P/'test_manifest.json'),sha256=digest(P/'test_manifest.json'),json_digest=json_digest(m)),primary_metric='截断时延；先把每个场景全部 stream 窗口池化，再对 20 个场景等权平均，配对口径同 hrl_v2.gates.paired',test_scenarios_master_independent=True),
-  baselines={name:dict(internal_name=internal,training='各自独立训练 200 个上层轮次（upper_phases 身份），配固定的确定性下层，取 episode-0200-anchor.pt',evaluation="paper_evaluation.comparison_policy(..., stochastic=False)",seed_4101=dict(path=str(PATHS[name]),sha256=metadata[name]['sha256'])) for name,internal in (('NC','moving_local'),('TOO','moving_equal'),('TFOO','moving_cvx'))},
-  ablations=dict(TwoStage='SD200 上层 + SAC-300',Alternating='P600 原交替（仅 seed 4101）',HoverSAC='SAC-300 悬停',UpperCVX='SD200 上层（与 C10 上层哈希相同）+ cvx 路由'),
-  statistics=dict(primary_endpoint="test 上 Proposed 对 TFOO-200：20 个场景的配对差，scipy.stats.wilcoxon 双侧，zero_method='wilcox'，α=0.05",also_reported='配对 t 检验（ttest_rel）、符号检验（binomtest）、相对差、胜场',secondary='Proposed 对 NC/CA/TOO，以及四个消融；只作报告，不做多重比较校正'),
-  multi_seed_plan=dict(seeds=[4102,4103,4104,4105],pipeline=['stage1：hrl_v2.run 跑到第 300 轮（G1 只报告）','stage2：同上驱动，只做 seed 参数化','stage3：同上驱动，只做 seed 参数化','NC/TOO/TFOO：按 v2.4.0 原基线协议各训练 200 个上层轮次','test 面板：与 4101 同构造（不含 Alternating）'],driver_rule='只允许 seed 参数化；其他任何差异须在启动前逐条列出并获批',aggregate_primary='4101–4105 共 5 个 seed：每个 test 场景先取 5 个 seed 的场景值均值，得到 20 个配对差，Proposed 对 TFOO 做 Wilcoxon 双侧检验，α=0.05；同时报告每个 seed 的场景等权均值、胜场，以及 Proposed 更优的 seed 数'),
-  source_contracts=dict(R2=r2contract,S3=s3contract))
- protocol['baselines']['CA']=dict(internal_name='hover_equal',description='悬停 + 均分，不用上层')
- path=OUT/'protocol_v2_4_0_three_stage.json';require(not path.exists(),'new_protocol_file');atomic_json(path,protocol)
- freeze=dict(path=str(path),sha256=digest(path),frozen_at=stamp,windows_exists_at_freeze=(OUT/'windows').exists());atomic_json(OUT/'protocol_freeze.json',freeze)
- require(not freeze['windows_exists_at_freeze'],'freeze_before_windows');CHECKS['protocol_freeze']='PASS';TIMING['protocol_seconds']=time.perf_counter()-start
- print('PROTOCOL_FROZEN',freeze['sha256'],flush=True);return freeze
-
-def main():
- sys.modules['test_4101']=sys.modules[__name__]
- config,binding,device,m,metadata=preflight();freeze=freeze_protocol(config,binding,m,metadata)
- from evaluate_test import replay_checks,test_panel,summarize,compare,report
- global STAGE
- STAGE='dev_replay';replay_checks(config,binding,freeze,metadata)
- report(summary,comparisons,freeze,git);print('TEST_PANEL_COMPLETE',flush=True)
+ report(summary,comparisons,freeze,git);atomic_json(OUT/'checks.json',dict(status='passed',checks=CHECKS));print('TEST_PANEL_COMPLETE',flush=True)
```

### evaluate_test_seed.py

来源：D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_test\evaluate_test.py

@@ -1,3 +1,4 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
+# Derived from D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_4101_test\evaluate_test.py; changes restricted to seed/path and approved D1-D9.
```

@@ -7,7 +8,9 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
-from test_4101 import WT,MAIN,OUT,P,B,A,R2,S3,CONFIG,SCENE,PATHS,GROUPS,CONSTRUCTIONS,OWNERS,CHECKS,TIMING,read,require,firstdiff,now
+from test_seed import WT,MAIN,OUT,P,B,R2,S3,CONFIG,SCENE,PATHS,GROUPS,CONSTRUCTIONS,OWNERS,CHECKS,TIMING,read,require,firstdiff,now
+from common import SEED,assert_imports
+assert_imports()
```

@@ -27,7 +30,7 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
-  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(4101,device,config)
+  learner=(UpperLearner if name in ('NC','TOO','TFOO') else Learner)(SEED,device,config)
```

@@ -39,11 +42,11 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
-  streams=Streams(4101,episode=scenario['scenario'],split=split,policy_stream=stream)
+  streams=Streams(SEED,episode=scenario['scenario'],split=split,policy_stream=stream)
-  elif group in ('NC','CA','TOO','TFOO','Abl_Alternating'):
-   owner=OWNERS[group][0];internal=dict(NC='moving_local',CA='hover_equal',TOO='moving_equal',TFOO='moving_cvx',Abl_Alternating='moving')[group]
+  elif group in ('NC','CA','TOO','TFOO'):
+   owner=OWNERS[group][0];internal=dict(NC='moving_local',CA='hover_equal',TOO='moving_equal',TFOO='moving_cvx')[group]
```

@@ -68,39 +71,23 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
-def replay_checks(config,binding,freeze,metadata):
- start=time.perf_counter();scenario=gate_manifest(20)['scenarios'][0]
- references=dict(Proposed=S3/'windows/S3-c10_01_1.json',NC=A/'deterministic_moving_local_01_1.json',CA=A/'deterministic_hover_equal_01_1.json',TOO=A/'deterministic_moving_equal_01_1.json',TFOO=A/'deterministic_moving_cvx_01_1.json',Abl_TwoStage=R2/'windows/E2_01_1.json',Abl_Alternating=A/'deterministic_moving_01_1.json',Abl_HoverSAC=R2/'windows/E3_01_1.json',Abl_UpperCVX=R2/'windows/E5_01_1.json')
- tasks=[]
- for i,group in enumerate(GROUPS,1):
-  ref=read(references[group]);tasks.append(dict(check=f'R{i}',group=group,scenario=scenario,stream=1,split='dev',identity=ref['identity'],reference=str(references[group]),replay=True))
- checks=[]
- def done(task,r):
-  old=read(task['reference']);candidate=r['result'];d=firstdiff(candidate['metrics'],old['metrics']);dd=firstdiff(candidate['diagnostics'],old['diagnostics']) if task['group'] in ('TFOO','Abl_UpperCVX') else None
-  row=dict(check=task['check'],group=task['group'],reference=task['reference'],metrics_equal=d is None,metrics_first_difference=d,diagnostics_equal=dd is None,diagnostics_checked=task['group'] in ('TFOO','Abl_UpperCVX'),diagnostics_first_difference=dd,wall_seconds=r['wall_seconds'])
-  checks.append(row);atomic_json(OUT/'replay_checks.json',dict(status='checking',rows=sorted(checks,key=lambda x:int(x['check'][1:]))))
-  require(d is None and dd is None,'dev_replay',row);CHECKS[task['check']]='PASS';print('REPLAY_PASS',task['check'],task['group'],flush=True)
- pool_tasks(tasks,done);require(len(checks)==9,'replay_check_count');TIMING['replay_seconds']=time.perf_counter()-start
- atomic_json(OUT/'replay_checks.json',dict(status='PASS',rows=sorted(checks,key=lambda x:int(x['check'][1:])),wall_seconds=TIMING['replay_seconds']))
- require(not (OUT/'windows').exists(),'no_test_window_before_replay_pass')
-
- cpids={name:metadata[name]['sha256'] for name in PATHS};contract=dict(protocol_sha256=freeze['sha256'],manifest_sha256=digest(P/'test_manifest.json'),manifest_digest=json_digest(m),constructions=CONSTRUCTIONS,checkpoint_sha256=cpids,windows=720,groups=GROUPS,created_at=now())
+ cpids={name:metadata[name]['sha256'] for name in PATHS};contract=dict(protocol_sha256=freeze['sha256'],manifest_sha256=digest(P/'test_manifest.json'),manifest_digest=json_digest(m),constructions=CONSTRUCTIONS,checkpoint_sha256=cpids,windows=640,groups=GROUPS,created_at=now())
-    item=dict(protocol='v2_4_0_three_stage_test_1',protocol_sha256=freeze['sha256'],group=group,checkpoints={name:metadata[name]['sha256'] for name in OWNERS[group]},manifest=json_digest(m),scenario=scenario,stream=stream,master=4101,split='test',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=config['objective']['reward_scale'])
+    item=dict(protocol='v2_4_0_three_stage_test_1',protocol_sha256=freeze['sha256'],group=group,checkpoints={name:metadata[name]['sha256'] for name in OWNERS[group]},manifest=json_digest(m),scenario=scenario,stream=stream,master=SEED,split='test',upper_policy='deterministic',lower_policy='deterministic',rollout_seconds=900,reward_scale=config['objective']['reward_scale'])
- require(len(tasks)==720,'test_window_budget');start=time.perf_counter();completed=[];first=None
+ require(len(tasks)==640,'test_window_budget');start=time.perf_counter();completed=[];first=None
-  atomic_json(OUT/'evaluation_progress.json',dict(completed_windows=len(completed),total_windows=720,last_completed=r,first_test_window_started_at=first))
-  if len(completed)%80==0:print('TEST_EVALUATION',len(completed),'/720',flush=True)
+  atomic_json(OUT/'evaluation_progress.json',dict(completed_windows=len(completed),total_windows=640,last_completed=r,first_test_window_started_at=first))
+  if len(completed)%80==0:print('TEST_EVALUATION',len(completed),'/640',flush=True)
```

@@ -120,7 +107,7 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
- summary=dict(protocol_sha256=freeze['sha256'],manifest=m,master=4101,groups=GROUPS,windows=720,by_mode=totals,per_scenario=per_scenario,final_distance_m=distances,diagnostics=diagnostics,stream_consistency=dict(per_group={g:dict(equal=sum(int(c['equal']) for r in consistency if r['group']==g for c in r['comparisons']),total=60) for g in GROUPS},comparisons=consistency),learning_updates=0,replay_writes=0)
+ summary=dict(protocol_sha256=freeze['sha256'],manifest=m,master=SEED,groups=GROUPS,windows=640,by_mode=totals,per_scenario=per_scenario,final_distance_m=distances,diagnostics=diagnostics,stream_consistency=dict(per_group={g:dict(equal=sum(int(c['equal']) for r in consistency if r['group']==g for c in r['comparisons']),total=60) for g in GROUPS},comparisons=consistency),learning_updates=0,replay_writes=0)
```

@@ -134,35 +121,12 @@

- seed/path: SEED和参数模块导入、导入路径检查
- D6: 删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组

```diff
- for ref in ('Abl_TwoStage','Abl_Alternating','Abl_HoverSAC','Abl_UpperCVX'):add('Proposed',ref,'ablation')
+ for ref in ('Abl_TwoStage','Abl_HoverSAC','Abl_UpperCVX'):add('Proposed',ref,'ablation')
- lines=['# v2.4.0 seed 4101 三阶段协议冻结与 test 面板','', '状态：全部步骤完成；仅评估，learning_updates=0，replay_writes=0。','', '## 检查结果','']
- fmt=lambda x:'缺失' if x is None else f'{x:.9g}' if isinstance(x,float) else str(x)
- def table(headers,rs):
-  lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']);lines.extend('| '+' | '.join(fmt(v) for v in r)+' |' for r in rs);lines.append('')
- replay=read(OUT/'replay_checks.json');pre=read(OUT/'preflight.json');main=comparisons[0];st=main['statistics']
- checkrows=[[f'P{i}',CHECKS[f'P{i}']] for i in range(1,7)]+[['协议冻结',CHECKS['protocol_freeze']]]+[[r['check'],f'{CHECKS[r["check"]]}；{r["group"]}；完整metrics相等'+('；diagnostics相等' if r['diagnostics_checked'] else '')] for r in replay['rows']]
- equal=sum(r['equal'] for r in summary['stream_consistency']['per_group'].values());total=sum(r['total'] for r in summary['stream_consistency']['per_group'].values())
- checkrows += [['跨stream一致性（只报告）',f'{equal}/{total} 完整metrics逐位相同'],['协议哈希复核',CHECKS['protocol_hash_recheck']]];table(['检查','结果'],checkrows)
- lines.extend(['P1：两棵工作树分支、HEAD、status。P2：WT/src导入、cuda:0、运行时、scipy。P3：五份identity相等且validation passed。P4：排除自身后的GPU/进程检查。P5：test manifest完全一致且与dev种子三元组不相交。P6：checkpoint身份、SHA256、类型、权重哈希及基线预算。','', f'解释器：{pre["runtime"]["executable"]}；torch={pre["runtime"]["torch"]}；CUDA={pre["runtime"]["cuda"]}；scipy={pre["scipy_version"]}；torch线程数={pre["torch_threads"]}。','', '## 协议冻结','',f'路径：{freeze["path"]}','',f'SHA256：`{freeze["sha256"]}`','',f'冻结时间：{freeze["frozen_at"]}','',f'冻结时 OUT/windows 存在：{freeze["windows_exists_at_freeze"]}','',f'第一个 test 窗口开始时间：{TIMING["first_test_window_started_at"]}','', '本协议在看过 4101 dev 结果后制定；4101 test 为 test 划分的首次使用。多seed方案仅写入协议，本任务未执行其他seed。','', '## 主终点：Proposed 对 TFOO-200',''])
- table(['Proposed均值s','TFOO均值s','相对差%','胜场/20','Wilcoxon统计量','W+','Wilcoxon p','t检验p','符号检验p'],[[main['candidate_mean_s'],main['reference_mean_s'],main['relative_difference_percent'],main['wins'],st['wilcoxon']['statistic'],st['wilcoxon']['W_plus'],st['wilcoxon']['pvalue'],st['ttest_rel']['pvalue'],st['binomtest']['pvalue']]])
- lines.extend(['Wilcoxon p<0.05：'+('是' if st['wilcoxon']['pvalue']<0.05 else '否')+'。','',"20场景配对差；Wilcoxon双侧，zero_method='wilcox'，α=0.05。次要比较只作报告，不做多重比较校正。",'', '## 主比较',''])
- headers=['候选','参考','候选均值s','参考均值s','相对差%','胜场/20','mean d','sd d','Wilcoxon统计量','W+','Wilcoxon p','t统计量','t p','符号p']
- def comparisonrows(rs):return [[r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['mean_d'],r['statistics']['sd_d'],r['statistics']['wilcoxon']['statistic'],r['statistics']['wilcoxon']['W_plus'],r['statistics']['wilcoxon']['pvalue'],r['statistics']['ttest_rel']['statistic'],r['statistics']['ttest_rel']['pvalue'],r['statistics']['binomtest']['pvalue']] for r in rs]
- table(headers,comparisonrows([r for r in comparisons if r['category']=='main']))
- lines.extend(['### 主比较 cost/s（场景等权）','']);table(['参考','Proposed均值','参考均值','相对差%','胜场/20'],[[r['reference'],r['cost_per_s']['candidate_mean'],r['cost_per_s']['reference_mean'],r['cost_per_s']['relative_difference_percent'],r['cost_per_s']['wins']] for r in comparisons if r['category']=='main'])
- lines.extend(['## 消融比较','']);table(headers,comparisonrows([r for r in comparisons if r['category']=='ablation']))
- lines.extend(['## 补充比较','']);table(headers,comparisonrows([r for r in comparisons if r['category']=='supplementary']))
- lines.extend(['gates.unfinished_per_episode 是20个场景各自每窗口未完成数之和，不是原始窗口总数。下表按80窗口归一。','', '## 各组指标（80窗口池化）',''])
- table(['组','截断s','upload s','ready s','u2u s','cpu s','cost/s','高优未完成/窗口','低优未完成/窗口','最终距离m','solver_calls','fallback_calls'],[[g,m['truncated_delay_s'],*[m['stage_delay_s'][k] for k in ('upload','ready','u2u','cpu')],m['cost_per_s'],*[p['unfinished']/m['windows'] for p in m['priority']],summary['final_distance_m'][g],summary['diagnostics'][g]['solver_calls'] if g in ('TFOO','Abl_UpperCVX') else '不适用',summary['diagnostics'][g]['fallback_calls'] if g in ('TFOO','Abl_UpperCVX') else '不适用'] for g,m in summary['by_mode'].items()])
- lines.extend(['## 主终点逐场景','']);table(['场景','Proposed s','TFOO s','差s','胜'],[[r['scenario'],r['candidate_s'],r['reference_s'],r['candidate_s']-r['reference_s'],r['win']] for r in main['pairs']])
- lines.extend(['## 跨stream一致性','', '对每组每场景，将stream 2/3/4完整metrics与stream 1逐位比较；此项只报告，不作为停止条件。',''])
- table(['组','相同数量','比较总数'],[[g,r['equal'],r['total']] for g,r in summary['stream_consistency']['per_group'].items()])
- lines.extend(['## 耗时','']);atomic_json(OUT/'timing.json',TIMING)
- table(['阶段','墙钟秒'],[[k,TIMING[k]] for k in ('preflight_seconds','protocol_seconds','replay_seconds','test_evaluation_seconds','summary_seconds','total_wall_seconds_before_report_write')])
- lines.extend(['每窗口墙钟是worker内从窗口开始到写完窗口JSON的实测时间（含该窗口的读取、初始化与计算）；并行运行，不能将窗口时间求和当成面板墙钟。',''])
- table(['组','窗口数','平均每窗口墙钟秒'],[[g,80,TIMING['test_per_group_mean_window_wall_seconds'][g]] for g in GROUPS])
- lines.extend(['## 脚本实现错误与修正','', '无。','', '## 最终工作树状态','']);table(['工作树','git status','branch','HEAD'],[[root,'空' if not r['status'] else r['status'],r['branch'],r['head']] for root,r in git.items()])
+ atomic_json(OUT/'timing.json',TIMING)
+ lines=[f'# v2.4.0 seed {SEED} test 面板','',f"协议SHA256：{freeze['sha256']}",'','检查：'+json.dumps(CHECKS,ensure_ascii=False),'','|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|','|---|---|---|---|---|---|---|']
+ for r in comparisons:lines.append('| '+' | '.join(map(str,(r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue'])))+' |')
+ lines+=['','## 实测耗时','',json.dumps(TIMING,ensure_ascii=False,indent=2),'','## 最终Git状态','',json.dumps(git,ensure_ascii=False,indent=2),'','脚本错误与修正：无。']
```

### stage1_wrapper.py

来源：D:\HRL_UAV_V2_speedup\src\hrl_v2\run.py

@@ -1,390 +1,62 @@

- D1: 替代CLI调用train并在300 evaluated存在时只阻断301；包含指定后检
- seed/path: SEED、WORK、seed目录；导入路径断言

```diff
-"""Explicit fresh/resume CLI. Default action is a zero-step dry run."""
-import argparse
-import copy
-import json
+# Derived from hrl_v2.run.main; D1 wrapper, unchanged 1-300 training call.
+import sys
+sys.dont_write_bytecode=True
-import sys
-import time
-import uuid
+import json,gc,traceback
+from common import *
+assert_imports()
-from . import SCHEMA
-from .checkpoint import atomic_json,digest,identity,json_digest,load,save,runtime_record
-from .schedule import phase,expected_counts
-from .randomness import derive,Streams
-from .replay import Replay
-from .paths import source_root, default_config
-from .checkpoint import runtime_signature
-from .timing import ClockConfig
-from .baseline_protocol import reference_modes_complete
-from .archive import (checkpoint_interval, should_checkpoint, publish_checkpoint,
-                      canonical_rows, supersede_episodes, recover_metric, save_network_snapshot)
+import hrl_v2.run as run
+from hrl_v2.schedule import phase as original_phase
+from hrl_v2.checkpoint import load,identity,atomic_json
+from hrl_v2.runtime import Learner,configure,parameter_hash
+from hrl_v2.evaluate import manifest
+class StopStage1(Exception):pass
-SCENE_SHA="6933abadbb8e34a532de6429477d52441e131996015126e458cbc97494e4c582"
-# Configs written before a physics field existed describe the v2.3.0 physics.
-LEGACY_PHYSICS=dict(u2u_model="paper_eq13_fixed_noise",terminal_mobility=None)
+def guard_for(output):
+ def guarded(number):
+  if number==301 and (Path(output)/'checkpoints/episode-0300-evaluated.pt').exists():raise StopStage1()
+  return original_phase(number)
+ return guarded
-
-def read_config(path,root):
-    if digest(source_root()/"hrl_v2/configs/physical_scene.json")!=SCENE_SHA:
-        raise ValueError("Physical scene differs from the reviewed physical parameters")
-    config=json.loads(Path(path).read_text(encoding="utf-8"))
-    if config.get('encoder')=='lite' and config.get('observation_version')!='lite_110_729_v2_2_1':
-        raise ValueError('This Lite implementation requires the v2_2_1 observation version; archived configs are not upgraded implicitly')
-    if config.get("schema")!=SCHEMA:
-        raise ValueError("Expected continuing-service configuration")
-    clocks=ClockConfig.from_config(config)
-    scene=json.loads((source_root()/"hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
-    if clocks!=ClockConfig.from_scene(scene,rollout_seconds=clocks.rollout_seconds):
-        raise ValueError("Control and physical clocks must match the physical scene")
-    objective=config["objective"]
-    physical=scene["base_scene"]["objective"]
-    import math
-    if (objective["priority_weights"]!=physical["priority_weights"] or
-        not math.isclose(objective["holding_coefficient"],physical["holding_weight"]/physical["holding_reference_task_s"],rel_tol=0.,abs_tol=1e-12) or
-        not math.isclose(objective["energy_coefficient"],(1-physical["holding_weight"])/physical["energy_reference_j"],rel_tol=0.,abs_tol=1e-12) or
-        objective.get('energy_mode','full')!=physical['energy_mode'] or
-        objective.get('propulsion_weight',1.)!=physical.get('propulsion_weight',1.)):
-        raise ValueError("Configured costs must match the reviewed physical objective and units")
-    base=scene["base_scene"]
-    if config.get("physics",LEGACY_PHYSICS)!=dict(u2u_model=base["resources"].get("u2u_model","psd_per_uav_budget"),
-                                                  terminal_mobility=base.get("terminal_mobility")):
-        raise ValueError("Configured physics version must match the physical scene; archived configs are not upgraded implicitly")
-    if (objective.get("normal_rollout_terminated") is not False or
-            objective.get("normal_rollout_truncated") is not True):
-        raise ValueError("Artificial rollout boundaries must truncate, never terminate")
-    if not isinstance(objective["reward_scale"],(float,int)) or not 0<objective["reward_scale"]<float("inf"):
-        raise ValueError("reward_scale must be finite and positive")
-    validate_discounts(config,clocks)
-    return config
-
-
-def validate_discounts(config,clocks):
-    """Both layers discount the same physical time: gamma_macro = gamma_micro ** lower_per_upper."""
-    import math
-    objective,ppo=config["objective"],config["ppo"]
-    micro,macro=objective["gamma_micro"],objective["gamma_macro"]
-    if not math.isclose(ppo["gamma"],macro,rel_tol=0.,abs_tol=1e-12):
-        raise ValueError("ppo.gamma must equal objective.gamma_macro")
-    if not math.isclose(macro,micro**clocks.lower_per_upper,rel_tol=0.,abs_tol=1e-9):
-        raise ValueError("gamma_macro must equal gamma_micro ** lower_per_upper (same physical horizon)")
-    if not 0<=ppo["gae_lambda"]<=1:
-        raise ValueError("gae_lambda must lie in [0, 1]")
-
-
-def require_research_protocol(config):
-    # Continuing service needs a strict discount: with gamma=1 the bootstrapped
-    # truncation target is an unbounded sum of costs.
-    research=config.get("research",{})
-    objective=config.get("objective",{})
-    missing=[k for k in ("training_enabled","discounts_confirmed","schedule_confirmed")
-             if research.get(k) is not True]
-    if missing:
-        raise RuntimeError("Research training is disabled: config research flags not confirmed: "+", ".join(missing))
-    if not (0<objective.get("gamma_micro",1.)<1 and 0<objective.get("gamma_macro",1.)<1):
-        raise RuntimeError("Continuing-service training requires 0 < gamma < 1 for both layers")
-
-
-def technical_status(root,config_path,config):
-    """Readiness is a technical report, never authorization to launch a run."""
-    try:
-        require_research_protocol(config)
-        confirmed=True
-    except RuntimeError:
-        confirmed=False
-    path=Path(root)/"validation/validation.json"
-    validation=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
-    ready=(validation.get("status")=="passed" and
-           validation.get("identity")==identity(root,config_path) and
-           runtime_signature(validation.get("runtime",{}))==runtime_signature(runtime_record()))
-    return dict(validation_ready=ready,research_protocol_confirmed=confirmed,
-                training_ready=bool(ready and confirmed),training_authorized=False,
-                training_started=False)
-
-
-def append(path,value):
-    with Path(path).open("a",encoding="utf-8",newline="\n") as f:
-        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
-
-
-def collection_policy(learner,streams,phase):
-    from .policy import Policy
-    return Policy(learner.sac.actor,learner.ppo.actor,
-                  learner.ppo.value if phase.layer=="upper" else None,streams,
-                  mode="hover" if phase.stage==1 else "moving")
-
-
-def train(root,config_path,seed,output,resume=None,baseline_reference=None,checkpoint_every=10):
-    checkpoint_interval(checkpoint_every)
-    require_research_protocol(read_config(config_path,root))
-    from .runtime import configure,Learner,parameter_hash
-    from .physics_adapter import PhysicsAdapter
-    from .collector import collect
-    from .evaluate import manifest,panel
-    root,output=Path(root),Path(output)
-    config=read_config(config_path,root); binding=identity(root,config_path)
-    clocks=ClockConfig.from_config(config)
-    validation_path=root/"validation/validation.json"
-    if not validation_path.exists():
-        raise ValueError("Run hrl-v2 validate with the same --workspace before research training")
-    validation=json.loads(validation_path.read_text(encoding="utf-8"))
-    if validation.get("status")!="passed" or validation.get("identity")!=binding:
-        raise ValueError("Implementation validation is missing, failed or belongs to another code/config version")
-    saved=load(resume,binding) if resume is not None else None
-    reference=None
-    if config.get('observation_version')=='lite_110_729_v2_2_1':
-        if saved is not None:
-            reference=saved['baseline_reference']
-        else:
-            if baseline_reference is None: raise ValueError('Freeze the paired baseline reference before this fresh Lite run')
-            reference=json.loads(Path(baseline_reference).read_text(encoding='utf-8'))
-        from .evaluate import gate_manifest
-        if reference['identity']!=binding or reference['master']!=seed:
-            raise ValueError('Baseline reference source/config/seed mismatch')
-        count=len(reference['manifest']['scenarios'])
-        if reference['manifest']!=gate_manifest(count) or reference['required_wins']!={6:4,20:12}[count]:
-            raise ValueError('Baseline gate contract mismatch')
-        if not reference_modes_complete(config,reference['per_scenario']):
-            raise ValueError('Incomplete baseline reference')
-    evaluation_manifest=reference['manifest'] if reference else manifest('dev')
-    if saved is not None:
-        seed=saved["seed"]
-        if saved["output_relative"]!=str(output.relative_to(root)).replace("\\","/"):
-            raise ValueError("Resume must use the checkpoint's original run directory")
-        latest_path=output/"latest.json"
-        if latest_path.exists():
-            latest=json.loads(latest_path.read_text(encoding="utf-8"))
-            if latest["sha256"]!=digest(resume):
-                raise ValueError("Resume from latest.json; older snapshots are evaluation-only and cannot overwrite later work")
-    elif output.exists(): raise FileExistsError("Fresh runs never overwrite an existing directory")
-    device=configure()
-    if runtime_signature(validation.get("runtime",{}))!=runtime_signature(runtime_record()):
-        raise ValueError("Run local validation in the current Python/PyTorch/CUDA environment")
-    learner=Learner(seed,device,config=config)
-    if saved is not None and runtime_signature(saved.get("runtime",{}))!=runtime_signature(runtime_record()):
-        raise ValueError("Resume runtime differs from the saved Python/PyTorch/CUDA/device contract")
-    scene=json.loads((source_root()/"hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
-    output.mkdir(parents=True,exist_ok=True)
-    replay=Replay(); ppo_batch=[]; completed=0; pending=None; dev_results={}; last_streams=None
-    attempt_id=uuid.uuid4().hex
-    if saved:
-        learner.load_state_dict(saved["learner"])
-        replay=saved["replay"]; ppo_batch=saved["ppo_batch"]
-        completed=saved["completed_episode"]; pending=saved["pending_evaluation"]
-        dev_results=saved["development_results"]; last_streams=saved["streams"]
-        torch.set_rng_state(saved["global_cpu_rng"])
-        torch.cuda.set_rng_state_all(saved["global_cuda_rng"])
-    else:
-        training_manifest=[]
-        for number in range(1,601):
-            p=phase(number)
-            training_manifest.append(dict(global_episode=number,stage=p.stage,cycle=p.cycle,block_episode=p.block_episode,
-                **{k+"_seed":derive(seed,p.stage,p.cycle,p.block_episode,purpose=k) for k in ("arrival","layout","load")}))
-        atomic_json(output/"training_manifest.json",training_manifest)
-        atomic_json(output/"development_manifest.json",evaluation_manifest)
-        if reference: atomic_json(output/'baseline_reference.json',reference)
-        atomic_json(output/"test_manifest.json",manifest("test"))
-        atomic_json(output/"contract.json",dict(identity=binding,config=config,seed=seed,
-            runtime=runtime_record(),expected_counts=expected_counts(config),checkpoint_every=checkpoint_every,
-            initial_parameters={name:parameter_hash(module) for name,module in
-                (("lower_actor",learner.sac.actor),("q1",learner.sac.q1),("q2",learner.sac.q2),
-                 ("upper_actor",learner.ppo.actor),("value",learner.ppo.value))},
-            physical_source="Bundled physical source derived from dbf2b97; see docs/PROVENANCE.md"))
-    checkpoint_path=Path(resume) if resume is not None else None
-    last_checkpoint_episode=completed
-    def persist(label):
-        nonlocal checkpoint_path,last_checkpoint_episode
-        state=dict(schema=SCHEMA,safe_episode_boundary=True,completed_episode=completed,
-            pending_evaluation=pending,identity=binding,seed=seed,
-            output_relative=str(output.relative_to(root)).replace("\\","/"),
-            learner=learner.state_dict(),replay=replay,ppo_batch=ppo_batch,streams=last_streams,
-            global_cpu_rng=torch.get_rng_state(),global_cuda_rng=torch.cuda.get_rng_state_all(),
-            development_results=dev_results,scene=scene,config=config,baseline_reference=reference,
-            runtime=runtime_record(),
-            discarded_work_policy="restart from last completed episode; incomplete episode never admitted")
-        path=output/"checkpoints"/f"episode-{completed:04d}-{label}.pt"
-        if path.exists():
-            # Resume of an earlier boundary must not overwrite evidence from a
-            # later attempt. A new immutable checkpoint gets a unique suffix.
-            path=path.with_name(path.stem+"-"+uuid.uuid4().hex[:8]+".pt")
-        save(path,state)
-        publish_checkpoint(root,output,path,completed,attempt_id)
-        checkpoint_path=path
-        last_checkpoint_episode=completed
-        # Keep two rolling episode boundaries and all registered development
-        # nodes. Delete only this run's own superseded complete snapshots after
-        # the new snapshot and latest pointer have both committed atomically.
-        rolling=sorted((output/"checkpoints").glob("episode-*-complete*.pt"),key=lambda p:p.stat().st_mtime)
-        for old in rolling[:-2]:
-            if old.parent.resolve()!=(output/"checkpoints").resolve(): raise RuntimeError("Checkpoint retention path escaped run")
-            old.with_suffix(".json").unlink()
-            old.unlink()
-        return path
-    def evaluate_pending(checkpoint_path):
-        nonlocal pending
-        evaluation_number=pending
-        # Gates use the primary execution; a deterministic primary also records
-        # the stochastic panel as secondary evidence, never used by a gate.
-        deterministic=config["evaluation"].get("primary_execution","stochastic").startswith("deterministic")
-        evaluate=lambda stochastic: panel(learner,scene,seed,evaluation_manifest,output/"evaluations"/f"episode-{pending:04d}",
-                     checkpoint_id=digest(checkpoint_path),
-                     modes=("hover",) if pending==300 else ("moving","hover"),stochastic=stochastic,
-                     timing=clocks,reward_scale=config["objective"]["reward_scale"])
-        result=evaluate(not deterministic)
-        if deterministic:
-            secondary=evaluate(True)
-            result['secondary_stochastic']=dict(by_mode=secondary['by_mode'],per_scenario=secondary['per_scenario'])
-        dev_results[str(pending)]=result
-        if reference:
-            from .gates import g1,g2
-            gate=(g1(result['per_scenario']['hover'],reference['per_scenario'],reference['required_wins'],
-                     unfinished=config["evaluation"].get("g1_unfinished","gate"))
-                  if pending==300 else g2(result['per_scenario']['moving'],result['per_scenario']['hover']))
-            result['gate']=gate
-            atomic_json(output/'gates'/f'episode-{evaluation_number:04d}.json',gate)
-            if not gate.get('passed',True) or gate.get('weak_upper_signal',False):
-                atomic_json(output/'diagnosis_required.json',dict(episode=pending,gate=gate,
-                    stage_delay_s={k:v['stage_delay_s'] for k,v in result['by_mode'].items()},
-                    candidate_training_authorized=False))
-        pending=None
-        return persist("evaluated")
-    if saved is not None: supersede_episodes(output,completed)
-    if saved is None: persist("initial")
-    append(output/"attempts.jsonl",dict(attempt_id=attempt_id,resumed=bool(saved),
-        restored_completed_episode=completed,checkpoint_sha256=digest(resume) if resume else None,
-        checkpoint_every=checkpoint_every,discarded_after_episode=completed if saved else None,
-        prior_uncommitted_work_excluded=bool(saved),unrecorded_discarded_physical_steps=None,
-        note="Uncommitted records from previous attempts are retained by attempt_id, excluded from completed-episode totals; unknown unlogged work is not fabricated."))
-    if pending is not None: evaluate_pending(resume)
-    if saved is not None: recover_metric(output,completed)
-    def paused_g1():
-        failed=reference and not dev_results.get('300',{}).get('gate',{}).get('passed',True)
-        if failed:
-            record=dict(status='paused_G1',completed_episode=completed,last_checkpoint_episode=last_checkpoint_episode,gate=dev_results['300']['gate'],
-                        resumable_checkpoint=json.loads((output/'latest.json').read_text()),
-                        note='PPO stage blocked. Resume preserves this gate; no automatic threshold override.')
-            atomic_json(output/'progress.json',record)
-            return record
-    paused=paused_g1()
-    if paused: return paused
-    for number in range(completed+1,601):
-        if identity(root,config_path)!=binding: raise RuntimeError("Source changed during training")
-        p=phase(number)
-        print(f'[{number}/600] {p.layer} episode started; stage={p.stage}',flush=True)
-        streams=Streams(seed,p.stage,p.cycle,p.block_episode)
-        adapter=PhysicsAdapter(scene,timing=clocks)
-        external={k+"_seed":derive(seed,p.stage,p.cycle,p.block_episode,purpose=k)
-                  for k in ("arrival","layout","load")}
-        adapter.reset(**external)
-        version="hover-0" if p.stage==1 else f"ppo-{learner.ppo.version}-stochastic"
-        if p.layer=="lower": replay.set_upper_version(version)
-        policy=collection_policy(learner,streams,p)
-        started=time.perf_counter()
-        counters_before=dict(learner.sac.counts)
-        residuals=[]
-        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
-        def event():
-            row=learner.sac.event(replay,p.alpha,p.actor_enabled,streams)
-            residuals.extend(v for fit in row.get('critics',[]) for v in fit['residual_mse'])
-            append(output/"learning.jsonl",dict(attempt_id=attempt_id,global_episode=number,
-                physical_tick=adapter.env.step_index,event=learner.sac.counts["triggers"],alpha=p.alpha,**row))
-        episode=collect(adapter,policy,upper_version=version,lower_version=learner.sac.version,
-                        identity=dict(global_episode=number,external=external),on_event=event if p.layer=="lower" else None,
-                        reward_scale=config["objective"]["reward_scale"],gamma_lower=config["objective"]["gamma_micro"])
-        if p.layer=="lower": replay.add(episode)
-        else:
-            ppo_batch.append(episode)
-            if len(ppo_batch)==5:
-                updates=learner.ppo.update(ppo_batch,streams); ppo_batch.clear()
-                append(output/"learning.jsonl",dict(attempt_id=attempt_id,global_episode=number,ppo=updates))
-        completed=number; last_streams=streams.state_dict()
-        metrics=episode.metrics
-        work_seconds=time.perf_counter()-started
-        from .telemetry import compact,resources
-        row=dict(attempt_id=attempt_id,global_episode=number,layer=p.layer,stage=p.stage,
-            reward_physical=float(sum(episode.rewards)),
-            reward_learning_macro_sum=float(sum(h['reward_learning'] for h in episode.high)),
-            work_seconds=work_seconds,checkpoint_seconds=None,evaluation_seconds=None,snapshot_seconds=None,
-            wall_seconds=None,alpha=p.alpha,
-            critic_residual_mse=sum(residuals)/len(residuals) if residuals else None,
-            actor_relative_entropy=policy.entropy_sum/policy.entropy_count if policy.entropy_count else None,
-            entropy_samples=policy.entropy_count,entropy_scope='executed lower samples plus real bootstrap successor',
-            sac_delta={k:v-counters_before[k] for k,v in learner.sac.counts.items()},
-            resources=resources(),**compact(metrics))
-        atomic_json(output/"episodes"/f"episode-{number:04d}.json",dict(attempt_id=attempt_id,phase=p.__dict__,metrics=metrics,
-            wall_seconds=work_seconds,storage_bytes=episode.storage_bytes,
-            sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts),train_metric=row))
-        pending=number if number in config["schedule"]["development_global_episodes"] else None
-        saved_at=time.perf_counter()
-        if should_checkpoint(completed,checkpoint_every,pending=pending):
-            checkpoint_path=persist("pending" if pending is not None else "complete")
-        checkpoint_seconds=time.perf_counter()-saved_at
-        snapshot_at=time.perf_counter()
-        if completed%25==0: save_network_snapshot(output,learner,completed,binding)
-        snapshot_seconds=time.perf_counter()-snapshot_at
-        evaluated_at=time.perf_counter()
-        if pending is not None: checkpoint_path=evaluate_pending(checkpoint_path)
-        evaluation_seconds=time.perf_counter()-evaluated_at
-        row.update(checkpoint_seconds=checkpoint_seconds,evaluation_seconds=evaluation_seconds,snapshot_seconds=snapshot_seconds,
-                   wall_seconds=time.perf_counter()-started,resources=resources())
-        append(output/'train_metrics.jsonl',row)
-        print(f'[{number}/600] {p.layer} reward={row["reward_physical"]:.6f} '
-              f'truncated_delay={row["truncated_delay_s"]:.6f}s '
-              f'work={work_seconds:.2f}s SAC_events={row["sac_delta"]["events"]} '
-              f'critic_fits={row["sac_delta"]["critic_fits"]}',flush=True)
-        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
-        paused=paused_g1()
-        if paused: return paused
-        atomic_json(output/"progress.json",dict(status="complete" if number==600 else "running",
-            completed_episode=number,last_checkpoint_episode=last_checkpoint_episode,
-            sac=learner.sac.counts,ppo=learner.ppo.counts,latest=str(checkpoint_path.relative_to(root))))
-    actual=dict(episodes=completed,physics_steps=completed*clocks.physical_steps,sac_events=learner.sac.counts["events"],
-                critic_fits=learner.sac.counts["critic_fits"],sac_actor_updates=learner.sac.counts["actor_updates"],
-                ppo_batches=learner.ppo.counts["batches"])
-    if actual!=expected_counts(config): raise RuntimeError(f"Budget mismatch: {actual}")
-    eligible=[(int(k),v["by_mode"]["moving"]["cost_per_s"]) for k,v in dev_results.items() if k!="300"]
-    best=min(eligible,key=lambda x:(x[1],x[0]))[0]
-    result=dict(status="training_complete_performance_not_yet_accepted",actual_counts=actual,
-                final_episode=600,development_best_episode=best,development_results=dev_results)
-    atomic_json(output/"result.json",result)
-    return result
-
+def postcheck(root,config_path,output):
+ binding=identity(root,config_path);config=run.read_config(config_path,root);device=configure()
+ ev=output/'checkpoints/episode-0300-evaluated.pt';pd=output/'checkpoints/episode-0300-pending.pt'
+ check(ev.is_file() and ev.with_suffix('.json').is_file(),'300_evaluated_files')
+ saved=load(ev,binding);check(saved['completed_episode']==300 and saved['pending_evaluation'] is None,'300_endpoint')
+ learner=Learner(SEED,device,config=config);learner.load_state_dict(saved['learner'])
+ hs=dict(sac_actor=parameter_hash(learner.sac.actor),ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value));counts=dict(learner.sac.counts)
+ check(learner.ppo.counts['batches']==0,'stage1_ppo_zero');del learner,saved;gc.collect();torch.cuda.empty_cache()
+ pending=load(pd,binding);learner=Learner(SEED,device,config=config);learner.load_state_dict(pending['learner'])
+ phs=dict(sac_actor=parameter_hash(learner.sac.actor),ppo_actor=parameter_hash(learner.ppo.actor),ppo_value=parameter_hash(learner.ppo.value))
+ equal(hs,phs,'300_pending_evaluated_hashes');del learner,pending;gc.collect();torch.cuda.empty_cache()
+ old=load(P4101/'checkpoints/episode-0300-evaluated.pt',binding);equal(counts,old['learner']['sac']['counts'],'stage1_sac_counts_vs4101');del old;gc.collect()
+ initial=read(output/'contract.json')['initial_parameters'];check(hs['ppo_actor']==initial['upper_actor'] and hs['ppo_value']==initial['value'],'stage1_upper_unchanged')
+ metrics=[json.loads(x) for x in (output/'train_metrics.jsonl').read_text().splitlines() if x.strip()]
+ equal([r['global_episode'] for r in metrics],list(range(1,301)),'stage1_exact_episodes')
+ forbidden=[]
+ for folder in ('episodes','checkpoints','gates','evaluations','snapshots'):
+  for path in (output/folder).glob('episode-*'):
+   n=int(path.name.split('-')[1].split('.')[0]);
+   if n>300:forbidden.append(str(path))
+ check(not forbidden,'no_301_outputs',forbidden)
+ equal(read(output/'test_manifest.json'),manifest('test'),'own_test_manifest');equal(read(output/'test_manifest.json'),read(P4101/'test_manifest.json'),'test_manifest_vs4101')
+ atomic_json(output/'checks.json',dict(status='passed',sac_counts=counts,ppo_batches=0,parameter_hashes=hs,gate=read(output/'gates/episode-0300.json')))
-    parser=argparse.ArgumentParser(description=__doc__)
-    parser.add_argument("--project-root",type=Path,default=Path.cwd()/"outputs")
-    parser.add_argument("--config",type=Path,default=default_config())
-    parser.add_argument("--seed",type=int,choices=range(4101,4106))
-    group=parser.add_mutually_exclusive_group()
-    group.add_argument("--fresh",action="store_true"); group.add_argument("--resume",type=Path)
-    parser.add_argument("--output",type=Path)
-    parser.add_argument("--checkpoint-every",type=int,default=10,help="Full checkpoint interval; mandatory anchors are always saved")
-    parser.add_argument('--baseline-reference',type=Path,help='Frozen paired baseline summary for the Lite gated run')
-    args=parser.parse_args(); root=args.project_root.resolve()
-    rooted=lambda p:p if p.is_absolute() else root/p
-    config_path=rooted(args.config); config=read_config(config_path,root)
-    binding=identity(root,config_path)
-    if not args.fresh and args.resume is None:
-        from .runtime import configure
-        configure(test_cpu=True)
-        print(json.dumps(dict(status="dry_run",physical_steps=0,updates=0,
-            **technical_status(root,config_path,config),rollout=ClockConfig.from_config(config).counts(),
-            rollout_seconds=config["task"]["rollout_seconds"],
-            config_sha256=digest(config_path),expected_counts=expected_counts(config)),indent=2)); return
-    require_research_protocol(config)
-    if args.fresh and args.seed is None: parser.error("--fresh requires --seed")
-    if args.resume is not None:
-        if args.seed is not None: parser.error("--resume reads the saved training seed")
-        saved=load(rooted(args.resume),binding)
-        output=root/saved["output_relative"]
-    else:
-        output=rooted(args.output) if args.output else root/"v2_runs"/json_digest(binding)[:12]/"ppo_sac"/f"seed_{args.seed}"
-    try:
-        result=train(root,config_path,args.seed,output,rooted(args.resume) if args.resume else None,
-                     rooted(args.baseline_reference) if args.baseline_reference else None,checkpoint_every=args.checkpoint_every)
-    except Exception as error:
-        if output.exists(): atomic_json(output/"technical_failure.json",dict(error=type(error).__name__,message=str(error)))
-        raise
-    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
-
-
-if __name__=="__main__": main()
+ root=WORK.resolve();rooted=lambda p:p if p.is_absolute() else root/p
+ config_path=rooted(CONFIG);config=run.read_config(config_path,root);binding=identity(root,config_path)
+ run.require_research_protocol(config)
+ output=rooted(envpath('P',run_path()));reference=rooted(WORK/f'references/seed_{SEED}/baselines/summary.json')
+ run.phase=guard_for(output)
+ try:
+  result=run.train(root,config_path,SEED,output,None,reference,checkpoint_every=10)
+  check(result.get('status')=='paused_G1','stage1_normal_return',result)
+ except StopStage1:
+  result=dict(status='stopped_after_300',completed_episode=300)
+  atomic_json(output/'stage1_stop.json',result)
+ except Exception as error:
+  if output.exists():atomic_json(output/'technical_failure.json',dict(error=type(error).__name__,message=str(error)))
+  raise
+ postcheck(root,config_path,output);print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
+if __name__=='__main__':main()
```

### launch_baselines_seed.py

来源：D:\HRL_UAV_V2_speedup\outputs\lite_v2_4_0\baselines_4101_20260930_221900\launch_baselines.py

@@ -1,194 +1,51 @@

- D7: 保留三基线并行及失败等待全部退出；删除面板A/B；保留typed anchor检查
- seed/path: SEED、WORK、日志及初始上层哈希对照
- D9: CLI入口加导入断言；队列父进程不初始化CUDA

```diff
-"""Fresh parallel independent upper baselines; frozen Proposed is read-only."""
-from contextlib import contextmanager
-from datetime import datetime
-import json
-import os
-from pathlib import Path
-import subprocess
+# Derived from launch_baselines.py; D7 no dev panels, seed/path parameterized.
-import time
-import traceback
-import hrl_v2
-from hrl_v2.checkpoint import atomic_json,digest,identity
-from hrl_v2.evaluate import gate_manifest
-from hrl_v2.paths import source_root
-from hrl_v2.runtime import configure
-from hrl_v2.upper_baselines import load_typed
-
-ROOT=source_root().parent
-WORK=Path(__file__).resolve().parent
-EXPECTED_ROOT=Path('D:/HRL_UAV_V2_speedup').resolve()
-MAIN_ROOT=Path('D:/HRL_UAV_V2')
-PY=Path('D:/HRL_UAV_V2/.venv/Scripts/python.exe')
-CONFIG=EXPECTED_ROOT/'src/hrl_v2/configs/v2_4_0_lite.json'
-RUN=EXPECTED_ROOT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'
-FROZEN=EXPECTED_ROOT/'outputs/lite_v2_4_0/baselines'
-REFERENCE=FROZEN/'summary.json'
-REFERENCE_RUN=EXPECTED_ROOT/'outputs/lite_v2_4_0/baseline_run.json'
-PREVIOUS_FAILED_WORK=EXPECTED_ROOT/'outputs/lite_v2_4_0/baselines_4101_20260930_220351'
-ANALYSIS_OUT=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_baselines')
-COMMIT='25d59ff9f7c3e27c060c63d462235a30ce28930a'
-MODES=('hover_local','hover_equal','hover_cvx')
+sys.dont_write_bytecode=True
+import time,subprocess,json
+from common import *
+assert_imports()
+from hrl_v2.checkpoint import identity,load,atomic_json
+from hrl_v2.upper_baselines import load_typed,UpperLearner
+from hrl_v2.runtime import configure,parameter_hash
-SEED=4101
-
-def read(path):
- return json.loads(Path(path).read_text(encoding='utf-8'))
-def stamp():
- return datetime.now().astimezone().isoformat()
-def state(status,**fields):
- row=dict(status=status,updated_at=stamp(),pid=os.getpid(),commit=COMMIT,seed=SEED,order=ORDER,**fields)
- atomic_json(WORK/'queue_status.json',row)
- print(json.dumps(row,ensure_ascii=False),flush=True)
-def require_clean():
- for root in (ROOT,MAIN_ROOT):
-  if subprocess.check_output(['git','status','--porcelain'],cwd=root).strip():
-   raise RuntimeError('Git status must be empty: '+str(root))
-def require_identity(binding):
- if ROOT.resolve()!=EXPECTED_ROOT:
-  raise RuntimeError(f'hrl_v2 imported from {ROOT}; check PYTHONPATH')
- if identity(WORK,CONFIG)!=binding:
-  raise RuntimeError('Source/config changed; halted without auto-resume')
- if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()!=COMMIT:
-  raise RuntimeError('Commit changed; halted')
-@contextmanager
-def lock():
- import msvcrt
- with (WORK/'queue.lock').open('a+b') as handle:
-  handle.seek(0,2)
-  if not handle.tell(): handle.write(b'0');handle.flush()
-  handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
-  try: yield
-  finally:
-   handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
-def check_reference(binding):
- run=read(REFERENCE_RUN)
- if run.get('status')!='passed' or run.get('identity')!=binding:
-  raise RuntimeError('Frozen reference run did not pass or differs from source/config')
- summary=read(REFERENCE)
- if summary['identity']!=binding or summary['master']!=SEED:
-  raise RuntimeError('Reference identity or master seed differs')
- if summary['manifest']!=gate_manifest(20) or summary['required_wins']!=12:
-  raise RuntimeError('Reference manifest or required wins differs')
- if not set(MODES)<=set(summary['per_scenario']):
-  raise RuntimeError('Reference is missing a required hover mode')
- return digest(REFERENCE)
-def record_failure(error,**fields):
- atomic_json(WORK/'queue_failure.json',dict(error=type(error).__name__,message=str(error),at=stamp(),traceback=traceback.format_exc(),**fields))
-def run_child(label,arguments):
- command=[str(PY),'-u']+list(arguments)
- stdout=WORK/'logs'/f'{label}.stdout.log';stderr=WORK/'logs'/f'{label}.stderr.log'
- out=stdout.open('x',encoding='utf-8');err=stderr.open('x',encoding='utf-8')
- started=time.perf_counter()
- try:
-  child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
- except BaseException:
-  out.close();err.close();raise
- return dict(label=label,child=child,out=out,err=err,started=started,command=command,stdout=str(stdout),stderr=str(stderr),started_at=stamp())
-def public_job(job):
- return {k:job[k] for k in ('label','command','stdout','stderr','started_at')}|dict(pid=job['child'].pid)
-def wait_children(jobs,status):
- pending=list(jobs);completed=[];failures=[]
- state(status,children=[public_job(j) for j in jobs],completed=completed)
- try:
-  while pending:
-   for job in list(pending):
-    code=job['child'].poll()
-    if code is None: continue
-    job['out'].close();job['err'].close()
-    atomic_json(WORK/'logs'/f"{job['label']}.process.json",dict(command=job['command'],pid=job['child'].pid,exit_code=code,wall_seconds=time.perf_counter()-job['started'],ended_at=stamp()))
-    pending.remove(job)
-    if code:
-     failures.append(dict(label=job['label'],exit_code=code,stderr=job['stderr']))
-     record_failure(RuntimeError(f"{job['label']} exited with {code}"),children=failures,remaining=[public_job(j) for j in pending],no_children_killed=True)
-    else: completed.append(job['label'])
-    state('failed_waiting_for_children' if failures else status,children=[public_job(j) for j in pending],completed=list(completed),failures=list(failures),auto_resume=False)
-   if pending: time.sleep(2)
- except BaseException:
-  # Never terminate sibling children. Reap those already started before reporting.
-  for job in pending:
-   code=job['child'].wait();job['out'].close();job['err'].close()
-   atomic_json(WORK/'logs'/f"{job['label']}.process.json",dict(command=job['command'],pid=job['child'].pid,exit_code=code,wall_seconds=time.perf_counter()-job['started'],ended_at=stamp()))
-  raise
- if failures: raise RuntimeError('A child process failed; all sibling processes exited naturally')
- return completed
-
-def verify_baseline(name,binding):
- output=WORK/'runs'/name/'seed_4101';result=read(output/'result.json')
- if result.get('status')!='complete_performance_not_evaluated' or result.get('completed_episode')!=200:
-  raise RuntimeError('Baseline completion check failed: '+name)
+def verify_baseline(name,binding,device):
+ output=WORK/f'runs/{name}/seed_{SEED}';result=read(output/'result.json')
+ check(result.get('status')=='complete_performance_not_evaluated' and result.get('completed_episode')==200,'baseline_completion',name)
-  path=output/f'checkpoints/episode-{n:04d}-anchor.pt'
-  if not path.is_file(): raise FileNotFoundError(path)
-  saved=load_typed(path,binding,name)
-  if saved['completed_episode']!=n or saved['pending_evaluation'] is not None or saved['learner']['ppo']['counts']['batches']!=batches:
-   raise RuntimeError('Anchor budget/type check failed: '+str(path))
-  checks.append(dict(episode=n,ppo_batches=batches,checkpoint=str(path),sha256=digest(path)))
-  del saved
- return checks
+  path=output/f'checkpoints/episode-{n:04d}-anchor.pt';saved=load_typed(path,binding,name)
+  check(saved['completed_episode']==n and saved['pending_evaluation'] is None and saved['learner']['ppo']['counts']['batches']==batches,'baseline_anchor_budget',str(path))
+  checks.append(dict(episode=n,ppo_batches=batches,sha256=sha(path)));del saved
+ initial=load_typed(output/'checkpoints/episode-0000-initial.pt',binding,name);learner=UpperLearner(SEED,device,read(CONFIG));learner.load_state_dict(initial['learner'])
+ hs=dict(upper_actor=parameter_hash(learner.ppo.actor),value=parameter_hash(learner.ppo.value));expected=read(run_path()/'contract.json')['initial_parameters']
+ return dict(anchors=checks,initial_parameters=hs,initial_upper_matches_proposed={k:hs[k]==expected[k] for k in hs})
- if Path(sys.executable).resolve()!=PY.resolve(): raise RuntimeError('Unexpected interpreter')
- if not Path(hrl_v2.__file__).resolve().is_relative_to(EXPECTED_ROOT/'src'): raise RuntimeError('Unexpected import path')
- if ROOT.resolve()!=EXPECTED_ROOT: raise RuntimeError('Unexpected source_root')
- if str(configure())!='cuda:0': raise RuntimeError('Expected cuda:0')
- if subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'],cwd=ROOT).decode().strip()!='codex/v2-lite-speedup': raise RuntimeError('Unexpected branch')
- binding=identity(WORK,CONFIG);require_identity(binding);require_clean()
- subprocess.check_call(['git','check-ignore',str(WORK)],cwd=ROOT)
- if ANALYSIS_OUT.exists(): raise FileExistsError('Analysis OUT already exists')
- if binding!=read(RUN/'contract.json')['identity']: raise RuntimeError('Proposed contract identity differs')
- reference_sha=check_reference(binding)
- for n in (400,600):
-  if not (RUN/f'checkpoints/episode-{n:04d}-pending.pt').is_file(): raise FileNotFoundError(f'Proposed pending checkpoint {n}')
- if (WORK/'queue_status.json').exists(): raise RuntimeError('Fresh queue already started')
- (WORK/'logs').mkdir(exist_ok=False)
- panels={}
- for key,proposed,baseline in (('A',600,200),('B',400,100)):
-  panels[key]=dict(proposed_checkpoint=str(RUN/f'checkpoints/episode-{proposed:04d}-pending.pt'),baseline_episode=baseline,baseline_checkpoints={name:str(WORK/'runs'/name/'seed_4101'/f'checkpoints/episode-{baseline:04d}-anchor.pt') for name in ORDER},output=str(WORK/f'paper_dev_{key}_P{proposed}_B{baseline}'),manifest=str(RUN/'development_manifest.json'),policy='deterministic',mode='paper')
- atomic_json(WORK/'launch_contract.json',dict(commit=COMMIT,identity=binding,script_sha256=digest(__file__),seed=SEED,order=ORDER,config=str(CONFIG),created_at=stamp(),fresh=True,auto_resume=False,previous_failed_work=str(PREVIOUS_FAILED_WORK),baseline_reference=str(REFERENCE),baseline_run=str(REFERENCE_RUN),baseline_reference_sha256=reference_sha,runs={name:str(WORK/'runs'/name/'seed_4101') for name in ORDER},panels=panels,independent_upper_episodes_each=200,periodic_notifications=False))
- validation=run_child('validation',['-m','hrl_v2.validate','--config',str(CONFIG),'--output',str(WORK/'validation'),'--skip-benchmark'])
- wait_children([validation],'validation')
- v=read(WORK/'validation/validation.json')
- if v.get('status')!='passed' or v.get('identity')!=binding: raise RuntimeError('Validation status/identity check failed')
- jobs=[]
+ binding=identity(WORK,CONFIG);git_states();jobs=[];failures=[];start=time.perf_counter()
+ # Queue checks GPU once before this launch; baseline child processes do not check GPU.
-   require_identity(binding);require_clean()
-   if digest(REFERENCE)!=reference_sha: raise RuntimeError('Frozen reference changed externally')
-   output=WORK/'runs'/name/'seed_4101'
-   if output.exists(): raise FileExistsError(output)
-   jobs.append(run_child(name,['-m','hrl_v2.train_upper','--project-root',str(WORK),'--config',str(CONFIG),'--baseline',name,'--seed',str(SEED),'--fresh','--output',str(output)]))
+   output=WORK/f'runs/{name}/seed_{SEED}';check(not output.exists(),'new_baseline_output',str(output))
+   cmd=[str(PY),'-B','-u',str(OUTM/'checked_entry.py'),'hrl_v2.train_upper','--project-root',str(WORK),'--config',str(CONFIG),'--baseline',name,'--seed',str(SEED),'--fresh','--output',str(output)]
+   logfile=WORK/f'logs/seed_{SEED}_{name}.log';handle=logfile.open('x',encoding='utf-8');t=time.perf_counter()
+   child=subprocess.Popen(cmd,cwd=WT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
+   job=dict(name=name,child=child,handle=handle,command=cmd,started=t,started_at=now());jobs.append(job)
-  record_failure(sys.exc_info()[1],no_children_killed=True)
-  if jobs: wait_children(jobs,'failed_waiting_for_children')
+  for j in jobs:j['child'].wait();j['handle'].close()
- wait_children(jobs,'training')
- require_identity(binding);require_clean()
- state('checking_baseline_anchors',completed=list(ORDER))
- anchors={name:verify_baseline(name,binding) for name in ORDER}
- atomic_json(WORK/'anchor_checks.json',anchors)
- jobs=[]
- try:
-  for key,p in panels.items():
-   require_identity(binding);require_clean()
-   if digest(REFERENCE)!=reference_sha: raise RuntimeError('Frozen reference changed externally')
-   arguments=['-m','hrl_v2.evaluate','--project-root',str(WORK),'--config',str(CONFIG),'--mode','paper','--policy','deterministic','--manifest',p['manifest'],'--checkpoint',p['proposed_checkpoint']]
-   for name,flag in (('NC','--nc-checkpoint'),('TOO','--too-checkpoint'),('TFOO','--tfoo-checkpoint')): arguments += [flag,p['baseline_checkpoints'][name]]
-   arguments += ['--output',p['output']]
-   jobs.append(run_child('paper_dev_'+key,arguments))
- except BaseException:
-  record_failure(sys.exc_info()[1],no_children_killed=True)
-  if jobs: wait_children(jobs,'failed_waiting_for_children')
-  raise
- wait_children(jobs,'paper_evaluation')
- require_identity(binding);require_clean()
- state('complete',completed=list(ORDER),panels=panels,analysis_started=False)
-
-if __name__=='__main__':
- try:
-  with lock(): main()
- except BaseException as error:
-  record_failure(error)
-  state('failed',error=type(error).__name__,message=str(error),auto_resume=False)
-  raise
+ pending=list(jobs);completed=[]
+ while pending:
+  for j in list(pending):
+   code=j['child'].poll()
+   if code is None:continue
+   j['handle'].close();pending.remove(j)
+   row=dict(name=j['name'],pid=j['child'].pid,command=j['command'],exit_code=code,started_at=j['started_at'],ended_at=now(),wall_seconds=time.perf_counter()-j['started']);completed.append(row)
+   write(WORK/f"logs/seed_{SEED}_{j['name']}.process.json",row)
+   if code:failures.append(row)
+  write(WORK/f'baseline_progress_seed_{SEED}.json',dict(status='failed_waiting_for_children' if failures else 'running',children=[dict(name=j['name'],pid=j['child'].pid) for j in pending],completed=completed,failures=failures,no_children_killed=True))
+  if pending:time.sleep(2)
+ check(not failures,'baseline_children',failures)
+ device=configure();checks={name:verify_baseline(name,binding,device) for name in ORDER}
+ write(OUTM/f'seed_{SEED}/baseline_checks.json',dict(status='passed',checks=checks,wall_seconds=time.perf_counter()-start,processes=completed))
+ git_states()
+if __name__=='__main__':main()
```

# v2.4.0 五 seed 汇总

4101 主终点完整对象复算逐位一致；协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a

## 主终点与次要比较

20场景；先对5 seed的场景池化值取算术均值，场景等权；Wilcoxon双侧，zero_method=wilcox；次要比较未作多重校正。

| 候选 | 参考 | 候选均值s | 参考均值s | 相对差% | 胜场 | Wilcoxon p | 配对t p | 符号p |
|---|---|---|---|---|---|---|---|---|
| Proposed | TFOO | 2.186135074488039 | 0.8238365867714856 | 165.36028013215991 | 5 | 0.001209259033203125 | 0.007743187595321528 | 0.04138946533203125 |
| Proposed | NC | 2.186135074488039 | 4.961494927565456 | -55.93797622683958 | 17 | 0.0003223419189453125 | 0.0002686666384176536 | 0.0025768280029296875 |
| Proposed | CA | 2.186135074488039 | 1.31684462025671 | 66.01313783412554 | 13 | 0.956329345703125 | 0.07347394176147057 | 0.26317596435546875 |
| Proposed | TOO | 2.186135074488039 | 0.9763757301512708 | 123.90305360717426 | 13 | 0.6476554870605469 | 0.018857903496378135 | 0.26317596435546875 |
| Proposed | Abl_TwoStage | 2.186135074488039 | 1.2325397624090162 | 77.36832037087449 | 13 | 0.7285060882568359 | 0.08690606502638973 | 0.26317596435546875 |
| Proposed | Abl_HoverSAC | 2.186135074488039 | 1.6141817854923999 | 35.43301591779311 | 13 | 0.8983173370361328 | 0.2904890064183814 | 0.26317596435546875 |
| Proposed | Abl_UpperCVX | 2.186135074488039 | 0.8904398533815179 | 145.51181825319395 | 12 | 0.5958194732666016 | 0.011055345891557421 | 0.5034446716308594 |
| Abl_TwoStage | TFOO | 1.2325397624090162 | 0.8238365867714856 | 49.60973841173868 | 2 | 6.29425048828125e-05 | 0.022622484067845006 | 0.0004024505615234375 |

## 主终点逐场景

| 场景 | Proposed s | TFOO s | 差s | 胜 |
|---|---|---|---|---|
| 1 | 0.9841629995948745 | 0.9185714485282821 | 0.06559155106659242 | False |
| 2 | 0.8020561009033205 | 0.7864488072265583 | 0.015607293676762168 | False |
| 3 | 0.7013282787781586 | 0.7309665378727137 | -0.029638259094555086 | True |
| 4 | 1.073732644896943 | 0.9356193851137157 | 0.13811325978322742 | False |
| 5 | 0.819852298787813 | 0.8334219722616574 | -0.013569673473844412 | True |
| 6 | 0.7950195679286667 | 0.7717287153977397 | 0.023290852530926998 | False |
| 7 | 0.8131930824462863 | 0.8012647724415386 | 0.0119283100047477 | False |
| 8 | 5.083377565685743 | 0.7740449579304276 | 4.309332607755316 | False |
| 9 | 5.326335359819035 | 0.838835289127668 | 4.487500070691367 | False |
| 10 | 6.087610961126887 | 1.0134499013834284 | 5.074161059743458 | False |
| 11 | 5.306308773394555 | 0.9203723648507618 | 4.3859364085437935 | False |
| 12 | 1.0968616105450335 | 1.0119040898698817 | 0.08495752067515183 | False |
| 13 | 1.3319439330543925 | 0.7158340306834036 | 0.616109902370989 | False |
| 14 | 0.7314850521846803 | 0.7701087917919708 | -0.038623739607290486 | True |
| 15 | 0.7765984137825293 | 0.7876523145115201 | -0.011053900728990818 | True |
| 16 | 0.7752793042887083 | 0.7795204237720237 | -0.004241119483315403 | True |
| 17 | 0.9409497501388115 | 0.8615263742365352 | 0.07942337590227633 | False |
| 18 | 0.7768046925070367 | 0.7449641931618695 | 0.031840499345167195 | False |
| 19 | 5.2029196210789115 | 0.7377919904702483 | 4.465127630608663 | False |
| 20 | 4.2968814788183884 | 0.7427053747977714 | 3.554176104020617 | False |

## 每seed主终点

| seed | Proposed s | TFOO s | 相对差% | 胜场 | Wilcoxon p |
|---|---|---|---|---|---|
| 4101 | 0.8511244224924497 | 0.9459839808136465 | -10.027607258170224 | 13 | 0.009435653686523438 |
| 4102 | 0.9134134471794786 | 0.8411420729393928 | 8.592053181638093 | 6 | 0.029575347900390625 |
| 4103 | 7.560720106158246 | 0.7689560310519243 | 883.24478914812 | 4 | 0.0003223419189453125 |
| 4104 | 0.7980702277519404 | 0.7669025218483826 | 4.064102674801173 | 7 | 0.15364646911621094 |
| 4105 | 0.8073471688580799 | 0.7961983272040823 | 1.400259366676604 | 12 | 0.8983173370361328 |

Proposed场景等权均值更低的seed数：1/5。

## 主比较cost/s

| 参考 | 候选cost/s | 参考cost/s | 相对差% | 胜场 |
|---|---|---|---|---|
| TFOO | 17.106744833608637 | 7.461895006260083 | 129.25469762382212 | 4 |
| NC | 17.106744833608637 | 36.94896929234326 | -53.70169950274207 | 17 |
| CA | 17.106744833608637 | 10.88314443480691 | 57.185682282201064 | 13 |
| TOO | 17.106744833608637 | 8.495679880018113 | 101.35816173869495 | 13 |

## 跨seed均值及样本标准差

| 组 | 均值s | 样本标准差s |
|---|---|---|
| Proposed | 2.186135074488039 | 3.004830201179013 |
| NC | 4.961494927565457 | 0.08259995654657365 |
| CA | 1.31684462025671 | 0.0 |
| TOO | 0.9763757301512707 | 0.19032980625998416 |
| TFOO | 0.8238365867714859 | 0.07455987905614804 |
| Abl_TwoStage | 1.232539762409016 | 0.8641185120538925 |
| Abl_HoverSAC | 1.6141817854923997 | 0.8030185678729194 |
| Abl_UpperCVX | 0.8904398533815178 | 0.049866792378822425 |

## 仅4101的Alternating

[
  {
    "candidate": "Proposed",
    "reference": "Abl_Alternating",
    "category": "ablation",
    "primary_endpoint": false,
    "scenarios": 20,
    "candidate_mean_s": 0.8511244224924497,
    "reference_mean_s": 1.5607963534607499,
    "relative_difference_percent": -45.468579510372784,
    "wins": 20,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 0.9720701842616247,
        "reference_s": 1.7623849237947562,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 0.8021967726598997,
        "reference_s": 1.3917750370556785,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 0.6724414959539258,
        "reference_s": 1.3710271925348159,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 0.8764095181406694,
        "reference_s": 2.267544746058334,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.7716719449601408,
        "reference_s": 1.3700147428196983,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 0.7338231288616196,
        "reference_s": 1.3548418079905977,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 0.7527184086273843,
        "reference_s": 1.625627916072978,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 0.9112562267152239,
        "reference_s": 1.408983214031568,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 0.9750007069136175,
        "reference_s": 1.553576982892692,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.23828700919022,
        "reference_s": 1.867159982654679,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.2225158282891522,
        "reference_s": 2.045092882487999,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.0086200570154382,
        "reference_s": 1.9424073499110899,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 0.7568647140864706,
        "reference_s": 1.1870627615062808,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 0.7021434046818776,
        "reference_s": 1.3934946046347108,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.7456142062640245,
        "reference_s": 1.5719038791240951,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 0.724563764092689,
        "reference_s": 1.6116763922723876,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 0.8235508051082733,
        "reference_s": 1.811021654636314,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 0.7254535068409165,
        "reference_s": 1.2352049488672785,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 0.8008948323784704,
        "reference_s": 1.204591865675871,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 0.8063919348073588,
        "reference_s": 1.2405341841931816,
        "win": true
      }
    ],
    "candidate_unfinished_per_episode": {
      "total": 1409.0,
      "by_priority": [
        128.0,
        1281.0
      ]
    },
    "reference_unfinished_per_episode": {
      "total": 3833.0,
      "by_priority": [
        395.0,
        3438.0
      ]
    },
    "statistics": {
      "mean_d": -0.7096719309683004,
      "sd_d": 0.2377698256790427,
      "nonzero_differences": 20,
      "ttest_rel": {
        "statistic": -13.347990434334804,
        "pvalue": 4.206000151821033e-11,
        "alternative": "two-sided"
      },
      "wilcoxon": {
        "statistic": 0.0,
        "W_plus": 0.0,
        "pvalue": 1.9073486328125e-06,
        "alternative": "two-sided",
        "zero_method": "wilcox"
      },
      "binomtest": {
        "wins": 20,
        "n": 20,
        "pvalue": 1.9073486328125e-06,
        "alternative": "two-sided"
      }
    }
  }
]

## G1（原始检查结果）

### seed 4101

```json
{
  "gate": "G1",
  "passed": true,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 17,
    "candidate_mean_s": 1.2631590689120074,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": 0.03527380599807617,
    "candidate_unfinished": 2701,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0877532188531789,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6157116030273704,
        "reference_s": 1.6444104592272226,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1939639728597455,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2667699911436137,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 1.002453195610072,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0483325001785309,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.8262278495243192,
        "reference_s": 1.7929975688930762,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.2158289414667018,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0759518403852775,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1266000168496686,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3961334137376493,
        "reference_s": 1.4235512864691027,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.302090489329355,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.1552128794491323,
        "reference_s": 2.1435304875514403,
        "win": false
      },
      {
        "scenario": "14",
        "candidate_s": 1.2709871912815802,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8890164241733861,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.229665576995397,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0598654805191055,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.029724181415606,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0992297138937326,
        "reference_s": 1.0874324749192534,
        "win": false
      },
      {
        "scenario": "20",
        "candidate_s": 1.3716628975467293,
        "reference_s": 1.3811481979789915,
        "win": true
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2631590689120074,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7785826587520077,
    "candidate_unfinished": 2701,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0877532188531789,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6157116030273704,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1939639728597455,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2667699911436137,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 1.002453195610072,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0483325001785309,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.8262278495243192,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.2158289414667018,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0759518403852775,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1266000168496686,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3961334137376493,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.302090489329355,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.1552128794491323,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2709871912815802,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8890164241733861,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.229665576995397,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0598654805191055,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.029724181415606,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0992297138937326,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.3716628975467293,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "continue",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 2701.0,
        "by_priority": [
          265.0,
          2436.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4102

```json
{
  "gate": "G1",
  "passed": false,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 6,
    "candidate_mean_s": 4.049713563872141,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": -2.092931721289405,
    "candidate_unfinished": 9647,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0769280678230053,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.957219882774707,
        "reference_s": 1.6444104592272226,
        "win": false
      },
      {
        "scenario": "3",
        "candidate_s": 1.1756442989942328,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2641245036139759,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 10.430198368448849,
        "reference_s": 1.0521553494923395,
        "win": false
      },
      {
        "scenario": "6",
        "candidate_s": 10.87548096836394,
        "reference_s": 1.1384760408483925,
        "win": false
      },
      {
        "scenario": "7",
        "candidate_s": 6.807377636626808,
        "reference_s": 1.7929975688930762,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.1935089145548354,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 11.717027863777094,
        "reference_s": 1.1692782937736506,
        "win": false
      },
      {
        "scenario": "10",
        "candidate_s": 1.1835898227976063,
        "reference_s": 1.173628857873011,
        "win": false
      },
      {
        "scenario": "11",
        "candidate_s": 1.4652366184240753,
        "reference_s": 1.4235512864691027,
        "win": false
      },
      {
        "scenario": "12",
        "candidate_s": 1.4560293372374566,
        "reference_s": 1.3679387915986276,
        "win": false
      },
      {
        "scenario": "13",
        "candidate_s": 2.1707330590077194,
        "reference_s": 2.1435304875514403,
        "win": false
      },
      {
        "scenario": "14",
        "candidate_s": 1.3149600169839422,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 9.712519286507373,
        "reference_s": 0.9609525732166311,
        "win": false
      },
      {
        "scenario": "16",
        "candidate_s": 4.577048919489682,
        "reference_s": 1.282487632713281,
        "win": false
      },
      {
        "scenario": "17",
        "candidate_s": 8.535695089176658,
        "reference_s": 1.117563702734941,
        "win": false
      },
      {
        "scenario": "18",
        "candidate_s": 1.3290555303019154,
        "reference_s": 1.1213220522896896,
        "win": false
      },
      {
        "scenario": "19",
        "candidate_s": 1.05034441363935,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.7015486788996053,
        "reference_s": 1.3811481979789915,
        "win": false
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 13,
    "candidate_mean_s": 4.049713563872141,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.2901315185103074,
    "candidate_unfinished": 9647,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0769280678230053,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.957219882774707,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1756442989942328,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2641245036139759,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 10.430198368448849,
        "reference_s": 5.147813838840298,
        "win": false
      },
      {
        "scenario": "6",
        "candidate_s": 10.87548096836394,
        "reference_s": 6.176136542169513,
        "win": false
      },
      {
        "scenario": "7",
        "candidate_s": 6.807377636626808,
        "reference_s": 3.8867722488441734,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.1935089145548354,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 11.717027863777094,
        "reference_s": 7.7487416580667166,
        "win": false
      },
      {
        "scenario": "10",
        "candidate_s": 1.1835898227976063,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4652366184240753,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.4560293372374566,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.1707330590077194,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.3149600169839422,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 9.712519286507373,
        "reference_s": 4.482054535753929,
        "win": false
      },
      {
        "scenario": "16",
        "candidate_s": 4.577048919489682,
        "reference_s": 3.2290551682240287,
        "win": false
      },
      {
        "scenario": "17",
        "candidate_s": 8.535695089176658,
        "reference_s": 3.7194262801626836,
        "win": false
      },
      {
        "scenario": "18",
        "candidate_s": 1.3290555303019154,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.05034441363935,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.7015486788996053,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "pause_before_PPO",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 9647.0,
        "by_priority": [
          264.0,
          9383.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4103

```json
{
  "gate": "G1",
  "passed": true,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 19,
    "candidate_mean_s": 1.2335955866077004,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": 0.05785264540698987,
    "candidate_unfinished": 2600,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0282696771775008,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.5927914983212932,
        "reference_s": 1.6444104592272226,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1324018563710505,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.1933877096248915,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9820837490463041,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0220195672355912,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.880220907519574,
        "reference_s": 1.7929975688930762,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.148566102265087,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0421891984864116,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1166278188098513,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3824517574086803,
        "reference_s": 1.4235512864691027,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2604721549636797,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.0761018052842286,
        "reference_s": 2.1435304875514403,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2690241313424389,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8697861540901841,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2269534527150194,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0126830600307548,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0650380515271893,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0340497899219707,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.336793290012309,
        "reference_s": 1.3811481979789915,
        "win": true
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2335955866077004,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7837647991577208,
    "candidate_unfinished": 2600,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0282696771775008,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.5927914983212932,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1324018563710505,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.1933877096248915,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9820837490463041,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0220195672355912,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.880220907519574,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.148566102265087,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0421891984864116,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1166278188098513,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3824517574086803,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2604721549636797,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.0761018052842286,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2690241313424389,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8697861540901841,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2269534527150194,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0126830600307548,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0650380515271893,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0340497899219707,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.336793290012309,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "continue",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 2600.0,
        "by_priority": [
          267.0,
          2333.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4104

```json
{
  "gate": "G1",
  "passed": true,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2540270696025186,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": 0.042248287007074836,
    "candidate_unfinished": 2556,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0721224966477774,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6031596767768734,
        "reference_s": 1.6444104592272226,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.2057355658687237,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2657972173813643,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911526498033929,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0354524030564856,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7587400401905509,
        "reference_s": 1.7929975688930762,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.189871607983852,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0582084623322985,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1595635934735598,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4134806455318152,
        "reference_s": 1.4235512864691027,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2982684835857918,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.085251534422322,
        "reference_s": 2.1435304875514403,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2860944023777514,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8861526481996849,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2232447131657367,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.029147324700709,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.068522075751133,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0709963700803165,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.379579480720236,
        "reference_s": 1.3811481979789915,
        "win": true
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2540270696025186,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7801833938115494,
    "candidate_unfinished": 2556,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0721224966477774,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6031596767768734,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.2057355658687237,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2657972173813643,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911526498033929,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0354524030564856,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7587400401905509,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.189871607983852,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0582084623322985,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1595635934735598,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4134806455318152,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2982684835857918,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.085251534422322,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2860944023777514,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8861526481996849,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2232447131657367,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.029147324700709,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.068522075751133,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0709963700803165,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.379579480720236,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "continue",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 2556.0,
        "by_priority": [
          263.0,
          2293.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4105

```json
{
  "gate": "G1",
  "passed": false,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 15,
    "candidate_mean_s": 1.3122700022100928,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": -0.0022342204497352025,
    "candidate_unfinished": 3522,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0333823550614951,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.7153644795993828,
        "reference_s": 1.6444104592272226,
        "win": false
      },
      {
        "scenario": "3",
        "candidate_s": 1.1378792794571935,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2044196211753266,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911966664710372,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0332671570377754,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7676817357822412,
        "reference_s": 1.7929975688930762,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.1503335949764566,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0702690058479523,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.149778145974333,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4626565012634927,
        "reference_s": 1.4235512864691027,
        "win": false
      },
      {
        "scenario": "12",
        "candidate_s": 1.3611197139478628,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.9489989885974737,
        "reference_s": 2.1435304875514403,
        "win": false
      },
      {
        "scenario": "14",
        "candidate_s": 1.5492449225107923,
        "reference_s": 1.3207359705611799,
        "win": false
      },
      {
        "scenario": "15",
        "candidate_s": 0.8830134536907527,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.221368646396858,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0408171771029466,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0858087599161266,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.035281676051104,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.4035181633412535,
        "reference_s": 1.3811481979789915,
        "win": false
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.3122700022100928,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7699740737014839,
    "candidate_unfinished": 3522,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0333823550614951,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.7153644795993828,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1378792794571935,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2044196211753266,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911966664710372,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0332671570377754,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7676817357822412,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.1503335949764566,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0702690058479523,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.149778145974333,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4626565012634927,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.3611197139478628,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.9489989885974737,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.5492449225107923,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8830134536907527,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.221368646396858,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0408171771029466,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0858087599161266,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.035281676051104,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.4035181633412535,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "pause_before_PPO",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 3522.0,
        "by_priority": [
          270.0,
          3252.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

## 第三阶段每cycle训练截断时延均值

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4101 | 0.9555016576594599 | 0.959105403225965 | 0.9858260507385145 | 0.9301807475694058 | 0.8551648214160885 | 0.8902548110934039 | 1.0137436074308321 | 0.833722843365074 | 0.904817172524174 | 0.8750504691367885 |
| 4102 | 4.56082029978258 | 1.267618035151656 | 1.0303367733611526 | 0.978222131248937 | 0.901785418122388 | 0.9341526155953932 | 0.9823592850386769 | 1.0255009840092628 | 1.0662655415993754 | 0.854430896865993 |
| 4103 | 0.8538041190749557 | 0.8929619567920938 | 0.7813592435547744 | 0.8188113641898896 | 0.7622005940361707 | 0.8493788715906959 | 0.8378443244234897 | 0.8105614348003645 | 0.8393920823501386 | 2.1765432792422597 |
| 4104 | 1.0636829310378408 | 0.9474282653016521 | 0.8733580095775949 | 0.800569367486841 | 0.9835933256234787 | 0.888086707359047 | 0.8046641726748609 | 0.7238240284909607 | 0.8313731974479813 | 0.7397451204098507 |
| 4105 | 0.896705206667046 | 0.9596474677082206 | 0.9714391674406133 | 1.0638001274566737 | 0.7995567334112247 | 0.9735088296014492 | 0.9126007031859669 | 0.9492399176773312 | 0.8362576928012105 | 0.7934096295412487 |

## 每步实测墙钟

| seed | 步骤 | 秒 |
|---|---|---|
| None | validation | 119.03670050000001 |
| None | validation_check | 1.99920809999999 |
| None | RP-W | 1.992780499999995 |
| None | RP-S2 | 194.2456917 |
| None | RP-S3 | 105.02618910000001 |
| None | RP-T | 163.72535520000002 |
| None | RP-REF | 28.847659999999905 |
| 4102 | reference | 639.2486046999999 |
| 4102 | stage1 | 14241.4202481 |
| 4102 | stage2 | 4166.973567300001 |
| 4102 | stage3 | 3520.6119131999985 |
| 4102 | baselines | 3971.784667 |
| 4102 | test | 2490.013923899998 |
| 4103 | reference | 586.3895402000016 |
| 4103 | stage1 | 10961.762414699999 |
| 4103 | stage2 | 2690.917072700002 |
| 4103 | stage3 | 2937.6100326000014 |
| 4103 | baselines | 3974.7364871000027 |
| 4103 | test | 2699.6804955000043 |
| 4104 | reference | 584.5319776000033 |
| 4104 | stage1 | 16445.877860200002 |
| 4104 | stage2 | 2805.417653800003 |
| 4104 | stage3 | 3024.710041600003 |
| 4104 | baselines | 4039.3930884 |
| 4104 | test | 2061.8538688000117 |
| 4105 | reference | 675.464304699999 |
| 4105 | stage1 | 12234.703220099997 |
| 4105 | stage2 | 2786.5228418999905 |
| 4105 | stage3 | 2952.7097021999944 |
| 4105 | baselines | 3940.0246805999923 |
| 4105 | test | 2145.030444899996 |

## 磁盘占用（文件字节数之和）

| seed | 字节 |
|---|---|
| 4102 | 24253680101 |
| 4103 | 24309708414 |
| 4104 | 24252275943 |
| 4105 | 24274743949 |
| 4101 | 49488325851 |



## 脚本错误与修正

{
  "number": 1,
  "maximum": 1,
  "step": "background_launch",
  "original_error": "Start-Process could not open logs/queue.out.log because logs was not yet created",
  "python_queue_started": false,
  "reason": "RedirectStandardOutput opens its file before queue main() can create logs",
  "fix": "Create WORK/logs before Start-Process, set ErrorActionPreference=Stop",
  "previous_failed_work": "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333",
  "previous_failed_out": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333",
  "retry_work": "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2",
  "retry_out": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2",
  "failure_directories_preserved_read_only": true
}

## 最终Git状态

```json
{
  "D:\\HRL_UAV_V2": {
    "status": "",
    "branch": "codex/v2-lite-paper-caliber",
    "head": "a9953dc36d9d65f99bcc35044cdfc250ca11bb49"
  },
  "D:\\HRL_UAV_V2_speedup": {
    "status": "",
    "branch": "codex/v2-lite-speedup",
    "head": "25d59ff9f7c3e27c060c63d462235a30ce28930a"
  }
}
```
