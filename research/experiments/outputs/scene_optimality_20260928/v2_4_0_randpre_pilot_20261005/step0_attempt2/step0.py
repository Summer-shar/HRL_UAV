# Derived from OUTM/common.py, stage2_seed.py, stage3_seed.py and NEW/logs_diag.py.
# Step 0 only: source checks, existing JSON/checkpoint reads and offline 10000-action statistics.
from pilot_common import *
import inspect,ast
import torch
from hrl_v2.replay import Replay,probabilities,resolve
from hrl_v2.sac import SAC
from hrl_v2.schedule import phase
from hrl_v2.upper_baselines import upper_phases
from hrl_v2.randomness import derive,PURPOSES
from hrl_v2.checkpoint import identity,load,runtime_record
from hrl_v2.runtime import configure
from hrl_v2.run import read_config
from hrl_v2.ppo import PPO
from hrl_v2.collector import _collect
from hrl_uav.industrial_deployment_env import IndustrialDeploymentEnv
FACTS={};STAGE='step0_rules'
def source(fn):return inspect.getsource(fn)
def evidence(fn):return dict(file=inspect.getsourcefile(fn),line=inspect.getsourcelines(fn)[1],source=source(fn))
def main():
 global STAGE
 FACTS.update(status='checking',time=now(),git=git_states(),imports=assert_imports(),gpu=gpu_check())
 device=configure();check(str(device)=='cuda:0','cuda_device',str(device))
 config=read_config(CONFIG,WT);binding=identity(WT,CONFIG);v=read(WORK/'validation/validation.json')
 check(v['status']=='passed','validation_passed');equal(binding,v['identity'],'validation_identity');check(sha(PROTO)==PROTO_SHA,'frozen_protocol_sha',sha(PROTO))
 FACTS.update(identity=binding,runtime=runtime_record(),torch_threads=torch.get_num_threads(),evidence={})
 for name,fn in [('replay_init',Replay.__init__),('replay_set',Replay.set_upper_version),('replay_add',Replay.add),('replay_sample',Replay.sample),('probabilities',probabilities),('sac_event',SAC.event),('phase',phase),('upper_phases',upper_phases),('collect',_collect),('ppo_update',PPO.update),('motion',IndustrialDeploymentEnv.start_motion)]:FACTS['evidence'][name]=evidence(fn)
 check(config['replay']['max_completed_episodes']==55 and 'len(self.episodes)>55' in source(Replay.add),'replay_capacity_55_FIFO')
 r=Replay();r.set_upper_version('a');check(r.flushes==1 and r.upper_version=='a','first_version_flush');r.episodes=[object()];r.set_upper_version('b');check(r.flushes==2 and not r.episodes,'change_version_flush');r.set_upper_version('b');check(r.flushes==2,'same_version_no_flush')
 check('not episode.complete or episode.upper_version!=self.upper_version' in source(Replay.add),'add_rejects_incomplete_or_wrong_version')
 check('size=4' in source(Replay.sample) and 'size=32' in source(Replay.sample) and '(1/count)/q[i]' in source(Replay.sample),'sample_4x32_IS')
 check(np.allclose(probabilities([0,1,0]),np.array([.3,.4,.3]),rtol=0,atol=1e-16),'sample_09_uniform_01_severe')
 ev=source(SAC.event);check('for _ in range(2)' in ev and 'self.counts["events"]%2==0' in ev and 'if not replay.episodes:' in ev and 'lerp_(b,.005)' in ev,'SAC_event_rules')
 check(config['task']['rollout_seconds']/config['task']['lower_interval']/20==90 and 'completed_lower%20==0' in source(_collect),'90_triggers_each_episode')
 check((config['ppo']['batch_episodes'],config['ppo']['epochs'],config['ppo']['minibatch_size'])==(5,6,150),'PPO_5x6x150')
 check('range(6)' in source(PPO.update) and '150' in source(PPO.update),'PPO_code_epochs_minibatch')
 for n in range(1,601):
  p=phase(n);expect=.005 if n<=100 else .0025 if n<=200 else .0005 if n<=500 else 0
  check(p.alpha==expect,'phase_alpha_'+str(n))
  check(p.actor_enabled==(p.layer=='lower' and p.block_episode>2),'phase_actor_'+str(n))
 upp=upper_phases();expected=list(range(301,401))+[n for n in range(401,601) if (n-401)%20<10]
 check([p.global_episode for p in upp]==expected and len(upp)//5==40,'upper_mapping_200_40')
 for fn in (Replay.sample,resolve,SAC.event,SAC._critic_fit,SAC._actor_fit,SAC._critic_targets):check('upper_version' not in source(fn),'no_episode_version_read_'+fn.__name__)
 from hrl_v2.lite_batch import LiteBatch
 check('upper_version' not in inspect.getsource(LiteBatch),'no_episode_version_read_LiteBatch')
 FACTS['episode_upper_version_reads']='Replay.add checks episode labels; Replay.sample/resolve/LiteBatch/SAC update do not. PPO.update checks versions only in upper training.'
 scene=read(SCENE);step=scene['deployment']['max_target_step_m'];check(step==20,'deployment_step_20')
 rng=np.random.default_rng(derive(4104,1,0,1,purpose='upper'));u=rng.uniform(-1,1,(10000,6));a=u.reshape(-1,3,2);d=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1)*step;lengths=np.linalg.norm(d,axis=-1)
 FACTS['offline_motion_distribution']=dict(samples=10000,rng_seed=derive(4104,1,0,1,purpose='upper'),purpose='step0 offline only; never reused by training',before_boundary_and_safety=True,lengths_per_uav=[dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),p10=float(np.quantile(x,.1)),p50=float(np.quantile(x,.5)),p90=float(np.quantile(x,.9)),at_cap=int(np.count_nonzero(x==step))) for x in lengths.T],formula='s=clip(u.reshape(3,2),-1,1); s/=max(norm(s),1); requested=clip(x+20*s,0,map_size); actual=unchanged safety projection(requested-x)',accepts_post_tanh=True,atanh_injection=False)
 FACTS['random_upper_purpose']=dict(original_PURPOSES=PURPOSES,registered=('random_upper' in PURPOSES),new_script_only_mapping='random_upper=13, first unused identifier; existing identifiers unchanged; derive original SeedSequence fields',reason='P1 explicitly requests a dedicated random_upper generator; original source only defines identifiers 1..12')
 STAGE='step0_history';FACTS['seeds']={};inventory=[]
 for s in (4104,4105):
  equal(binding,read(old_run(s)/'contract.json')['identity'],'Proposed_identity_'+str(s))
  z=load(old_run(s)/'checkpoints/episode-0300-evaluated.pt',binding)
  check(z['replay'].upper_version=='hover-0' and len(z['replay'].episodes)==55 and z['replay'].flushes==1,'S300_replay_'+str(s),dict(length=len(z['replay'].episodes),flushes=z['replay'].flushes))
  sc=dict(z['learner']['sac']['counts']);del z;gc.collect()
  orig=read(OUTM/f'seed_{s}/stage3/training_summary.json');check(orig['replay_flushes_delta']==1 and orig['skipped_global_episodes']==[411],'v1_stage3_flush_'+str(s),orig['replay_flushes_delta'])
  epfiles=[old_l(s)/f'episodes/episode-{n:04d}.json' for n in range(411,421)];first=[]
  for n,p in enumerate(epfiles):
   rr=read(p);first.append(dict(global_episode=411+n,sac=rr['sac'],sac_delta=rr['train_metric']['sac_delta'],event_sha256=rr['metrics']['event_sha256'],available_completed_episodes_before_collection=n,availability_evidence='source run_lower: initial flush then exactly one Replay.add per completed rollout'))
   check(rr['train_metric']['sac_delta']['skipped']==(90 if n==0 else 0),'original_first_block_skip_'+str(s)+'_'+str(n))
  c0=hist(s,'E2')['mean_s'];c1=hist(s,'S3-c1')['mean_s'];e5=hist(s,'E5')['mean_s'];tf=hist(s,'TFOO-200')['mean_s']
  expected0,expected1=(.8625,.9154) if s==4104 else (.8912,.8659)
  check(round(c0,4)==expected0 and round(c1,4)==expected1,'v1_c0_c1_vs_NEW_'+str(s),dict(c0=c0,c1=c1))
  curves={}
  for label,root,width in [('stage1',old_run(s),25),('SD',old_sd(s),20),('TFOO',old_tfoo(s),20)]:
   data=rows(root/'train_metrics.jsonl');data=[x for x in data if x.get('global_episode',0)<=300] if label=='stage1' else data
   keyed={x.get('episode',x.get('global_episode')):x for x in data};data=[keyed[k] for k in sorted(keyed)]
   check(len(data)==(300 if label=='stage1' else 200),'original_curve_rows_'+str(s)+'_'+label)
   bins=[dict(first=data[i].get('episode',data[i].get('global_episode')),last=data[i+width-1].get('episode',data[i+width-1].get('global_episode')),rows=width,truncated_delay_s=avg([x['truncated_delay_s'] for x in data[i:i+width]])) for i in range(0,len(data),width)]
   curves[label]=dict(source=str(root/'train_metrics.jsonl'),field='truncated_delay_s (same name as dev; training samples distinct)',bins=bins,tail=tail_label([b['truncated_delay_s'] for b in bins]))
  tfoo=[p for p in sorted((old_tfoo(s)/'checkpoints').glob('*.pt')) if '0000-initial' not in p.name]
  FACTS['seeds'][str(s)]=dict(S300_counts=sc,v1_first_block=first,v1_stage3_flushes_delta=orig['replay_flushes_delta'],dev=dict(c0=c0,c1=c1,E5=e5,TFOO200=tf),curves=curves,TFOO_checkpoints=[str(x) for x in tfoo],TFOO_checkpoint_count=len(tfoo),expected_TFOO_count=10,actual_count_different=len(tfoo)!=10)
  paths=[old_run(s)/'checkpoints/episode-0300-evaluated.pt',old_sd(s)/'checkpoints/episode-0200.pt',*[old_l(s)/f'checkpoints/cycle-{n:02d}.pt' for n in range(1,11)]]
  inventory.extend(dict(seed=s,path=str(p),bytes=p.stat().st_size) for p in paths)
 STAGE='step0_disk';free=shutil.disk_usage(PILOT).free;maxfull=max(x['bytes'] for x in inventory if '0300-evaluated' in x['path']);maxlearner=max(x['bytes'] for x in inventory if 'cycle-' in x['path'] or '0200.pt' in x['path'])
 # Conservative disk estimate: complete S300r plus 40 learner archives per seed,
 # all dev JSON windows and logs budgeted at 12 GB per seed; not measured usage.
 estimate=2*(maxfull+40*maxlearner+12*1024**3)
 FACTS['disk']=dict(free_bytes=free,original_checkpoints=inventory,estimated_two_seed_bytes=estimate,estimate_not_measurement=True,basis='per seed: one full replay checkpoint + 40 full learner checkpoints + 12 GiB evaluation JSON/log reserve')
 check(free>estimate,'disk_space_sufficient',dict(free=free,estimated=estimate))
 FACTS.update(status='passed',checks=CHECKS,time_finished=now());put(PILOT/'step0_facts/facts.json',FACTS)
 report=['# 第0步事实核对','',f'状态：{FACTS["status"]}；时间：{now()}。未运行训练或环境。','',f'原冻结协议 SHA256：{sha(PROTO)}。','', '所有 Replay/SAC/PPO/phase 规则检查通过；代码证据及行号保存在 facts.json。','', '运动映射直接接受 tanh 后的 [-1,1]^6；不需 atanh。离线抽10000个六维向量，未运行环境。','', 'P1新增 random_upper 在原 PURPOSES 中未注册；新脚本单独用13作为该新目的标识，原1–12不改。','', '|seed|v1-c0|v1-c1|E5|TFOO-200|TFOO现有checkpoint数|','|---|---:|---:|---:|---:|---:|']
 for s,d in FACTS['seeds'].items():
  v=d['dev'];report.append(f'|{s}|{v["c0"]:.12f}|{v["c1"]:.12f}|{v["E5"]:.12f}|{v["TFOO200"]:.12f}|{d["TFOO_checkpoint_count"]}|')
  report.extend(['',f'## seed {s} 原训练曲线','', '|运行|窗口|truncated_delay_s|','|---|---|---:|'])
  for label,c in d['curves'].items():
   report.extend(f'|{label}|{b["first"]}–{b["last"]}|{b["truncated_delay_s"]:.12f}|' for b in c['bins']);report.append(f'{label}末段：{c["tail"]["label"]}；改进={c["tail"]["improvement"]:.12f}。')
  report.extend(['','实际 TFOO 存档（照用全部，未补建）：',*[f'- {p}' for p in d['TFOO_checkpoints']]])
 report.extend(['','## 离线单步长度（边界/安全投影前）','',json.dumps(FACTS['offline_motion_distribution'],ensure_ascii=False,indent=2),'',f'磁盘剩余 {free} bytes；两个seed试点估算 {estimate} bytes；估算依据和原存档大小逐项见 facts.json。'])
 textfile(PILOT/'step0_facts/facts.md','\n'.join(report)+'\n');print('STEP0_PASS',json.dumps(dict(disk=FACTS['disk']['free_bytes'],estimate=estimate,TFOO_counts={s:d['TFOO_checkpoint_count'] for s,d in FACTS['seeds'].items()})),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  FACTS.update(status='failed',stage=STAGE,checks=CHECKS);put(PILOT/'step0_facts/facts.json',FACTS);failure(STAGE,e);raise