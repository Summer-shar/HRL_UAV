# -*- coding: utf-8 -*-
"""Read archived JSON only; never load a checkpoint or run training/evaluation."""
import os
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json, statistics as st, hashlib, collections, datetime
ROOT=Path('D:/HRL_UAV_V2_speedup')
WORK=ROOT/'outputs/lite_v2_4_0/training_4101_20260930_010036'
RUN=WORK/'runs/Proposed/seed_4101'
REF=ROOT/'outputs/lite_v2_4_0/baselines/summary.json'
NODES=[300,400,440,480,520,560,600]
M='缺失'
def read(p):
 return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
def jl(p):
 with p.open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def mean(v):
 v=[x for x in v if isinstance(x,(float,int)) and not isinstance(x,bool)]
 return st.mean(v) if v else None
def fmt(v):
 if v is None:return M
 if isinstance(v,float):return f'{v:.6f}'
 if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
 return str(v)
def schema(d):
 return {k:list(v) if isinstance(v,dict) else sorted(set(q for x in v if isinstance(x,dict) for q in x)) if isinstance(v,list) else [] for k,v in d.items()}
def no_pairs(d):
 if isinstance(d,dict):return {k:no_pairs(v) for k,v in d.items() if k!='pairs'}
 if isinstance(d,list):return [no_pairs(x) for x in d]
 return d
train=jl(RUN/'train_metrics.jsonl'); learn=jl(RUN/'learning.jsonl')
ref=read(REF); result=read(RUN/'result.json'); contract=read(RUN/'contract.json'); manifest=read(RUN/'development_manifest.json')
# Inventory first, before any numerical aggregation.
structures={'gates/episode-0300.json':schema(read(RUN/'gates/episode-0300.json'))}
for name,rows in [('train_metrics.jsonl',train),('learning.jsonl',learn)]:
 for tag,d in [('first',rows[0]),('last',rows[-1])]:structures[name+':'+tag]=schema(d)
windows={}; summaries={}; episode_schema={}; mobility={}; pair_checks=[]
for n in NODES:
 folder=RUN/'evaluations'/f'episode-{n:04d}'
 for f in sorted(folder.glob('*.json')):
  d=read(f)
  if n==600:structures[str(f.relative_to(RUN))]=schema(d)
  if f.name.startswith('summary_'):
   summaries[(n,d['identity']['policy'])]=d
  else:
   i=d['identity'];mm=d['metrics'];key=(n,i['policy'],i['mode'],i['scenario'],i['stream'])
   windows[key]={'identity':i,'event_sha256':mm.get('event_sha256'),'canonical_event_sha256':mm.get('canonical_event_sha256'),'mobility':mm.get('terminal_mobility'),'keys':list(mm)}
 for policy in ['deterministic','stochastic']:
  h={k[3:]:v for k,v in windows.items() if k[:3]==(n,policy,'hover')};v={k[3:]:v for k,v in windows.items() if k[:3]==(n,policy,'moving')}
  pair_checks.append(dict(node=n,policy=policy,windows=len(h),same_scenarios_streams=(h.keys()==v.keys()) if v else None,same_event_sha256=all(h[k]['event_sha256']==v[k]['event_sha256'] for k in h) if v and h.keys()==v.keys() else None,same_manifest=all(h[k]['identity']['manifest']==v[k]['identity']['manifest'] for k in h) if v and h.keys()==v.keys() else None))
print('evaluation JSON read complete',flush=True)
# All episode files are inspected serially; no model import and no workers.
episode_counts={}; action_keys=set(); episode_shapes=collections.Counter()
for f in sorted((RUN/'episodes').glob('episode-*.json')):
 d=read(f);n=int(f.stem.split('-')[1]);episode_counts[n]=d.get('ppo',{})
 episode_shapes[json.dumps(schema(d),sort_keys=True)]+=1
 for k in list(d)+list(d.get('metrics',{})):
  if any(s in k.lower() for s in ['position','velocity','speed','target','distance','displacement','trajectory','action']):action_keys.add(k)
print('episode JSON read complete',flush=True)
gates={str(n):read(RUN/'gates'/f'episode-{n:04d}.json') for n in NODES}
phases=[]
for lo,hi in [(1,300),(301,400),(401,600)]:
 rr=[r for r in train if lo<=r['global_episode']<=hi];back=[r for r in rr if r.get('unfinished_parents',-1)>=500]
 def totals(rr):return {k:sum(r[k] for r in rr if r.get(k) is not None) for k in ['work_seconds','checkpoint_seconds','evaluation_seconds','snapshot_seconds','wall_seconds']}
 phases.append(dict(range=[lo,hi],episodes=len(rr),**totals(rr),backlog_episodes=[r['global_episode'] for r in back],backlog_times=totals(back)))
# Learning events: retain observed names; do not fabricate missing losses/advantages.
learning_keys=collections.Counter(k for r in learn for k in r)
ppo_keys=sorted(set(k for r in learn for x in r.get('ppo',[]) for k in x))
curves=[]
for lo in range(1,601,10):
 hi=lo+9;rr=[r for r in train if lo<=r['global_episode']<=hi];ll=[r for r in learn if lo<=r['global_episode']<=hi];pe=[x for r in ll for x in r.get('ppo',[])]
 a=episode_counts.get(lo-1,{});b=episode_counts.get(hi,{})
 curves.append(dict(start=lo,end=hi,truncated_delay_s=mean([r.get('truncated_delay_s') for r in rr]),unfinished=mean([r.get('unfinished_parents') for r in rr]),cost_per_s=mean([r.get('cost_per_s') for r in rr]),propulsion_j=mean([r.get('energy_j',{}).get('propulsion') for r in rr]),alpha=mean([r.get('alpha') for r in ll]),entropy_learning=None,actor_relative_entropy_train=mean([r.get('actor_relative_entropy') for r in rr]),critic_residual_mse=mean([v for r in ll for c in r.get('critics',[]) for v in c.get('residual_mse',[])]),ppo_kl=mean([x.get('kl') for x in pe]),ppo_accepted_epochs=sum(x.get('accepted',False) for x in pe),ppo_attempted=b.get('actor_steps_attempted',0)-a.get('actor_steps_attempted',0),ppo_accepted=b.get('actor_steps_accepted',0)-a.get('actor_steps_accepted',0),ppo_rollbacks=b.get('rollbacks',0)-a.get('rollbacks',0)))
details=[];comparisons=[]
manifest_equal=manifest==ref['manifest'];source_identity_equal=contract['identity']==ref['identity']
ref_events={}
for f in REF.parent.glob('hover_*.json'):
 d=read(f);i=d['identity'];ref_events[f.name]=dict(identity=i,event=d['metrics'].get('event_sha256'))
for (n,policy),s in sorted(summaries.items()):
 for mode,b in s['by_mode'].items():
  mm=[v['mobility'] for k,v in windows.items() if k[:3]==(n,policy,mode) and v['mobility']]
  mob={k:mean([x.get(k) for x in mm]) for k in sorted(set(k for x in mm for k in x))}
  row=dict(node=n,policy=policy,mode=mode,windows=b.get('windows'),truncated_delay_s=b.get('truncated_delay_s'),scenario_mean=mean([x['truncated_delay_s'] for x in s['per_scenario'][mode].values()]),weighted_completed_delay_s=b.get('weighted_completed_delay_s'),cost_per_s=b.get('cost_per_s'),stage_delay_s=b.get('stage_delay_s'),priority=b.get('priority'),energy_j=b.get('energy_j'),unfinished_per_episode=b['unfinished_parents']/b['windows'],terminal_mobility_mean=mob)
  details.append(row)
  for label,base in ref['per_scenario'].items():
   cand=s['per_scenario'][mode];compatible=manifest_equal and source_identity_equal and cand.keys()==base.keys()
   # Compare exogenous event hashes against both reference policy streams per scenario.
   matching=[]
   for k,v in windows.items():
    if k[:3]!=(n,policy,mode):continue
    for stream in ref['manifest']['policy_streams']:
     rv=ref_events.get(f'{label}_{k[3]:02d}_{stream}.json')
     matching.append(bool(rv and rv['event']==v['event_sha256']))
   compatible=compatible and bool(matching) and all(matching)
   comparisons.append(dict(node=n,policy=policy,mode=mode,baseline=label,comparable=compatible,event_match=all(matching),candidate_mean=mean([x['truncated_delay_s'] for x in cand.values()]) if compatible else None,reference_mean=mean([x['truncated_delay_s'] for x in base.values()]) if compatible else None,wins=sum(cand[k]['truncated_delay_s']<base[k]['truncated_delay_s'] for k in cand) if compatible else None,candidate_unfinished_per_episode=mean([x['unfinished_parents']/x['episodes'] for x in cand.values()]) if compatible else None,reference_unfinished_per_episode=mean([x['unfinished_parents']/x['episodes'] for x in base.values()]) if compatible else None))
# Capture queue and NC once, never wait for it.
queue=read(WORK/'queue_status.json');nc=read(WORK/'runs/NC/seed_4101/progress.json');process=read(WORK/'logs/Proposed.process.json');diagnosis=no_pairs(read(RUN/'diagnosis_required.json'))
now=datetime.datetime.now().astimezone().isoformat()
data=dict(captured_at=now,run=str(RUN),reference=str(REF),structures=structures,queue=queue,process=process,phases=phases,gates=gates,diagnosis=diagnosis,details=details,comparisons=comparisons,curves=curves,pair_checks=pair_checks,manifest=manifest,reference_manifest=ref['manifest'],source_identity_equal=source_identity_equal,source_identity=contract['identity'],evaluation_identities={f'{n}/{p}':s['identity'] for (n,p),s in summaries.items()},learning_keys=dict(learning_keys),ppo_epoch_keys=ppo_keys,nc=nc,episode_action_keys=sorted(action_keys),episode_schema_groups={k:v for k,v in episode_shapes.items()})
(OUT/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,2,figsize=(14,14))
fields=[('truncated_delay_s','Training truncated delay (s)'),('unfinished','Unfinished parents'),('cost_per_s','Cost / s'),('propulsion_j','Propulsion (J)'),('alpha','SAC alpha (learning log)'),('actor_relative_entropy_train','Actor relative entropy (train log)'),('critic_residual_mse','Critic residual MSE'),('ppo_kl','PPO epoch KL')]
for ax,(key,title) in zip(axes.flat,fields):
 ax.plot([x['end'] for x in curves],[float('nan') if x[key] is None else x[key] for x in curves],marker='.',markersize=3)
 ax.axvline(300,color='gray',linestyle='--');ax.axvline(400,color='gray',linestyle='--');ax.set_title(title);ax.set_xlabel('Episode (10-episode mean)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
lines=[]
def add(t=''):lines.append(t)
def table(headers,rows):
 add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']*len(headers))+'|')
 for row in rows:add('| '+' | '.join(fmt(x).replace('|','/') for x in row)+' |')
 add()
def block(d):add('```json\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n```\n')
add('# v2.4.0 seed 4101 只读结果分析\n')
table(['项目','结果'],[['Proposed状态',result['status']],['最终轮次',result['final_episode']],['开发最佳轮次',result['development_best_episode']],['G1',gates['300']['passed']],['600轮移动/悬停场景均值(s)',[gates['600']['paired']['candidate_mean_s'],gates['600']['paired']['reference_mean_s']]],['600轮改善/胜场',[gates['600']['paired']['relative_improvement'],gates['600']['paired']['wins']]],['Proposed进程wall_seconds',process.get('wall_seconds')],['采样时间',now]])
add('仅读取既有JSON及选择规则源码；无训练、无评估、无checkpoint加载。表格六位小数；summary.json保留原精度。D=确定性(stream 0)，S=随机(stream 1、2)，H=hover，M=moving。缺失不补造。\n')
add('## 1. 队列与耗时\n');add('queue_status.json.wall_seconds：缺失。队列状态及当前任务：'+fmt({k:queue.get(k) for k in ['status','current','updated_at','pid','child_pid']})+'。Proposed.process.json.exit_code='+fmt(process.get('exit_code'))+'。\n')
table(['阶段','轮数','work秒','checkpoint秒','eval秒','snapshot秒','wall秒'],[[str(x['range']),x['episodes'],*[x[k] for k in ['work_seconds','checkpoint_seconds','evaluation_seconds','snapshot_seconds','wall_seconds']]] for x in phases])
add('上述为日志字段求和；eval包括评价路径自身工作，不另把内部存档重复计入checkpoint。进程墙钟与逐轮wall统计范围不同，不将差额猜成某项开销。\n')
table(['阶段','积压轮数(unfinished≥500)','work秒','checkpoint秒','eval秒','wall秒'],[[str(x['range']),len(x['backlog_episodes']),*[x['backlog_times'][k] for k in ['work_seconds','checkpoint_seconds','evaluation_seconds','wall_seconds']]] for x in phases])
add('积压轮编号：'+fmt([i for x in phases for i in x['backlog_episodes']])+'。\n')
add('## 2. Gate与最佳轮次\n')
g=gates['300'];table(['G1对比','candidate均值s','reference均值s','wins'],[[k,g[k]['candidate_mean_s'],g[k]['reference_mean_s'],g[k]['wins']] for k in ['equal','local']]);add('G1 unfinished report_only原文：\n');block(g['unfinished'])
table(['轮次','moving s','hover s','wins/20','改善比例','moving未完成','hover未完成','弱信号'],[[n,*[gates[str(n)]['paired'][k] for k in ['candidate_mean_s','reference_mean_s','wins','relative_improvement','candidate_unfinished','reference_unfinished']],gates[str(n)]['weak_upper_signal']] for n in NODES[1:]])
add('diagnosis_required.json（递归仅移除pairs，其他值原样）：\n');block(diagnosis)
add('result.json.development_best_episode=400。选择规则见src/hrl_v2/run.py:344–345：排除300，在其余节点的主评估moving.cost_per_s中选最小，平局取较早轮次；不是按G2胜场或随机面板选择。\n')
table(['节点','确定性moving.cost_per_s'],[[n,summaries[(n,'deterministic')]['by_mode']['moving']['cost_per_s']] for n in NODES[1:]])
add('注意：G1文件中的per_episode.total仍是20个场景归一化后的合计，不是每个窗口均值；candidate 2701/20=135.05，equal 2474/20=123.7，local 13545/20=677.25。第4节统一按单episode报告。\n')
add('## 3. 各节点评估明细\n');add('以下时延为summary.by_mode的池化值；stage为加权驻留时延分解。priority的mean_delay_ms只含已完成任务。energy_j是该面板全部窗口原始焦耳总和，不是每窗口；未乘推进权重。\n')
def label(d):return f"{d['node']}/{d['policy'][0].upper()}/{d['mode'][0].upper()}"
table(['节点/面板/模式','窗口','截断s','已完成加权s','cost/s','upload s','ready s','u2u s','cpu s'],[[label(d),d['windows'],d['truncated_delay_s'],d['weighted_completed_delay_s'],d['cost_per_s'],*[d['stage_delay_s'].get(k) for k in ['upload','ready','u2u','cpu']]] for d in details])
table(['节点/面板/模式','高优均值ms','高优未完成','低优均值ms','低优未完成','upload J','cpu J','u2u J','propulsion J'],[[label(d),d['priority'][0].get('mean_delay_ms'),d['priority'][0]['unfinished'],d['priority'][1].get('mean_delay_ms'),d['priority'][1]['unfinished'],*[d['energy_j'][k] for k in ['upload','cpu','u2u','propulsion']]] for d in details])
add('terminal_mobility未保留在summary.by_mode中，但逐窗口文件存在。下面是逐窗口等权均值；位移是终端位移，不是UAV位移；距离只在终点测量。\n')
table(['节点/面板/模式','终端位移m','反弹终端数均值','终点终端至关联UAV距离m'],[[label(d),*[d['terminal_mobility_mean'].get(k) for k in ['mean_displacement_m','terminals_reflected','final_mean_terminal_to_serving_uav_m']]] for d in details])
table(['节点/面板','hover窗口数','相同场景+stream','相同event_sha256','相同manifest'],[[f"{x['node']}/{x['policy'][0].upper()}",x['windows'],x['same_scenarios_streams'],x['same_event_sha256'],x['same_manifest']] for x in pair_checks]);add('300只有hover，无moving，配对列为缺失。\n')
add('### 1.265与1.402的口径\n')
table(['节点','池化hover截断s','场景等权hover截断s'],[[n,summaries[(n,'deterministic')]['by_mode']['hover']['truncated_delay_s'],mean([x['truncated_delay_s'] for x in summaries[(n,'deterministic')]['per_scenario']['hover'].values()])] for n in [300,600]])
add('两者均为确定性hover、stream 0、同一20个dev场景；checkpoint不同。1.265是300池化结果，G2的1.402是600场景等权结果；即便统一场景等权口径，600也较300退化，不能归因于评估集或随机/确定性切换。两个identity原文：\n')
for n in [300,600]:add(f'{n}：');block(summaries[(n,'deterministic')]['identity'])
add('两者manifest均对应development_manifest.json，下列种子表对两者及REF完全相同；声明policy_streams='+fmt(manifest['policy_streams'])+'，确定性执行实际使用0。schema='+manifest['schema']+'，split='+manifest['split']+'。\n')
table(['scenario','layout_seed','load_seed','arrival_seed'],[[x[k] for k in ['scenario','layout_seed','load_seed','arrival_seed']] for x in manifest['scenarios']])
add('## 4. 与冻结基线配对\n');add('RUN契约与REF源码/config/scene identity完整相等：'+fmt(source_identity_equal)+'；manifest完整相等：'+fmt(manifest_equal)+'。逐窗口外生event_sha256也逐场景核对。D与REF的stream标签不同，但基线使用两条等价策略流；先按场景聚合，绝不把stream算成独立场景。\n')
table(['基线','场景均值s','未完成/episode'],[[k,mean([x['truncated_delay_s'] for x in v.values()]),mean([x['unfinished_parents']/x['episodes'] for x in v.values()])] for k,v in ref['per_scenario'].items()])
add('下表每个基线单元为“胜场/20”；候选均值与每episode未完成同列给出。若不满足场景/身份/事件配对，显示不可比。\n')
table(['节点/面板/模式','候选场景均值s','候选未完成/episode','local胜场','equal胜场','cvx胜场','greedy胜场'],[[label(d),d['scenario_mean'],d['unfinished_per_episode'],*[next((x['wins'] if x['comparable'] else '不可比' for x in comparisons if (x['node'],x['policy'],x['mode'],x['baseline'])==(d['node'],d['policy'],d['mode'],k)),None) for k in ['hover_local','hover_equal','hover_cvx','hover_greedy']]] for d in details])
add('## 5. 学习曲线\n');add('![每10轮学习曲线](learning_curves.png)\n')
add('learning.jsonl中有alpha、critics[].residual_mse（critic残差平方均值，未另存名为loss的字段）、PPO各epoch的kl/accepted。SAC熵、PPO value loss、优势均值/方差、actor attempted/accepted计数、rollbacks在learning.jsonl中缺失。熵替代展示train_metrics.actor_relative_entropy，不能冒称来自learning日志；PPO计数从episode JSON的累计ppo计数相减获得，不从accepted epoch数猜算。\n')
add('每10轮均值：alpha与critic按已记录学习事件/fit平均，KL按已记录epoch平均；PPO A/B/R为该10轮尝试步数/接受步数/回滚次数，属于增量总数。无事件的alpha、critic、KL写缺失。\n')
table(['轮次','截断s','未完成','cost/s','propulsion J','alpha','相对熵(train)','critic MSE','PPO KL','PPO A/B/R'],[[f"{c['start']}-{c['end']}",c['truncated_delay_s'],c['unfinished'],c['cost_per_s'],c['propulsion_j'],c['alpha'],c['actor_relative_entropy_train'],c['critic_residual_mse'],c['ppo_kl'],f"{c['ppo_attempted']}/{c['ppo_accepted']}/{c['ppo_rollbacks']}"] for c in curves])
add('301–400阶段需结合上表首尾10轮看训练波动：场景随轮次变化，不能当作配对因果证据。固定SAC的开发评价从300 hover到400 moving反而更慢（见G2），未证明PPO收益。401–600阶段确定性hover在440–560显著变差、600回落但仍差于300；固定场景hover也退化，说明问题不限于移动执行。仅凭日志不能断言熵、critic或动作塌缩是根因。\n')
add('重点观察：301–310至391–400训练截断时延为1.449643→1.221074 s、cost/s为11.865205→10.343819，但未完成数125.9→133.7，且区间波动而非单调改善。300/400确定性hover指标完全相同，符合SAC固定。交替阶段411–420截断均值跃至6.748858 s，461–470达12.353714 s；这些不同训练场景不能直接做因果对照。固定开发集520轮hover的CPU段为2.153110 s（300轮0.249435 s），低优已完成时延11211.793655 ms（300轮2516.305649 ms），支持低优计算排队恶化这一现象。600轮moving上传段1.145275 s高于hover 0.905631 s，而CPU段0.243097 s略低于hover 0.263211 s；最终终端距离481.011142 m对413.191659 m。上传/几何退化与移动效果不佳一致，但没有轨迹，不能由此确认具体动作根因。\n')
add('## 6. 上层动作行为\n');add('已检查全部episode JSON的顶层和metrics字段，以及全部节点逐窗口评价。UAV逐宏步位移、速度、目标点、逐宏步终端距离：未记录，不能计算时间平均位移/距离。存在terminal_mobility.final_mean_terminal_to_serving_uav_m，仅可比较最终时刻的距离，已在第3节列出；mean_displacement_m描述终端运动，不可误当UAV运动。未重跑评价。\n')
add('## 7. NC当前进度\n');add('读取时刻：'+now+'，不等待结束。\n');block(nc)
add('## 附录：实际文件结构\n');add('每项格式为“顶层key: 第二层keys”；标量用[]，列表展示直接元素对象的key并集。相同结构的文件合并列出，但覆盖episode-0600目录所有文件。summary.json.structures保留逐文件映射。\n')
groups=collections.defaultdict(list)
for filename,s in structures.items():groups[json.dumps(s,ensure_ascii=False,sort_keys=True)].append(filename)
for idx,(s,names) in enumerate(groups.items(),1):
 add(f'### 结构组{idx}（{len(names)}项）\n')
 if len(names)>10:
  # Exact filename coverage via contiguous scenario groups, with validation.
  prefixes=collections.defaultdict(list)
  for name in names:
   stem=Path(name).stem;parts=stem.rsplit('_',2);prefixes[(parts[0],parts[-1])].append(parts[-2])
  for (prefix,stream),scenarios in prefixes.items():add(f'- evaluations/episode-0600/{prefix}_{{'+','.join(sorted(scenarios))+f'}}_{stream}.json')
 else:
  for name in names:add('- '+name)
 add();obj=json.loads(s)
 for k,v in obj.items():add('- `'+k+'`: '+(', '.join(v) if v else '[]'))
 add()
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('DONE',len(lines),'lines',len('\n'.join(lines)),'chars',flush=True)
