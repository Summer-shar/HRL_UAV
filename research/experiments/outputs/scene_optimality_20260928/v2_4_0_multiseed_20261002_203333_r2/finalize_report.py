import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,shutil,subprocess,hashlib
import hrl_v2,hrl_uav
WT=Path('D:/HRL_UAV_V2_speedup');MAIN=Path('D:/HRL_UAV_V2')
for mod in (hrl_v2,hrl_uav):assert Path(mod.__file__).resolve().is_relative_to(WT/'src')
OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_multiseed_20261002_203333_r2'
WORK=WT/'outputs/lite_v2_4_0/multiseed_20261002_203333_r2'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
a=read(OUT/'aggregate/aggregate.json');q=read(WORK/'queue_status.json');fc=read(OUT/'final_checks.json')
assert q['status']=='complete' and a['status']=='passed' and fc['status']=='passed'
PROTO=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_test/protocol_v2_4_0_three_stage.json'
actual=hashlib.sha256(PROTO.read_bytes()).hexdigest();assert actual==fc['protocol_sha256']
for root,state in fc['git'].items():
 assert not subprocess.check_output(['git','-C',root,'status','--porcelain']).strip()
 assert subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()==state['head']
lines=[]
def f(x):
 if isinstance(x,bool):return '是' if x else '否'
 if isinstance(x,float):return f'{x:.9g}'
 return str(x)
def table(headers,rows):
 lines.extend(['','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
 lines.extend('| '+' | '.join(f(x) for x in row)+' |' for row in rows);lines.append('')
def section(title):lines.extend(['',f'**{title}**',''])
lines+=['v2.4.0 三阶段协议 seeds 4102–4105 全流程与五 seed 汇总','',f"状态：全部完成。北京时间 {q['started_at']} 启动，{q['ended_at']} 结束；实测墙钟 {q['total_wall_seconds']:.6f} 秒（29 小时 46 分 47.6 秒）。当前无该队列训练进程。",'','主终点：Proposed 场景等权截断时延 2.186135074 s，TFOO 0.823836587 s，相对差 +165.360280%；Proposed 胜 5/20 场景，均值更低的 seed 为 1/5（4101）。']
section('检查结果')
startup=read(OUT/'startup_queue.json');repro=read(OUT/'repro_4101/checks.json')
rows=[['两棵工作树分支、HEAD、干净状态','通过'],['新目录事先不存在、git check-ignore','通过'],['启动磁盘≥150 GiB及GPU/祖先链检查','通过'],['validation及identity','通过'],['协议启动/每seed/汇总前后SHA256','通过'],['队列父进程未初始化CUDA','通过'],['4101主终点完整对象复算逐位一致',a['seed4101_recompute_exact']]]
for name,r in repro['results'].items():rows.append([name,f"{r['status']}；{len(r['checks'])}项；{r['wall_seconds']:.3f} s"])
table(['检查','结果'],rows)
lines+=['RP-S2：10轮事件SHA、截断时延、训练日志非计时字段、两次完整PPO更新日志逐位一致。RP-S3：411–413轮指标、计数增量、每轮90行学习日志逐位一致。RP-T：8组完整metrics及两组CVX diagnostics一致。RP-REF：equal/cvx各一窗口完整metrics一致。RP-W：无evaluated时不停止；有evaluated时只在301停止。','',f'协议SHA256：`{actual}`。未重新冻结协议。']
checks=[]
for s in (4102,4103,4104,4105):
 paths=[WORK/f'references/seed_{s}/checks.json',WORK/f'runs/Proposed/seed_{s}/checks.json',OUT/f'seed_{s}/stage2/checks.json',OUT/f'seed_{s}/stage3/checks.json',OUT/f'seed_{s}/baseline_checks.json',OUT/f'seed_{s}/test/checks.json']
 values=[read(p) for p in paths];assert all(r['status']=='passed' for r in values)
 checks.append([s]+[r['status'] for r in values])
table(['seed','参考','第一阶段','第二阶段','第三阶段','NC/TOO/TFOO','test'],checks)
lines+=['四个新seed的第一阶段均止于300，无301输出；pending/evaluated参数哈希相等，SAC counts等于4101第300轮，PPO batches=0，上层保持初始化。第二阶段200轮/40批PPO，SAC actor哈希与counts冻结，100/200 checkpoint存在。第三阶段上层actor/value/counts/version冻结，SAC增量均为 triggers=9000、skipped=90、events=8910、critic_fits=17820、target_updates=8910、actor_updates=3600；仅411跳过，replay flush增加1，cycle-10 completed=600/pending=None。','', '三组独立基线均完成200轮，100/200 anchor的PPO batches为20/40，typed加载通过；各组初始上层actor/value哈希均与本seed Proposed初始化相同。每seed test为640窗口、6 worker，学习更新和replay写入均为0。']
reportrows=[]
for s in (4102,4103,4104,4105):
 r=read(WORK/f'references/seed_{s}/checks.json');sm=read(OUT/f'seed_{s}/test/test_summary.json');n=sum(v['equal'] for v in sm['stream_consistency']['per_group'].values());total=sum(v['total'] for v in sm['stream_consistency']['per_group'].values())
 reportrows.append([s,f"{r['matching_REF4101_full_metrics']}/{r['total']}",f'{n}/{total}',sm['diagnostics']['TFOO']['solver_calls'],sm['diagnostics']['Abl_UpperCVX']['solver_calls'],sm['diagnostics']['TFOO']['fallback_calls'],sm['diagnostics']['Abl_UpperCVX']['fallback_calls']])
table(['seed','参考与4101完整metrics一致','跨stream完整metrics一致','TFOO求解','UpperCVX求解','TFOO fallback','UpperCVX fallback'],reportrows)
section('主终点与逐场景结果')
lines+=['每个seed先池化每场景4个stream；每场景再对5个seed算术平均，得到20个配对差；最后按场景等权。Wilcoxon双侧、zero_method=wilcox、α=0.05；未剔除seed或场景。']
r=a['primary'];st=r['statistics']
table(['Proposed s','TFOO s','相对差%','胜场/20','Wilcoxon统计量','W+','Wilcoxon p','t统计量','t p','符号p'],[[r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],st['wilcoxon']['statistic'],st['wilcoxon']['W_plus'],st['wilcoxon']['pvalue'],st['ttest_rel']['statistic'],st['ttest_rel']['pvalue'],st['binomtest']['pvalue']]])
table(['场景','Proposed s','TFOO s','差s','Proposed胜'],[[x['scenario'],x['candidate_s'],x['reference_s'],x['candidate_s']-x['reference_s'],x['win']] for x in r['pairs']])
section('每seed主终点')
table(['seed','Proposed s','TFOO s','相对差%','胜场/20','Wilcoxon p'],[[s,r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue']] for s,r in a['per_seed'].items()])
lines+=[f"Proposed场景等权均值更低的seed数：{a['proposed_mean_better_seeds']}/5。"]
section('次要比较与消融')
lines+=['次要比较只报告，不作多重比较校正。相对差为100×(候选均值/参考均值−1)。']
table(['候选','参考','候选s','参考s','相对差%','胜场/20','Wilcoxon p','配对t p','符号p'],[[r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue'],r['statistics']['ttest_rel']['pvalue'],r['statistics']['binomtest']['pvalue']] for r in a['comparisons'] if not r['primary_endpoint']])
table(['参考','Proposed cost/s','参考 cost/s','相对差%','胜场/20'],[[r['reference'],r['cost_per_s']['candidate_mean'],r['cost_per_s']['reference_mean'],r['cost_per_s']['relative_difference_percent'],r['cost_per_s']['wins']] for r in a['comparisons'] if 'cost_per_s' in r])
lines+=['跨seed标准差以各seed的20场景等权均值为单位，ddof=1。']
table(['组','跨seed均值s','样本标准差s'],[[g,r['mean'],r['std']] for g,r in a['across_seed'].items()])
x=a['only4101_Alternating']['comparisons'][0]
lines+=[f"仅4101的Abl_Alternating：Proposed={x['candidate_mean_s']:.9f} s；Alternating={x['reference_mean_s']:.9f} s；相对差={x['relative_difference_percent']:.6f}%；胜场={x['wins']}/20；Wilcoxon p={x['statistics']['wilcoxon']['pvalue']:.9g}。未纳入五seed主终点。"]
section('每seed的G1')
lines+=['G1保持原值；本任务按冻结协议将未通过只作报告并继续。未完成数采用gates.unfinished_per_episode：20场景各自每窗口未完成数之和，不是20场景平均；高/低优先级顺序为[0,1]。']
table(['seed','G1通过','SAC s','equal s','胜equal/20','local s','胜local/20'],[[s,g['passed'],g['equal']['candidate_mean_s'],g['equal']['reference_mean_s'],g['equal']['wins'],g['local']['reference_mean_s'],g['local']['wins']] for s,g in a['G1'].items()])
table(['seed','候选total','高优','低优','equal total','equal高/低','local total','local高/低'],[[s,g['unfinished']['per_episode']['candidate']['total'],*g['unfinished']['per_episode']['candidate']['by_priority'],g['unfinished']['per_episode']['hover_equal']['total'],str(g['unfinished']['per_episode']['hover_equal']['by_priority']),g['unfinished']['per_episode']['hover_local']['total'],str(g['unfinished']['per_episode']['hover_local']['by_priority'])] for s,g in a['G1'].items()])
section('第三阶段每cycle的训练截断时延均值（秒）')
table(['seed']+[str(c) for c in range(1,11)],[[s]+[r[str(c)] for c in range(1,11)] for s,r in a['stage3'].items()])
section('实测耗时与磁盘')
steps=q['completed_steps']
table(['seed','参考s','阶段1 s','阶段2 s','阶段3 s','三基线并行s','test s','六步总小时'],[[s]+[next(r['wall_seconds'] for r in steps if r.get('seed')==s and r['step']==name) for name in ('reference','stage1','stage2','stage3','baselines','test')]+[sum(r['wall_seconds'] for r in steps if r.get('seed')==s)/3600] for s in (4102,4103,4104,4105)])
lines+=['三基线耗时为并行步骤墙钟，未将三个子进程时间相加。六步总小时不含步骤间检查。']
table(['公共步骤','实测墙钟s'],[[r['step'],r['wall_seconds']] for r in steps if r.get('seed') is None])
table(['seed','文件字节数之和','GiB'],[[s,v,v/1024**3] for s,v in a['disk_bytes'].items()])
lines+=['磁盘为aggregate执行时记录；4101为其已有P/B/R2/S3/test五目录总和，4102–4105为各自reference、四组训练及stage2/stage3/test目录总和。']
section('驱动差异逐hunk对照')
lines+=['源码未修改。以下H编号按driver_diffs/hunks.json和各unified diff出现顺序；完整diff另见driver_diffs/。§0/§5/§7表示prompt明确要求的导入、GPU祖先链、失败检查，均不改变训练数值。']
notes={
'stage1_wrapper.py':[('D1；seed/路径；§3.1','原CLI等价rooted/read_config/identity/train调用；301守卫；300后检；无源码改动')],
'reference_seed.py':[('来源说明','添加4101来源文件头'),('seed/路径；§0','导入参数及WT/src断言，禁用pyc写入'),('路径','OUTPUT改为本seed参考目录'),('seed','measure_case的master参数化'),('D2；§3.2','freeze(master=seed)，原check保留，后检并记录160个完整metrics对照')],
'stage2_seed.py':[('seed/路径；§0','来源及参数模块、两个包导入断言'),('路径；D8','输入输出和共用WORK validation'),('路径元数据','删除4101旧失败目录说明'),('seed；D3；§5','Learner/Streams/derive/payload参数化，完整祖先链检查，删除对照日志工具'),('D3','删除随机SAC对照的逐轮比较'),('D3/D4；§3.3/§7','不跑随机对照和dev评估；记录冻结counts、检查100/200 checkpoint，保存失败信息')],
'stage3_seed.py':[('seed/路径；§0','来源及参数模块、两个包导入断言'),('路径；D5/D8','本seed输入输出、共用validation；删除P590对照输入'),('seed','Learner初始化参数化'),('§5','GPU排除扩展为完整祖先链'),('seed；D5','contract移除P590及4101判据，保留100轮映射和冻结协议'),('seed','采集Streams及三个外生种子参数化；run_lower其余AST相同'),('D5；RP-S3；§3.4/§7','删除LOOP/dev，formal增加phases_limit；完整运行增量及cycle-10后检；记录cycle均值')],
 'test_seed.py':[('来源说明','添加4101来源文件头'),('seed/路径；D6；§0','本seed checkpoint路径，删除Alternating/P600，保留8组构造，导入断言'),('路径；D6/D8；§5','替换identity链与manifest路径；祖先链检查；去掉面板A SHA'),('D6；§3.5','只读PROTO并核对SHA，去掉freeze_protocol和dev replay，保留P1-P6其余检查')],
'evaluate_test_seed.py':[('来源说明','添加4101来源文件头'),('seed/路径；§0','使用参数模块、导入断言'),('seed','缓存Learner的seed参数化'),('seed；D6','Streams参数化，删除Alternating分支'),('seed；D6','删除dev replay，640窗口及master/identity参数化'),('seed；D6','汇总改为本seed和640窗口，池化计算不变'),('D6','删除Alternating比较，8组报告；Wilcoxon/t/符号统计函数其余不变')],
'launch_baselines_seed.py':[('seed/路径；D7/D9；§3.6','三基线并行，失败等待全部退出；不跑A/B；typed anchor后检及初始上层哈希对照；原CLI经仅加导入断言的入口调用')]
}
hunks=read(OUT/'driver_diffs/hunks.json')
for name in ('stage1_wrapper.py','reference_seed.py','stage2_seed.py','stage3_seed.py','launch_baselines_seed.py','test_seed.py','evaluate_test_seed.py'):
 record=hunks[name];assert len(notes[name])==len(record['hunks'])
 lines+=['',f'`{name}`；来源：`{record["source"]}`。']
 table(['hunk','范围','批准范围','说明'],[[i+1,h['hunk'],*note] for i,(h,note) in enumerate(zip(record['hunks'],notes[name]))])
lines+=['D8：validation仅在WORK执行一次，四seed共用。D9：queue_multiseed.py按seed串行、按步骤串行，只有三个基线并行；msvcrt锁、失败停止、stdout/stderr和每步状态/时间记录均按prompt。common.py仅统一路径、导入/Git/协议/祖先链检查；checked_entry.py只验证导入后调用原模块CLI。aggregate.py保留4101统计函数口径，完成4101精确复算和冻结的五seed场景平均。']
section('脚本实现错误与修正')
fix=read(OUT/'implementation_correction.json')
lines+=['发生1次，已用完允许的修正机会。首次Start-Process因WORK/logs尚未创建，不能打开queue.out.log；Python队列当时未启动。修正在新的_r2 WORK/OUTM中先创建logs再启动，并启用PowerShell ErrorActionPreference=Stop。原失败目录保留只读。没有其他实现错误或实质失败。','', '修正diff：','','```diff',(OUT/'implementation_correction.diff').read_text(encoding='utf-8').strip(),'```','', '原失败WORK：`'+fix['previous_failed_work']+'`。原失败OUT：`'+fix['previous_failed_out']+'`。']
section('最终Git状态及产物')
table(['工作树','status','分支','HEAD'],[[root,'空',r['branch'],r['head']] for root,r in fc['git'].items()])
lines+=['未commit、切分支、merge、pip install；原源码、配置、场景、4101 checkpoint和协议未改。','',f'WORK：`{WORK}`。',f'OUTM：`{OUT}`。','完整数值：`aggregate/aggregate.json`；各seed结果：`seed_S/test/test_summary.json`及`test_comparisons.json`；复现：`repro_4101/checks.json`；最终检查：`final_checks.json`；队列：WORK下`queue_status.json`。','', '原自动生成、含完整源码diff的报告保留为`report_with_full_diffs.md`；本文件按相同数据整理为最终交付报告。数值展示有舍入，JSON保留原精度。']
backup=OUT/'report_with_full_diffs.md';assert not backup.exists();shutil.copy2(OUT/'report.md',backup)
text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text,encoding='utf-8')
print(text)
