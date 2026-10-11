import sys
sys.dont_write_bytecode=True
import json,re,difflib
from pathlib import Path
from common import *
changes={}
def replace(s,a,b):
 assert a in s, a[:150]
 return s.replace(a,b)
def emit(name,src,s,categories):
 original=Path(src).read_text(encoding='utf-8-sig')
 header='# Derived from '+str(src)+'; changes restricted to seed/path and approved D1-D9.\n'
 s=header+s
 compile(s,name,'exec')
 (OUTM/name).write_text(s,encoding='utf-8')
 dif=list(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=name))
 dest=OUTM/'driver_diffs';dest.mkdir(exist_ok=True)
 (dest/(name+'.diff')).write_text(''.join(dif),encoding='utf-8')
 hunks=[];cur=None
 for line in dif:
  if line.startswith('@@'):
   cur=dict(hunk=line.strip(),allowed_categories=categories,changes=[]);hunks.append(cur)
  elif cur is not None and line[:1] in ('+','-'):cur['changes'].append(line.rstrip())
 changes[name]=dict(source=str(src),categories=categories,hunks=hunks)

def replace_gpu(s,start,end):
 i=s.index(start);j=s.index(end,i)
 return s[:i]+" record.update(**gpu_check());atomic_json(OUT/'preflight.json',record)\n"+s[j:]

src=R24101/'sacdet_4101.py';s=src.read_text(encoding='utf-8')
s=replace(s,'import torch,hrl_v2','import torch,hrl_v2,hrl_uav\nfrom common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports\nassert_imports()')
s=replace(s,"OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'","OUT=envpath('OUT',stage_path('stage2'))")
s=replace(s,"P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'","P=envpath('P',run_path())")
s=replace(s,"B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'","B=envpath('VALIDATION_ROOT',WORK)")
s=s.replace('Learner(4101,','Learner(SEED,').replace('Streams(4101,','Streams(SEED,').replace('derive(4101,','derive(SEED,').replace('seed=4101','seed=SEED')
s=replace_gpu(s,' command="Get-CimInstance',' hs={}')
s=replace(s,"record=dict(status='checking',previous_failed_attempt=str(PREVIOUS),previous_failure_reason='GPU 检查未排除自身进程')","record=dict(status='checking')")
s=replace(s,"dict(previous_failed_attempt=str(PREVIOUS),previous_failure_reason='GPU 检查未排除自身进程',identity=binding","dict(identity=binding")
s=s.replace('control_episodes=25,','control_episodes=0,')
a=s.index('\ndef original_ppo():');b=s.index('\ndef train_run',a);s=s[:a]+s[b:]
s=replace(s,"phases=upper_phases()[:total];batch=[];comparisons=[];oldppo=original_ppo() if lower_stochastic else {};first_difference=None;updates_count=0","phases=upper_phases()[:total];batch=[];updates_count=0")
a=s.index(' def compare(episode,');b=s.index(' payload_save(run,',a);s=s[:a]+s[b:]
a=s.index('  if lower_stochastic:');b=s.index('  if n%25==0:',a);s=s[:a]+s[b:]
a=s.index(' if lower_stochastic:',s.index(' hs=hashes(learner);require'));b=s.index(' result=dict(',a);s=s[:a]+s[b:]
s=replace(s,"final_hashes=hs,wall_seconds=","final_hashes=hs,initial_sac_counts=initial_counts,final_sac_counts=dict(learner.sac.counts),final_ppo_counts=dict(learner.ppo.counts),wall_seconds=")
a=s.index('\ndef main():');s=s[:a]+'''
def main():
 config,binding,device=preflight()
 mainrun=train_run('sacdet',False,200,config,binding,device)
 for n in (100,200):require((OUT/f'runs/sacdet/checkpoints/episode-{n:04d}.pt').is_file(),'stage2_checkpoint',n)
 atomic_json(OUT/'training_summary.json',dict(sacdet=mainrun))
 atomic_json(OUT/'checks.json',dict(status='passed',preflight=CHECKS,sacdet=mainrun))
 atomic_json(OUT/'git_after.json',git_states())
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));raise
'''
emit('stage2_seed.py',src,s,{'seed/path':'SEED及本seed输入输出、WORK validation','D3':'删除随机SAC对照及比较分支；复现由RP-S2执行','D4':'删除dev评估；保留训练与冻结检查','D9':'所有子进程导入检查、完整祖先链GPU排除、后检记录'})

src=S34101/'stage3_4101.py';s=src.read_text(encoding='utf-8')
s=replace(s,'import torch,hrl_v2,hrl_uav','import torch,hrl_v2,hrl_uav\nfrom common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports\nassert_imports()')
s=replace(s,"OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_stage3'","OUT=envpath('OUT',stage_path('stage3'))")
s=replace(s,"R2=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'","R2=envpath('R2',stage_path('stage2'))")
s=replace(s,"P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'","P=envpath('P',run_path())")
s=replace(s,"B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'","B=envpath('VALIDATION_ROOT',WORK)")
s=s.replace('Learner(4101,','Learner(SEED,').replace('Streams(4101,','Streams(SEED,').replace('derive(4101,','derive(SEED,').replace('seed=4101','seed=SEED')
s=replace_gpu(s,' cmd="Get-CimInstance',' learner,sd=')
s=s.replace("C590=P/'checkpoints/episode-0590-complete.pt'\n",'')
s=replace(s,"(('SACDET-200',SD),('S300',S300),('P590',C590))","(('SACDET-200',SD),('S300',S300))")
s=re.sub(r",criterion=dict\(candidate='S3-c10'.*?intermediate_models='S3-c1\.\.S3-c9 report only; no selection'\)",',intermediate_models="S3-c1..S3-c9 report only; no selection"',s)
a=s.index('\ndef loop_equiv(');b=s.index('\ndef formal(',a);s=s[:a]+s[b:]
s=replace(s,'def formal(config,binding,device,lower):','def formal(config,binding,device,lower,phases_limit=None):')
s=replace(s,"timing=run_lower(learner,replay,lower,OUT/'runs/stage3');STAGE='formal_checks'","selected=lower if phases_limit is None else lower[:phases_limit]\n timing=run_lower(learner,replay,selected,OUT/'runs/stage3');STAGE='formal_checks'")
s=replace(s,"require(delta==EXPECTED and result['skipped_global_episodes']==[411],'SAC_counts',result);CHECKS['SAC_counts']='PASS'","if phases_limit is None:\n  require(delta==EXPECTED and result['skipped_global_episodes']==[411],'SAC_counts',result);CHECKS['SAC_counts']='PASS'\n  side=read(OUT/'runs/stage3/checkpoints/cycle-10.json');require(side['completed_episode']==600 and side['pending_evaluation'] is None,'cycle10_sidecar')\n metrics=rows(OUT/'runs/stage3/train_metrics.jsonl')\n result['cycle_mean_truncated_delay_s']={str(c):sum(r['truncated_delay_s'] for r in metrics if phase(r['global_episode']).cycle//2==c)/sum(phase(r['global_episode']).cycle//2==c for r in metrics) for c in sorted({phase(r['global_episode']).cycle//2 for r in metrics})}\n atomic_json(OUT/'training_summary.json',result)")
a=s.index('\ndef main():');s=s[:a]+'''
def main():
 config,binding,device,lower=preflight();result=formal(config,binding,device,lower)
 atomic_json(OUT/'checks.json',dict(status='passed',checks=CHECKS,result=result))
 atomic_json(OUT/'git_after.json',git_states())
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(stage=STAGE,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_completed=CHECKS,substantive_check=isinstance(error,HardCheckError)));raise
'''
emit('stage3_seed.py',src,s,{'seed/path':'SEED及本seed输入输出、WORK validation','D5':'删除LOOP_EQUIV、评估及4101判据；formal仅增加RP所需phases_limit；保留cycle存档和冻结检查','D9':'导入检查、祖先链GPU排除及后检记录'})

src=WT/'tools/run_v2_4_0_reference.py';s=src.read_text(encoding='utf-8')
s=replace(s,'import argparse','import sys\nsys.dont_write_bytecode=True\nfrom common import SEED,WORK,envpath,assert_imports,REF4101,read\nassert_imports()\nimport argparse')
s=replace(s,"OUTPUT=ROOT/'outputs/lite_v2_4_0'","OUTPUT=envpath('OUT',WORK/f'references/seed_{SEED}')")
s=replace(s,"label,*COMBINATIONS[label],4101)","label,*COMBINATIONS[label],SEED)")
s=replace(s,"freeze(CONFIG,OUTPUT/'baselines',20,12,include_greedy=True)","freeze(CONFIG,OUTPUT/'baselines',20,12,master=SEED,include_greedy=True)")
s=replace(s,'    checked=check()','''    checked=check()
    if not (summary['identity']==binding and summary['master']==SEED and summary['manifest']==gate_manifest(20) and summary['required_wins']==12 and checked['passed']):raise AssertionError('Reference hard checks failed')
    same=0
    for scenario,stream,label in jobs:
        name=f'{label}_{scenario["scenario"]:02d}_{stream}.json'
        same+=int(read(OUTPUT/'baselines'/name)['metrics']==read(REF4101/name)['metrics'])
    atomic_json(OUTPUT/'checks.json',dict(status='passed',check=checked,matching_REF4101_full_metrics=same,total=160))''')
emit('reference_seed.py',src,s,{'seed/path':'master=SEED，OUTPUT=本seed reference；导入路径检查；160窗口后检和与4101只读比较','D2':'每seed冻结自身参考，原check及聚合不变'})

src=T4101/'test_4101.py';s=src.read_text(encoding='utf-8')
s=replace(s,'import torch,hrl_v2,hrl_uav','import torch,hrl_v2,hrl_uav\nfrom common import SEED,WORK,OUTM,envpath,stage_path,run_path,gpu_check,assert_imports,PROTO,PROTO_SHA,protocol_check,P4101\nassert_imports()')
s=replace(s,"P=WT/'outputs/lite_v2_4_0/training_4101_20260930_010036/runs/Proposed/seed_4101'","P=envpath('P',run_path())")
s=replace(s,"B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900';A=B/'paper_dev_A_P600_B200'","B=envpath('B',WORK)")
s=replace(s,"R2=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_sacdet_r2'","R2=envpath('R2',stage_path('stage2'))")
s=replace(s,"S3=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_stage3'","S3=envpath('S3',stage_path('stage3'))")
s=replace(s,"OUT=MAIN/'outputs/scene_optimality_20260928/v2_4_0_4101_test'","OUT=envpath('OUT',stage_path('test'))")
s=s.replace(",P600=P/'checkpoints/episode-0600-evaluated.pt',P600_pending=P/'checkpoints/episode-0600-pending.pt'",'').replace('seed_4101/checkpoints','seed_{SEED}/checkpoints')
s=s.replace(",'Abl_Alternating'",'').replace(',Abl_Alternating=[\'P600\']','').replace(',Abl_Alternating="comparison_policy(P600 learner, scene, streams, \'moving\', stochastic=False)"','')
s=s.replace(')(4101,device,config)',')(SEED,device,config)')
s=replace(s,"validation=read(B/'validation/validation.json')","validation=read(envpath('VALIDATION_ROOT',WORK)/'validation/validation.json')")
s=replace_gpu(s,' cmd="Get-CimInstance'," m=read(P/'test_manifest.json')")
s=replace(s,"require(m==manifest('test'),'test_manifest_exact')","require(m==manifest('test')==read(P4101/'test_manifest.json'),'test_manifest_exact')")
s=replace(s,"metadata={};panel=read(A/'summary_deterministic.json')","metadata={}")
s=s.replace("   require(sha==panel['identity']['checkpoints'][name],'baseline_A_checkpoint_sha',name)\n",'')
s=s.replace(" else 'Proposed' if name in ('P600','P600_pending') else None"," else None")
s=s.replace(" require(metadata['P600']['parameter_hashes']==metadata['P600_pending']['parameter_hashes'],'P600_pending_evaluated_hashes')\n",'')
a=s.index('\ndef freeze_protocol(');b=s.index('\ndef main():',a);s=s[:a]+s[b:]
s=s.replace("sys.modules['test_4101']=sys.modules[__name__]","sys.modules['test_seed']=sys.modules[__name__]")
s=replace(s,"config,binding,device,m,metadata=preflight();freeze=freeze_protocol(config,binding,m,metadata)","config,binding,device,m,metadata=preflight();freeze=dict(path=str(PROTO),sha256=protocol_check());CHECKS['protocol_hash']='PASS'")
s=replace(s,'from evaluate_test import replay_checks,test_panel,summarize,compare,report','from evaluate_test_seed import test_panel,summarize,compare,report')
s=replace(s," STAGE='dev_replay';replay_checks(config,binding,freeze,metadata)\n",'')
s=replace(s,"report(summary,comparisons,freeze,git);print('TEST_PANEL_COMPLETE',flush=True)","report(summary,comparisons,freeze,git);atomic_json(OUT/'checks.json',dict(status='passed',checks=CHECKS));print('TEST_PANEL_COMPLETE',flush=True)")
emit('test_seed.py',src,s,{'seed/path':'SEED及输入/输出、身份链WORK validation、test manifest对4101检查','D6':'8组保留原构造；去掉Alternating及600检查；只读已有PROTO；去掉dev replay及Panel A SHA','D9':'导入及祖先链GPU检查、记录硬检查'})

src=T4101/'evaluate_test.py';s=src.read_text(encoding='utf-8')
s=replace(s,'from test_4101 import WT,MAIN,OUT,P,B,A,R2','from test_seed import WT,MAIN,OUT,P,B,R2')
s=replace(s,'from hrl_v2.checkpoint import identity','from common import SEED,assert_imports\nassert_imports()\nfrom hrl_v2.checkpoint import identity')
s=s.replace(')(4101,device,config)',')(SEED,device,config)').replace('Streams(4101,','Streams(SEED,').replace('master=4101','master=SEED')
s=s.replace(",'Abl_Alternating'",'').replace(",Abl_Alternating='moving'",'')
a=s.index('\ndef replay_checks(');b=s.index('\ndef test_panel',a);s=s[:a]+s[b:]
s=s.replace('720','640')
a=s.index('\ndef report(');s=s[:a]+'''
def report(summary,comparisons,freeze,git):
 atomic_json(OUT/'timing.json',TIMING)
 lines=[f'# v2.4.0 seed {SEED} test 面板','',f"协议SHA256：{freeze['sha256']}",'','检查：'+json.dumps(CHECKS,ensure_ascii=False),'','|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|','|---|---|---|---|---|---|---|']
 for r in comparisons:lines.append('| '+' | '.join(map(str,(r['candidate'],r['reference'],r['candidate_mean_s'],r['reference_mean_s'],r['relative_difference_percent'],r['wins'],r['statistics']['wilcoxon']['pvalue'])))+' |')
 lines+=['','## 实测耗时','',json.dumps(TIMING,ensure_ascii=False,indent=2),'','## 最终Git状态','',json.dumps(git,ensure_ascii=False,indent=2),'','脚本错误与修正：无。']
 (OUT/'report.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
'''
emit('evaluate_test_seed.py',src,s,{'seed/path':'SEED和参数模块导入、导入路径检查','D6':'删除Alternating和dev回放；640窗口；原窗口/汇总/比较计算保留；报告只读协议与8组'})
write(OUTM/'driver_diffs/hunks.json',changes)
