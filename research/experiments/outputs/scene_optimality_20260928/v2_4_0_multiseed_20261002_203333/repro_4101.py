# Reproduction uses the same parameterized drivers; all artifacts remain in repro_4101.
import sys
sys.dont_write_bytecode=True
import argparse,os,tempfile,json,time
from common import *
assert_imports()
RP=OUTM/'repro_4101'
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('step',choices=('RP-W','RP-S2','RP-S3','RP-T','RP-REF'));args=parser.parse_args()
 step=args.step;out=RP/step;check(not out.exists(),'new_repro_step',str(out));out.mkdir(parents=True)
 os.environ.update(MULTI_SEED='4101',MULTI_OUT=str(out),MULTI_P=str(P4101),MULTI_R2=str(R24101),MULTI_S3=str(S34101),MULTI_B=str(B4101),MULTI_VALIDATION_ROOT=str(WORK))
 start=time.perf_counter();checks=[]
 def eq(a,b,label):equal(a,b,label);checks.append(dict(check=label,status='PASS'))
 if step=='RP-W':
  import stage1_wrapper as s
  with tempfile.TemporaryDirectory(dir=out) as td:
   root=Path(td);guard=s.guard_for(root)
   for n in range(1,601):eq(guard(n).__dict__,s.original_phase(n).__dict__,f'phase_absent_{n}')
   (root/'checkpoints').mkdir();(root/'checkpoints/episode-0300-evaluated.pt').touch()
   caught=[]
   for n in range(1,601):
    try:eq(guard(n).__dict__,s.original_phase(n).__dict__,f'phase_present_{n}')
    except s.StopStage1:caught.append(n)
   eq(caught,[301],'only_301_stops')
 elif step=='RP-S2':
  import stage2_seed as s
  config,binding,device=s.preflight();s.train_run('sacdet',False,10,config,binding,device)
  run=out/'runs/sacdet';old=R24101/'runs/sacdet'
  for n in range(1,11):
   a=read(run/f'episodes/episode-{n:04d}.json');b=read(old/f'episodes/episode-{n:04d}.json')
   for k in ('event_sha256','truncated_delay_s'):eq(a['metrics'][k],b['metrics'][k],f'{n}.metrics.{k}')
  a=rows(run/'train_metrics.jsonl');b=rows(old/'train_metrics.jsonl')[:10]
  strip=lambda r:{k:v for k,v in r.items() if k not in ('work_seconds','wall_seconds','attempt_id')}
  eq([strip(r) for r in a],[strip(r) for r in b],'all_non_timing_train_metrics')
  eq([strip(r) for r in rows(run/'learning.jsonl')],[strip(r) for r in rows(old/'learning.jsonl') if r['episode'] in (5,10)],'ppo_learning_5_10')
 elif step=='RP-S3':
  import stage3_seed as s
  config,binding,device,lower=s.preflight();s.formal(config,binding,device,lower,phases_limit=3)
  run=out/'runs/stage3';old=S34101/'runs/stage3';aa=rows(run/'train_metrics.jsonl');bb={r['global_episode']:r for r in rows(old/'train_metrics.jsonl')}
  expected={411:dict(triggers=90,skipped=90,events=0,critic_fits=0,target_updates=0,actor_updates=0),412:dict(triggers=90,skipped=0,events=90,critic_fits=180,target_updates=90,actor_updates=0),413:dict(triggers=90,skipped=0,events=90,critic_fits=180,target_updates=90,actor_updates=45)}
  logs=rows(run/'learning.jsonl');oldlogs=rows(old/'learning.jsonl')
  for a in aa:
   n=a['global_episode'];b=bb[n]
   for k in ('event_sha256','truncated_delay_s','reward_physical','sac_delta'):eq(a[k],b[k],f'{n}.{k}')
   eq(a['sac_delta'],expected[n],f'{n}.expected_sac_delta')
   strip=lambda r:{k:v for k,v in r.items() if k!='attempt_id'}
   x=[strip(r) for r in logs if r['global_episode']==n];y=[strip(r) for r in oldlogs if r['global_episode']==n]
   eq(len(x),90,f'{n}.learning_count');eq(x,y,f'{n}.learning_rows')
 elif step=='RP-T':
  import test_seed as s
  config,binding,device,m,metadata=s.preflight()
  import evaluate_test_seed as e
  for g in s.GROUPS:
   old=read(T4101/f'windows/{g}_01_1.json');r=e.window(dict(group=g,scenario=m['scenarios'][0],stream=1,split='test',identity=old['identity'],replay=True))
   eq(r['result']['metrics'],old['metrics'],g+'.full_metrics')
   if g in ('TFOO','Abl_UpperCVX'):eq(r['result']['diagnostics'],old['diagnostics'],g+'.diagnostics')
 elif step=='RP-REF':
  import reference_seed as s
  from hrl_v2.checkpoint import identity
  config=s.read_config(CONFIG,WORK);scene=read(WT/'src/hrl_v2/configs/physical_scene.json');binding=identity(WORK,CONFIG);scenario=s.gate_manifest(20)['scenarios'][0]
  for label in ('hover_equal','hover_cvx'):
   metrics=s.measure_case(config,scene,binding,out/'baselines',scenario,1,label,*s.COMBINATIONS[label],4101)
   eq(metrics,read(REF4101/f'{label}_01_1.json')['metrics'],label+'.full_metrics')
 result=dict(status='passed',step=step,checks=checks,wall_seconds=time.perf_counter()-start);write(out/'checks.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
