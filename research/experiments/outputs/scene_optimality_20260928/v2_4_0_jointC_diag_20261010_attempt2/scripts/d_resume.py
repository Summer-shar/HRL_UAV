from d_guard import install,ROOT
install()
from j_common import *
from d_main import verify_inputs,compress_if_needed
from d_eval import group,summarize_group,whole_metrics_equal
from d_pool import save_error
import zipfile,datetime

def validate_completed(g,s,index):
 equal(s['group'],g,'resumed_group_identity')
 check(s['stream_consistency']['passed'] and s['anchor_checks']['passed'],'resumed_group_passed')
 check(len(s['window_files'])==40,'resumed_group_40_windows')
 handles={}
 try:
  for w in s['window_files']:
   path=Path(w['path']);h=hashlib.sha256()
   if path.exists():stream=path.open('rb')
   else:
    a=index[str(path)];equal(a['sha256'],w['sha256'],'archive_record_sha')
    if a['archive'] not in handles:handles[a['archive']]=zipfile.ZipFile(a['archive'])
    stream=handles[a['archive']].open(a['member'])
   with stream:
    for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
   equal(h.hexdigest(),w['sha256'],'completed_window_sha_'+g['label'])
 finally:
  for z in handles.values():z.close()
 for ap in g['anchors']:
  old=read(ap);whole_metrics_equal(s['per_scenario'],old['per_scenario'],'resumed_anchor_per_scene')
 return dict(group=g['label'],windows=40,sha_pass=True,anchors_pass=True)

def finish_existing(g):
 root=CPIL/'groups'/g['label'];files=list((root/'windows').glob('*.json'))
 expected={f'{sid:02d}_{st}.json' for sid in range(1,21) for st in (1,2)}
 equal({p.name for p in files},expected,'interrupted_group_all_40_present')
 anchors=[{(w['scenario'],w['stream']):w for w in read(ap)['window_files']} for ap in g['anchors']]
 records=[];start=time.perf_counter();recorded_times=[]
 for sid in range(1,21):
  for st in (1,2):
   p=root/'windows'/f'{sid:02d}_{st}.json';before=sha(p);d=read(p)
   equal((d['identity']['group'],d['identity']['seed'],d['identity']['scenario']['scenario'],d['identity']['stream']),(g['label'],g['seed'],sid,st),'existing_window_identity')
   check(d['parameter_hash_check']['passed'] and d['anchor_checks']['passed'],'existing_window_checks')
   equal(d['parameter_hash_check']['before'],d['parameter_hash_check']['after'],'existing_parameters_same')
   equal((d['learning_updates'],d['replay_writes']),(0,0),'existing_readonly')
   equal(d['rng_source'],g['upper'],'existing_upper_rng')
   for a in anchors:
    w=a[(sid,st)];equal(sha(w['path']),w['sha256'],'existing_anchor_sha');old=read(w['path'])
    whole_metrics_equal(d['metrics'],old['metrics'],'existing_anchor_full_metrics')
   equal(sha(p),before,'existing_output_sha_stable');records.append(dict(path=str(p),sha256=before));recorded_times.append(d['wall_seconds'])
   del d
 put(CPIL/'recovery'/('validated_'+g['label']+'.json'),dict(time=now(),windows=records,full_anchor_metrics_pass=True,recorded_rollout_seconds=recorded_times))
 return summarize_group(g,root,list(range(1,21)),start)

def main():
 c=contract();pre=read(CPIL/'preflight.json');check(pre['status']=='passed','resume_preflight_passed')
 equal(pre['script_sha256'],{str(p):sha(p) for p in sorted((CPIL/'scripts').glob('*.py'))},'resume_scripts_match_preflight')
 gpu_check();verify_inputs(c)
 original=read(CPIL/'recovery/execution_start.json.before_resume')
 elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(original['time'])).total_seconds()
 started=time.perf_counter()-elapsed
 order=sorted(c['groups'],key=lambda g:(0 if g['label'] in ('A1','A2') else 1 if g['anchors'] else 2,c['groups'].index(g)))
 index=read(CPIL/'archive_index.json');checks=[];done=[]
 put(CPIL/'execution_start.json',dict(time=now(),first_started_at=original['time'],pid=os.getpid(),script=str(Path(__file__).resolve()),order=[g['label'] for g in order],training=False,workers=6,resumed=True))
 for g in order:
  p=CPIL/'groups'/g['label']/'summary.json'
  if p.exists():
   put(CPIL/'progress.json',dict(status='running',stage='resume_sha_verification',current_group=g['label'],completed_groups=11,total_groups=33,time=now()))
   checks.append(validate_completed(g,read(p),index));done.append(g['label']);print('RESUME_SHA_PASS',g['label'],flush=True)
 put(CPIL/'recovery/completed_groups_sha_check.json',dict(status='passed',time=now(),groups=checks))
 for g in order:
  if g['label'] in done:continue
  put(CPIL/'progress.json',dict(status='running',stage='evaluation',current_group=g['label'],seed=g['seed'],completed_groups=len(done),total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
  root=CPIL/'groups'/g['label']
  if root.exists() and any(root.iterdir()):finish_existing(g)
  else:group(g)
  done.append(g['label']);compress_if_needed()
  put(CPIL/'progress.json',dict(status='running',stage='group_complete',last_group=g['label'],completed_groups=len(done),total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
 verify_inputs(c)
 put(CPIL/'progress.json',dict(status='running',stage='report',completed_groups=33,total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
 from d_report import report
 report(c,started);verify_inputs(c)
 put(CPIL/'progress.json',dict(status='complete',stage='complete',completed_groups=33,total_groups=33,time=now(),wall_seconds=time.perf_counter()-started,elapsed_includes_interruption=True))
 print('DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save_error('resume',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  failure('diagnostic_resume',e);traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
