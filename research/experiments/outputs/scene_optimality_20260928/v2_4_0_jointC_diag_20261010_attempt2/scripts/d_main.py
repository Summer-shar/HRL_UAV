from d_guard import install,ROOT
install()
from j_common import *
import zipfile
from d_eval import group

def compress_if_needed(force=False):
 files=list((CPIL/'groups').glob('*/windows/*.json'));size=sum(p.stat().st_size for p in files)
 if size<=1024**3:return
 begin=time.perf_counter();aroot=output_path(CPIL/'archives');aroot.mkdir(exist_ok=True)
 count=len(list(aroot.glob('batch_*.zip')))+1;name=f'batch_{count:03d}'
 records=[dict(path=str(p),member=p.relative_to(CPIL).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
 put(aroot/(name+'_sha256.json'),dict(time=now(),files=records))
 archive=output_path(aroot/(name+'.zip'))
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for r in records:z.write(r['path'],r['member'])
 with zipfile.ZipFile(archive) as z:
  for r in records:
   h=hashlib.sha256()
   with z.open(r['member']) as f:
    for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
   equal(h.hexdigest(),r['sha256'],'decompressed_sha_'+r['member'])
 index=read(CPIL/'archive_index.json') if (CPIL/'archive_index.json').exists() else {}
 for r in records:index[r['path']]=dict(archive=str(archive),member=r['member'],sha256=r['sha256'])
 put(CPIL/'archive_index.json',index)
 for r in records:
  p=output_path(r['path']);equal(sha(p),r['sha256'],'pre_delete_generated_window_hash');p.unlink()
 stats=dict(archive=str(archive),file_count=len(records),original_bytes=size,archive_bytes=archive.stat().st_size,ratio=archive.stat().st_size/size,wall_seconds=time.perf_counter()-begin,decompressed_hash_check=True,originals_deleted_after_validation=True)
 put(aroot/(name+'_result.json'),stats);print('ARCHIVE_PASS',stats,flush=True)

def verify_inputs(c):
 results=[]
 for p,expected in c['protected_inputs'].items():
  actual=sha(p);equal(actual,expected,'input_unchanged_'+p);results.append(dict(path=p,sha256=actual,unchanged=True))
 old=Path(c['paths']['R'])/'v2_4_0_jointC_diag_20261010';manifest=read(CPIL/'old_attempt_frozen_manifest.json')['files']
 equal(sorted(str(p) for p in old.rglob('*') if p.is_file()),sorted(manifest),'old_attempt_file_inventory_unchanged')
 equal(git_states(),c['git_before'],'git_before_after')
 put(CPIL/'readonly_checks_final.json',dict(status='passed',time=now(),files=results,git=git_states(),test_scenes_accessed=False,training=False))
 return True

def main():
 c=contract();check(read(CPIL/'preflight.json')['status']=='passed','preflight_passed');equal(read(CPIL/'preflight.json')['script_sha256'],{str(p):sha(p) for p in sorted((CPIL/'scripts').glob('*.py'))},'scripts_unchanged_since_selftests');gpu_check();print('PREFLIGHT_CURRENT_GPU_PASS',flush=True)
 for g in c['groups']:
  for p in [g['upper'],g['lower']]:
   if p:equal(sha(p),c['protected_inputs'][str(Path(p).resolve())],'checkpoint_before_first_eval')
 # Anchors first; preserve all 33 registered groups and do not reselect any model.
 order=sorted(c['groups'],key=lambda g:(0 if g['label'] in ('A1','A2') else 1 if g['anchors'] else 2,c['groups'].index(g)))
 started=time.perf_counter();put(CPIL/'execution_start.json',dict(time=now(),pid=os.getpid(),script=str(Path(__file__).resolve()),order=[g['label'] for g in order],training=False,workers=6))
 out=[]
 for g in order:
  put(CPIL/'progress.json',dict(status='running',stage='evaluation',current_group=g['label'],seed=g['seed'],completed_groups=len(out),total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
  summary_path=CPIL/'groups'/g['label']/'summary.json'
  out.append(read(summary_path) if summary_path.exists() else group(g));compress_if_needed()
  put(CPIL/'progress.json',dict(status='running',stage='group_complete',last_group=g['label'],completed_groups=len(out),total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
 verify_inputs(c)
 put(CPIL/'progress.json',dict(status='running',stage='report',completed_groups=33,total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
 from d_report import report
 report(c,started)
 verify_inputs(c)
 put(CPIL/'progress.json',dict(status='complete',stage='complete',completed_groups=33,total_groups=33,time=now(),wall_seconds=time.perf_counter()-started))
 print('DIAGNOSTIC_COMPLETE',time.perf_counter()-started,flush=True)

if __name__=='__main__':
 try:
  main()
 except BaseException as e:
  from d_pool import save_error
  save_error('main',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  failure('diagnostic_evaluation_or_report',e)
  traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)

