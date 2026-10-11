from d_guard import install,ROOT
install()
from j_common import *
import subprocess,sys,copy
from d_compare import exact_equal
from d_pool import save_error
def run_child(mode,limit=25):
 before=set((ROOT/'errors').glob('*.json'))
 start=time.monotonic()
 p=subprocess.run([str(PY),'-B',str(ROOT/'scripts/d_test_child.py'),mode],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8',timeout=limit)
 elapsed=time.monotonic()-start
 traces=[str(x) for x in set((ROOT/'errors').glob('*.json'))-before]
 r=dict(mode=mode,returncode=p.returncode,wall_seconds=elapsed,stdout=p.stdout,stderr=p.stderr,traceback_files=traces)
 if mode in ('worker_failure','worker_timeout'):
  check(p.returncode!=0,mode+'_nonzero')
  check(elapsed<limit,mode+'_bounded_exit')
  check(bool(traces),mode+'_traceback_persisted')
  text=' '.join(Path(x).read_text(encoding='utf-8') for x in traces)
  check('Traceback' in text,mode+'_full_traceback')
  check(('intentional_worker_failure_for_preflight' if mode=='worker_failure' else 'TimeoutError') in text,mode+'_original_exception')
 else:check(p.returncode==0,mode+'_passed',r)
 return r
def main():
 begun=time.perf_counter();c=contract()
 run_id='run_'+str(time.time_ns());smoke_root=ROOT/'selftest'/run_id
 smoke_root.mkdir(parents=True)
 result=dict(status='running',started_at=now(),selftest_root=str(smoke_root),tests={})
 put(ROOT/'preflight.json',result)
 scripts=sorted((ROOT/'scripts').glob('*.py'))
 p=subprocess.run([str(PY),'-B','-m','py_compile',*[str(p) for p in scripts]],capture_output=True,text=True,encoding='utf-8')
 result['tests']['py_compile']=dict(returncode=p.returncode,files=[str(p) for p in scripts],stdout=p.stdout,stderr=p.stderr)
 check(p.returncode==0,'all_scripts_compile',result['tests']['py_compile'])
 for mode in ('guard','success','worker_failure','worker_timeout'):
  result['tests'][mode]=run_child(mode)
  put(ROOT/'preflight.json',result)
 exact_equal(np.float64(349.74321325224406),float(349.74321325224406),'same_numeric_value')
 exact_equal(float(349.74321325224406),np.float64(349.74321325224406),'same_numeric_value_reverse')
 x=np.float64(1.0);y=np.nextafter(x,np.float64(2.0))
 try:exact_equal(x,float(y),'one_ulp')
 except HardCheckError as e:one_ulp_detail=str(e)
 else:raise AssertionError('one ulp difference not rejected')
 result['tests']['numeric_comparison']=dict(passed=True,equal_types_ignored=True,one_ulp_rejected=True,one_ulp_detail=one_ulp_detail,sample=exact_equal(np.float64(349.74321325224406),float(349.74321325224406),'repr_sample',record=True))
 put(ROOT/'preflight.json',result)
 from d_main import verify_inputs
 verify_inputs(c)
 put(ROOT/'progress.json',dict(status='preparing',stage='smoke_A1_scenes_1_2_two_streams',completed_groups=0,total_groups=33,time=now()))
 from d_eval import group
 a1=next(g for g in c['groups'] if g['label']=='A1')
 smoke=group(a1,scenario_ids=[1,2],output_root=smoke_root/'groups')
 result['tests']['smoke']=dict(passed=smoke['anchor_checks']['passed'],scenarios=[1,2],streams=[1,2],summary=str(smoke_root/'groups/A1/summary.json'),mean_s=smoke['mean_s'],wall_seconds=smoke['wall_seconds'],anchors=smoke['anchor_checks'],stream_consistency=smoke['stream_consistency'])
 verify_inputs(c)
 from d_report import report
 subset=copy.deepcopy(c);subset['groups']=[a1]
 report(subset,begun,partial=True,run_root=smoke_root)
 check((smoke_root/'report.md').is_file() and (smoke_root/'report.json').is_file(),'smoke_report_generated')
 check(read(smoke_root/'report.json')['partial_smoke'] is True,'smoke_report_marked_partial')
 result['tests']['smoke_report']=dict(passed=True,report_md=str(smoke_root/'report.md'),report_json=str(smoke_root/'report.json'))
 verify_inputs(c)
 result.update(status='passed',completed_at=now(),wall_seconds=time.perf_counter()-begun,script_sha256={str(p):sha(p) for p in scripts},contract_sha256=sha(ROOT/'contract.json'),checkpoint_count=len(c['checkpoints']),task_timeout_seconds=c['task_timeout']['task_timeout_seconds'])
 put(ROOT/'preflight.json',result);put(smoke_root/'preflight.json',result)
 put(ROOT/'progress.json',dict(status='ready',stage='preflight_passed',completed_groups=0,total_groups=33,time=now()))
 print('ALL_PREFLIGHT_PASS',json.dumps(dict(seconds=result['wall_seconds'],smoke_mean_s=smoke['mean_s'],report=result['tests']['smoke_report'])),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  traceback.print_exc();save_error('preflight',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  result=read(ROOT/'preflight.json') if (ROOT/'preflight.json').exists() else {}
  result.update(status='failed',failed_at=now(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc())
  put(ROOT/'preflight.json',result);failure('preflight',e)
  sys.stdout.flush();sys.stderr.flush();os._exit(1)
