from d_guard import install,ROOT
install()
from j_common import *
from d_setup import input_checks,write_diff
from d_storage import save_candidate,load_evaluation,save_full,load_resume
from j_train import hashes,NETWORKS,EXPECTED200,EXPECTED_FINAL,budget_counts,SCHEMA
from hrl_v2.runtime import Learner
from hrl_v2.replay import Replay
from hrl_v2.lite_observation import LOWER_DIM,UPPER_DIM
from unittest.mock import patch
import ast,py_compile,torch

def main():
    checks={};base=ROOT/'selftest/lean'/str(time.time_ns());base.mkdir(parents=True,exist_ok=True)
    backup=Path(contract()['lean_runtime']['original_contract_backup']).parent
    for name,functions in [('d_train.py',['random_external','replay_mix','add_complete','collect_extras','collect_base']),('j_train.py',['run_joint_batch','ordinal_plan','source_phase','external_seeds','validate_behavior_episode'])]:
        def selected(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        old=selected(backup/name);new=selected(ROOT/'scripts'/name)
        for name1 in functions:equal(old[name1],new[name1],'numerical_AST_unchanged_'+name1)
    checks['numerical_functions_AST_unchanged']=True
    for p in (ROOT/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True,cfile=str(base/('compile_'+p.name+'c')))
    checks['compile_all']=True
    equal(budget_counts(200),EXPECTED200,'lean_base_budget');equal(budget_counts(),EXPECTED_FINAL,'lean_final_budget');checks['budget_alpha_frozen_unchanged']=True
    fixture=base/'reader.json';put(fixture,dict(value=17));original=Path.read_text;attempt=[0]
    def transient(path,*args,**kwargs):
        if path==fixture and attempt[0]<2:attempt[0]+=1;raise PermissionError('intentional sharing-conflict selftest')
        return original(path,*args,**kwargs)
    with patch.object(Path,'read_text',transient):equal(read(fixture),dict(value=17),'bounded_reader_retry')
    checks['PermissionError_reader_retry']=True
    jsonl=base/'live.jsonl';jsonl.write_text('{"a":1}\n{"a":',encoding='utf-8');equal(rows(jsonl),[{'a':1}],'live_partial_line_snapshot');checks['incomplete_JSONL_tail_safe']=True
    config=read(CONFIG);l=Learner(4104,torch.device('cpu'),config=config);before=hashes(l);p=base/'minimal.pt';save_candidate(p,l,Replay(),4104,220);z=load_evaluation(p,contract()['identity']);check('replay' not in z and not any('optimizer' in k for v in z['learner'].values() if isinstance(v,dict) for k in v),'minimal_no_replay_optimizer')
    second=Learner(4104,torch.device('cpu'),config=config);second.load_state_dict(z['learner'])
    for owner,name in [('sac','actor'),('ppo','actor'),('ppo','value')]:equal(getattr(getattr(l,owner),name).state_dict(),getattr(getattr(second,owner),name).state_dict(),'minimal_loaded_'+owner+'_'+name)
    lower=torch.zeros(LOWER_DIM);upper=torch.zeros(UPPER_DIM)
    with torch.no_grad():
        for original_net,loaded_net,data in [(l.sac.actor,second.sac.actor,lower),(l.ppo.actor,second.ppo.actor,upper),(l.ppo.value,second.ppo.value,upper)]:equal(original_net(data),loaded_net(data),'minimal_network_output_equal')
    checks['minimal_network_parameters_and_outputs_equal']=True;checks['minimal_candidate_bytes']=p.stat().st_size
    payload=dict(schema=SCHEMA,safe_episode_boundary=True,identity=contract()['identity'],completed_episode=0,pending_evaluation=None,resumable=True,kind='jointD_full_resume',learner=l.state_dict(),replay=Replay())
    p=base/'resume.pt';save_full(p,payload);restored=load_resume(p);equal(restored['learner'],payload['learner'],'lean_storage_loadable_full_state');checks['lean_resume_full_state_loadable']=True
    # Restore scheduler-compatible writer policy and input proof; no training occurs here.
    checks['input_start_once']=input_checks();write_diff()
    from d_eval import group
    tf=str(old_tfoo(4104)/'checkpoints/episode-0200-anchor.pt')
    g=dict(label='lean-TFOO-4104-scene1',seed=4104,upper=tf,lower=None,upper_kind='TFOO',route='cvx',anchors=[str(NEW/'summaries/seed_4104/TFOO-200.json')],diagnostic_only=True)
    result=group(g,scenario_ids=[1],output_root=base/'groups')
    check(result['stream_consistency']['passed'] and result['anchor_checks']['passed'],'lean_TFOO_streams_anchor')
    check(not result['window_files'] and not list((base/'groups').rglob('windows/*.json')),'no_raw_window_files')
    for file in result['scenario_files']:check('samples_5s' not in read(file['path'])['observations'],'scene_summary_only')
    checks['anchor_scene1_both_streams']=True;checks['raw_windows_not_written']=True
    put(ROOT/'recovery/lean_preflight.json',dict(passed=True,time=now(),checks=checks,training_ran=False,decision_path_changed=False))
    print('LEAN_PREFLIGHT_PASS',json.dumps(checks,ensure_ascii=False),flush=True)
    return 0
if __name__=='__main__':
    import sys
    try:sys.exit(main())
    except Exception as e:put(ROOT/'recovery/lean_preflight_failure.json',dict(time=now(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),training_ran=False));raise
