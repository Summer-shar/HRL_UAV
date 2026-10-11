from j_common import *
import torch
from hrl_v2 import SCHEMA
from hrl_v2.checkpoint import save as original_save
from hrl_v2.runtime import parameter_hash
from hrl_v2.upper_baselines import require_checkpoint_type

def load_evaluation(path,identity,expected=None):
    z=torch.load(path,map_location='cpu',weights_only=False)
    check(z.get('schema')==SCHEMA and z.get('safe_episode_boundary'),'evaluation_complete_checkpoint')
    equal(z['identity'],identity,'evaluation_checkpoint_identity')
    return require_checkpoint_type(z,expected) if expected else z

def save_candidate(path,learner,replay,seed,n):
    path=output_path(path);check(not path.exists(),'new_lean_candidate')
    cpu=torch.get_rng_state().clone();cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
    state=learner.state_dict()
    minimal={owner:{k:state[owner][k] for k in keys} for owner,keys in [('sac',('actor','counts','version')),('ppo',('actor','value','counts','version'))]}
    minimal['discounts']=state['discounts']
    payload=dict(schema=SCHEMA,safe_episode_boundary=True,identity=contract()['identity'],completed_episode=n,pending_evaluation=None,learner=minimal,global_cpu_rng=cpu,global_cuda_rng=cuda,seed=seed,resumable=False,kind='jointD_minimal_evaluation_pair',replay_included=False,optimizer_included=False,protocol_contract_sha256=sha(CPIL/'contract.json'))
    original_save(path,payload)
    equal(torch.get_rng_state(),cpu,'candidate_output_CPU_RNG_unchanged');equal(torch.cuda.get_rng_state_all(),cuda,'candidate_output_CUDA_RNG_unchanged')
    return str(path)

def save_full(path,payload):
    path=output_path(path);tmp=path.with_name(path.name+'.writing')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    put(path.with_suffix('.json'),dict(schema=SCHEMA,completed_episode=payload['completed_episode'],kind=payload['kind'],identity=payload['identity'],write_complete=True,sha256_not_computed=True))

def load_resume(path):
    z=torch.load(path,map_location='cpu',weights_only=False)
    check(z.get('schema')==SCHEMA and z.get('safe_episode_boundary') and z.get('resumable'),'resumable_complete_boundary')
    equal(z['identity'],contract()['identity'],'resume_recorded_code_config_identity')
    return z
