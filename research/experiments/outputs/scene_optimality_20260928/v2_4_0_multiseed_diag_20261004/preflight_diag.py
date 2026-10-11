# Read-only preflight and seed-only fresh manifest.
from diag_shared import *
from hrl_v2.evaluate import manifest,gate_manifest
from hrl_v2.schedule import phase
from hrl_v2.upper_baselines import upper_phases
from hrl_v2.randomness import derive

def preflight():
 start=time.perf_counter();record=dict(status='checking',started_at=now(),git_before=git_states(),imports=assert_imports())
 device=configure();config=read_config(CONFIG,WT);binding=identity(WT,CONFIG)
 record.update(identity=binding,runtime=runtime_record(),sys_prefix=sys.prefix,torch_threads=torch.get_num_threads(),device=str(device))
 check(str(device)=='cuda:0' and torch.get_num_threads()==10,'device_threads')
 record['gpu']=gpu_check();record['protocol_sha256']=protocol_check()
 trace=read(R24101/'evaluate_sacdet.py') if False else (R24101/'evaluate_sacdet.py').read_text(encoding='utf-8')
 tracepaths=(R24101/'sacdet_4101.py').read_text(encoding='utf-8')
 check("B/'runs/TFOO/seed_4101/checkpoints/episode-0200-anchor.pt'" in trace,'TFOO4101_E4_trace')
 check("B=WT/'outputs/lite_v2_4_0/baselines_4101_20260930_221900'" in tracepaths,'TFOO4101_B_trace')
 panel=read(B4101/'launch_contract.json')['panels']['A']
 equal(str(Path(panel['baseline_checkpoints']['TFOO']).resolve()),str(paths(4101)['TFOO'].resolve()),'TFOO4101_panel_path')
 record['TFOO4101_trace']=dict(script=str(R24101/'evaluate_sacdet.py'),path_definition=str(R24101/'sacdet_4101.py'),checkpoint=str(paths(4101)['TFOO']),panel_A=panel['output'])
 chain=[];metadata={}
 for seed in SEEDS:
  rt=roots(seed);stage2=rt['stage2'].parent.parent;stage3=rt['stage3'].parent.parent
  validation=(B4101 if seed==4101 else WORK)/'validation/validation.json'
  contracts=[validation,rt['stage1']/'contract.json',stage2/'contract.json',stage3/'contract.json']
  for p in contracts:
   r=read(p);actual=r.get('identity');used=p
   if actual is None:used=p.parent/'preflight.json';actual=read(used).get('identity')
   equal(actual,binding,'identity_chain_'+str(p));chain.append(dict(seed=seed,path=str(p),identity_source=str(used),passed=True))
   if p==validation:check(r['status']=='passed','validation_passed_'+str(seed),r['status'])
  frozen=read(stage3/'preflight.json')['frozen_upper']['hashes'];hh={};md={}
  oldtest=T4101 if seed==4101 else OUTM/f'seed_{seed}/test'
  testmeta=read(oldtest/'preflight.json').get('checkpoint_metadata',{});testids=read(oldtest/'test_contract.json').get('checkpoint_sha256',{})
  for name,path in paths(seed).items():
   check(path.is_file(),'checkpoint_exists',str(path));side=read(path.with_suffix('.json'));actual=sha(path)
   if 'sha256' in side:equal(actual,side['sha256'],'checkpoint_sidecar_sha')
   if name.startswith('C'):
    n=int(name[1:]);check(side['cycle']==n and side['completed_episode']==420+20*(n-1),'cycle_mapping',dict(seed=seed,name=name,cycle=side.get('cycle'),completed=side.get('completed_episode')))
   l=model(path,seed,binding,config,device,'TFOO' if name=='TFOO' else None);h=hashes(l)
   if name.startswith('C'):
    equal(h,side['parameter_hashes'],'cycle_all_network_hashes')
   if name.startswith('C') or name=='SD200':equal({k:h[k] for k in frozen},frozen,'frozen_upper_hashes')
   for recordname,old in testmeta.items():
    if str(Path(old.get('path','')).resolve())==str(path.resolve()):equal(actual,old['sha256'],'old_test_checkpoint_sha')
   if name in testids:equal(actual,testids[name],'old_test_contract_sha')
   md[name]=dict(path=str(path),sha256=actual,parameter_hashes=h,sidecar_cycle=side.get('cycle'),completed_episode=side['completed_episode']);hh[name]=h
   del l;clean_gpu()
  equal({k:v for k,v in hh['SD200'].items() if k.startswith('sac_')},{k:v for k,v in hh['S300'].items() if k.startswith('sac_')},'SD200_S300_five_SAC_networks')
  metadata[str(seed)]=md;record.update(identity_chain=chain,checkpoints=metadata);put(NEW/'preflight.json',record);print('PREFLIGHT_SEED_PASS',seed,flush=True)
 record.update(status='PASS',wall_seconds=time.perf_counter()-start);put(NEW/'preflight.json',record)
 CHECKS['preflight']=dict(status='PASS',identity_chains=len(chain),checkpoints=sum(len(v) for v in metadata.values()),frozen_upper_network_checks=110,SAC_network_match_checks=25)
 return config,binding,record

def fresh_manifest():
 old=manifest('test');formula=dict(schema=old['schema'],split='test',scenarios=[dict(scenario=i,**{k+'_seed':derive(0,episode=i,purpose=k,split='test') for k in ('layout','load','arrival')}) for i in range(1,21)],policy_streams=[1,2,3,4])
 equal(formula,old,'fresh_formula_1_20')
 m=dict(schema='hrl_v2_fresh_test_manifest_candidate_1_0',split='test',scenarios=[dict(scenario=i,**{k+'_seed':derive(0,episode=i,purpose=k,split='test') for k in ('layout','load','arrival')}) for i in range(21,41)],policy_streams=[1,2,3,4])
 fresh=[r[k] for r in m['scenarios'] for k in ('layout_seed','load_seed','arrival_seed')];check(len(fresh)==60 and len(set(fresh))==60,'fresh_seed_unique')
 sets={name:{r[k] for r in mm['scenarios'] for k in ('layout_seed','load_seed','arrival_seed')} for name,mm in [('test',old),('dev',manifest('dev')),('gate_dev20',gate_manifest(20))]}
 sets['training_5_seeds']={derive(s,p.stage,p.cycle,p.block_episode,purpose=k) for s in SEEDS for p in map(phase,range(1,601)) for k in ('layout','load','arrival')}
 sets['upper_baselines_5_seeds']={derive(s,p.stage,p.cycle,p.block_episode,purpose=k) for s in SEEDS for p in upper_phases() for k in ('layout','load','arrival')}
 checks={k:dict(seed_count=len(v),intersection=sorted(set(fresh)&v),passed=not(set(fresh)&v)) for k,v in sets.items()}
 check(all(v['passed'] for v in checks.values()),'fresh_disjoint',checks)
 p=NEW/'fresh_test_manifest_21_40.json';put(p,m)
 result=dict(status='PASS',scenarios=20,external_seeds=60,unique_external_seeds=60,policy_streams=m['policy_streams'],formula_1_20_equal=True,no_environments_constructed=True,no_windows_executed=True,json_digest=json_digest(m),sha256=sha(p),disjoint_checks=checks)
 put(NEW/'fresh_manifest_checks.json',result);CHECKS['fresh_manifest']=result;return result
