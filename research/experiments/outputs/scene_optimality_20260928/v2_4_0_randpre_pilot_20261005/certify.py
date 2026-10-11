# Lightweight final evidence watcher; no training/evaluation/model loading.
from pilot_common import *
from hrl_v2.checkpoint import identity
while True:
 if (PILOT/'failure.json').exists():
  target='failure';break
 p=PILOT/'progress.json'
 if p.exists() and read(p).get('status')=='complete':target='complete';break
 time.sleep(30)
try:
 before=read(PILOT/'old_checkpoint_sha_before.json');after={p:sha(p) for p in before};equal(after,before,'old_checkpoint_hashes_unchanged')
 equal(identity(WT,CONFIG),read(PILOT/'pilot_contract.json')['identity'],'final_source_config_scene_identity_unchanged')
 check(sha(PROTO)==PROTO_SHA,'final_frozen_protocol_unchanged');git=git_states()
 result=dict(status='PASS',checked_at=now(),run_status=target,old_checkpoints_verified=len(before),old_checkpoint_sha256_before=before,old_checkpoint_sha256_after=after,source_config_scene_identity_unchanged=True,frozen_protocol_sha256=sha(PROTO),pilot_contract_sha256=sha(PILOT/'pilot_contract.json'),git=git)
 put(PILOT/'final_readonly_checks.json',result)
 if target=='complete':
  p=PILOT/'summary.md';textfile(p,p.read_text(encoding='utf-8')+f'\n## 只读保护终检\n\n旧checkpoint哈希逐项一致：{len(before)}/{len(before)}；源码/config/物理场景identity未变；原冻结协议SHA256未变；两棵Git工作树干净。核对时间：{now()}。\n')
 print('FINAL_READONLY_CHECKS_PASS',target,len(before),flush=True)
except BaseException as e:failure('final_readonly_checks',e);raise