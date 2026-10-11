
import json
from pathlib import Path
import hrl_v2,hrl_uav
assert all(Path(m.__file__).resolve().is_relative_to(Path('D:/HRL_UAV_V2_speedup/src').resolve()) for m in (hrl_v2,hrl_uav))
p=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_curriculum_pilot_20261008')
f=json.loads((p/'step0_facts/facts.json').read_text(encoding='utf-8'))
e=json.loads((p/'failure.json').read_text(encoding='utf-8'))
c=json.loads((p/'final_readonly_checks.json').read_text(encoding='utf-8'))
assert f['status']=='failed' and e['substantive_check'] is True
assert 'disk_free_greater_than_two_times_estimate' in e['message']
assert c['status']=='PASS' and not c['training_started'] and not c['evaluation_started']
d=f['disk']
lines=[
 '# 4104 \u8bfe\u7a0b\u5f0f\u9884\u8bad\u7ec3\u8bd5\u70b9\uff1a\u7b2c 0 \u6b65\u505c\u6b62\u62a5\u544a','',
 f'\u505c\u6b62\u8bb0\u5f55\u65f6\u95f4\uff1a{e["time"]}\u3002','',
 '\u78c1\u76d8\u786c\u6761\u4ef6\u672a\u901a\u8fc7\u3002\u672c\u6b21\u672a\u542f\u52a8\u8bad\u7ec3\u6216\u4efb\u4f55\u8bc4\u4f30\uff0c\u672a\u521b\u5efa\u5408\u540c\u6216 checkpoint\u3002','',
 '## \u78c1\u76d8\u68c0\u67e5','',
 '\u9644\u4ef6\u7b2c 4 \u8282\u7b2c 5 \u6761\u8981\u6c42 D: \u5269\u4f59\u7a7a\u95f4\u4e25\u683c\u5927\u4e8e\u4f30\u7b97\u7528\u91cf\u7684 2 \u500d\u3002\u4ee5\u4e0b GB \u6309 10^9 \u5b57\u8282\u6362\u7b97\uff1b\u8f93\u51fa\u7528\u91cf\u662f\u57fa\u4e8e\u65e7\u6587\u4ef6\u5b9e\u6d4b\u5927\u5c0f\u7684\u4f30\u7b97\u3002','',
 '|\u9879\u76ee|\u5b57\u8282|GB|','|---|---:|---:|',
 f'|D: \u5b9e\u6d4b\u5269\u4f59|{d["free_bytes"]}|{d["free_bytes"]/1e9:.6f}|',
 f'|\u672c\u6b21\u8f93\u51fa\u4f30\u7b97|{d["total_estimated_bytes"]}|{d["total_estimated_bytes"]/1e9:.6f}|',
 f'|\u987b\u5927\u4e8e\u7684\u4e24\u500d\u95e8\u69db|{d["required_free_bytes_strictly_greater_than"]}|{d["required_free_bytes_strictly_greater_than"]/1e9:.6f}|',
 f'|\u4e0e\u95e8\u69db\u7684\u5dee\u989d|{d["shortfall_bytes"]}|{d["shortfall_bytes"]/1e9:.6f}|','',
 f'\u4ec5\u8ba1\u7b97\u89c4\u5b9a\u5b58\u6863\u548c 26 \u7ec4\u9a8c\u8bc1\u8bc4\u4f30\u5c31\u7ea6\u4e3a {d["subset_estimated_bytes"]/1e9:.6f} GB\uff0c\u4e24\u500d\u4e3a {d["subset_required_double_bytes"]/1e9:.6f} GB\uff0c\u4ecd\u8d85\u8fc7\u5269\u4f59\u7a7a\u95f4\u3002\u8be5\u5b50\u96c6\u8fd8\u672a\u8ba1\u8bad\u7ec3\u548c\u590d\u73b0\u65e5\u5fd7\u3001\u4f4d\u7f6e\u8bb0\u5f55\u53ca\u62a5\u544a\u3002','',
 '\u4f30\u7b97\u5206\u9879\uff1a','',
 '|\u9879\u76ee|\u4f9d\u636e|GB|','|---|---|---:|',
 f'|\u5e26\u7ecf\u9a8c\u6c60\u5b58\u6863|7 \u00d7 \u539f S300 \u7684 {d["full_checkpoint_basis"]["bytes"]} \u5b57\u8282|{d["full_checkpoint_estimated_bytes"]/1e9:.6f}|',
 f'|\u5176\u4ed6 learner \u5b58\u6863|13 \u00d7 \u8bd5\u70b9 U200 \u7684 {d["other_learner_checkpoint_basis"]["bytes"]} \u5b57\u8282|{d["other_checkpoint_estimated_bytes"]/1e9:.6f}|',
 f'|\u9a8c\u8bc1\u8bc4\u4f30|26 \u00d7 \u65e7\u7ec4\u6700\u5927\u5b9e\u6d4b\u5927\u5c0f|{d["evaluation_estimated_bytes"]/1e9:.6f}|',
 f'|\u8bad\u7ec3\u8f93\u51fa\uff0c\u4e0d\u542b\u5b58\u6863|\u65e7\u8bd5\u70b9 4104 \u7b2c 1\u20135 \u6b65\u5b9e\u6d4b|{sum(x["bytes"] for x in d["training_non_checkpoint_measured"])/1e9:.6f}|',
 f'|RP-U / RP-L \u65e5\u5fd7|\u65e7\u590d\u73b0\u76ee\u5f55\u5b9e\u6d4b|{sum(x["bytes"] for x in d["RP_non_checkpoint_measured"])/1e9:.6f}|',
 f'|\u62a5\u544a\u4e0e\u5143\u6570\u636e\u9884\u7559|\u4f30\u8ba1 32 MiB|{d["report_metadata_reserve_estimate_bytes"]/1e9:.6f}|','',
 '\u6240\u6709\u4f9d\u636e\u8def\u5f84\u3001\u9010\u9879\u6587\u4ef6\u6570\u548c\u5b57\u8282\u6570\u89c1 step0_facts/facts.json \u7684 disk \u6bb5\u3002','',
 '## \u5df2\u901a\u8fc7\u4e0e\u672a\u6267\u884c','',
 '- Git \u5206\u652f\u3001HEAD\u3001\u5e72\u51c0\u72b6\u6001\u3001\u51bb\u7ed3\u534f\u8bae\u3001\u5bfc\u5165\u8def\u5f84\u3001GPU \u68c0\u67e5\u901a\u8fc7\u3002',
 '- identity \u4e0e PILOT \u5408\u540c\u3001ORIG \u5408\u540c\u3001WORK validation \u4e00\u81f4\uff1bvalidation \u4e3a passed\u3002',
 '- \u539f 300 \u8f6e JSON \u7684\u5fc5\u8981\u5b57\u6bb5\u53ca\u5916\u90e8\u79cd\u5b50\u9f50\u5168\uff1b\u6ca1\u6709\u53ef\u7eed\u8bad\u7684\u7b2c 200 \u8f6e\u5b58\u6863\u3002',
 '- 25\u2013200 \u8f6e 8 \u4efd\u5feb\u7167\u7684\u54c8\u5e0c\u3001\u8eab\u4efd\u3001\u8f6e\u6b21\u53ca\u7ed3\u6784\u6838\u5bf9\u901a\u8fc7\uff0cSAC \u4e94\u4e2a\u7f51\u7edc\u4e0e PPO \u4e24\u4e2a\u7f51\u7edc\u53ef\u9010\u53c2\u6570\u6bd4\u8f83\u3002',
 '- \u5feb\u7167\u5b9e\u9645\u542b learner.state_dict\uff0c\u5305\u62ec\u4f18\u5316\u5668\u3001\u8ba1\u6570\u548c\u7248\u672c\uff1b\u7f3a\u5c11\u7ecf\u9a8c\u6c60\u53ca\u5168\u5c40 RNG\uff0c\u4ecd\u4e0d\u53ef\u7eed\u8bad\u3002',
 '- \u539f\u8bad\u7ec3\u5faa\u73af\u4e0e Replay \u6807\u7b7e\u5207\u6362\u3001\u76f4\u63a5\u8d4b\u503c\u4fdd\u7559\u6c60\u3001sample \u4e0d\u6309\u6807\u7b7e\u8fc7\u6ee4\u300155 \u8f6e FIFO \u5df2\u6838\u5bf9\u3002',
 '- \u78c1\u76d8\u95e8\u69db\u5931\u8d25\u540e\u672a\u7ee7\u7eed\u53c2\u8003\u66f2\u7ebf\u4e0e\u8017\u65f6\u4f30\u8ba1\uff1bRP\u3001DG\u3001S1\u2013S5\u3001D-a\u3001D-b\u3001A-H/A-F \u5747\u672a\u8fd0\u884c\u3002',
 '- \u5224\u636e 1 \u548c\u5224\u636e 2 \u672a\u8bc4\u4f30\uff0c\u6ca1\u6709\u672c\u6b21\u4e3b\u8868\u7ed3\u679c\uff1btest \u8bc4\u4f30\u4e3a 0 \u6b21\u3002','',
 '## \u5931\u8d25\u53ca\u5b9e\u73b0\u4fee\u6b63','',
 '\u7b2c\u4e00\u5904\u5931\u8d25\u4e3a HardCheckError\uff1adisk_free_greater_than_two_times_estimate\uff0c\u5c5e\u4e8e\u5b9e\u8d28\u6761\u4ef6\u4e0d\u6ee1\u8db3\u3002\u6ca1\u6709\u653e\u5bbd\u6761\u4ef6\u6216\u81ea\u52a8\u91cd\u8dd1\u3002','',
 '\u7b2c\u4e00\u6b21\u5185\u8054\u751f\u6210\u4e2d\u6587\u62a5\u544a\u51fa\u73b0 SyntaxError: Non-UTF-8 code starting with byte 0xe5\u3002\u6309\u5141\u8bb8\u7684\u4e00\u6b21\u5b9e\u73b0\u4fee\u6b63\u89c4\u5219\uff0c\u5c06\u62a5\u544a\u751f\u6210\u811a\u672c\u7684\u4e2d\u6587\u5b57\u7b26\u4e32\u6539\u4e3a ASCII Unicode \u8f6c\u4e49\uff1b\u4ec5\u8865\u5199\u62a5\u544a\uff0c\u4e0d\u91cd\u8dd1\u9884\u68c0\uff0c\u4e0d\u8bad\u7ec3\uff0c\u4e0d\u8bc4\u4f30\u3002\u9519\u8bef\u548c diff \u89c1 implementation_corrections.json\u3001report_attempt1/failure.json\u3001report_attempt2/encoding_correction.diff\u3002','',
 '## \u53ea\u8bfb\u4fdd\u62a4\u4e0e\u8bc1\u636e','',
 f'\u5df2\u8bfb\u8f93\u5165 SHA256 \u524d\u540e\u4e00\u81f4\uff1a{c["protected_files"]}/{c["protected_files"]}\uff1b\u51bb\u7ed3\u534f\u8bae\u672a\u53d8\uff1b\u7ed3\u675f\u65f6\u4e24\u68f5 Git \u5de5\u4f5c\u6811\u5e72\u51c0\u3002\u6240\u6709\u65b0\u6587\u4ef6\u4ec5\u5728 CPIL\uff0c\u65e7\u6e90\u7801\u3001\u914d\u7f6e\u3001checkpoint \u548c\u7ed3\u679c\u672a\u6539\u52a8\u3002','',
 '- failure.json\uff1a\u5931\u8d25\u9636\u6bb5\u3001traceback\u3001\u5df2\u901a\u8fc7\u68c0\u67e5\u53ca\u5b9e\u8d28\u6027\u6807\u8bb0\u3002',
 '- step0_facts/facts.json\uff1a\u5b8c\u6574\u9884\u68c0\u3001\u5feb\u7167\u7ed3\u6784\u3001\u78c1\u76d8\u4f9d\u636e\u53ca\u8f93\u5165\u54c8\u5e0c\u3002',
 '- git_after_failure.json\uff1a\u505c\u6b62\u540e\u7684\u5206\u652f\u3001\u63d0\u4ea4\u53ca\u5de5\u4f5c\u6811\u72b6\u6001\u3002',
 '- final_readonly_checks.json\uff1a\u8f93\u5165 SHA256 \u524d\u540e\u5bf9\u7167\u3002',
 '- cp_contract.json / cp_contract.sha256\uff1a\u672a\u521b\u5efa\u3002'
]
text='\n'.join(lines)+'\n'
for rel in ('summary.md','seed_4104/report.md'):
 q=(p/rel).resolve();assert q.is_relative_to(p.resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
(p/'seed_4104/report.json').write_text(json.dumps(dict(status='stopped_step0',time=e['time'],seed=4104,disk=d,criterion1='not_evaluated',criterion2='not_evaluated',training_started=False,evaluation_started=False,implementation_correction='report encoding only',protected_files_unchanged=c['protected_files'],git=c['git'],sources=['step0_facts/facts.json','failure.json','final_readonly_checks.json']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='STOPPED',time=e['time'],report=str(p/'seed_4104/report.md'),disk_free_GB=d['free_bytes']/1e9,estimated_GB=d['total_estimated_bytes']/1e9,required_free_GB=d['required_free_bytes_strictly_greater_than']/1e9,protected_inputs_verified=c['protected_files']),ensure_ascii=False))

