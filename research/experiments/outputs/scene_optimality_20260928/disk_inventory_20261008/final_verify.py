import json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
P=Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/disk_inventory_20261008')
d=json.loads((P/'inventory.json').read_text(encoding='utf-8'))
assert all(x['category'] in 'ABCD' for x in d['entries'])
assert len({x['path'].lower() for x in d['entries']})==len(d['entries'])
assert d['reference_search']['complete'] and d['reference_search']['exit_code']==0
assert not d['C_ranked_nonoverlapping'] and d['C_reclaimable_allocated_total_bytes']==0
assert all(x['references'] for x in d['entries'] if x.get('initial_category')=='C')
assert d['git_matches_before']
missing=[x['path'] for x in d['A_required_roots'] if not x['exists']]
for root,old in d['git_after'].items():
 actual={k:subprocess.check_output(['git','--no-optional-locks','-C',root,*cmd],text=True,encoding='utf-8').strip() for k,cmd in [('status',['status','--porcelain']),('branch',['branch','--show-current']),('head',['rev-parse','HEAD'])]}
 assert actual==old
oldfree=d['disk']['free_current_bytes'];oldstamp=d['disk']['measured_at'];u=shutil.disk_usage('D:/')
stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
d['disk'].update(measured_at=stamp,free_current_bytes=u.free,used_current_bytes=u.used,free_after_C_calculated_bytes=u.free,current_meets_strict_gt=u.free>d['threshold_bytes'],after_C_meets_strict_gt=u.free>d['threshold_bytes'])
written=sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
d['observations']=dict(inventory_written_logical_bytes_before_final_refresh=written,free_space_changed_during_task=True,cause_not_investigated=True,allocation_measurements_required=False,C_empty=True,generic_name_hits_do_not_establish_actual_runtime_dependency=True,missing_specified_roots=missing)
(P/'inventory.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
s=(P/'inventory.md').read_text(encoding='utf-8').replace(str(oldfree),str(u.free)).replace(f'{oldfree/1e9:.3f}',f'{u.free/1e9:.3f}').replace(oldstamp,stamp)
notes='\n## \u6700\u7ec8\u590d\u6838\u8865\u5145\n\n'
notes+=f'\u6700\u7ec8\u7a7a\u95f4\u91c7\u6837\uff1a{stamp}\uff0c{u.free:,}\u5b57\u8282\uff08{u.free/1e9:.3f}GB\uff09\u3002\u626b\u63cf\u5f00\u59cb\u4e3a\u7a7a\u95f2{d["metadata_scan"]["free_before_bytes"]:,}\u5b57\u8282\uff1b\u6e05\u70b9\u671f\u95f4\u7a7a\u95f2\u7a7a\u95f4\u53d8\u5316\u672a\u6838\u67e5\u539f\u56e0\uff0c\u4e0d\u80fd\u5f52\u56e0\u4e8e\u5220\u9664\u3002\u672c\u8f6e\u6e05\u5355/\u8bc1\u636e\u903b\u8f91\u5927\u5c0f\u7ea6{written/1e9:.3f}GB\uff0c\u4e14\u9879\u76ee\u5927\u5c0f\u7edf\u8ba1\u6392\u9664\u4e86\u672c\u6b21INV\u3002\n\n'
notes+='\u5f15\u7528\u547d\u4e2d\u5305\u62ec\u7b2c\u4e09\u65b9\u4ee3\u7801\u91cc\u7684\u901a\u7528\u53d8\u91cf\u540d\uff0c\u5982pkgs\uff1b\u5b83\u4e0d\u7b49\u4e8e\u786e\u8ba4\u4e86\u771f\u5b9e\u8fd0\u884c\u4f9d\u8d56\u3002\u4f9d\u7528\u6237\u201c\u8def\u5f84\u6216\u76ee\u5f55\u540d\u51fa\u73b0\u5373\u964dD\u201d\u7684\u89c4\u5b9a\u4ecd\u7136\u964d\u7ea7\u3002C\u4e3a\u7a7a\uff0c\u672a\u6253\u5f00\u4efb\u4f55\u7f13\u5b58\u5206\u914d\u5c5e\u6027\u53e5\u67c4\uff0c\u91ca\u653e\u91cf\u4e3a0\u3002\n\n'
notes+='\u7528\u6237\u5217\u51fa\u7684\u90e8\u5206\u6839\u8def\u5f84\u5b9e\u9645\u4e0d\u5b58\u5728\uff1a'+ '\u3001'.join(missing)+'\u3002\u5df2\u5728A\u6307\u5b9a\u8303\u56f4\u8868\u6807\u4e3a\u7f3a\u5931\uff0c\u6ca1\u6709\u65b0\u5efa\u5b83\u4eec\u3002\u5b9e\u9645src\u4e0b\u7684\u914d\u7f6e\u4ecd\u5904\u4e8eA\u4fdd\u62a4\u8303\u56f4\u3002\n\n'
notes+='\u590d\u6838\u901a\u8fc7\uff1a\u5168\u90e8532\u4e2a\u5927\u4e8e200MB\u9879\u5df2\u5f52\u7c7b\uff1b\u6e05\u5355\u8def\u5f84\u65e0\u91cd\u590d\uff1b\u6240\u6709\u7531C\u964dD\u9879\u5747\u6709\u5f15\u7528\u8bc1\u636e\uff1b\u4e24\u4e2aGit\u5de5\u4f5c\u6811\u4fdd\u6301\u6307\u5b9a\u5206\u652f\u3001HEAD\u4e0e\u7a7astatus\u3002\n\n'
notes+='\u9996\u6b21\u6700\u7ec8\u590d\u6838\u628a\u201c\u6240\u6709\u6307\u5b9a\u6839\u8def\u5f84\u5b58\u5728\u201d\u8bef\u5199\u4e3a\u65ad\u8a00\uff0c\u56e0\u4e0a\u8ff03\u5904\u539f\u672c\u4e0d\u5b58\u5728\u800c\u5931\u8d25\uff1b\u672a\u5199\u5165\u65e7\u6587\u4ef6\u3002\u5df2\u6539\u4e3a\u5982\u5b9e\u8bb0\u5f55\u7f3a\u5931\uff0c\u518d\u6267\u884c\u590d\u6838\uff0c\u672a\u6539\u53d8\u5206\u7c7b\u6216\u4fdd\u7559\u6761\u4ef6\u3002\n'
(P/'inventory.md').write_text(s+notes,encoding='utf-8')
checks=dict(checked_at=stamp,current_free_bytes=u.free,total_bytes=u.total,C_count=0,C_total_bytes=0,large_entries=sum(x['bytes']>200_000_000 for x in d['entries']),missing_roots=missing,reference_search_complete=True,git_after=d['git_after'],passed=True)
(P/'final_readonly_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(P/'verification_attempt1_error.txt').write_text('AssertionError: assertion that all user-named A roots exist. Actual absent roots: '+json.dumps(missing)+'\nCorrection: retain/protect those paths and report absence; do not create them.\n',encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
