# Final disk inventory, existing files remain read-only.
import os,json,shutil,subprocess,collections
from pathlib import Path
from datetime import datetime,timezone,timedelta
INV=Path(__file__).resolve().parent
TZ=timezone(timedelta(hours=8))
def norm(p):return str(p).replace('\\','/').rstrip('/').lower()
def below(p,q):return norm(p)==norm(q) or norm(p).startswith(norm(q)+'/')
def gb(n):return f'{n/1e9:.3f}'
def clean(s):return str(s).replace('|','\\|').replace('\n',' ')
def dump(n,x):(INV/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 d=json.loads((INV/'classified_inventory.json').read_text(encoding='utf-8'))
 expected=[('D:/HRL_UAV_V2','codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),('D:/HRL_UAV_V2_speedup','codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')]
 after={}
 for root,branch,head in expected:
  state={k:subprocess.check_output(['git','--no-optional-locks','-C',root,*cmd],text=True,encoding='utf-8').strip() for k,cmd in [('status',['status','--porcelain']),('branch',['branch','--show-current']),('head',['rev-parse','HEAD'])]}
  if state!=dict(status='',branch=branch,head=head):raise RuntimeError('Unexpected final git state '+repr(state))
  after[root]=state
 entries=d['entries'];ranked=d['C_ranked_nonoverlapping']
 # Maximal D roots that are not mixed A/B containers: do not double count parents.
 uncertain=[e for e in entries if e['category']=='D']
 roots=[]
 for e in sorted(uncertain,key=lambda x:len(norm(x['path']))):
  if '\u6df7\u5408\u6c47\u603b\u76ee\u5f55' in e['reason']:continue
  if any(below(e['path'],x['path']) for x in roots):continue
  roots.append(e)
 roots.sort(key=lambda e:e['bytes'],reverse=True)
 # User-specified small source/config roots must be listed too.
 for item in d['A_required_roots']:
  match=next((e for e in entries if norm(e['path'])==norm(item['path'])),None)
  if match:item.update(bytes=match['bytes'],modified=match['modified'])
  elif item['exists']:
   total=0
   for parent,ds,fs in os.walk(item['path'],followlinks=False):
    for fn in fs:total+=os.stat(Path(parent)/fn,follow_symlinks=False).st_size
   item.update(bytes=total,modified=datetime.fromtimestamp(Path(item['path']).stat().st_mtime,TZ).isoformat())
 # Space is sampled after this inventory's earlier evidence files were written.
 u=shutil.disk_usage('D:/');c=d['C_reclaimable_allocated_total_bytes'];limit=d['threshold_bytes']
 d['disk']=dict(measured_at=datetime.now(TZ).isoformat(),total_bytes=u.total,free_current_bytes=u.free,used_current_bytes=u.used,free_after_C_calculated_bytes=u.free+c,current_meets_strict_gt=u.free>limit,after_C_meets_strict_gt=u.free+c>limit,threshold_bytes=limit,units='GB=10^9 bytes; GiB=2^30 bytes',no_deletion_performed=True)
 d['git_after']=after;d['git_matches_before']=after==d['metadata_scan']['git_before'];d['largest_D_nonoverlapping']=roots[:20]
 d['actions']=dict(new_directory=str(INV),deleted=0,moved=0,renamed=0,compressed=0,modified_preexisting=0,training_or_evaluation_started=False,project_modules_imported=False,cleaner_commands=False,attempt2_not_received=True)
 top=d['metadata_scan']['top_level'];rootfiles=[x for x in top if x['kind']=='file'];top=[x for x in top if x['kind']!='file']
 d['metadata_scan']['top_level']=top
 d['root_loose_files']=dict(count=len(rootfiles),bytes=sum(x['bytes'] for x in rootfiles),names_withheld=True,contents_not_read=True)
 d['reference_search']['basename_coverage']='\u6309\u76ee\u5f55\u540d\u68c0\u67e5\u5df2\u8986\u76d6\u7edd\u5bf9\u8def\u5f84\uff1a\u4efb\u4f55\u5b8c\u6574\u8def\u5f84\u90fd\u542b\u8be5\u76ee\u5f55\u540d\uff1b\u6587\u5b57\u63d0\u53ca\u4e5f\u6309\u7528\u6237\u8981\u6c42\u964dD\u3002'
 d['reference_search']['first_search_error']='\u7b2c\u4e00\u6b2133101\u4e2a\u8def\u5f84/\u540d\u79f0\u6a21\u5f0f\u8d85\u8fc7rg\u7f16\u8bd1\u4e0a\u9650\uff1b\u5931\u8d25\u8bc1\u636e\u4fdd\u7559\u4e8ereference_search_attempt1.json\u4e0ereference_stderr.txt\u3002\u7b2c\u4e8c\u6b21\u4ec5\u76ee\u5f55\u540d\u641c\u7d22\uff0c\u6761\u4ef6\u4e0d\u53d8\u3002'
 dump('inventory.json',d)
 lines=[]
 def add(s=''):lines.append(s)
 def table(head,rows):
  add('| '+' | '.join(head)+' |');add('| '+' | '.join(['---']*len(head))+' |')
  for row in rows:add('| '+' | '.join(clean(x) for x in row)+' |')
  add()
 add('# D\u76d8\u78c1\u76d8\u6e05\u70b9\uff08\u53ea\u8bfb\uff0c2026-10-08\uff09');add()
 add('\u6ca1\u6709\u5220\u9664\u3001\u79fb\u52a8\u3001\u91cd\u547d\u540d\u3001\u538b\u7f29\u6216\u4fee\u6539\u4efb\u4f55\u5df2\u6709\u6587\u4ef6\uff0c\u6ca1\u6709\u8fd0\u884c\u6e05\u7406\u547d\u4ee4\u3001\u8bad\u7ec3\u6216\u8bc4\u4f30\u3002\u552f\u4e00\u65b0\u589e\u76ee\u5f55\u4e3a'+str(INV)+'\u3002\u4e0b\u9762\u7684C\u7c7b\u53ea\u662f\u5019\u9009\uff0c\u5c1a\u672a\u83b7\u5f97\u6216\u6267\u884c\u5220\u9664\u6307\u4ee4\u3002')
 add('GB=10^9\u5b57\u8282\u3002\u76ee\u5f55\u5927\u5c0f\u4e3a\u903b\u8f91\u6587\u4ef6\u957f\u5ea6\u4e4b\u548c\uff0c\u7236\u5b50\u884c\u91cd\u53e0\uff0c\u4e0d\u80fd\u76f8\u52a0\u3002C\u7c7b\u6839\u76ee\u5f55\u4e92\u4e0d\u91cd\u53e0\uff1b\u9884\u8ba1\u53ef\u91ca\u653e\u91cf\u6309\u5b9e\u9645NTFS\u5206\u914d\u5927\u5c0f\u53ca\u5916\u90e8\u786c\u94fe\u63a5\u6838\u7b97\u3002')
 add()
 add('## \u7a7a\u95f4\u4e0eattempt2');add()
 table(['\u9879\u76ee','\u5b57\u8282','GB'],[('D\u603b\u5bb9\u91cf',u.total,gb(u.total)),('\u626b\u63cf\u5f00\u59cb\u5269\u4f59',d['metadata_scan']['free_before_bytes'],gb(d['metadata_scan']['free_before_bytes'])),('\u5f53\u524d\u5269\u4f59',u.free,gb(u.free)),('C\u7c7b\u903b\u8f91\u5408\u8ba1',d['C_logical_total_bytes'],gb(d['C_logical_total_bytes'])),('C\u7c7b\u9884\u8ba1\u53ef\u56de\u6536\u5206\u914d\u7a7a\u95f4',c,gb(c)),('\u5220\u5b8cC\u540e\u7684\u8ba1\u7b97\u5269\u4f59',u.free+c,gb(u.free+c)),('attempt2\u4e25\u683c\u5927\u4e8e\u7684\u95e8\u69db',limit,gb(limit))])
 add('\u5269\u4f59\u7a7a\u95f4\u91c7\u6837\u65f6\u95f4\uff1a'+d['disk']['measured_at']+'\u3002\u6309\u5f53\u524d\u5269\u4f59\uff1a'+('\u6ee1\u8db3' if u.free>limit else '\u4e0d\u6ee1\u8db3')+'\uff1b\u6309\u5220\u5b8cC\u540e\u7684\u8ba1\u7b97\u503c\uff1a'+('\u6ee1\u8db3' if u.free+c>limit else '\u4e0d\u6ee1\u8db3')+'\u3002\u95e8\u69db\u6cbf\u75282 \u00d7 21,821,911,722 = 43,643,823,444\u5b57\u8282\uff08\u7ea643.64GB\uff09\u3002\u6ca1\u6709\u542f\u52a8attempt2\uff0c\u5c1a\u672a\u6536\u5230\u540e\u7eed\u6267\u884c\u6307\u4ee4\u3002')
 add('\u5206\u914d\u7a7a\u95f4\u4ee5FILE_READ_ATTRIBUTES\u67e5\u8be2\uff0c\u4e0d\u8bfb\u53d6\u7f13\u5b58\u5185\u5bb9\uff1b\u4ecd\u6709\u5916\u90e8\u4fdd\u7559\u786c\u94fe\u63a5\u7684inode\u8ba10\uff0c\u9009\u4e2d\u94fe\u63a5\u5168\u90e8\u5220\u9664\u65f6\u53ea\u8ba1\u4e00\u6b21\uff1b\u67e5\u8be2\u5931\u8d25\u8ba10\u3002\u76ee\u5f55\u5143\u6570\u636e\u5f00\u9500\u4e0d\u8ba1\uff0c\u8ba1\u7b97\u5269\u4f59\u4e0d\u4fdd\u8bc1\u5176\u4ed6\u8fdb\u7a0b\u4e0d\u518d\u5199\u5165\u3002')
 add()
 add('## D\u9876\u5c42\u6587\u4ef6\u5939');add()
 table(['\u9876\u5c42\u6587\u4ef6\u5939','GB','\u6587\u4ef6\u6570','\u6700\u540e\u5185\u5bb9\u4fee\u6539\u65f6\u95f4','\u5b8c\u6574\u6027'],[(x['path'],gb(x['bytes']),x.get('files',0),x.get('latest_descendant_modified',x.get('modified')),'\u4e0d\u5b8c\u6574/\u4e0b\u754c' if x['incomplete'] else '\u5b8c\u6210') for x in top])
 add('D\u6839\u76ee\u5f55\u6563\u6587\u4ef6'+str(len(rootfiles))+'\u4e2a\uff0c\u5408\u8ba1'+gb(sum(x['bytes'] for x in rootfiles))+'GB\uff1b\u4e0d\u516c\u5f00\u540d\u79f0\uff0c\u4e0d\u8bfb\u53d6\u5185\u5bb9\u3002\u4e24\u4e2a\u9879\u76ee\u548cAnaconda\u4ee5\u5916\u7684\u4e2a\u4eba\u3001\u5176\u4ed6\u9879\u76ee\u4e0e\u8f6f\u4ef6\u76ee\u5f55\u53ea\u7edf\u8ba1\u9876\u5c42\u5927\u5c0f\uff0c\u4e0d\u5217\u4efb\u4f55\u5220\u9664\u5019\u9009\u3002')
 add()
 add('## \u5927\u4e8e1GB\u76ee\u5f55\u5c55\u5f00\u81f3\u7b2c3\u5c42\uff08\u5927\u4e8e500MB\u9879\uff09');add()
 add('\u9075\u5faa\u4e2a\u4eba\u76ee\u5f55\u9690\u79c1\u89c4\u5219\uff0c\u53ea\u5c55\u5f00\u4e24\u68f5\u6307\u5b9a\u4ed3\u5e93\u548cAnaconda\u3002D\u6839\u4e3a\u7b2c0\u5c42\uff0c\u9876\u5c42\u6587\u4ef6\u5939\u4e3a\u7b2c1\u5c42\uff1b\u5176\u4ed6\u76ee\u5f55\u4e0d\u5c55\u5f00\u3002')
 table(['\u8def\u5f84','\u7c7b\u578b','\u5c42\u6570','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4'],[(x['path'],x['kind'],x['depth_from_D'],gb(x['bytes']),x['modified']) for x in sorted(d['metadata_scan']['expanded_depth3_over500MB'],key=lambda x:x['path'])])
 add('## A \u5fc5\u7559\u8f93\u5165\u4e0e\u73af\u5883');add()
 add('\u4e0b\u5217\u6307\u5b9a\u6839\u76ee\u5f55\u53ca\u5168\u90e8\u5b50\u76ee\u5f55\u4f18\u5148\u4fdd\u7559\uff0c\u5305\u62ec\u5c0f\u4e8e200MB\u9879\u3002')
 table(['\u6307\u5b9a\u8def\u5f84','\u6807\u7b7e','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u5b58\u5728'],[(x['path'],x['label'],gb(x.get('bytes',0)) if x['exists'] else '\u7f3a\u5931',x.get('modified','\u672a\u53d6\u5f97'),x['exists']) for x in d['A_required_roots']])
 add('\u6240\u6709\u5927\u4e8e200MB\u7684A\u9879\uff08\u5d4c\u5957\u884c\u4e0d\u76f8\u52a0\uff09\uff1a');add()
 table(['\u8def\u5f84','\u7c7b\u578b','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u7406\u7531'],[(e['path'],e['kind'],gb(e['bytes']),e['modified'],e['reason']) for e in entries if e['category']=='A'])
 add('## B \u5fc5\u7559\u5b9e\u9a8c\u8bb0\u5f55');add()
 add('\u5df2\u6709checkpoint\u3001\u7f51\u7edc\u5feb\u7167\u3001G1/gates\u3001\u8bc4\u4f30/test\u9762\u677f\u3001\u51bb\u7ed3\u53c2\u7167\u3001\u5408\u540c/\u534f\u8bae/\u62a5\u544a/summary\u548c\u8bba\u6587\u5907\u4efd\u5747\u4fdd\u7559\u3002\u5931\u8d25\u3001\u65e7\u7248\u672c\u6216\u5185\u5bb9\u76f8\u540c\u4e0d\u80fd\u8986\u76d6B\u89c4\u5219\u3002\u4ee5\u4e0b\u5217\u5927\u4e8e200MB\u9879\u3002')
 table(['\u8def\u5f84','\u7c7b\u578b','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u7406\u7531'],[(e['path'],e['kind'],gb(e['bytes']),e['modified'],e['reason']) for e in entries if e['category']=='B'])
 add('## C \u53ef\u5220\u5019\u9009\uff08\u672a\u5220\u9664\uff09');add()
 add('\u6309\u9884\u8ba1\u53ef\u56de\u6536\u7a7a\u95f4\u4ece\u5927\u5230\u5c0f\uff0c\u5019\u9009\u6839\u76ee\u5f55\u4e92\u4e0d\u91cd\u53e0\u3002\u5404\u9879\u5747\u5df2\u901a\u8fc7\u4e24\u4e2aoutputs\u5185\u6240\u6709.py/.json/.md\u7684\u8def\u5f84/\u76ee\u5f55\u540d\u5f15\u7528\u68c0\u67e5\uff1b\u672c\u6b21INV\u6392\u9664\uff0c\u9632\u6b62\u6e05\u5355\u81ea\u5f15\u7528\u3002')
 if ranked:
  table(['\u6392\u540d','\u8def\u5f84','\u903b\u8f91GB','\u53ef\u56de\u6536GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u7406\u7531/\u8bc1\u636e','\u5f15\u7528\u7ed3\u679c'],[(i,e['path'],gb(e['bytes']),gb(e['reclaimable_bytes']),e['modified'],e['reason']+'\uff1b'+'\uff1b'.join(e['evidence']),str(e['reference_check'])) for i,e in enumerate(ranked,1)])
  table(['\u8def\u5f84','\u6587\u4ef6\u6570','\u72ec\u5360inode','\u5916\u90e8\u786c\u94fe\u63a5inode','\u5916\u90e8\u94fe\u63a5\u5206\u914dGB\uff08\u4e0d\u91ca\u653e\uff09','\u5c5e\u6027\u9519\u8bef\u6570'],[(e['path'],e['allocation']['files'],e['allocation']['exclusive_inodes'],e['allocation']['external_hardlink_inodes'],gb(e['allocation']['external_link_allocated_bytes']),len(e['allocation']['metadata_errors'])) for e in ranked])
 else:add('\u65e0\u901a\u8fc7\u5168\u90e8\u68c0\u67e5\u7684C\u7c7b\u5019\u9009\u3002');add()
 add('C\u7c7b\u5408\u8ba1\uff1a\u903b\u8f91'+gb(d['C_logical_total_bytes'])+'GB\uff0c\u9884\u8ba1\u53ef\u56de\u6536'+gb(c)+'GB\u3002')
 add()
 children=[e for e in entries if e['category']=='C' and not e.get('counted_candidate') and e['bytes']>200_000_000]
 if children:
  add('C\u6839\u76ee\u5f55\u5185\u5927\u4e8e200MB\u9879\uff0c\u5df2\u5305\u542b\u5728\u5408\u8ba1\u4e2d\uff0c\u4e0d\u91cd\u590d\u8ba1\u7b97\uff1a')
  table(['\u8def\u5f84','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u6240\u5c5e\u5019\u9009','\u7406\u7531/\u8bc1\u636e','\u5f15\u7528\u7ed3\u679c'],[(e['path'],gb(e['bytes']),e['modified'],e['candidate_root'],e['reason']+'\uff1b'+'\uff1b'.join(e['evidence']),str(e['reference_check'])) for e in children])
 add('## D \u4e0d\u786e\u5b9a\u6216\u88ab\u5f15\u7528');add()
 add('\u6df7\u5408\u6c47\u603b\u76ee\u5f55\u542bA/B\uff0c\u4e0d\u53ef\u6574\u4f53\u5220\u9664\u3002\u5176\u4ed6\u73af\u5883\u6216\u5b89\u88c5\u7ec4\u4ef6\u672a\u786e\u8ba4\u65e0\u4eba\u4f7f\u7528\u3002\u4e0b\u8868\u5217\u5168\u90e8\u5927\u4e8e200MB\u9879\uff1b\u5c0f\u4e8e200MB\u7684\u88ab\u5f15\u7528\u7f13\u5b58\u53e6\u5217\u6392\u540d\u548c\u7edf\u8ba1\uff0c\u5b8c\u6574\u9010\u6761\u6e05\u5355\uff08\u542b\u6bcf\u6761\u8bc1\u636e\u4e0e\u5168\u90e8\u5f15\u7528\uff09\u5728inventory.json\u3002\u5f15\u7528\u547d\u4e2d\u5373\u6309\u89c4\u5219\u964dD\uff0c\u4e0d\u628a\u6587\u5b57\u63d0\u53ca\u89e3\u91ca\u6210\u65e0\u6548\u5f15\u7528\u3002')
 def d_row(e):
  refs='\uff1b'.join(f"{x['file']}:{x['line']} [{x['matched']}]" for x in e['references'][:3])
  return (e['path'],e['kind'],gb(e['bytes']),e['modified'],e['reason'],'\uff1b'.join(map(str,e['evidence']))+'\uff1b'+str(e['reference_check'])+'\uff1b'+refs)
 table(['\u8def\u5f84','\u7c7b\u578b','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u7406\u7531','\u8bc1\u636e/\u5f15\u7528'],[d_row(e) for e in uncertain if e['bytes']>200_000_000])
 small=[e for e in uncertain if e['bytes']<=200_000_000 and e.get('initial_category')=='C']
 add('\u5c0f\u4e8e200MB\u7684\u521d\u9009\u7f13\u5b58\u964dD\uff1a'+str(len(small))+'\u9879\uff0c\u903b\u8f91\u5927\u5c0f'+gb(sum(e['bytes'] for e in small))+'GB\uff08\u8fd9\u4e9b\u7f13\u5b58\u6839\u76ee\u5f55\u4e92\u4e0d\u91cd\u53e0\uff09\u3002\u524d10\u9879\uff1a');add()
 table(['\u8def\u5f84','\u7c7b\u578b','GB','\u6700\u540e\u4fee\u6539\u65f6\u95f4','\u7406\u7531','\u8bc1\u636e/\u5f15\u7528'],[d_row(e) for e in small[:10]])
 add('\u6700\u5927D\u9879\uff08\u5254\u9664\u542bA/B\u7684\u6df7\u5408\u6c47\u603b\u7236\u76ee\u5f55\uff1b\u4e92\u4e0d\u91cd\u53e0\uff0c\u4ec5\u7528\u4e8e\u770b\u5927\u5c0f\uff09\uff1a')
 table(['\u8def\u5f84','GB','\u7406\u7531'],[(e['path'],gb(e['bytes']),e['reason']) for e in roots[:10]])
 add('## \u5931\u8d25\u5c1d\u8bd5\u4e0e\u91cd\u590d\u6838\u67e5');add()
 add('failure\u53ca\u539f\u59cb\u62a5\u544a\u672c\u8eab\u4e3aB\u8bb0\u5f55\uff1b\u5b58\u5728\u5931\u8d25\u6807\u8bb0\u6216\u540c\u540dr2\u4e0d\u8db3\u4ee5\u628a\u6574\u76ee\u5f55\u5224\u4e3a\u53ef\u5220\u3002\u672c\u6b21\u672a\u5c06\u4efb\u4f55\u5931\u8d25\u5c1d\u8bd5\u6574\u76ee\u5f55\u5217C\u3002')
 table(['\u65e7\u5c1d\u8bd5','GB','A\u4fdd\u62a4','\u53ef\u80fd\u66ff\u4ee3\u76ee\u5f55','\u5df2\u8bfb\u66ff\u4ee3\u8bc1\u636e','\u5904\u7406'],[(x['attempt'],gb(x['attempt_bytes']) if x['attempt_bytes'] is not None else '\u672a\u53d6\u5f97',x['protected_A'],x['possible_replacement'] or '\u672a\u6838\u5b9e','\uff1b'.join(map(str,x['replacement_evidence'])),x['classification']) for x in d['failed_attempt_evidence']])
 add('\u91cd\u590d\u68c0\u67e5\u53ea\u7b5b\u9009\u5927\u4e8e200MB\u3001\u975eA/B\u7684\u53ef\u7591\u6570\u636e/\u538b\u7f29\u526f\u672c\uff08\u540c\u6587\u4ef6\u540d\u3001\u540c\u957f\u5ea6\uff09\uff0c\u518d\u8ba1\u7b97SHA256\uff1b\u4e0d\u6563\u5217\u4e2a\u4eba\u6587\u4ef6\uff0c\u4e0d\u5c06\u5df2\u6709checkpoint\u964d\u7ea7\u3002\u5b9e\u9645\u7b5b\u9009\u7ec4\u6570\uff1a'+str(len(d['duplicate_checks']))+'\u3002')
 if d['duplicate_checks']:add(json.dumps(d['duplicate_checks'],ensure_ascii=False,indent=2))
 add()
 add('## \u65b9\u6cd5\u3001\u7f3a\u53e3\u4e0e\u68c0\u67e5\u8bb0\u5f55');add()
 add('\u9608\u503c\uff1a200MB=200,000,000\u5b57\u8282\uff0c500MB=500,000,000\u5b57\u8282\uff0c1GB=1,000,000,000\u5b57\u8282\u3002\u626b\u63cf\u53ea\u8bfb\u53d6\u5143\u6570\u636e\uff1b\u4ec5\u5728\u4e24\u4ed3\u5e93\u8bfb\u53d6\u5931\u8d25/\u66ff\u4ee3\u8bc1\u636e\u548coutputs\u4f9d\u8d56\u6587\u672c\u3002')
 add('\u5f15\u7528\u641c\u7d22\u56fa\u5b9a\u5b57\u7b26\u4e32\u3001\u5ffd\u7565\u5927\u5c0f\u5199\uff0c\u5305\u542b\u9690\u85cf/\u5ffd\u7565\u7684.py/.json/.md\u3002\u641c\u7d22\u76ee\u5f55\u540d\u5df2\u8986\u76d6\u5b8c\u6574\u8def\u5f84\u5339\u914d\uff0c\u56e0\u4e3a\u4efb\u4f55\u5b8c\u6574\u8def\u5f84\u90fd\u542b\u76f8\u5e94\u76ee\u5f55\u540d\u3002\u672c\u6b21\u4f7f\u7528'+str(d['reference_search']['patterns_count'])+'\u4e2a\u4e0d\u540c\u76ee\u5f55\u540d\uff1b\u9000\u51fa\u7801'+str(d['reference_search']['exit_code'])+'\uff0c\u5b8c\u6574='+str(d['reference_search']['complete'])+'\u3002')
 add(d['reference_search']['first_search_error'])
 add('\u5b8c\u6574\u547d\u4e2d\u4fdd\u5b58\u4e8e'+d['reference_search']['stdout']+'\uff0c\u9519\u8bef\u8f93\u51fa\u4fdd\u5b58\u4e8e'+d['reference_search']['stderr']+'\uff1binventory.json\u9010\u9879\u8bb0\u5f55\u5f15\u7528\u6587\u4ef6\u4e0e\u884c\u53f7\u3002\u53ea\u6709\u521d\u9009C\u6309\u89c4\u5219\u641c\u7d22\uff1b\u5176\u4ed6D\u56e0\u7528\u9014/\u4fdd\u7559\u8bc1\u636e\u4e0d\u8db3\u76f4\u63a5\u4fdd\u7559\u3002')
 add('\u4e2a\u4eba\u76ee\u5f55\u4e0d\u6253\u5f00\u5185\u5bb9\uff0c\u4e0d\u5c55\u5f00\u5185\u5c42\u3002\u5185\u90e8\u91cd\u89e3\u6790\u70b9\u4e0d\u8ddf\u968f\uff0c\u6743\u9650\u5931\u8d25\u5904\u4e0d\u8865\u731c\u3002\u903b\u8f91\u5927\u5c0f\u4e0e\u78c1\u76d8\u5360\u7528\u4e0d\u76f8\u7b49\uff0c\u786c\u94fe\u63a5\u3001\u538b\u7f29/\u7a00\u758f\u6587\u4ef6\u3001\u5377\u5143\u6570\u636e\u3001\u6743\u9650\u7f3a\u53e3\u53ca\u5e76\u53d1\u6d3b\u52a8\u5747\u53ef\u9020\u6210\u5dee\u5f02\u3002')
 table(['\u9876\u5c42\u76ee\u5f55','\u95ee\u9898','\u6570\u91cf'],[(x['top'],x['type'],x['count']) for x in d['metadata_scan']['issue_summary']])
 add('## \u5f00\u59cb/\u7ed3\u675fGit\u590d\u6838\u4e0e\u4ea4\u4ed8');add()
 table(['\u4ed3\u5e93','\u5f00\u59cb\u5206\u652f/HEAD','\u7ed3\u675f\u5206\u652f/HEAD','status'],[(p,d['metadata_scan']['git_before'][p]['branch']+' / '+d['metadata_scan']['git_before'][p]['head'],v['branch']+' / '+v['head'],'\u7a7a\uff1b\u5f00\u59cb\u7ed3\u675f\u5b8c\u5168\u4e00\u81f4') for p,v in after.items()])
 add('\u672a\u8fd0\u884ccommit\u3001checkout\u3001reset\u3001stash\u3001push\u3001conda clean\u6216pip cache purge\u3002\u4ec5\u521b\u5efa\u672c\u8f6e\u6e05\u70b9\u76ee\u5f55\u548c\u62a5\u544a/\u8bc1\u636e\u811a\u672c\uff1ainventory.json\u3001inventory.md\u3001metadata_scan.json\u3001classified_inventory.json\u3001reference_*\u4e0einventory_scan.py\u3001classify_inventory.py\u3001write_inventory.py\u3001creation_guard.json\u3002')
 add('\u672c\u8f6e\u7ed3\u675f\u72b6\u6001\uff1a\u6e05\u70b9\u5b8c\u6210\uff0c\u65e7\u6587\u4ef6\u4fdd\u6301\u539f\u6837\uff0c\u4e24\u4e2aGit\u5de5\u4f5c\u6811\u5e72\u51c0\u3002\u672a\u6536\u5230\u540e\u7eedattempt2\u6307\u4ee4\uff0c\u672a\u542f\u52a8\u4efb\u4f55\u65b0\u8fd0\u884c\u3002')
 (INV/'inventory.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 result=dict(current_free_bytes=u.free,C_logical_bytes=d['C_logical_total_bytes'],C_reclaimable_bytes=c,after_C_free_bytes=u.free+c,category_counts=dict(collections.Counter(e['category'] for e in entries)),C_top10=[dict(path=e['path'],bytes=e['bytes'],reclaimable_bytes=e['reclaimable_bytes']) for e in ranked[:10]],D_top10=[dict(path=e['path'],bytes=e['bytes'],reason=e['reason']) for e in roots[:10]],git_after=after,report=str(INV/'inventory.md'))
 dump('completion_check.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
