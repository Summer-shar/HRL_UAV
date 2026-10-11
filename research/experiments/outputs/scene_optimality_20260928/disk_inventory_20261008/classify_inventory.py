# Inventory classification and reference search; reads existing files, writes INV only.
import os,json,re,hashlib,subprocess,shutil,time,ctypes,collections
from pathlib import Path
from datetime import datetime,timezone,timedelta
INV=Path(__file__).resolve().parent
MAIN=Path('D:/HRL_UAV_V2');SPEED=Path('D:/HRL_UAV_V2_speedup')
TZ=timezone(timedelta(hours=8))
def now():return datetime.now(TZ).isoformat()
def norm(p):return str(p).replace('\\','/').rstrip('/').lower()
def below(p,q):return norm(p)==norm(q) or norm(p).startswith(norm(q)+'/')
def dump(n,x):(INV/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read_json(p):
 try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
 except Exception as e:return {'read_error':repr(e)}
A_ROOTS=[
('D:/HRL_UAV_V2_speedup/outputs/lite_v2_4_0/multiseed_20261002_203333_r2','WORK'),
('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_multiseed_20261002_203333_r2','OUTM'),
('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_multiseed_diag_20261004','NEW'),
('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_randpre_pilot_20261005','PILOT'),
('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_curriculum_pilot_20261008','CPIL'),
('D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_4101_test','\u51bb\u7ed3\u534f\u8bae\u4e0etest\u9762\u677f'),
('D:/Anaconda3/envs/pytorch-gpu','\u6307\u5b9aPython/GPU\u73af\u5883')]
for root in (MAIN,SPEED):
 for sub in ('src','configs','.venv'):A_ROOTS.append((str(root/sub),'\u4ed3\u5e93\u6e90\u7801/\u914d\u7f6e/\u73af\u5883'))
def a_reason(p):
 for root,label in A_ROOTS:
  if below(p,root):return '\u7528\u6237\u660e\u786e\u6307\u5b9a\u5fc5\u7559\uff1a'+label,root
 return None
def b_reason(r):
 p=r['path'];s=norm(p)
 if not (below(p,MAIN) or below(p,SPEED)):return None
 parts=s.split('/');name=parts[-1]
 if any(x in ('checkpoints','snapshots','gates','evaluations','baselines','baselines_task2','baseline_reference','references') for x in parts):
  return 'checkpoint/\u7f51\u7edc\u5feb\u7167/\u95e8\u69db/\u8bc4\u4f30/\u51bb\u7ed3\u53c2\u7167\u8bb0\u5f55\uff0c\u6309\u89c4\u5219\u4fdd\u7559'
 if any('backup_before_writing' in x for x in parts):return '\u8bba\u6587\u5907\u4efd\uff0c\u6309\u89c4\u5219\u4fdd\u7559'
 if r['kind']=='file':
  if Path(p).suffix.lower() in ('.pt','.pth','.ckpt','.safetensors'):return '\u5df2\u6709\u6a21\u578b/checkpoint/\u7f51\u7edc\u5feb\u7167\uff0c\u6309\u89c4\u5219\u4fdd\u7559'
  if '/outputs/' in s or '/scratch/' in s:
   if Path(p).suffix.lower() in ('.json','.jsonl','.md','.txt','.csv','.tsv','.log','.tex','.bib','.pdf'):
    return '\u5408\u540c/\u534f\u8bae/\u62a5\u544a/summary/\u8fd0\u884c\u6216\u6307\u6807\u8bb0\u5f55\uff1b\u4fdd\u5b88\u4fdd\u7559\u539f\u59cb\u5b9e\u9a8c\u8bb0\u5f55'
 return None
def cache_kind(p):
 s=norm(p);parts=s.split('/')
 if below(p,'D:/Anaconda3/pkgs'):return 'conda pkgs \u5305\u7f13\u5b58\uff1b\u73af\u5883\u672c\u4f53\u4e0d\u5728\u6b64\u76ee\u5f55'
 if '__pycache__' in parts:return 'Python \u5b57\u8282\u7801\u7f13\u5b58'
 if '.pytest_cache' in parts:return 'pytest \u7f13\u5b58'
 if any(x in ('pip_cache','pip-cache') for x in parts) or '/.cache/pip' in s:return 'pip \u7f13\u5b58'
 if parts[-1] in ('tmp','temp'):return '\u4e34\u65f6\u76ee\u5f55\uff1b\u4ecd\u987b\u6392\u9664\u4fdd\u7559\u8bb0\u5f55\u548c\u4f9d\u8d56'
 return None
def main():
 start=time.perf_counter()
 data=read_json(INV/'metadata_scan.json');rows=data['metadata']
 bypath={norm(r['path']):r for r in rows if r['kind']=='directory' or r['bytes']>200_000_000}
 def ancestors(path):
  p=norm(path)
  while '/' in p:
   p=p.rsplit('/',1)[0]
   if p in bypath:yield p
   if p=='d:':break
 protected=collections.defaultdict(list)
 for root,label in A_ROOTS:
  if norm(root) in bypath:
   for par in ancestors(root):
    if len(protected[par])<5:protected[par].append(root)
 for r in rows:
  if not (below(r['path'],MAIN) or below(r['path'],SPEED)):continue
  if a_reason(r['path']):continue
  if b_reason(r):
   for par in ancestors(r['path']):
    if len(protected[par])<5:protected[par].append(r['path'])
 entries=[];cache_roots=[]
 for r in rows:
  if r['kind']=='directory' and cache_kind(r['path']) and not a_reason(r['path']) and not b_reason(r) and not protected[norm(r['path'])] and not r['incomplete'] and r['bytes']>0:
   cache_roots.append(r)
 cache_roots.sort(key=lambda r:(len(norm(r['path'])),norm(r['path'])))
 candidates=[]
 for r in cache_roots:
  if not any(below(r['path'],x['path']) for x in candidates):candidates.append(r)
 candidate_root_ids={norm(x['path']) for x in candidates}
 candidate_for=lambda p:next((x for x in candidates if below(p,x['path'])),None)
 for r in rows:
  if r['bytes']<=200_000_000 and norm(r['path']) not in candidate_root_ids:continue
  ent={k:r.get(k) for k in ('path','kind','bytes','modified','latest_descendant_modified','files','incomplete')}
  aa=a_reason(r['path']);bb=b_reason(r);cc=candidate_for(r['path'])
  ent['evidence']=[];ent['references']=[];ent['reference_check']='\u975eC\u521d\u9009\u9879\uff0c\u4e0d\u8981\u6c42\u8def\u5f84\u5f15\u7528\u641c\u7d22'
  if aa:ent.update(category='A',reason=aa[0]);ent['evidence'].append(aa[1])
  elif bb:ent.update(category='B',reason=bb);ent['evidence'].append('\u5143\u6570\u636e\u4e0e\u5b9e\u9645\u6587\u4ef6/\u76ee\u5f55\u540d')
  elif cc:
   ent.update(category='C',reason=cache_kind(r['path']),candidate_root=cc['path'],counted_candidate=norm(cc['path'])==norm(r['path']))
   ent['evidence'].append('\u5b9e\u9645\u76ee\u5f55\u5b58\u5728\uff1b\u53ea\u5305\u542b\u53ef\u518d\u751f\u6210\u7f13\u5b58/\u975e\u53d7\u4fdd\u62a4\u4e34\u65f6\u5185\u5bb9\uff1b\u672a\u8fdb\u5165A/B\u8303\u56f4')
  else:
   reason='\u7528\u9014/\u53ef\u91cd\u5efa\u6027\u6216\u66ff\u4ee3\u5173\u7cfb\u672a\u5145\u5206\u6838\u5b9e\uff0c\u4e0d\u5217\u53ef\u5220'
   if protected[norm(r['path'])]:
    reason='\u6df7\u5408\u6c47\u603b\u76ee\u5f55\uff0c\u542bA/B\u4fdd\u7559\u9879\uff1b\u4e0d\u80fd\u6574\u4f53\u5220\u9664'
    ent['evidence'].extend(protected[norm(r['path'])])
   elif below(r['path'],'D:/Anaconda3'):
    reason='\u5176\u4ed6\u73af\u5883\u6216Anaconda\u5b89\u88c5\u7ec4\u4ef6\uff1b\u672a\u786e\u8ba4\u65e0\u4eba\u4f7f\u7528\uff0c\u4e0d\u80fd\u5224\u4e3a\u7f13\u5b58'
   if r['incomplete']:reason+='\uff1b\u5143\u6570\u636e\u626b\u63cf\u4e0d\u5b8c\u6574'
   ent.update(category='D',reason=reason)
  ent['aggregate_not_additive']=r['kind']=='directory'
  entries.append(ent)
 # Inspect failure records only inside repositories; existence alone never permits deletion.
 failures=[]
 for r in rows:
  p=Path(r['path'])
  if r['kind']!='file' or p.name.lower() not in ('failure.json','technical_failure.json'):continue
  if not (below(p,MAIN) or below(p,SPEED)):continue
  f=read_json(p);attempt=p.parent;replacement=None;replacement_evidence=[]
  sibling=attempt.with_name(attempt.name+'_r2')
  if sibling.exists():
   evidence=[]
   for name in ('progress.json','report.json','summary.json','summary.md','report.md','contract.json','launch_contract.json'):
    fp=sibling/name
    if fp.is_file() and fp.stat().st_size<8_000_000:
     text=fp.read_text(encoding='utf-8-sig',errors='replace')
     if attempt.name in text:evidence.append({'path':str(fp),'explicit_previous_attempt_mention':True})
     if name in ('progress.json','report.json','summary.json'):
      j=read_json(fp);status=j.get('status') if isinstance(j,dict) else None
      if status:evidence.append({'path':str(fp),'status':status})
   replacement=str(sibling);replacement_evidence=evidence
  failures.append(dict(path=str(p),attempt=str(attempt),attempt_bytes=bypath.get(norm(attempt),{}).get('bytes'),failure_record=f,protected_A=bool(a_reason(attempt)),possible_replacement=replacement,replacement_evidence=replacement_evidence,classification='\u4fdd\u7559failure/\u5408\u540c/\u62a5\u544a\uff1b\u672a\u5c06\u6574\u6b21\u5c1d\u8bd5\u5217\u4e3aC',reason='\u5931\u8d25\u6807\u8bb0\u4e0d\u80fd\u53d6\u4ee3B\u7c7b\u4fdd\u7559\u89c4\u5219\uff1b\u66ff\u4ee3\u5173\u7cfb\u548c\u5b8c\u6574\u7ed3\u679c\u53ea\u6309\u8bc1\u636e\u8bb0\u5f55'))
 # No eligible duplicate payload is assumed from filename or size.
 dup_groups=collections.defaultdict(list)
 for ent in entries:
  if ent['kind']=='file' and ent['category']=='D' and Path(ent['path']).suffix.lower() in ('.npz','.npy','.zip','.7z','.tar','.gz','.bak'):
   dup_groups[(ent['bytes'],Path(ent['path']).name.lower())].append(ent['path'])
 duplicates=[]
 for paths in dup_groups.values():
  if len(paths)<2:continue
  hs={}
  for p in paths:
   h=hashlib.sha256()
   with open(p,'rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
   hs[p]=h.hexdigest()
  duplicates.append(dict(files=hs,policy='\u4ec5\u76f8\u540cSHA256\u624d\u662f\u91cd\u590d\uff1b\u5c1a\u987b\u786e\u8ba4\u53ef\u5220\u8bed\u4e49\u4e0e\u5f15\u7528\u3002\u6ca1\u6709\u81ea\u52a8\u6539\u53d8\u4efb\u4f55\u6587\u4ef6\u3002'))
 # Reference search every pre-existing .py/.json/.md in BOTH outputs, including ignored files.
 rg=shutil.which('rg')
 if not rg:raise RuntimeError('rg unavailable; cannot complete candidate dependency check')
 pattern_map=collections.defaultdict(set)
 for c in candidates:
  p=c['path'];variants={str(Path(p).name)}
  for v in variants:pattern_map[v.casefold()].add(norm(p))
 patterns=sorted(pattern_map)
 (INV/'reference_patterns_basename.txt').write_text('\n'.join(patterns)+'\n',encoding='utf-8')
 cmd=[rg,'--no-ignore','--hidden','--text','-i','-F','-f',str(INV/'reference_patterns_basename.txt'),'--glob','*.py','--glob','*.json','--glob','*.md','--glob','!**/disk_inventory_20261008/**','--no-heading','--line-number','--only-matching','--color','never',str(MAIN/'outputs'),str(SPEED/'outputs')]
 with (INV/'reference_matches_basename.txt').open('w',encoding='utf-8') as out,(INV/'reference_stderr_basename.txt').open('w',encoding='utf-8') as err:res=subprocess.run(cmd,stdout=out,stderr=err,encoding='utf-8')
 refs=collections.defaultdict(list);counts=collections.Counter();search_ok=res.returncode in (0,1)
 with (INV/'reference_matches_basename.txt').open(encoding='utf-8') as f:
  for line in f:
   m=re.match(r'^(.*?):(\d+):(.*)$',line.rstrip('\r\n'))
   if not m:continue
   file,ln,matched=m.groups()
   for target in pattern_map.get(matched.casefold(),()):
    counts[target]+=1;refs[target].append(dict(file=file,line=int(ln),matched=matched))
 for ent in entries:
  if ent['category']!='C':continue
  target=norm(ent['candidate_root'])
  ent['reference_check']=dict(complete=search_ok,match_count=counts[target],method='\u4e24\u4e2aoutputs\u5185\u6240\u6709.py/.json/.md\uff0c\u8def\u5f84\u6216\u76ee\u5f55\u540d\uff0c\u56fa\u5b9a\u5b57\u7b26\u4e32\u5ffd\u7565\u5927\u5c0f\u5199\uff1b\u6392\u9664\u672c\u6b21INV\u81ea\u5f15\u7528')
  ent['references']=refs[target]
  if counts[target] or not search_ok:
   ent['initial_category']='C';ent['category']='D';ent['reason']+='\uff1b'+('\u88ab\u5df2\u6709\u8f93\u51fa\u5f15\u7528\uff0c\u6309\u89c4\u5219\u964d\u4e3aD' if counts[target] else '\u5f15\u7528\u641c\u7d22\u4e0d\u5b8c\u6574\uff0c\u964d\u4e3aD')
 selected=[e for e in entries if e['category']=='C' and e.get('counted_candidate')]
 # Measure NTFS allocation using attribute-only handles; no contents opened.
 from ctypes import wintypes
 kernel=ctypes.WinDLL('kernel32',use_last_error=True)
 kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];kernel.CreateFileW.restype=wintypes.HANDLE
 kernel.GetFileInformationByHandleEx.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];kernel.GetFileInformationByHandleEx.restype=wintypes.BOOL
 kernel.CloseHandle.argtypes=[wintypes.HANDLE]
 class FILE_STANDARD_INFO(ctypes.Structure):
  _fields_=[('AllocationSize',ctypes.c_longlong),('EndOfFile',ctypes.c_longlong),('NumberOfLinks',wintypes.DWORD),('DeletePending',wintypes.BOOLEAN),('Directory',wintypes.BOOLEAN)]
 def allocation(p):
  path='\\\\?\\'+str(Path(p).resolve())
  h=kernel.CreateFileW(path,0x80,7,None,3,0,None)
  if h==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_last_error(),'CreateFileW attribute-only')
  try:
   info=FILE_STANDARD_INFO()
   if not kernel.GetFileInformationByHandleEx(h,1,ctypes.byref(info),ctypes.sizeof(info)):raise OSError(ctypes.get_last_error(),'FileStandardInfo')
   return info.AllocationSize,info.NumberOfLinks
  finally:kernel.CloseHandle(h)
 selected_map={norm(x['path']):x for x in selected};links=collections.defaultdict(list)
 def selected_root(p):
  for q in selected_map:
   if below(p,q):return q
  return None
 for r in rows:
  if r['kind']!='file':continue
  q=selected_root(r['path'])
  if q:links[(r['device'],r['inode'])].append((q,r))
 allocations={q:dict(reclaimable_allocated_bytes=0,logical_bytes=selected_map[q]['bytes'],files=0,exclusive_inodes=0,external_hardlink_inodes=0,external_link_allocated_bytes=0,metadata_errors=[]) for q in selected_map}
 for group in links.values():
  q,r=group[0];alloc=allocations[q]
  for qq,rr in group:allocations[qq]['files']+=1
  try:size,nlink=allocation(r['path'])
  except OSError as e:
   alloc['metadata_errors'].append(dict(path=r['path'],error=repr(e)));continue
  if r['inode'] and nlink<=len(group):
   alloc['reclaimable_allocated_bytes']+=size;alloc['exclusive_inodes']+=1
  else:
   alloc['external_hardlink_inodes']+=1;alloc['external_link_allocated_bytes']+=size
 for ent in selected:
  ent['allocation']=allocations[norm(ent['path'])]
  ent['reclaimable_bytes']=ent['allocation']['reclaimable_allocated_bytes']
 selected.sort(key=lambda e:e['reclaimable_bytes'],reverse=True)
 total=sum(e['reclaimable_bytes'] for e in selected);logical=sum(e['bytes'] for e in selected)
 del rows;data.pop('metadata',None)
 output=dict(created_at=now(),metadata_scan=data,threshold_bytes=43_643_823_444,threshold_display_GB=43.64,A_required_roots=[dict(path=p,label=l,exists=Path(p).exists()) for p,l in A_ROOTS],entries=sorted(entries,key=lambda e:(e['category'],-e['bytes'],e['path'])),C_ranked_nonoverlapping=selected,C_logical_total_bytes=logical,C_reclaimable_allocated_total_bytes=total,failed_attempt_evidence=failures,duplicate_checks=duplicates,reference_search=dict(command=cmd,exit_code=res.returncode,complete=search_ok,patterns_count=len(patterns),stdout='reference_matches_basename.txt',stderr='reference_stderr_basename.txt',excluded_inventory_self_reference=str(INV)),allocation_method='NTFS FileStandardInfo AllocationSize; FILE_READ_ATTRIBUTES only; selected hardlinks counted once; external kept links give zero; unreadable metadata gives zero. Estimates exclude directory metadata and concurrent disk changes.',classification_wall_seconds=time.perf_counter()-start)
 dump('classified_inventory.json',output)
 print('CLASSIFICATION_COMPLETE',len(entries),'C',len(selected),'C_LOGICAL',logical,'C_RECLAIMABLE',total,'MATCHES',sum(counts.values()),flush=True)
 print('C_TOP10',json.dumps([dict(path=x['path'],bytes=x['bytes'],reclaimable_bytes=x['reclaimable_bytes']) for x in selected[:10]],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
