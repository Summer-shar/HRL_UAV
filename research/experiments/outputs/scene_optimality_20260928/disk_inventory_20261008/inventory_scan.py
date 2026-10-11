
# Read-only metadata inventory; no project/environment import or mutation.
import os,json,stat,shutil,time,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
INV=Path(__file__).resolve().parent
ROOTS=['D:/HRL_UAV_V2','D:/HRL_UAV_V2_speedup','D:/Anaconda3']
TZ=timezone(timedelta(hours=8))
def now():return datetime.now(TZ).isoformat()
def normalize(path):return os.path.abspath(path).replace('\\','/').rstrip('/').lower()
def timestamp(value):
 try:return datetime.fromtimestamp(value,TZ).isoformat()
 except Exception:return None
def dump(name,value):
 p=(INV/name).resolve()
 if not p.is_relative_to(INV.resolve()):raise RuntimeError('Output outside INV')
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def git_states():
 expected=[('D:/HRL_UAV_V2','codex/v2-lite-paper-caliber','a9953dc36d9d65f99bcc35044cdfc250ca11bb49'),('D:/HRL_UAV_V2_speedup','codex/v2-lite-speedup','25d59ff9f7c3e27c060c63d462235a30ce28930a')]
 result={}
 for root,branch,head in expected:
  actual={k:subprocess.check_output(['git','--no-optional-locks','-C',root,*cmd],text=True,encoding='utf-8').strip() for k,cmd in [('status',['status','--porcelain']),('branch',['branch','--show-current']),('head',['rev-parse','HEAD'])]}
  if actual!=dict(status='',branch=branch,head=head):raise RuntimeError('Unexpected git state: '+json.dumps(actual))
  result[root]=actual
 return result
def scan():
 started=time.perf_counter();usage=shutil.disk_usage('D:/');before=git_states();detail=[];top=[];expanded=[];issues=[]
 def walk(path,topname,detailed,depth):
  size=0;count=0;latest=0.;incomplete=False
  try:
   ds=os.stat(path,follow_symlinks=False);own_mtime=timestamp(ds.st_mtime)
  except OSError as exc:
   issues.append(dict(top=topname,type=type(exc).__name__,path=str(path) if detailed else None));return 0,0,0.,True
  try:
   with os.scandir(path) as entries:
    for entry in entries:
     if normalize(entry.path)==normalize(INV):continue
     try:
      s=entry.stat(follow_symlinks=False);attrs=getattr(s,'st_file_attributes',0)
      if stat.S_ISLNK(s.st_mode) or attrs&0x400:
       issues.append(dict(top=topname,type='reparse_point_not_followed',path=entry.path if detailed else None));incomplete=True;continue
      if stat.S_ISDIR(s.st_mode):
       sub,nn,ll,bad=walk(entry.path,topname,detailed,depth+1)
       size+=sub;count+=nn;latest=max(latest,ll,s.st_mtime);incomplete|=bad
      elif stat.S_ISREG(s.st_mode):
       size+=s.st_size;count+=1;latest=max(latest,s.st_mtime)
       if detailed:
        record=dict(path=str(Path(entry.path)),kind='file',bytes=s.st_size,modified=timestamp(s.st_mtime),latest_descendant_modified=timestamp(s.st_mtime),inode=s.st_ino,device=s.st_dev,nlink=getattr(s,'st_nlink',1),attributes=attrs,depth_from_D=depth+1,incomplete=False)
        detail.append(record)
        if depth+1<=3 and s.st_size>500_000_000:expanded.append(record.copy())
     except OSError as exc:
      issues.append(dict(top=topname,type=type(exc).__name__,path=entry.path if detailed else None));incomplete=True
  except OSError as exc:
   issues.append(dict(top=topname,type=type(exc).__name__,path=str(path) if detailed else None));incomplete=True
  if detailed:
   record=dict(path=str(Path(path)),kind='directory',bytes=size,modified=own_mtime,latest_descendant_modified=timestamp(latest or ds.st_mtime),files=count,depth_from_D=depth,incomplete=incomplete)
   detail.append(record)
   if depth<=3 and depth>1 and size>500_000_000:expanded.append(record.copy())
  return size,count,latest,incomplete
 with os.scandir('D:/') as entries:
  root_entries=list(entries)
 for entry in root_entries:
  try:
   ss=entry.stat(follow_symlinks=False)
   if stat.S_ISDIR(ss.st_mode):
    detailed=normalize(entry.path) in {normalize(p) for p in ROOTS}
    size,count,latest,incomplete=walk(entry.path,entry.path,detailed,1)
    top.append(dict(path=str(Path(entry.path)),kind='directory',bytes=size,files=count,modified=timestamp(ss.st_mtime),latest_descendant_modified=timestamp(latest or ss.st_mtime),incomplete=incomplete,details_allowed=detailed))
    print('SCANNED_TOP',entry.name,size,count,incomplete,flush=True)
   elif stat.S_ISREG(ss.st_mode):
    top.append(dict(path=str(Path(entry.path)),kind='file',bytes=ss.st_size,files=1,modified=timestamp(ss.st_mtime),incomplete=False,details_allowed=False))
  except OSError as exc:issues.append(dict(top=entry.path,type=type(exc).__name__,path=None))
 top.sort(key=lambda r:r['bytes'],reverse=True)
 large_roots={normalize(r['path']) for r in top if r['kind']=='directory' and r['bytes']>1_000_000_000 and r['details_allowed']}
 expanded=[r for r in expanded if any(normalize(r['path']).startswith(k+'/') for k in large_roots)]
 issue_summary={}
 for r in issues:
  key=(r['top'],r['type']);issue_summary[key]=issue_summary.get(key,0)+1
 out=dict(started_at=now(),scan_wall_seconds=time.perf_counter()-started,total_bytes=usage.total,free_before_bytes=usage.free,used_before_bytes=usage.used,git_before=before,top_level=top,expanded_depth3_over500MB=expanded,detailed_roots=ROOTS,metadata=detail,issues=issues,issue_summary=[dict(top=k[0],type=k[1],count=v) for k,v in issue_summary.items()],methods=dict(size='logical file sizes summed recursively; may count hardlinks multiple times',private_data='metadata only; non-project/non-Anaconda content never opened; only top-level sizes published',reparse='not followed; incomplete totals flagged',INV='excluded from scan and from dependency search to prevent inventory self-reference'))
 dump('metadata_scan.json',out);print('METADATA_SCAN_COMPLETE',len(detail),time.perf_counter()-started,flush=True)
 return out
if __name__=='__main__':scan()

