import sys
sys.dont_write_bytecode=True
import ast,difflib
from common import *
# The statistical function is verbatim, except its output-file side effect.
s=(T4101/'evaluate_test.py').read_text(encoding='utf-8');a=s.index('def compare(summary):');b=s.index('\ndef report(',a)
fn=s[a:b].replace(";atomic_json(OUT/'test_comparisons.json',comparisons);return comparisons",";return comparisons").replace("'Abl_TwoStage','Abl_Alternating','Abl_HoverSAC'","'Abl_TwoStage','Abl_HoverSAC'")
p=OUTM/'aggregate.py';p.write_text(p.read_text(encoding='utf-8-sig')+'\n'+fn+'\n'+(OUTM/'aggregate_tail.txt').read_text(encoding='utf-8-sig'),encoding='utf-8')
(OUTM/'aggregate_tail.txt').unlink()
# Correct Markdown newline escaping during construction, before any execution.
p=OUTM/'evaluate_test_seed.py';s=p.read_text(encoding='utf-8');s=s.replace("'\\\\n'.join(lines)+'\\\\n'","'\\n'.join(lines)+'\\n'");p.write_text(s,encoding='utf-8')
changes=read(OUTM/'driver_diffs/hunks.json')
for name,src,cat in [('stage1_wrapper.py',WT/'src/hrl_v2/run.py',{'D1':'替代CLI调用train并在300 evaluated存在时只阻断301；包含指定后检','seed/path':'SEED、WORK、seed目录；导入路径断言'}),('launch_baselines_seed.py',B4101/'launch_baselines.py',{'D7':'保留三基线并行及失败等待全部退出；删除面板A/B；保留typed anchor检查','seed/path':'SEED、WORK、日志及初始上层哈希对照','D9':'CLI入口加导入断言；队列父进程不初始化CUDA'}),('evaluate_test_seed.py',T4101/'evaluate_test.py',changes['evaluate_test_seed.py']['categories'])]:
 old=src.read_text(encoding='utf-8-sig');new=(OUTM/name).read_text(encoding='utf-8-sig');diff=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(src),tofile=name));(OUTM/f'driver_diffs/{name}.diff').write_text(''.join(diff),encoding='utf-8')
 hunks=[];cur=None
 for line in diff:
  if line.startswith('@@'):cur=dict(hunk=line.strip(),allowed_categories=cat,changes=[]);hunks.append(cur)
  elif cur is not None and line[:1] in ('+','-'):cur['changes'].append(line.rstrip())
 changes[name]=dict(source=str(src),categories=cat,hunks=hunks)
write(OUTM/'driver_diffs/hunks.json',changes)
write(OUTM/'startup.json',dict(checked_at=now(),directories_absent_before_creation=True,initial_git=git_states(),protocol_sha256=protocol_check(),disk_free_bytes=206982926336,work=str(WORK),out=str(OUTM)))
text=(Path('C:/Users/PC/.codex/attachments/1642bd42-b82b-448b-bd2c-338281608b6c/已粘贴的文本.txt')).read_text(encoding='utf-8-sig')
lines=[line[2:] for line in text.splitlines() if line.startswith('- D') and len(line)>4 and line[3].isdigit()]
write(OUTM/'deviations.json',lines)
for p in OUTM.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
print('All scripts compile; deviations:',len(lines))
