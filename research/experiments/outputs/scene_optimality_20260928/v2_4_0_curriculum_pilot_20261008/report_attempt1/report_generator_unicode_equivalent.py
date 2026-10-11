
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
 '# 4104 课程式预训练试点：第 0 步停止报告','',
 f'停止记录时间：{e["time"]}。','',
 '磁盘硬条件未通过。本次未启动训练或任何评估，未创建合同或 checkpoint。','',
 '## 磁盘检查','',
 '附件第 4 节第 5 条要求 D: 剩余空间严格大于估算用量的 2 倍。以下 GB 按 10^9 字节换算；输出用量是基于旧文件实测大小的估算。','',
 '|项目|字节|GB|','|---|---:|---:|',
 f'|D: 实测剩余|{d["free_bytes"]}|{d["free_bytes"]/1e9:.6f}|',
 f'|本次输出估算|{d["total_estimated_bytes"]}|{d["total_estimated_bytes"]/1e9:.6f}|',
 f'|须大于的两倍门槛|{d["required_free_bytes_strictly_greater_than"]}|{d["required_free_bytes_strictly_greater_than"]/1e9:.6f}|',
 f'|与门槛的差额|{d["shortfall_bytes"]}|{d["shortfall_bytes"]/1e9:.6f}|','',
 f'仅计算规定存档和 26 组验证评估就约为 {d["subset_estimated_bytes"]/1e9:.6f} GB，两倍为 {d["subset_required_double_bytes"]/1e9:.6f} GB，仍超过剩余空间。该子集还未计训练和复现日志、位置记录及报告。','',
 '估算分项：','',
 '|项目|依据|GB|','|---|---|---:|',
 f'|带经验池存档|7 × 原 S300 的 {d["full_checkpoint_basis"]["bytes"]} 字节|{d["full_checkpoint_estimated_bytes"]/1e9:.6f}|',
 f'|其他 learner 存档|13 × 试点 U200 的 {d["other_learner_checkpoint_basis"]["bytes"]} 字节|{d["other_checkpoint_estimated_bytes"]/1e9:.6f}|',
 f'|验证评估|26 × 旧组最大实测大小|{d["evaluation_estimated_bytes"]/1e9:.6f}|',
 f'|训练输出，不含存档|旧试点 4104 第 1–5 步实测|{sum(x["bytes"] for x in d["training_non_checkpoint_measured"])/1e9:.6f}|',
 f'|RP-U / RP-L 日志|旧复现目录实测|{sum(x["bytes"] for x in d["RP_non_checkpoint_measured"])/1e9:.6f}|',
 f'|报告与元数据预留|估计 32 MiB|{d["report_metadata_reserve_estimate_bytes"]/1e9:.6f}|','',
 '所有依据路径、逐项文件数和字节数见 step0_facts/facts.json 的 disk 段。','',
 '## 已通过与未执行','',
 '- Git 分支、HEAD、干净状态、冻结协议、导入路径、GPU 检查通过。',
 '- identity 与 PILOT 合同、ORIG 合同、WORK validation 一致；validation 为 passed。',
 '- 原 300 轮 JSON 的必要字段及外部种子齐全；没有可续训的第 200 轮存档。',
 '- 25–200 轮 8 份快照的哈希、身份、轮次及结构核对通过，SAC 五个网络与 PPO 两个网络可逐参数比较。',
 '- 快照实际含 learner.state_dict，包括优化器、计数和版本；缺少经验池及全局 RNG，仍不可续训。',
 '- 原训练循环与 Replay 标签切换、直接赋值保留池、sample 不按标签过滤、55 轮 FIFO 已核对。',
 '- 磁盘门槛失败后未继续参考曲线与耗时估计；RP、DG、S1–S5、D-a、D-b、A-H/A-F 均未运行。',
 '- 判据 1 和判据 2 未评估，没有本次主表结果；test 评估为 0 次。','',
 '## 失败及实现修正','',
 '第一处失败为 HardCheckError：disk_free_greater_than_two_times_estimate，属于实质条件不满足。没有放宽条件或自动重跑。','',
 '第一次内联生成中文报告出现 SyntaxError: Non-UTF-8 code starting with byte 0xe5。按允许的一次实现修正规则，将报告生成脚本的中文字符串改为 ASCII Unicode 转义；仅补写报告，不重跑预检，不训练，不评估。错误和 diff 见 implementation_corrections.json、report_attempt1/failure.json、report_attempt2/encoding_correction.diff。','',
 '## 只读保护与证据','',
 f'已读输入 SHA256 前后一致：{c["protected_files"]}/{c["protected_files"]}；冻结协议未变；结束时两棵 Git 工作树干净。所有新文件仅在 CPIL，旧源码、配置、checkpoint 和结果未改动。','',
 '- failure.json：失败阶段、traceback、已通过检查及实质性标记。',
 '- step0_facts/facts.json：完整预检、快照结构、磁盘依据及输入哈希。',
 '- git_after_failure.json：停止后的分支、提交及工作树状态。',
 '- final_readonly_checks.json：输入 SHA256 前后对照。',
 '- cp_contract.json / cp_contract.sha256：未创建。'
]
text='\n'.join(lines)+'\n'
for rel in ('summary.md','seed_4104/report.md'):
 q=(p/rel).resolve();assert q.is_relative_to(p.resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
(p/'seed_4104/report.json').write_text(json.dumps(dict(status='stopped_step0',time=e['time'],seed=4104,disk=d,criterion1='not_evaluated',criterion2='not_evaluated',training_started=False,evaluation_started=False,implementation_correction='report encoding only',protected_files_unchanged=c['protected_files'],git=c['git'],sources=['step0_facts/facts.json','failure.json','final_readonly_checks.json']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='STOPPED',time=e['time'],report=str(p/'seed_4104/report.md'),disk_free_GB=d['free_bytes']/1e9,estimated_GB=d['total_estimated_bytes']/1e9,required_free_GB=d['required_free_bytes_strictly_greater_than']/1e9,protected_inputs_verified=c['protected_files']),ensure_ascii=False))

