# 4104 课程式预训练试点：第 0 步停止报告

停止记录时间：2026-10-08T10:21:25.481739+08:00。

磁盘硬条件未通过。本次未启动训练或任何评估，未创建合同或 checkpoint。

## 磁盘检查

附件第 4 节第 5 条要求 D: 剩余空间严格大于估算用量的 2 倍。以下 GB 按 10^9 字节换算；输出用量是基于旧文件实测大小的估算。

|项目|字节|GB|
|---|---:|---:|
|D: 实测剩余|25352577024|25.352577|
|本次输出估算|21821911722|21.821912|
|须大于的两倍门槛|43643823444|43.643823|
|与门槛的差额|18291246420|18.291246|

仅计算规定存档和 26 组验证评估就约为 14.966433 GB，两倍为 29.932866 GB，仍超过剩余空间。该子集还未计训练和复现日志、位置记录及报告。

估算分项：

|项目|依据|GB|
|---|---|---:|
|带经验池存档|7 × 原 S300 的 477964475 字节|3.345751|
|其他 learner 存档|13 × 试点 U200 的 9552011 字节|0.124176|
|验证评估|26 × 旧组最大实测大小|11.496506|
|训练输出，不含存档|旧试点 4104 第 1–5 步实测|6.678571|
|RP-U / RP-L 日志|旧复现目录实测|0.143353|
|报告与元数据预留|估计 32 MiB|0.033554|

所有依据路径、逐项文件数和字节数见 step0_facts/facts.json 的 disk 段。

## 已通过与未执行

- Git 分支、HEAD、干净状态、冻结协议、导入路径、GPU 检查通过。
- identity 与 PILOT 合同、ORIG 合同、WORK validation 一致；validation 为 passed。
- 原 300 轮 JSON 的必要字段及外部种子齐全；没有可续训的第 200 轮存档。
- 25–200 轮 8 份快照的哈希、身份、轮次及结构核对通过，SAC 五个网络与 PPO 两个网络可逐参数比较。
- 快照实际含 learner.state_dict，包括优化器、计数和版本；缺少经验池及全局 RNG，仍不可续训。
- 原训练循环与 Replay 标签切换、直接赋值保留池、sample 不按标签过滤、55 轮 FIFO 已核对。
- 磁盘门槛失败后未继续参考曲线与耗时估计；RP、DG、S1–S5、D-a、D-b、A-H/A-F 均未运行。
- 判据 1 和判据 2 未评估，没有本次主表结果；test 评估为 0 次。

## 失败及实现修正

第一处失败为 HardCheckError：disk_free_greater_than_two_times_estimate，属于实质条件不满足。没有放宽条件或自动重跑。

第一次内联生成中文报告出现 SyntaxError: Non-UTF-8 code starting with byte 0xe5。按允许的一次实现修正规则，将报告生成脚本的中文字符串改为 ASCII Unicode 转义；仅补写报告，不重跑预检，不训练，不评估。错误和 diff 见 implementation_corrections.json、report_attempt1/failure.json、report_attempt2/encoding_correction.diff。

## 只读保护与证据

已读输入 SHA256 前后一致：336/336；冻结协议未变；结束时两棵 Git 工作树干净。所有新文件仅在 CPIL，旧源码、配置、checkpoint 和结果未改动。

- failure.json：失败阶段、traceback、已通过检查及实质性标记。
- step0_facts/facts.json：完整预检、快照结构、磁盘依据及输入哈希。
- git_after_failure.json：停止后的分支、提交及工作树状态。
- final_readonly_checks.json：输入 SHA256 前后对照。
- cp_contract.json / cp_contract.sha256：未创建。
