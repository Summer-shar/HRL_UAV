# 方案 D 精简切换记录

记录时间：2026-10-11T22:22:14.144788+08:00。两 seed 后台并行运行，训练内容未改变。

## 切换与验证

切换检查时已存在 21:34:53 的外围 PermissionError；原错误已保留。4104/4103 从最近完整续训档第 60/15 轮恢复。参数、优化器、计数及 CPU/CUDA 随机数状态一致，经验池成功载入。重跑的 4104 第 61–64 轮、4103 第 16–18 轮全部非耗时字段及 event_sha256 与中断前一致。

自检通过：数值函数 AST、预算/α/冻结表、全部脚本语法、外围读写重试、最少评估状态、完整续训载入、TFOO 场景 1 两条流完全相同、逐窗口数据不落盘。

## 实测速度

| seed | 范围 | 样本数 | 实际平均周期（秒/轮） | collection（秒/轮，含在线 SAC） |
|---|---:|---:|---:|---:|
| 4104 | 61–65 | 5 | 179.800 | 179.654 |
| 4103 | 16–24 | 9 | 97.678 | 95.787 |

旧 4104 单独运行基线为 267.710 秒/轮，当前相对变化 -32.84%，未超过 +60% 阈值。样本含恢复后首次回合；除去第一回合的 collection 均值详见 JSON，不能保证该剔除值就是稳态速度。

4103 新第 20 轮完整续训档：181,751,843 字节，15.747 秒，未整份重读、未计算整份 SHA。旧第 15 轮档 138,516,771 字节、43.758 秒。两者大小及状态不同，不能当作同一存档的优化前后对照。4104 新第 80 轮档尚未到达，不填虚拟耗时。

10 秒只读 CPU 样本：系统 34.4%；两个训练进程各约一个核（4104 98.91%、4103 98.13%，单核 100%）。显存快照 2367 / 8151 MiB，GPU 利用率 11%。

## 设置与偏差

完整续训每 20 正常轮一份，仅留最新档；十份选档保留，最少评估状态不含经验池和优化器。诊断 14–15 组/seed，3 workers，仅保留逐场景和组汇总；两条流仍严格比对。

event_sha256 仍逐轮计算：它嵌在 WT 环境的事件生成中，没有独立关闭开关；按用户允许的例外保留，未修改决策路径。输入 SHA 仅在切换启动时核对实际 6 个文件。

## 剩余时长估计

还需约 **12–18 小时**，包含两个 seed 剩余训练与精简评估。打底并行估计 6.5–9 小时，联合及附加回合 3–6 小时，评估约 2–3 小时；区间总和取整。依据当前实测打底速度、JC 联合工作耗时、历史评估耗时与 worker 数缩减。D 联合阶段未实测，经验池增长及训练/评估竞争均会影响结果；不是完成时间承诺。

## 修复记录

- lean_runtime_and_permission_error_recovery：Windows transient PermissionError reading atomically replaced seed progress
- lean_transition_static_review_fix：restore replacement omitted interrupted suffix directory initialization
- lean_preflight_import_fix：selftest SCHEMA NameError
- lean_background_launcher_fix：CIM embedded startup object type mismatch

完整状态比对、逐样本耗时、旧/新存档记录、原错误路径和修复明细见 lean_switch_report.json。MAIN 代码/git、WT/src、已有其他输出保持只读，未访问测试场景 21–40。
