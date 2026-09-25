# PPO 逐设备观测候选

本目录是独立、可执行的候选入口，修补旧 85 维摘要中逐设备任务量与坐标关联丢失的问题。新增实现位于 `C:/Users/PC/Desktop/HRL_UAV/industrial_device_pilot`；原 `hrl_uav`、原 pilot、历史结果和检查点保持原样。原源码目录实际为 D 盘联接，本候选通过正常导入复用其服务与学习过程，不覆盖联接目标。

## 已实现的改变

PPO 输入为 **523 维 = 39×12 + 55**。每个终端的一条记录包含：相对所属 UAV 的二维位置，以及分别按高、低优先级统计的剩余上传 bits、尚未完整上传父任务的完整 cycles、任务数、近期实际到达 bit/s 与 cycle/s。每个终端的位置和任务量在进入网络前始终配对。

近期需求来自过去 `min(当前时间, 10秒)` 的实际到达，除以实际观测时长；轮初无历史时为零。当前队列单独保留，不用近期平均替代。三类业务的计算差异通过 bits 与 cycles 表达，不向网络提供隐藏模板编号、未来到达或终端编号特征。

全局 55 维包括三机坐标、已上传待分配的 bits/cycles/数量、CPU 队列、六条有向链路的转发 bits、目的地在途 cycles、来源未完成父任务数和有效历史长度。上传完成后，终端 pending_cycles 中的该任务移到全局 ready_cycles，之后进入在途或 CPU 阶段。

PPO 动作网络、评价网络各自使用共享的 `12→32→32` 设备编码器；在每个来源的 13 个终端内做均值与最大值汇聚，再拼接全局信息，经过 `247→256→256` 输出。设备报告在同一来源内换序不会改变输出，除浮点归约误差。三个 UAV 的动作与全局负载仍按来源排列，并未声称具备 UAV 置换等变性。

宏观输入不含预测速率或 `bits/rate`。SAC 仍使用旧 86 维编码和原动作映射；其输入仍有原来的链路/服务时间特征，不能把整套系统称为完全不依赖无线模型。

## 影响范围

|项目|本候选的处理|
|---|---|
|系统模型与服务规则|保持 24.725 Gcycles/s 的随机场景、任务过程、无线与能耗模型|
|优化问题|保持父任务加权驻留成本和能耗的原目标|
|MDP / 观测|PPO 改为逐设备公开报告编码；这仍是部分信息，不能宣称充分马尔可夫状态|
|动作|保留 PPO 6 维运动和 SAC 18 维分配分数及执行映射|
|奖励与折扣|沿原 control，不采用 long150，不加入朝中心奖励|
|时钟|5 秒上层、1 秒调整加 4 秒驻留，持续服务；0.5 秒下层、0.1 秒物理步|
|训练|完整迁移 joint-0600 SAC，重建 PPO 及其优化器，直接联合更新|
|恢复|独立协议、523/86 维 spec、源码指纹；网络、5 个优化器、经验池、随机状态和待更新 PPO 样本完整保存|
|图与实验|新结果需另存，与原 control 和本次自身 SAC 悬停配对；功能测试权重禁止用于研究训练|
|论文与贡献|本轮未修改；新表示尚无性能验证，不作为已证明贡献|

新输入仍未保留逐任务 FIFO 顺序、所有年龄和父任务分片关联；集合汇聚也会压缩信息。这次实现修补明确的信息缺口，不保证消除外飞或达到 600 ms。

## 奖励

每个 5 秒大时隙含 10 个 0.5 秒下层周期。当前 control 折扣下：

`R_macro = r_0 + 0.99*r_1 + ... + 0.99^9*r_9`。

每个下层奖励又是该周期 5 个物理步成本的负折扣和。下一大时隙折扣为 `0.99^10`。PPO 没有累加 SAC 的熵、损失或 Q。实现和目标关系见 [reward_audit.md](C:/Users/PC/Desktop/HRL_UAV/industrial_device_pilot/reward_audit.md)。

## 入口和运行边界

从仓库根目录运行，环境为已有 `D:/Anaconda3/envs/pytorch-gpu/python.exe`，CPU、Torch 单线程。

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -B -m industrial_device_pilot.experiment describe
```

`describe` 只核实合同，服务步和更新均为零。

有界功能检查入口为 `industrial_device_pilot.validate`。验证保存到独立 attempt 目录，所有尝试共最多 22000 个物理步、800 次 Adam 调用；异常中断且无完成账本的尝试保守占用整笔登记预算。两轮 900 秒仅用于核查正常批量更新和测速，其权重不是研究结果。

正式短程训练入口如下，本轮交付时**尚未执行**：

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -B -m industrial_device_pilot.experiment train --output industrial_hotspot_records/load24725_device_pilot_20260924/phase25 --acceptance industrial_device_pilot/validation/attempt01/verification.json --stop-after 25
```

新任务必须先停在 25 轮，含六场景运动/悬停评价，不能直接从零跑 50 轮。继续到 50 需要显式 `--checkpoint` 指向已完成第 25 轮评价后的完整检查点，以及 `--stop-after 50` 和新的输出目录。不存在冻结 SAC 单独训练 PPO 的阶段，也不重新做 300 轮预训练。

每轮 900 秒，300 个宏观样本更新 PPO，跨轮待更新样本保留。25 轮为 225000 个成功训练物理步、15 次 PPO 更新；全部 50 轮最多 450000 个成功训练物理步、30 次 PPO 更新。每个评价节点另用 108000 个物理步，0 学习/经验写入。每个训练阶段墙钟上限 3600 秒；异常中断不自动重试，已完成整轮可恢复。研究 runner 的累计训练步计数是已成功收集的步，人工恢复前须另核算失败半轮开销，不能把它当成含所有失败尝试的实际算量上限；本轮功能验证器已另行逐步计数。

运行会保留完整检查点、每轮指标、每 5 秒轨迹及队列、学习统计、普通总成本负值 `physical_return` 和 `upper_discounted_return`。评价同时报告成本、加权时延、未完成数、上传积压面积以及本次同一 SAC 的运动/悬停差异。`bad_performance_node` 是沿用旧试验的联合阈值标签，并非收敛或优于基线的证明；阶段完成后需要查看全部指标。

## SAC 独立审计

本轮检查用于迁移的原 `candidate-joint-0600.pt` 下层。结论是已具备有效分配能力，但低优任务仍存在路线偏好与算力使用不均，不能说完全学好了。见 [SAC 实际工作分配报告](C:/Users/PC/Desktop/HRL_UAV/industrial_hotspot_analysis/load24725_sac_action_audit_20260924/findings.md)。

## 本轮实测交付状态

2026-09-24 的 attempt01 验收通过：18700 物理步、572 次 Adam 调用、48.109 秒，无失败重跑。包括两条相同动作的 20 秒旧新物理过程对照、真实短更新及完整保存恢复、正常配置两轮 900 秒联合更新。完整 SAC 迁移及后续网络/优化器/经验池/RNG 一致性通过；7 类损坏合同均在学习状态突变前拒绝。奖励恒等最大误差 0。正常两轮新增 180 次 SAC 更新、1 批 300 样本 PPO 更新，保留 60 条待更新宏观样本；最终 KL=0.014064。

两轮纯训练段耗时 17.425 / 18.095 秒，不含完整评价和每轮全部检查点写盘，不能当作后续训练耗时保证。主代理又独立核对当前源码身份与 verification 一致。完整依据见 [verification.json](C:/Users/PC/Desktop/HRL_UAV/industrial_device_pilot/validation/attempt01/verification.json) 和 [integration_review.md](C:/Users/PC/Desktop/HRL_UAV/industrial_device_pilot/integration_review.md)。

正式 25 轮性能对照未启动，新候选完整六场景评价入口仅完成静态审查，尚未实跑。SAC 审计另有 108000 个冻结物理步、0 学习；本轮新增物理服务总量 126700 步，其中任何功能测试权重都不用于研究训练。

## 后续25轮运行结果（2026-09-24）

上述“未启动”为实现交付时快照。用户随后批准的25轮及全部六场景运动/悬停评价现已完成，运行目录 `industrial_hotspot_records/load24725_device_pilot_20260924`。训练225000步，评价108000步；新增15批PPO、2250次SAC事件。原作业总耗时593.227秒。新运动/同端点SAC悬停加权时延702.951/665.512ms、成本10.895399/10.636783每秒。旧control同25轮运动时延786.382ms；改动减轻运动损失，但仍未获得优于悬停的服务收益。

原评价已完成全部轨迹与冻结/配对检查，最后写report时因`bad_performance_node`为NumPy布尔值而失败。已用12个完整原始JSON离线补齐report，保存原失败现场并逐项核对身份，新增服务步和学习均0。恢复说明与原始文件SHA在运行目录 `summary_recovery/receipt.json`，可严格恢复的完成检查点在 `phase25/checkpoints/recovered-episode-0025/final.pt`。原训练检查点及所有封存源码保持原样；本次是报告和完成游标恢复，后续扩展前仍应在保持历史身份兼容的前提下处理评价入口的该序列化缺陷。

## 第26–50轮续训与序列化修复（2026-09-24）

用户批准继续至50轮后，仅将 `evaluation.py` 的报告字段改为 `bad_performance_node=bool(bad)`。所有训练、场景、奖励和网络源码保持原样；原25轮 evaluator 副本保存于运行目录 `serialization_fix/evaluation_before.py`。该一行改变更新了源码身份，因此旧25轮检查点及原验收记录不能直接冒充新源码下的检查点和验收。

`serialization_fix/migrate.py` 明确校验41文件清单只发生这一处字节/语义变化，按旧身份严格恢复完成的25轮状态，再发布新身份下的等价检查点；学习器、5个优化器、经验池、全局及经验池随机状态、探索、更新计数和完整边界游标不变。四种布尔短路路径及完整评价JSON读写检查通过；另拒绝非白名单源码身份变化。所有验证为0服务步、0学习。派生验收 `serialization_fix/acceptance.json` 引用原18700步结果，不将其计为新增运行；完整当前身份恢复依据为 `serialization_fix/receipt.json`。

实际续训命令：

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -B -m industrial_device_pilot.experiment train --output industrial_hotspot_records/load24725_device_pilot_20260924/phase50 --acceptance industrial_hotspot_records/load24725_device_pilot_20260924/serialization_fix/acceptance.json --checkpoint industrial_hotspot_records/load24725_device_pilot_20260924/serialization_fix/checkpoint/final.pt --stop-after 50
```

2026-09-24 10:14:33启动，PID42636。从已完成25轮继续，额外25轮/225000训练物理步/15批PPO/2250次SAC事件；50轮端点另做108000步配对评价，完成即停。启动前的两次迁移检查因错误计入加载器追加的`_load_metadata`而停在检查点发布前，均0服务/学习；最终比较保留全部持久化学习字段，仅单列加载器提供的路径/SHA，见execution_attempts.json。当前训练状态以phase50/progress.json为准。

本批现已完成，phase50状态为phase_complete，总耗时595.954秒，评价154.402秒；报告正常写出且冻结指纹相同。50轮运动/悬停时延756.276/731.502ms、成本11.275326/11.108073每秒，运动高/低未完成9/189、悬停4/194。25轮对应运动/悬停时延702.951/665.512ms，因此本次续训未改善绝对性能，下层SAC悬停表现也退化。运动相对自身悬停的差距缩小并不表示整体服务改善。当前已停50轮，未继续扩量。原始结果：`industrial_hotspot_records/load24725_device_pilot_20260924/phase50/evaluations/episode-0050/report.json`。
