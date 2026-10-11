# 参考环境落地与验证交付

日期：2026-09-27。结论：指定参考源码已核对；当前环境与持续采样接口通过本轮确定性验证，未发现需要修复的语义、容量或守恒缺陷，因此保留现有生产源码。本轮新增 14 项、复用 12 项测试，26 项全部通过；加强一项活跃分片断言后单独复测通过。实际累计执行 **1,451 个物理步 / 上限 20,000**。未训练、未扫参、未修改网络、学习参数、物理配置或训练日程。

## 工作区与参考来源

开始时 `git status --short` 为空，分支为 `codex/hrl-uav-v2-standalone`，基线提交为 `3b858f9e248f95c489e88f72ce4e525e2deea64f`。工作始终留在该分支，没有切换或覆盖文件。交付仅包含本轮新增文件。

三个仓库均只通过只读接口读取指定 commit 的指定源码：LyDROO 两文件；FANET 的 drone、uav_env、packet、event 四文件；UAV-DDPG 的 DDPG/UAV_env.py。未安装、执行或复制整个外部环境。附件 ZIP 仅阅读其说明与 manifest，作为参考数据；其历史基线不替代本地源码或本轮用户指令。具体代码位置与取舍见[一页映射表](REFERENCE_ENVIRONMENT_MAPPING.md)。

## 当前场景、观测、动作与服务

生产场景仍为 3 UAV、39 固定终端、1000 m 方形地图、50 m 高度，每 UAV 固定关联 13 个终端。热点布局、业务 profile 和隐藏活动状态影响持续随机到达；动作不控制外生任务流。每 UAV CPU 为 10 GHz；上传总 20 MHz 均分 39 个终端，每终端 0.1 W；U2U 总 6 MHz 在活跃有向链路之间分配，每个发送 UAV 的 1 W 功率在其活跃出链间共享。

物理步为 0.1 s，下层每 0.5 s 更新 2×3×3 的 priority/source/destination 分片比例，各行非负且和为 1。`[0.3,0.5,0.2]` 创建三个真实分片，精确 0 不创建对象，每个父任务只分片一次。**Local 指来源 UAV 的 CPU**。上层每 5 s 提交 3×2 运动请求，按原有安全投影执行 1 s 调整、4 s 驻留，最多 10 m 位移、10 m/s 速度、至少 5 m UAV 间距。

上层观测来自新运动提交和该边界新到达之前；下层观测来自提交运动与准备当前到达之后。报告含当前位置、真实运动阶段、已发生到达前缀、父子身份、完整有序队列与剩余工作；不提供隐藏负载模式、RNG、未来到达或完成时刻，也没有 rollout 剩余时长。已复用测试检查报告不读隐藏生成器、到达特征仅使用观察前缀。完整模型说明见[设计说明](design_specification.md)。

每物理步依次执行：步首父任务 holding → 保存步首 ready → 上传 → 对原 ready 一次分片 → CPU → U2U → 推进能量、位置和时钟更新。新上传完成的任务下一步才能路由；本步转发完成的分片下一步才能用目的 CPU。上传保持每终端统一 FIFO，CPU 和有向 U2U 为高优先级优先、同优先级 FIFO。没有服务能力时保留原任务与残量；所有正分片完成后才记录父任务完成，转发本身不是完成。

成本保持 `H_k = dt × (10 N_high + N_low)`、`c_k = 0.09 H_k + E_k/140`。未完成父任务无论有几个分片只计一次；E 包含实际上传、转发、CPU 和全步推进/悬停能量。CPU 动态能量为 κf³dt，通信按实际服务时间计费。上下层奖励都由同一物理成本累计并除以固定 100 取负；没有距离 shaping、完成 bonus 或拥堵罚项。

## 确定性验证覆盖

| 要求 | 直接验证当前源码的测试 |
|---|---|
| 无服务时持久积压、恢复后继续 | 新增 `ReferenceCapacityTests.test_zero_service_preserves_all_backlogs_until_capacity_returns`，三类服务暂停 3 步后恢复 |
| 积压与 CPU/upload/U2U 预算；共享带宽/功率 | 复用 `PhysicalContractTests` 容量、优先级与微小残差用例；新增四活跃链、三发送源资源检查与小积压不超服务；新增按实际工作核对 CPU/通信能量 |
| 分片守恒、零对象、一次路由 | 复用 split/zero/tiny-positive 用例；新增改变后续 route 不改变既有分片、再次路由同父任务明确拒绝 |
| 转发进入目的 CPU、身份与 birth 不变 | 新增 `test_forward_keeps_identity_birth_and_waits_for_destination_cpu`，非零出生时间、转发步不获 CPU |
| 末分片完成且只完成一次 | 新增 `test_deliberately_late_fragment_blocks_parent_without_double_completion`，人为延迟一条链，检查前两分片完成时父任务仍未完成及后续唯一完成事件 |
| 优先级/FIFO、上传不抢占 | 复用 upload FIFO/CPU priority-FIFO；新增后来高优 U2U 分片优先、类内保持 FIFO、同链预算共享 |
| 位置只按物理公式改变速率/服务 | 新增 upload/U2U near/far 隔离用例，固定活动链、带宽、功率、其他端点与关联，手写 Shannon 公式计算速率和 `min(backlog, rate×dt)` |
| 独立 holding 与窗口删失 | 从外部 tape 的出生和 priority、CPU 完成事件、窗口边界计算驻留；不使用成本函数或 metrics 的 residence 字段作为答案；覆盖后来完成的 carry-in 和一直未完成的父任务 |
| 持续采样、真实后继、边界无重复 | 新增 `ReferenceContinuationTests` 五项，详见下段 |

所有到达均来自固定外部任务。验证器将复用用例的随机 frame 生成器替换为空 tape；新增任务显式注入，脚本动作确定。容量暂停、已上传 ready 状态与已知链路速率属于隔离测试夹具，生产配置未改。60 秒对照完整使用现有物理资源、几何、能耗与调度，只替换外部任务源和采样长度。

## 60 秒与 30＋30 秒的实测结果

两条轨迹分别执行 600 步和 300＋300 步，中间不 reset，共 **1,200 步**，由五个测试共享，未重复模拟。固定 tape 包含小任务、跨 30 秒后完成的任务、始终积压的 CPU/upload 任务，以及恰在 30/60 秒到达的任务。

- 逐物理步动作、位置、速度、目标、队列顺序、所有父/子剩余工作、成本与能量 reading 完全相同；全事件台账、最终报告和完成结果完全相同。
- 120 个下层动作与奖励一致；上层请求及真实提交时刻均为 0, 5, …, 60 秒，共 13 次，包含 60 秒真实后继所需的一次承诺。
- 到达 prepare 时刻均为 0, 0.5, …, 60 秒，共 121 次，每帧恰好一次；固定任务源不消费隐藏随机流，运行前后 RNG 状态相同。分段不重新开始策略、不多提交上层动作。
- 30 秒统计先于 30 秒到达，60 秒统计先于 60 秒到达；边界后继已包含该时刻真实到达与运动承诺，下段直接接续。窗口末未完成任务保留并继续服务。
- 普通窗口 `terminated=False`，最后 transition 的 `truncated=True`；上下层 successor 均等于下一窗口相应决策观测。显式 reset 单独验证会清空独立场景，未用于构造后继。
- 独立 holding 及成本对比容差 `rtol=1e-12, atol=1e-8`；相邻窗口成本之和等于连续窗口成本。两段的 cutoff 未完成计数不是可加的末端唯一任务数，未把重复 carry-in 当作新任务或丢失任务。

30/60 秒仅用于工程验证，不改变默认科研配置。环境接口和固定控制连续性已验证，不代表整个持续任务 RL 方法或训练恢复协议已完成。

## 真实执行记录与复现

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m hrl_v2.tests.reference_validation --output-dir docs/evidence/reference_environment
```

验证器在 `IndustrialEnv.micro_step` 统计实际 `step_index` 增量，单独记录未推进物理时间的非法动作拒绝；同一输出目录累计已记录运行的预算，达到 20,000 后拒绝继续。正常返回的失败测试也计入预算。本轮两次均正常串行完成；预算文件不是进程中断恢复或并发锁机制。

| 执行组 | 测试数 | 物理步 |
|---|---:|---:|
| 复用物理合同测试 | 10 | 210 |
| 新增父任务、能量、FIFO、训练 guard | 5 | 20 |
| 新增资源与几何隔离 | 4 | 9 |
| 新增连续采样（共享 fixture） | 5 | 1200 |
| 复用因果观测测试 | 2 | 5 |
| 主验证合计 | 26 通过，0 失败，0 错误 | 1444 |
| 加强活跃分片断言后的定向复测 | 1 通过 | 7 |
| **本轮累计** | **26 个不同测试，27 次执行全部通过** | **1451** |

主验证：2026-09-27 07:48:53 UTC（北京时间 15:48:53），Python 3.9.25，unittest 用时 1.858 s；另有 3 次非法动作调用被拒绝、0 物理步。静态复核后，将“改变后续 route 不会再次拆分”断言扩展到仍有阻塞分片的时刻；07:52:10 UTC 单独复测该项，用时 0.012 s、7 步。不运行包含 18,000 步旧对照的全套验证，也未执行优化器测试。

原始证据：[主验证输出](evidence/reference_environment/run_01.txt)、[主验证步数](evidence/reference_environment/run_01.json)、[定向复测输出](evidence/reference_environment/run_02.txt)、[定向复测步数](evidence/reference_environment/run_02.json)、[累计预算](evidence/reference_environment/budget.json)、[来源和本地文件指纹](evidence/reference_environment/provenance.json)。

## 修改文件与未决项

仅新增：`src/hrl_v2/tests/reference_validation.py`、`test_reference_parents.py`、`test_reference_physics.py`、`test_reference_continuation.py`；本文、`REFERENCE_ENVIRONMENT_MAPPING.md`；`docs/evidence/reference_environment/` 下上述六份证据。现有环境、场景、网络、学习器、采样器和配置文件均保留。

本轮环境范围内没有待研究者裁决的容量、身份、完成或成本歧义。恢复正式学习前仍需研究者确定持续任务的学习目标及 gamma/lambda、连续采样与参数更新的衔接和独立场景 reset/评价口径；持久环境/RNG/未消费上层承诺的恢复协议需要后续实现与验证。本轮不选择这些参数或日程。当前 `run.train` 在创建 Learner 或载入 checkpoint 前调用无条件 guard，train/resume 均被阻止；已用零物理步测试验证即使更改确认标志仍拒绝训练。
