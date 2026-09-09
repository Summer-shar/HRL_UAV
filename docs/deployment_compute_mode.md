# 候选模式：部署期间继续计算

日期：2026-09-09。基准提交：`4850092f4d198e53c4826754c12a232cee5df569`。

状态：可切换模型扩展；不是新的主实验结果，不代表性能已改善。实现与测试阶段未修改论文、历史配置、训练网络或已有结果，也未提交或推送；本次本地提交由作者后续单独授权，不包含推送。

## 开关与兼容

在配置已有的 `environment` 对象中增加：

```json
"compute_during_deployment": true
```

- 不设置或设置为 `false`：保持基准部署期上传、U2U 和 CPU 全部暂停的语义。
- 设置为 `true`：部署期只有已经进入 UAV 计算队列的分片继续计算。
- 开启时要求原选项 `pause_service_during_flight=true`。若同时设置为 `false`，构造环境时报错；不能用全服务模式替代本候选。
- 只接受布尔值，字符串 `"false"` 等无效输入会报错。
- 当前任何已存在的 JSON 配置均未改动，新模式没有被默认启用。`QueueMDP` 已将 `environment` 原样传给环境构造函数，因此无需修改入口或增加第二套配置覆盖逻辑。

## 服务与代价语义

1. 复用正常服务步的 CPU 分配、优先队列服务和计算能耗公式，不增加新的资源调度器。
2. 完整转发并已经进入计算队列的远端分片可以计算；仍在上传、上传完成但尚未路由、仍在转发或仅部分转发的任务/分片不能计算。
3. 部署期不调用上传、路由或 U2U 服务函数。新到达仍按旧协议进入待处理记录，移动和时钟按旧协议推进。已有通信分片不迁移、不重分配。
4. 分片完成沿用父任务登记逻辑，所有分片完成后才登记父任务完成。驻留代价仍按本步服务后未完成父任务计算。
5. `total_energy`、计算能量、分 UAV 能量和 UAV 服务电池账本包含本步实际计算能量；不把已经计入的推进能量重复加入。
6. `QueueMDP.step_frame` 的代价公式、系数、归一化和帧内折扣原样保留。部署奖励通过原有跨周期 `boundary_reward` 纳入前一个 SAC 边界经验；后继仍为部署结束后的真实服务观测。PPO 周期回报仍包含原来的部署帧。

## 统计接口

`service_paused` 保留原有部署通信/微观决策暂停标记。候选部署步额外报告 `compute_service_paused=false`，不能再仅凭 `service_paused=true` 推断 CPU 服务量为零。

`MeteredMDP` 分别累计通信 `service_steps` 和 CPU `compute_steps`，包含部署期 CPU 服务量并继续核验队列收支。原字段 `cpu_utilization_service` 以 CPU 可服务步为分母；缺少新字段的旧记录仍使用原分母。通信统计与带宽共享规则不变。

名义 CPU 容量记账同步区分两种模式：默认模式扣除部署暂停，候选模式不扣 CPU 的部署时间。但这不是系统稳定性或性能保证，通信仍只在原服务阶段推进。

## 影响范围

- 系统模型、MDP/SMDP 转移：候选模式部署期计算队列和任务完成状态可变化，CPU 服务能量可非零。
- 优化问题、奖励：控制变量与目标定义不变，按候选真实转移计算出的代价值可以变化。
- 观测与动作：维数、特征定义、动作映射不变；队列变化自然反映到观测数值。
- 训练流程：不改变 PPO/SAC、决策频率、更新时钟、折扣或经验衔接；未启动训练。
- 图、实验、论文贡献：不改图或论文，不把历史全停服成绩当作候选模式成绩，不声称新增性能贡献。

## 验证范围

新增单元测试位于 `tests/test_deployment_compute.py`，只构造小队列和短时间片，不实例化学习网络、不加载权重、不调用训练或性能评估入口。

人工算例：UAV 0 已有高优 0.6 Mbit、低优 0.8 Mbit，CPU 上限 10 GHz，每 bit 需要 1000 cycles，物理步长 0.1 s。屏蔽随机到达以隔离该队列，不修改正式负载配置。

- 第一步：高优服务 0.6 Mbit、低优服务 0.4 Mbit；频率分别为 6、4 GHz，总计算能量 10 J；高优在 0.1 s 完成。
- 第二步：低优服务剩余 0.4 Mbit、频率 4 GHz；计算能量 0.64 J；低优在 0.2 s 完成。
- 同时构造的上传任务、待路由任务和部分转发分片保持原阶段；不因部署期 CPU 开启而提前转入计算。

兼容人工检查使用 `git show` 只读加载指定基准的环境源码，与当前缺省模式在同一构造队列、到达随机种子、路由和部署动作下对照两个宏观周期，共 200 个物理步（20 s）。逐步原始状态、全部返回统计字段和宏观输出完全一致。这不是 900 s 性能评测。

实际执行结果：25 项定向测试通过，其中本候选新增 10 项，其余覆盖原通信/分片、资源账本、边界折扣及奖励定义。单元测试运行时间约 1.4 s（不含解释器和模块导入）。命令为：

```powershell
.venv/Scripts/python.exe -m unittest tests.test_deployment_compute tests.test_tcom_communication_model tests.test_bandwidth tests.test_clean_mdp.CleanMDPTests.test_boundary_discount_includes_deployment_and_never_reset tests.test_clean_mdp.CleanMDPTests.test_reward_unclipped_and_priority_weighted -v
```

`git diff --check` 通过。没有执行全量测试、`scripts/run.py` 的任何模式或正式训练/性能评估。

## 追加集成测试：真实 rollout 与尾部经验补齐

2026-09-09，基准 HEAD：`95633816bdbc7e7a48718259b8c14e3b7b5886fb`。原有专项测试覆盖手动推进和 `boundary_reward` 数值，但没有候选模式的完整 rollout 覆盖；本次只补下面一项，不重复添加原有测试，不修改生产代码或正式配置。

测试：`tests/test_deployment_compute.py::DeploymentComputeTests.test_rollout_completes_deployment_boundaries_without_tail_metric_leak`。

- 显式开启 `compute_during_deployment`，调用真实 `rollout(..., macro_count=2, train=True, complete_tail=True)`。只在测试夹具中将记录窗口设为一个大周期，屏蔽随机到达，并初始放入已上传的高优 100.6 Mbit、低优 100.8 Mbit 计算任务。它们是用于跨越边界的人工工作量，不是正式负载或性能样本。
- 固定宏观/微观动作替身，无学习网络；store 只复制到内存，更新替身仅记录调用时刻，不计算梯度、不修改参数、不加载权重。真实环境计算、观测、回报、pending 和 complete_tail 路径没有被替代。
- 高优任务在第二周期部署期的 10.1 s 完成。11 s 后继计算队列剩余低优 91.4 Mbit、一个未完成父任务；第二任务在尾部额外部署的 20.2 s 完成，21 s 后继队列和未完成任务数为零。逐条检查实际 store 的观测与真实部署后观测一致，且不是 reset 状态。
- 两周期共 36 条服务经验，两个 pending 分别在 11/21 s 写入且各一次；源观测、动作与执行路由逐条匹配。两条边界回报均等于末服务帧回报加 gamma、gamma^2 加权的两个部署帧回报，折扣为 gamma^3；其余 34 条为单帧折扣。
- 完整采样窗口截至 20 s：完成高优一项、低优零项，剩余队列 1.4 Mbit。尾部后环境到 21 s、完成两项；但两个窗口行仍对应 10/20 s，summary 的完成计数、时延、J、功率及时间范围仍使用尾部之前的快照。前 20 s 计算能量为 2000 J，尾部计算能量为 10.64 J；回报与统计同时核对原推进能量账本，尾部能量不计入已结束窗口。
- 微观动作调用恰好 36 次、实际路由函数调用 180 次，均在服务阶段；SAC 空更新调用只在 12 s，PPO 空更新调用只在 10/20 s。额外宏观动作只用于尾部后继构造，不新增 PPO 经验或更新；记录中的所有实际学习更新计数为零。

实际执行：

```powershell
.venv/Scripts/python.exe -m unittest tests.test_deployment_compute.DeploymentComputeTests.test_rollout_completes_deployment_boundaries_without_tail_metric_leak -v
.venv/Scripts/python.exe -m unittest tests.test_deployment_compute -v
```

结果：新增单测首次运行 1 项通过，用时 0.137 s；部署专项文件 11 项全部通过，用时 0.936 s（均不含解释器和模块导入）。未运行其他测试、训练或性能评估；本次未提交或推送。此处人工任务的完成时延不能作为算法性能结果。

## 边界与未解决问题

- “移动期间 CPU 可以继续工作”是候选协议假设；实际设备的供电、散热和控制开销尚未建模或验证。
- 100 ms 物理步末完成登记、完整上传后路由、完整转发后计算等原假设不变。
- 本轮只验证实现和账本一致性，没有验证时延、功耗、队列稳定性或学习收敛是否改善。
- 旧权重未加载或续训，历史成绩继续只对应原模型。
