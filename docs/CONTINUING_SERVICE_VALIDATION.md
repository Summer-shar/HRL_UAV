# Continuing-service 环境验证交付

日期：2026-09-27。本轮只依据当前工作目录的代码、配置、测试和文档。没有使用其他项目的实验或 checkpoint，没有运行正式强化学习训练。

## 交付范围与当前结论

本轮把数据收集截止从物理终止中分离，保留跨窗口真实队列与 successor，移除策略 rollout 倒计时，补齐父任务/fragment 物理守恒、删失与奖励聚合验证。已有场景、资源参数、连续 class-level 分片含义和物理成本权重保持；修复 tiny-positive 工作被 epsilon 免除或遗漏服务的行为。

场景、MDP/POMDP、观测、动作、Mermaid 任务状态图和成本公式的完整定义见 [当前设计说明](design_specification.md)。配置和有界运行方法见 [配置说明](CONFIGURATION.md) 与 [脚本示例](../examples/README.md)。

默认 900 s 是 rollout window。当前系统无正常真实 termination：截止 transition 为 `terminated=False,truncated=True`；窗口 metrics 先于下一 frame 到达准备，upper successor 是下一决策 precommit 报告，lower successor 包含下一 upper 已提交动作。`_continuation` 让下一窗口直接消费该动作和同一个世界。

## 修改文件与主要函数

| 文件 | 主要变更 |
|---|---|
| `src/hrl_v2/timing.py` | 新增 `ClockConfig`、配置读入、整除检查和物理/上下层步数推导 |
| `src/hrl_uav/industrial_env.py` | `_upload_step/_compute_step/_transfer_step` 按真实零剩余判断完成，保证 tiny-positive 工作受容量约束并获得服务 |
| `src/hrl_v2/physics_adapter.py` | `__init__/reset/start_macro/prepare/step/report` 接入独立时钟；允许跨 cutoff 服务；保留 continuation；报告真实 dt |
| `src/hrl_v2/reports.py` | `ReportV2` 移除 `remaining_ticks`，校验真实物理时间与 motion 计时 |
| `src/hrl_v2/features.py` | `build` 的 clock 改为物理 macro/motion 时间；`arrival_array` 拒绝非法 cursor |
| `src/hrl_v2/collector.py` | `collect/collect_metrics/_collect/_commit_macro` 真实 successor、两层 termination/truncation、cutoff 统计与 continuation 消费 |
| `src/hrl_v2/replay.py` | `Episode.seal` 保存独立 flags、上下层 successor，按配置步数和成本恒等式验证窗口 |
| `src/hrl_v2/metrics.py` | `snapshot/finish/PooledMetrics` 支持 carry-in、窗口差值、censored、完整完成时延和能耗/队列 pooled 指标 |
| `src/hrl_v2/sac.py` | `SAC.__init__/_critic_fit` 显式读取 discount，仅真正 termination 关闭 bootstrap |
| `src/hrl_v2/ppo.py` | `generalized_advantage/PPO.update` 对 truncation bootstrap、截断 trace，使用真实 upper successor value |
| `src/hrl_v2/runtime.py` | `Learner` 传递保留的 gamma/lambda 配置；未改网络和优化器结构 |
| `src/hrl_v2/run.py` | `read_config` 检查 clocks/物理成本/语义；`require_research_protocol` 无条件禁止正式 train/resume |
| `src/hrl_v2/cli.py` | `check` 明确报告 `training_ready=False` |
| `src/hrl_v2/evaluate.py` | 评价适配可配置窗口、当前 cutoff 指标 |
| `src/hrl_v2/schedule.py` | 保留 schedule 数值，仅将预算计数按窗口/控制步数推导 |
| `src/hrl_v2/benchmark_optimization.py` | `fixture_clock` 从实际轨迹推导时钟；采样、历史、上下层计数动态化，使用真实 cutoff value；拒绝不兼容的归档 baseline 分支 |
| `src/hrl_v2/budget.py` | `require_archived_measurement/budget` 对归档预算外推增加显式 opt-in 与元数据校验，拒绝将 continuing-service 记录套入归档固定时域预算 |
| `.github/workflows/ci.yml` | CPU CI 增加物理合同、因果观测/metrics 与 rollout 语义三组有界测试 |
| `src/hrl_v2/fixture.py` | 脚本 fixture 接收配置，物理步数自动推导，learning updates 为零 |
| `src/hrl_v2/validate.py` | 自动发现新增测试；验证记录明确零 research episodes、无性能结论 |
| `src/hrl_v2/configs/v2_1_0.json` | 当前 schema、独立时钟、continuing flags、未确认研究参数状态 |
| `src/hrl_v2/__init__.py`、`VERSION.json` | 当前数据/项目元信息，避免把当前合同与归档快照混用 |
| `pyproject.toml` | 项目 description 从 finite-horizon 更新为 continuing control；依赖版本未修改 |
| `README.md`、`docs/design_specification.md`、`docs/CONFIGURATION.md`、`examples/README.md` | 更新为本轮实际场景与 continuing-service 口径 |
| `src/hrl_v2/tests/`、`tests/` | 新增合同测试并更新受边界语义影响的集成、配置、包装和学习公式断言 |

网络、encoder、attention 层数、critic encoder 共享方式、optimizer、学习率、batch、gradient clipping、alpha、baseline 本轮没有重新设计或调优。SAC/PPO 修改仅为 termination/truncation 与显式参数输入的数据语义。

最终变更的完整文件列表以本次 Git commit 为准。最新 commit SHA 在最终交付回复中提供；本地可执行 `git rev-parse HEAD` 查询。

## 用户要求的测试映射

| 要求 | 覆盖位置与检查 |
|---|---|
| 1. bits/cycles/fragment/parent 守恒 | `test_physical_contract`：分片总量、各阶段残差和完成事件守恒 |
| 2. 全部 positive fragment 完成才完成 parent | 同模块：检查部分 child 已完成时 parent 仍 live，最终 finish 取最大值 |
| 3. zero allocation 无 fragment | 同模块：精确零不创建，tiny-positive 创建并实际获得服务 |
| 4. queue 跨 lower/upper 保留 | 同模块跨周期积压；`test_rollout_semantics` 比较拆窗与连续运行 |
| 5. 动作不能绕过容量 | 同模块检查 upload/U2U/CPU capacity、能耗、无效 simplex 和运动约束 |
| 6. reward 与 physical cost 聚合一致 | `Episode.seal` 与 collector 检查上下层/物理 cost 恒等式，集成测试执行验证 |
| 7. cutoff terminated=False,truncated=True | `test_rollout_semantics` 上下层窗口末 flags |
| 8. 截断帧有真实 successor | 同模块核对下一真实 upper/lower 决策报告 |
| 9. successor 非 reset 新状态 | 同模块比较相邻窗口、保留队列、到达流和 RNG；显式 reset 清 pending |
| 10. 修改 rollout 自动更新 counts | `test_configuration` 和 `test_rollout_semantics`：60 s、步数及错误整除关系 |
| 11. 无 rollout countdown | `test_observation_metrics`：不同窗口长度同物理状态输入一致；绝对时钟平移不改策略特征 |
| 12. age/motion phase 仍存在 | 同模块：task age 增长，调整倒计时变化，adjustment→hold |
| 13. cutoff 未完成记 censored | 同模块：未完成 delay=None，具有 cutoff age，没有 failed/deadline 标签 |
| 14. 统计前不清队列 | 同模块 report 前后一致；collector 测试 cutoff metrics 先于边界 arrival |
| 15. 不读 future/hidden state | 同模块用禁止访问的 hidden-generator 对象验证 report；future history 后缀修改不影响已观察前缀 |

附加测试涵盖 upload FIFO、CPU 高优/FIFO、极小 upload/CPU/U2U 残差不被免除、安全距离对连续路径成立、carry-in population 守恒、窗口 holding 生命周期恒等式、pooled 成本/能耗、SAC termination mask、GAE 截断 bootstrap 与 trace mask、内存状态恢复、upper version 不可重新标记、跨窗口采样事件计数。

## 测试命令与结果

以下命令由当前项目虚拟环境执行：

```powershell
.\.venv\Scripts\python.exe -m hrl_v2 --workspace ./outputs/continuing_service validate
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m hrl_v2 --workspace ./outputs/continuing_service validate-cuda
```

最终实测环境为 Windows、Python 3.9.25、NumPy 1.26.4、PyTorch 2.7.0.dev20250227+cu128、CUDA 12.8、NVIDIA GeForce RTX 5060。本轮新增 37 项实现测试和 6 项配置测试，共新增 43 项；当前共 99 个不同测试（88 项实现 + 11 项配置/包装）。CUDA 4 项是不同 device 上的专项复跑，不计作新增独立测试。以下秒数是测试执行用时，不代表策略推理、学习或服务性能。

| 检查 | 最终结果 |
|---|---|
| 完整实现测试套件 | 88/88 通过，50.096 s；0 failure、0 error、0 skip |
| 配置与包装测试 | 11/11 通过，5.109 s；0 failure、0 error、0 skip |
| CUDA 专项复验 | 4/4 通过，11.266 s；0 failure、0 error、0 skip |
| CLI 状态检查 | validation_ready=true、training_ready=false、training_started=false、updates=0 |
| 当前文档连续采样示例 | 通过：同一世界两个 5 s 窗口，真实 successor 相同，累计物理时间 10 s，零 learning updates；5 份文档本地链接和代码围栏检查通过 |
| wheel 构建与包内容 | pip wheel . --no-deps --no-build-isolation 成功；wheel 中 timing.py、当前配置与 4 个新增 implementation test modules 同源码逐字节一致 |
| 正式研究训练 | 未执行；train/resume 无条件 fail-fast |
| 策略性能比较 | 未执行，无性能结论 |

详细记录：[实现测试日志](evidence/continuing_service/tests.txt)、[实现验证 JSON](evidence/continuing_service/validation.json)、[配置与包装日志](evidence/continuing_service/configuration_tests.txt)、[CUDA 日志](evidence/continuing_service/cuda_tests.txt)、[CUDA JSON](evidence/continuing_service/cuda_tests.json)；受版本控制的本轮记录副本见 [continuing-service evidence](evidence/continuing_service/)。这些是本轮本地运行输出，不保证在其他设备上自动成立。

已完成的专项观测/metrics 测试：`test_observation_metrics` 的 7 项测试全部通过。完整首轮测试执行 82 项，其中 81 项通过，1 项现有 prepared-optimization Q-head 梯度对照失败。调查发现该测试输入依赖全局随机状态；修复为每测试隔离 RNG 并使用固定 927 seed，不修改网络或误差容限。隔离复验及该 7 项模块已通过，最终全套重跑结果以上表实测记录为准。保留这一过程记录，避免把首轮失败隐藏为一次性全通过。

有界学习测试可能创建临时模型并执行固定数量优化器步骤，只验证公式、梯度或工程等价性；不是研究训练，不产生经训练策略的性能主张。`_reference/` 可以继续服务于限定的工程对照测试，但其中归档的有限时域假设和运行证据不用于证明当前 continuing-service 模型正确。

## 仍需研究者决定

1. 持续服务目标采用长期平均成本还是折扣形式；SAC/PPO gamma、GAE lambda 尚未科研确认。当前数值 1.0 仅保留，不作为本轮新决定。
2. 上下层更新、预训练、训练 schedule、reset/warm-up 与评价统计设计尚未确定。现有 schedule 配置属于保留参数；不能直接启动。
3. 完整物理环境、隐藏到达 RNG、arrival 历史、任务队列、已提交动作和累计 hash 的持久化 checkpoint 协议。当前环境含 hashlib 对象，普通 pickle 不能直接完成完整环境恢复。
4. 无限运行下父/fragment 台账与 arrival 历史持续增长，尚无有界归档方案；当前保证的是保留物理任务。
5. 是否扩展瞬加速/直线运动、0.1 s 网格完成、结果回传、控制遥测开销和 CPU 静态能耗等物理假设。
6. 修改 lower/upper 物理控制时钟后，需审查未修改 baseline 的固定历史窗口、需求期和位移假设。
7. 尚未训练或验证学习性能，也未证明负载下长期队列稳定性；这些不能由本轮实现测试替代。
