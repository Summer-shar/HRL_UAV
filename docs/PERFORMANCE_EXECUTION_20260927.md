# 2026-09-27 正确性与等价性能执行报告

默认 critic 分块仍为 16。本轮完成 61 组相同输入、初态和更新计数的前后比较：长历史 critic 中位数从 6.233–7.514 秒降至 4.694–5.534 秒，actor 从 3.376–4.106 秒降至 2.628–3.023 秒。按真实单 seed 日程汇总的训练加开发评价条件估计从 **154.48–192.48 小时**降至 **120.69–159.76 小时，约 5.0–6.7 天**；对应端点节省约 17%–22%。这些是已测工作负载下的条件估计，尚有未定价的残差未计入，不是完整 seed 实测、置信区间或上下界。

64/128 分块未通过既有数值验收，不能作为默认配置。本轮没有启动正式 seed 或研究训练；性能测量的物理步为 0。候选诊断计时和 current 独立 trace 均已完成，其结果不改变上述默认值及验收结论。

## 范围与基线

正确性基线：`4fa6e650fe57fbdbfba4d7962ed52776ed9bfdf0`。本轮承接工作区原有六个未提交文件，没有切换分支或回退；这些相关修改连同双奖励、入口和版本检查一起纳入正确性提交。基线包内 111 项、仓库入口 13 项测试通过。完整验证调用了 20,639 个物理步（含原实现对照）；入口测试为 0 步。

物理奖励与学习奖励的定义见 [正确性契约](CORRECTNESS_AND_PERFORMANCE_20260927.md)。所有性能比较使用同一份配置、物理参数和保存的模型/优化器初态，不比较不同数学目标。没有启动任何正式 seed、研究训练日程或后台长训。

## 固定工作负载

`tools/profile_equivalent.py` 生成两条真正独立的 900 秒轨迹：固定外部到达 tape、独立布局/负载种子、hover + local/equal 脚本，共 18,000 物理步。原始轨迹保存在 `outputs/equivalent_20260927/fixtures/workloads.pkl`。之后的 GPU 性能测量不推进物理环境。

生成器最初遗漏了显式 gamma 传递，已在任何学习测速前修正；使用保存的下层 reward 以 gamma=0.99 重算上层学习 reward 并重新执行 Episode.seal，不重跑仿真、不改任何物理成本/队列。迁移前后文件哈希和原件均保留，见 `outputs/equivalent_20260927/reward_repair/reward_repair.json` 及 fixture 清单。

SAC 128 样本覆盖 120/600/1800 帧及实际队列负载分位；主比较使用两条真实独立轨迹。四个 replay 对象的压力案例明确复制自这两条轨迹，仅测四组独立计算图的负载。PPO 正式尺寸为 5×180=900 条，来源序列为 `[0,1,0,1,0]`，其行为统计来自同一临时模型；这是结构压力样本，不是五条独立科研轨迹或 on-policy 学习结果。

报告/历史/边/节点/FIFO 大小逐案例保存。普通和积压指保存轨迹的经验负载分位，不是假设任务参数改变。短采集开销额外测量 stored/streamed 各 60 秒，共 1,200 物理步；两者成本均为 627.8011450432558。其时长向 900 秒外推有队列规模风险，不能冒称实测完整窗口。

## 测量方法与限制

墙钟计时与 profiler trace 分开，预热后保留少量原始重复值、中位数、RAM 采样峰值及 CUDA allocated/reserved 峰值。少量重复计算的分位数不作为可靠尾延迟。CPU/GPU 包含时间、prepare/build 子区间不能相加。

测量使用已有 PyTorch 2.7.0.dev20250227+cu128 与 RTX 5060，确定性开启、TF32 关闭。当前 CUPTI 报 `CUPTI_ERROR_INVALID_DEVICE`；CPU trace 有效，但没有可用 CUDA kernel 时间，不能据此声称 scatter、kernel 启动或 GPU 占用是主因。没有安装或升级依赖。初次 trace 后的墙钟样本全部排除并在新进程重测，原记录保留于 `times_before_trace_separation.json`；正式比较不会把 profiler 造成的内存常驻混入后续计时。

每例预热一次、计时三次，每次恢复相同模型、Adam 状态、计数和随机流；初态的 Adam slots 由一次有界 critic/actor 更新及两条 PPO 样本初始化。该初始化不推进物理环境，也不是科研训练。峰值 RSS 每 10 ms 采样，可能漏掉短暂峰值；CUDA reserved 会受此前分配历史影响。

普通/积压推理与缓存重建分别测量。冻结评价采用真实报告序列回放，量取推理、汇总和写盘；它不执行新动作，不能代替闭环评价效果或完整物理耗时。checkpoint 的小回放测量外推到 55 个 episode 时，字节和耗时均是估计。

汇总工具要求 baseline/current 的 `completion.json` 完成清单与 `times.json` 一致，fixture 哈希、初态哈希、配置、所有共享 case 的 workload 和 `counts_after` 完全一致。生产源码身份分别记录，允许被比较的生产实现不同；PPO 停止或更新次数不同会直接拒绝比较。当前汇总为 61 个共享 case、0 个缺失 case。完整原始三次计时、图规模、来源映射、内存和计数见[性能比较 JSON] — 历史附件（未包含）：`evidence/equivalent_20260927/performance/performance_comparison.json`及[完整表](evidence/equivalent_20260927/performance/performance_comparison.md)。

baseline 的独立 CPU trace 给出 `data/fixed_graph_features` 包含时间 3.781 秒、`autograd/backward` 2.541 秒，支持优先检查特征构造的工程开销。这些是嵌套区间，不能相加或转写成 GPU 时间占比。`aten::copy_` / `_to_copy` 的 CPU 区间也不等于 CUDA 传输时间。详见[critic trace 摘要](evidence/equivalent_20260927/performance/baseline_critic_trace_summary.md)和[推理 trace 摘要](evidence/equivalent_20260927/performance/baseline_inference_trace_summary.md)；没有可用 kernel 数据，因此未将瓶颈归因于确定性 scatter。

current 的独立 trace 已完成，critic 的 CPU 包含区间如下。特征构造区间降低，与标量检查改动的方向一致；它不是一次只改变该函数的独立归因实验，不能将整套墙钟收益全部分配给该函数。

| CPU 嵌套区间 / 秒 | baseline trace | current trace |
| --- | ---: | ---: |
| 固定图特征构造 | 3.780930 | 1.452558 |
| 图批处理与数据准备 | 4.360825 | 2.341822 |
| autograd backward | 2.540982 | 2.520483 |

缓存推理 trace 中，固定图特征构造为 0.004154→0.002051 秒，整个推理标注为 0.040565→0.039216 秒；这些带 profiler 的区间不替代下表无 profiler 的墙钟样本。两份 current trace 的 kernel/memcpy/memset 事件仍均为 **0**，CUDA activity 不可用。详见 [current critic 摘要](evidence/equivalent_20260927/performance/current_critic_trace_summary.md)及[current 推理摘要](evidence/equivalent_20260927/performance/current_inference_trace_summary.md)。

## 61 组实测中的主要结果

以下均为预热 1 次、独立恢复初态后测量 3 次的中位数，单位为秒；倍数为 baseline/current，大于 1 表示 current 更快。SAC 每次 128 个样本，`e2` 明确是两个真实独立轨迹对象。`h120/h600/h1800` 是训练使用的历史上界，不能将旧短窗 profile 的 3.29 秒直接套用到所有长历史 fit。

| 实际 case | baseline / 秒 | current / 秒 | 倍数 |
| --- | ---: | ---: | ---: |
| critic.e2.h120.normal | 3.292224 | 3.071952 | 1.072× |
| critic.e2.h120.backlog | 3.215042 | 3.109871 | 1.034× |
| critic.e2.h600.normal | 5.540605 | 4.455338 | 1.244× |
| critic.e2.h600.backlog | 5.868996 | 4.758194 | 1.233× |
| critic.e2.h1800.normal | 6.233273 | 4.693965 | 1.328× |
| critic.e2.h1800.backlog | 7.513558 | 5.533634 | 1.358× |
| actor.e2.h1800.normal | 3.375751 | 2.628046 | 1.285× |
| actor.e2.h1800.backlog | 4.106361 | 3.022665 | 1.359× |
| ppo.rows900 | 82.719713 | 79.793618 | 1.037× |
| inference.lower.h1800.normal.cached | 0.033822 | 0.028788 | 1.175× |
| inference.lower.h1800.normal.rebuild | 0.036684 | 0.036324 | 1.010× |
| inference.lower.h1800.backlog.cached | 0.033657 | 0.030231 | 1.113× |
| inference.lower.h1800.backlog.rebuild | 0.041646 | 0.039759 | 1.047× |
| inference.lower.h1800.after_actor_update | 0.032507 | 0.030250 | 1.075× |
| checkpoint.save0 | 0.599161 | 0.590668 | 1.014× |
| checkpoint.save2 | 1.332744 | 1.346935 | 0.989× |
| checkpoint.load2 | 0.679149 | 0.594998 | 1.141× |
| evaluation.replay900s | 99.534243 | 78.627407 | 1.266× |

普通/积压推理从真实报告的 40%–60% / 90%–100% 队列分位选取。在 `h1800` 上界内，两者实际 history cursor 分别为 **1795 / 1769**，图分别为 **78 节点、356 边 / 117 节点、679 边**；没有伪造为恰好 1800 帧的同一状态。`after_actor_update` 使用另一份真实 1800 帧报告，先填充缓存，再在 setup 中执行真实 actor 更新，使参数版本失效；只计下一次推理。不同报告行的耗时差不能单独解释为缓存成本。缓存预热后恢复随机流，使同一 case 的当前动作抽样可比。

PPO 是完整 900 宏行的结构工作负载，6 个 epoch、minibatch 150。两端扣除初态计数后均为 **12 次 actor 尝试、6 次接受、36 次 value 更新、1 次 rollback**。82.720→79.794 秒包含此次真实停止行为，不能视为所有未来 PPO 批次的固定价格。评价回放包含 1800 次 lower、181 次 upper、指标汇总及一次 JSON 写入；不重复加计这些子项，也不把它称作闭环评价实测。

并非所有 case 都改善：`checkpoint.save2` 的中位数略升。其 0/2 表示回放 episode 占用，pending PPO 对象均为 0；`load2` 包含校验、CPU 反序列化，不含恢复到 GPU。三次重复不能支持尾延迟或稳定 I/O 改善的结论。

| 全部 61 case 中观察到的最大值 / MiB | baseline | current |
| --- | ---: | ---: |
| 进程 RSS | 3362.55 | 3426.78 |
| CUDA allocated | 1167.12 | 1168.24 |
| CUDA reserved | 1410.00 | 1392.00 |

这是各 case、各次重复的整体观察最大值，包含驻留模型和分配器历史，不是操作增量，也不证明完整训练的内存上界。本轮不声称显著降低内存。baseline 的完整图规模与 FIFO 附表见 [workload_summary.md](evidence/equivalent_20260927/performance/workload_summary.md) 和 [JSON](evidence/equivalent_20260927/performance/workload_summary.json)：e2 的 h1800 样本图覆盖 64–660 个节点、256–4424 条边，普通/积压 CPU FIFO 最大长度分别为 171/181。报告条目数没有被当作独立 episode 数；e4 的来源为 `[0,1,0,1]`，PPO 五对象来源为 `[0,1,0,1,0]`，都只有两条真实轨迹。

## 验证记录

首轮 CUDA 分块测试保存于 `outputs/equivalent_20260927/optimization_cuda`，后续完整诊断在 `optimization_cuda_diagnostics`。target、抽样输入/输出及随机流最大误差均为 0，但 64/128 的更新后参数分别出现约 `9.62e-5` / `3.46e-5` 差异，超过既有参数容差 `atol=3e-5, rtol=2e-5`。诊断显示最大差异处为接近零的梯度，Adam 的 `epsilon=1e-8` 放大了首步更新差异；这解释原因，不豁免验收。

因此默认 critic 分块保留 16，actor 与 target 抽样也保持 16。64/128 仅留作显式实验审计，`V2_AUDIT_CANDIDATE_CHUNKS=1` 会重新执行并保留原失败门限，不能把常规测试中的明确跳过写成候选通过。

性能阶段的生产修改仅在 `src/hrl_v2/sac.py` 与 `src/hrl_v2/features.py`；对应测试为 `test_optimizer_reuse.py`、`test_features_scalar.py`、冻结的 `_sac_reference_4fa6e65.py` 和日程计数测试 `test_throughput_budget.py`。

采用的工程改动限于两处：先按原 16 个样本槽位和真实终止 mask 生成全部 target，再独立分块计算当前 Q；以及标量特征检查跳过两次数组归约，仍使用完全相同的 float64 转换、除法和 `np.log1p`。数组路径、网络、分布后端、归一化公式、缓存和更新日程保持原样。完整图特征必须逐字节相同；默认 SAC 对照经过 hash 固定的 `4fa6e65` 原始 SAC 控制流，而不是另写目标公式模拟正确答案。

最终验证结果如下；测试中的 `skip` 明确表示未执行候选验收，不能计为候选通过。

| 已保存执行目录 | 结果 | 物理步 |
| --- | --- | ---: |
| `final_package` | 119 项：118 通过、1 明确 skip；0 failure/error | 20,640 |
| `final_cuda` | 7 项：6 通过、1 明确 skip；0 failure/error | 5 |
| `final_entry_budget_fix` | 17 项全部通过 | 0 |
| `final_candidate_audit` | 7 个测试方法；64/128 两个子测试失败，0 error，整套审计失败 | 5 |

新增 128 样本默认 critic=16 对照 `4fa6e65` 时，target、抽样、随机流、loss、未裁剪/裁剪梯度、Q1/Q2 参数和 Adam 状态的最大绝对误差均为 **0**。这项零误差结论只针对该明确的默认 critic 对照，不扩展为所有历史优化检查均逐位相等。图 single-versus-batch 最大误差为 **4.76837158203125e-6**，permutation 误差为 **0**，沿用原 **2e-5** 容差；完整原始图特征与 FIFO 的标量优化测试逐字节相等。

候选最终 Q2 参数误差分别为 **9.616697207093239e-5（64）**和 **3.46451997756958e-5（128）**，保留 `atol=3e-5, rtol=2e-5`，两者都被拒绝。参数与裁剪后梯度容差没有放宽，也没有因计时更快而启用。未裁剪梯度的绝对容差按参考联合裁剪范数换算为同一尺度，并另外要求联合相对 L2≤1e-5；这不豁免参数门限，原始失败记录仍保留。完整结果见 `outputs/equivalent_20260927/final_cuda/results.json`、`final_candidate_audit/results.json`；包内和入口的文本报告及各自 `execution.json` 同目录保存。

64/128 的诊断计时也已分别完成 6 个 e2 case；共享 workload 与更新计数一致。下表只报告测得耗时，**不是数值验收通过表**，两个候选均不采用。actor、target 的分块仍为 16。

| critic e2 case | current 16 / 秒 | 候选 64 / 秒 | 64 相对 16 | 候选 128 / 秒 | 128 相对 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| h120.normal | 3.071952 | 1.995108 | 1.540× | 1.938160 | 1.585× |
| h120.backlog | 3.109871 | 2.118480 | 1.468× | 2.043413 | 1.522× |
| h600.normal | 4.455338 | 3.473487 | 1.283× | 3.725618 | 1.196× |
| h600.backlog | 4.758194 | 3.971666 | 1.198× | 4.317614 | 1.102× |
| h1800.normal | 4.693965 | 3.899579 | 1.204× | 4.104132 | 1.144× |
| h1800.backlog | 5.533634 | 4.758168 | 1.163× | 5.004335 | 1.106× |

候选原始记录位于 `outputs/equivalent_20260927/candidate64`、`candidate128` 的 `times.json`、`metadata.json`、`completion.json`。诊断计时更快与数值不满足容差可以同时成立，不能据此自动改默认值。

## 本轮物理执行账本

逐个核对已保存的 `execution.json` 和两个 `physical_budget.json`，合计 **60,499 个实际物理步**，不是保留额度之和：

| 来源 | 实际物理步 |
| --- | ---: |
| baseline 完整包内验证 | 20,639 |
| final 完整包内验证 | 20,640 |
| `optimization_cuda`、`optimization_cuda_diagnostics`、`final_candidate_audit`、`final_cuda`，各 5 步 | 20 |
| 两条 900 秒固定轨迹 | 18,000 |
| stored / streamed 各 60 秒采集开销 | 1,200 |
| 入口验证、奖励代数迁移、GPU 性能 run、汇总和预算工具 | 0 |
| **合计** | **60,499** |

GPU 性能和数值审计中的临时 optimizer 更新不等于研究训练日程；所有执行记录的 `research_training_episodes` 均为 0。固定轨迹只生成一次；奖励迁移、后续 baseline/current 和候选测速复用其原始报告及到达历史。

## 日程预算

预算工具逐窗口枚举真实调用顺序：600 窗口、36,000 个 SAC 触发，其中空回放跳过 990 个；35,010 个有效 event、70,020 次 critic fit、17,010 次 actor fit、40 次正式 PPO 批次。包括 11 次回放清池、actor warmup、16,632 次窗口内部 actor 更新后的缓存重建，以及末帧更新后不再推理的 378 次。

上层实际采样含 cutoff 后继：训练 upper 54,300 次、value 36,200 次；开发评价 156 窗口（72 moving/84 hover），upper 13,032 次。checkpoint 共 608 次；累计序列化 replay episode 记录 22,675 份、pending PPO 记录 400 份。预算已采用本轮代表性测量系数，未知项单列。

预算计算已经完成，分别见 [baseline 预算](evidence/equivalent_20260927/performance/budget_baseline.md) 和 [current 预算](evidence/equivalent_20260927/performance/budget_current.md)。完成计算不表示所有成本均已定价；两份 JSON 都明确 `complete_seed_wall_time_measured=false`、训练项 `all_listed_components_priced=false`，总墙钟上下界仍为未知。

| 同一单 seed 范围 | baseline 条件小时 | current 条件小时 |
| --- | ---: | ---: |
| 训练已定价小计 | 150.97–188.52 | 117.64–156.26 |
| 开发评价主模型 | 3.52–3.96 | 3.05–3.51 |
| **训练 + 开发评价主模型** | **154.48–192.48** | **120.69–159.76** |
| 单独的 hover 按 moving 计价情景：训练 + 开发 | 155.67–193.28 | 121.45–160.07 |
| 可选最终 primary test，单列 | 3.69–4.12 | 3.18–3.62 |

主模型两个端点分别减少约 33.79 / 32.72 小时，即约 21.9% / 17.0%；这是条件端点比较，不是统计收益区间。单独的 hover-as-moving 情景包含额外 upper 工作，不能与主模型混合，也不能称为纯 hover 实测或严格上界。可选最终测试、额外 deterministic panel、更多 checkpoint 和多 seed 总量均不混入训练加开发的主结果。

SAC 价格按每个窗口的实际回放占用，计算四次有放回 episode 抽样产生 1–4 个独立对象的概率；3 对象用已测 2/4 对象价格插值，明确不是独立实测。采用长历史普通/积压样本的观察范围；h120/h600 仅说明长度敏感性。样本只来自两条固定脚本轨迹，不能代表训练后策略的队列分布。

保留的未定价与外推项包括：初始化/每窗 reset、5 次初始参数哈希、部分完整文件哈希和分配、GC/清理、文件系统争用；60→900 秒物理/报告规模变化；checkpoint 0/2→55 记录的非线性；以及未来 PPO KL/rollback 和队列负载变化。PPO 本例 40 次批次价格只固定了已测的停止行为，理论 actor 尝试次数仍可能为 240–1440，value 更新为 1440。没有用任意冗余系数把未知项包装为已测时间。

current 的完整 source-0 moving 报告回放为 78.50–78.66 秒，而 source-1 固定切片推理模型折算为 50.31–71.70 秒/窗，差比约 1.09–1.56。它反映轨迹、图规模、顺序、汇总与写盘边界差异，未作为训练乘数，也未与已经计价的推理重复相加。旧的 95 小时估计被本轮同规模系数替代，不再称为下限。

## 复现命令

以下从仓库根目录运行，使用已保存的真实 fixture 和共同初态，输出到新的 `outputs/reproduce_equivalent`；不会重新生成物理轨迹或启动正式训练。实际本轮启动参数保存在 baseline `metadata.json.command` 与 current `wall_metadata.json.command` 中；current 的最终 `metadata.json.command` 记录后续独立 trace 调用。墙钟计时和 trace 使用不同进程，先完成无 trace 的计时。

```powershell
$perfPython = (Resolve-Path .venv/Scripts/python.exe).Path
$perfFixture = 'outputs/equivalent_20260927/fixtures'
$perfInitial = 'outputs/equivalent_20260927/baseline/initial_state.pt'
$perfCases = @('critic.e2.*','critic.e1.h1800.*','critic.e4.h1800.*',
  'actor.e1.h1800.*','actor.e2.h1800.*','actor.e4.h1800.*','inference.*',
  'ppo.*','checkpoint.*','evaluation.*','io.*','source.*','replay.*','target.*')

& $perfPython tools/profile_equivalent.py run `
  --source-dir outputs/equivalent_20260927/baseline_source/src `
  --fixtures $perfFixture --initial-state $perfInitial `
  --output outputs/reproduce_equivalent/baseline --label baseline `
  --critic-microbatch 16 --cases $perfCases --trace-cases

& $perfPython tools/profile_equivalent.py run --source-dir src `
  --fixtures $perfFixture --initial-state $perfInitial `
  --output outputs/reproduce_equivalent/current --label current `
  --critic-microbatch 16 --cases $perfCases --trace-cases

& $perfPython tools/summarize_performance.py `
  --baseline outputs/reproduce_equivalent/baseline `
  --current outputs/reproduce_equivalent/current `
  --output outputs/reproduce_equivalent/performance_comparison.json

& $perfPython tools/estimate_seed_budget.py `
  --profile outputs/reproduce_equivalent/current `
  --fixture-manifest outputs/equivalent_20260927/fixtures/manifest.json `
  --collection-overhead outputs/equivalent_20260927/collection_overhead/collection_overhead.json `
  --output outputs/reproduce_equivalent/budget_current.json
```

baseline 预算复现时，将最后命令的 `--profile` 和 `--output` 改为 baseline 对应路径；日程使用同一当前协议，只有实测系数不同。如需只重建本次交付摘要，直接将 profile 和 comparison 的输入指向既有 `outputs/equivalent_20260927/baseline`、`current` 即可，不运行测量。

验证复现入口为 `tools/validate_execution.py --suite package|entry|cuda --output <新目录>`，它会真实执行测试并记录物理步，不能归类为零步整理。显式候选审计还需设置 `V2_AUDIT_CANDIDATE_CHUNKS=1`，预期保留当前失败结果；文档不会自动执行这些命令。

## 交付与证据导航

轻量归档目录为 [docs/evidence/equivalent_20260927/performance](evidence/equivalent_20260927/performance/)，[MANIFEST.json](evidence/equivalent_20260927/performance/MANIFEST.json) 记录文件哈希、实际物理步与生产源码差异。大型原始 fixture、初态、checkpoint 与 trace 留在本机 `outputs/equivalent_20260927`，通过哈希关联；此报告不嵌入巨大 trace，也不写尚未形成的提交 SHA。

| 内容 | 证据入口 |
| --- | --- |
| 正确性基线与数学定义 | [正确性契约](CORRECTNESS_AND_PERFORMANCE_20260927.md)，基线 `4fa6e650fe57fbdbfba4d7962ed52776ed9bfdf0` |
| 全部 61 case 原始重复、计数和身份 | [performance_comparison.json] — 历史附件（未包含）：`evidence/equivalent_20260927/performance/performance_comparison.json`、[可读表](evidence/equivalent_20260927/performance/performance_comparison.md)；原件为 baseline/current 的 `times.json`、`metadata.json`、`completion.json` |
| 图规模、FIFO、来源映射和内存 | [workload_summary.json](evidence/equivalent_20260927/performance/workload_summary.json)、[附表](evidence/equivalent_20260927/performance/workload_summary.md) |
| 单 seed 逐窗口调用数与条件预算 | [budget_baseline.json] — 历史附件（未包含）：`evidence/equivalent_20260927/performance/budget_baseline.json`、[budget_current.json] — 历史附件（未包含）：`evidence/equivalent_20260927/performance/budget_current.json` |
| 已完成 CPU trace 结论及 CUDA 缺失标记 | [baseline critic 摘要](evidence/equivalent_20260927/performance/baseline_critic_trace_summary.json)、[baseline inference 摘要](evidence/equivalent_20260927/performance/baseline_inference_trace_summary.json)、[current critic 摘要](evidence/equivalent_20260927/performance/current_critic_trace_summary.json)、[current inference 摘要](evidence/equivalent_20260927/performance/current_inference_trace_summary.json) |
| 数值通过与候选失败原始证据 | `outputs/equivalent_20260927/final_package`、`final_cuda`、`final_entry_budget_fix`、`final_candidate_audit` 的结果和文本报告 |
| 实际物理步与异常迁移 | 各执行目录的 `execution.json`；`fixtures/physical_budget.json`、`collection_overhead/physical_budget.json`；`reward_repair/reward_repair.json` 与保留的迁移前原件 |
| 可复用工具 | `tools/profile_equivalent.py`、`summarize_performance.py`、`estimate_seed_budget.py`、`summarize_trace.py`、`validate_execution.py` |

## 完成状态与训练入口

baseline/current 的 61 case 比较、64/128 候选的各 6 case 诊断、独立 baseline/current CPU trace、最终验证及预算汇总均已完成。CUPTI 不可用和候选数值失败是已经记录的结果，不作为待填空白处理。

实际 CLI `check` 与 `dry-run` 结果保存在 `outputs/equivalent_20260927/final_status/check.json`、`dry_run.json`：`validation_ready=true`、`research_protocol_confirmed=true`、`training_ready=true`；`training_authorized=false`、`training_started=false`，dry-run 的 `physical_steps=0`、两者 `updates=0`。入口已具备验证就绪条件，本轮没有授权或启动研究训练。
