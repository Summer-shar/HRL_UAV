# 候选无线模型：固定等效噪声谱密度与每机 U2U 发射预算

日期：2026-09-09。状态：已实现并通过短程功能测试，未做性能评估。

## 1. 基准、范围与来源

实际分支 `codex/u2u-model`，修改前和验收时 HEAD 均为
`07a727d8189be4f73c0218fca47603d8cbf6956b`。
没有 checkout、reset、clean、提交或推送。

这是显式启用的模型扩展，不表示旧模型错误。旧配置、旧结果和历史参数均保留。
部署期 CPU 继续计算与无线模式是两个独立选项；本次不改变前者默认值。
没有修改任务、负载、CPU、优先级、关联、动作、时间步、网络、更新流程或奖励公式。
通信服务率与发送能量改变，会改变环境转移、奖励中的实测能量和后继观测数值；
这不等于修改奖励目标，也不意味着已有策略适用于候选模型。

本次修改：

| 文件 | 改动 |
|---|---|
| `hrl_uav/env.py` | 模式和参数校验；参考质量接口；候选上行/U2U公式；统一分配、服务与能量；实际资源 info |
| `hrl_uav/mdp.py` | 观测调用明确的参考质量接口，维数和归一化不变 |
| `hrl_uav/bandwidth.py` | 复用 MeteredMDP，候选模式追加链路功率、噪声、时长和能量统计 |
| `tests/test_wireless_model.py` | 新增聚焦的功能测试，复用现有人工任务和 CRN 记录工具 |
| `PROJECT_FACTS.md` | 只追加候选记录，不改历史事实 |
| 本文件 | 定义、来源、功能验收及未验证边界 |

保留的既有未提交文件是 `hrl_uav/scene_prescreen.py`、
`scripts/run_scene_prescreen.py`、`tests/test_scene_prescreen.py`。
它们的 SHA256 修改前后相同。两个预筛选结果目录及 configs 共 169 个文件逐项 SHA256 相同。
本轮没有向 results 写入新性能结果。历史 588.47 ms、601.43 ms 仍属旧无线模式，不能当作本候选的成绩。

## 2. 启用方式与真实配置链

入口 `scripts/run.py:main` 读取 JSON，沿原调用链把配置传给 `QueueMDP`；
`QueueMDP.__init__` 将 `config["environment"]` 原样展开给 `UAV_MEC_Env`。
因此不需要新增命令行开关或修改训练入口。本轮没有执行该入口。

以下只是未来新配置的 environment 增量，不替换已有 environment 的其他字段；本轮未修改任何 JSON：

```json
{
  "wireless_model": "psd_per_uav_budget",
  "noise_psd_w_per_hz": 1.95e-19,
  "u2u_tx_budget_w": 1.0,
  "u2u_reference_bandwidth_hz": 512820.5128205128,
  "u2u_reference_power_w": 1.0
}
```

缺少 `wireless_model` 或显式设为 `legacy_fixed_noise`，均走旧无线定义。
四个新增数值参数的构造默认值与上面一致，但只有显式启用候选模式才用于无线计算；
未来实验应把它们显式写入有效配置。它们必须是有限、严格为正的数值标量，
字符串、布尔值、数组、零、负值、NaN、无穷值明确报错。
候选模式还校验带宽、上行功率、增益参数及派生噪声的有效性。

候选模式仅支持活跃有向链路均分总带宽。
`u2u_link_bandwidth_matrix_mhz` 非 None 时明确报不支持，包括全零自定义矩阵；旧模式仍兼容原固定矩阵。
旧 `noise_power` 和 `p_u2u_power`/`p_u2u` 不重命名、不改义：旧模式继续使用它们；
候选模式的噪声和 U2U 功率改用独立的新参数，不把旧值偷偷重新解释为谱密度或每机预算。

## 3. 两套无线定义

旧模式保留 `R=b*log2(1+p*g/noise_power)`，上行固定每用户带宽；
U2U 按原规则分带宽，发送能量为旧每链路功率 `p_u2u` 乘实际发送时长。
旧位置谱效率缓存和自定义带宽矩阵语义不变。

候选模式在每个 0.1 s 物理步的原 U2U 分配位置读取转发队列：

```text
A[i,j] = (i != j) and (Q_tx_high[i,j] + Q_tx_low[i,j] > 1e-12)
L = sum(A); d[i] = sum_j A[i,j]
active: b[i,j] = B_u2u_total/L; p[i,j] = u2u_tx_budget_w/d[i]
        noise[i,j] = N_eff*b[i,j]
        R[i,j] = b[i,j]*log2(1 + p[i,j]*g[i,j]/noise[i,j])
inactive/diagonal: b = p = noise = R = 0
capacity_mbit = R*micro_dt/1e6
t_tx = actual_sent_mbit*1e6/R; E_tx = p*t_tx
```

索引 i、j 分别是发送和接收 UAV；g 为原距离定义下的信道增益；
b、noise、R、p 的单位依次是 Hz、W、bit/s、W；时间为秒，能量为 J。
非空阈值、CPU 在 U2U 之前的服务顺序、完整分片转发后入计算队列以及父任务完成规则均不变。
活跃集合包含历史积压，与本帧动作是否产生新转发无关。

本物理步内只分配一次：高优先服务，低优使用剩余容量；链路提前清空不在步内重新分配。
下一物理步重新读取队列。同源两条链路各分 0.5 W；无外发的 UAV 分配 0 W。
空集合不传输、不计发送能量。短队列仅按实际时长付能量；若服务反推时长超过
`micro_dt + 1e-9 s` 则报错，不裁剪掩盖超量服务。
链路能量按源求和得到每机和系统发送能量，候选路径不再乘旧每链路 `p_u2u`。

候选上行采用同一个固定 N_eff：

```text
b_user = B_gt_total/M
R_user = b_user*log2(1 + p_gt*g_user/(N_eff*b_user))
```

M 是地面用户数。用户关联、发射功率、等分带宽、上传 FIFO 和信道增益不变。
参考参数的来源是 `b_ref=20e6/39 Hz`、`N_eff=1e-13/b_ref=1.95e-19 W/Hz`。
这是旧上行参考条件的等效换算，不是实测接收机参数，也不是理想热噪声谱密度。
运行时 N_eff 固定，不随 M、总带宽或链路数重新校准，不按历史时延反调。
每机 1 W 仅是 U2U 射频发射预算，不包括 CPU、推进或其他整机功耗。

## 4. 参考观测与实际资源接口

- `compute_u2u_reference_quality()`：候选下返回 `log2(1+P_ref*g/(N_eff*b_ref))`，对角为零；旧模式返回旧谱效率。
- `compute_u2u_spectral_efficiency()`：保留兼容入口，候选模式明确返回参考量，不是实际效率。
- `allocate_u2u_resources(active_link_mask, uav_coords=None)`：仅候选模式；返回实际带宽、功率、噪声和速率矩阵。
- `compute_u2u_rate_bps(...)`：候选模式委托真实资源分配，仍支持 `return_bandwidth=True`；不传 mask 的查询仍假设所有非对角链路活跃，不能把这种查询当作实际服务记录。
- `micro_step`：始终显式传入当步队列活跃 mask。部署暂停分支不调用资源分配，不产生实际无线服务；候选的资源 info 全为零。

候选模式不缓存实际谱效率或资源分配；每次计算几何增益并按当步条件重新分配。
参考带宽和参考功率只用于参考观测，不用于替代真实传输速率。
9 维特征的展平顺序和 `log1p` 归一化不变，宏观/微观布局仍为 67/68 维。
未增加带宽观测，未开启 routing_ready，未实现跨带宽自适应策略。
即使维数相同，旧权重也不能称为新无线模型上已训练的策略。

物理步 info 的实际量：
`u2u_link_bandwidth_hz`、`u2u_link_power_w`、`u2u_link_noise_w`、`u2u_rate_bps`、
`tx_service_matrix_mbits`、`u2u_active_time_s`、`u2u_link_energy_j`。
另有 `wireless_model` 和 `u2u_quality_semantics` 区分模式与参考语义。

MeteredMDP 复用原实际吞吐和容量统计，候选模式追加帧、每机及链路发送能量；
链路记录追加分配功率/噪声的物理步求和与实际发送时间。
`summarize_links` 的 `power_w_when_active`、`noise_w_when_active` 是活跃步条件均值，
不是整个窗口的平均功率；`summarize_meter.tx_power_w_wall` 才是发送能量除以完整统计时长。
旧模式不追加这些字段，原统计输出逐值兼容。

## 5. 理想化并发与未验证边界

地面上行与 U2U 使用不同频段；活跃 U2U 有向链路使用互不重叠的有效频率资源。
假设 UAV 支持频率隔离的多通道并发收发，忽略残余自干扰、保护带和控制开销。
`B_u2u_total` 是可用数据传输有效带宽预算。
这是模型假设，不是已验证硬件能力；未增加半双工 TDMA、同频全双工干扰模型、接收机、
射频电路或控制信令能量。部署期通信暂停规则不因候选接口而取消。

## 6. 独立标量算例

三机位置为 (100,100)、(400,100)、(100,400) m。
下表统一列 0→1 的 300 m 链路，g=1e-6/300²；不是场景性能结果。
双链路情形为 0→1 和 0→2，六链路情形为全部非对角链路。
标准答案通过独立 `math.log2` 标量公式计算，不调用待测分配函数。

| 总带宽 MHz | L | 本源外发数 | b Hz | p W | noise W | R Mbps | 满发0.1s能量 J | 仅发0.05Mbit能量 J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 不发送 |
| 3 | 1 | 1 | 3000000 | 1 | 5.85e-13 | 12.964345461 | 0.100 | 0.003856732 |
| 3 | 2 | 2 | 1500000 | 0.5 | 2.925e-13 | 6.482172730 | 0.050 | 0.003856732 |
| 3 | 6 | 2 | 500000 | 0.5 | 9.75e-14 | 2.928742423 | 0.050 | 0.008536087 |
| 6 | 1 | 1 | 6000000 | 1 | 1.17e-12 | 20.351163941 | 0.100 | 0.002456862 |
| 6 | 2 | 2 | 3000000 | 0.5 | 5.85e-13 | 10.175581971 | 0.050 | 0.002456862 |
| 6 | 6 | 2 | 1000000 | 0.5 | 1.95e-13 | 4.882155311 | 0.050 | 0.005120689 |

双链路同源满发时两条合计 0.1 J，不是各 0.1 J。
本轮人工队列还直接验证：双链路同源，0→1 用满容量、0→2 只用5%容量时，
链路能量为 0.05 J 与 0.0025 J，系统 U2U 能量 0.0525 J；不把提前空闲的资源回分。

## 7. 实际执行的功能验证

```powershell
.venv/Scripts/python.exe -m unittest tests.test_wireless_model -v
.venv/Scripts/python.exe -m unittest tests.test_wireless_model tests.test_deployment_compute tests.test_tcom_communication_model tests.test_bandwidth -v
.venv/Scripts/python.exe -m unittest tests.test_wireless_model tests.test_deployment_compute tests.test_tcom_communication_model tests.test_bandwidth -q
```

首轮新增文件 7 项通过；联合检查共 31 项通过。最后一次联合检查为 31 项、5.033 s、OK
（unittest 报告时间，不含解释器导入）。未实例化学习网络、加载权重或执行反向传播。
既有 rollout 边界测试输出的 `fixed-action-test` 使用内存替身和空更新，不是训练。

新增测试与覆盖：

| 测试方法（WirelessModelTests） | 实际检查 |
|---|---|
| `test_legacy_short_regression_against_pinned_source_both_cpu_modes` | 用 git show 读取基准 env/mdp/bandwidth，在内存执行；CPU开/关各做20s，同种子同动作逐值比较旧源码、缺省模式、显式旧模式的全部帧、原始状态、两层观测、奖励、能量、服务量及统计器输出 |
| `test_mode_validation_and_legacy_custom_bandwidth` | 未知模式、非数值/非有限/非正参数、派生溢出报错，候选拒绝固定矩阵，旧模式保持矩阵兼容；CPU默认关闭 |
| `test_scalar_allocations_and_refresh_at_fixed_geometry` | 空/单/同源双/六链路，3/6MHz独立标量核算，逐源功率和全网带宽预算，活跃集合变化后刷新、对角为零；参考质量不随带宽/活跃集合改变 |
| `test_uplink_reference_and_fixed_psd_when_user_count_or_bandwidth_changes` | 20MHz/39用户与旧上行容差一致；10MHz/39与20MHz/42用户按固定N_eff重新算速率，不重标定 |
| `test_backlog_priority_shared_budget_and_no_intra_step_reallocation` | 当前动作全本地但历史转发继续服务；高低优共用容量；同源分功率；提前清空不回分；下一步更新 |
| `test_short_send_energy_and_full_fragment_admission_before_parent_completion` | 满步后剩余20%容量的下一步发送仅计0.02s/0.02J；完整收到才入计算；父任务等待全部分片完成；无发送能量为零 |
| `test_real_rollout_meter_energy_observations_pause_and_crn` | CPU开/关各用真实固定规则rollout跑3个宏观周期；候选/旧模式到达、逐步运动、几何、负载和调度指纹配对；67/68维不变；每次54个路由决策、270个服务物理步、30个暂停物理步；分配/发送/能量/奖励逐项复算，统计只读且守恒 |

没有运行 scripts/run.py 训练/评估、900s比较、54组扫描、额外基线或参数选优。
上述只验证实现与协议一致，不证明吞吐可行、长期队列稳定、性能更好、模型已训练或硬件可部署。
三周期自然流不足以覆盖完整8–12周期驻留和长期负载切换后果；未在候选无线模式下重新做完整尾部经验测试。
既有部署CPU的尾部边界测试继续通过，跨更长窗口和候选组合的后续检查应另行限定范围。

另有与本轮扩展无关的历史文字问题：PROJECT_FACTS 原服务顺序概述写“路由、上传、计算、U2U”，
而实际 micro_step 先取出此前已上传任务，推进本步上传/运动，再路由此前取出的任务，随后计算和 U2U。
本步刚完成上传仍不能立即路由。本轮不改这些生产步骤，也不覆盖原历史概述；这里单独记录这一措辞差异。

## 8. 指纹与后续条件

修改前 SHA256（本轮快照，不覆盖 PROJECT_FACTS 早期指纹）：

```text
hrl_uav/env.py       30335cfb97e37a86312cf8cb45f3a9335cb095b31de4ec36969ac3f728e060d2
hrl_uav/mdp.py       7c4873f82e392d9203e81c1a8929bd3d6618ebf21a1c24f10cee3a18cfaccf98
hrl_uav/bandwidth.py 7123260cf1fc91c0fdce2c4e2102d84ff2bc7ed9a7aa09630cf7f26e1116429a
PROJECT_FACTS.md     d051fad8850322388f17235d4530d74e826e5fba761b9a24635fbb3332655c0d
```

修改后实现/测试 SHA256：

```text
hrl_uav/env.py               6bf01056edd63be2ab1317066158b7fac4ec907d7e12d498b88d0346159a9762
hrl_uav/mdp.py               7887fcc8fe73c109e2da61aeebb04c23ac5d041feced01a6db2d932ba2453ddb
hrl_uav/bandwidth.py         16a08f0a7b839ed6d71507e9ce0fb480ff241d5b28f7b160c6d5300973c8fce0
tests/test_wireless_model.py 20668a7553b4ca015efc4ccf7eeb63b9b0fab12a21a52d985f7b71d7363f71a9
```

保留的三个既有文件 SHA256：

```text
hrl_uav/scene_prescreen.py       d914323ffd41b091bf5bec03fd6211229830b608c41279c8f0d6bdd25d6033bd
scripts/run_scene_prescreen.py  7e75409ab738d3b226f90aefe79e614cd82088f2509690958e49d5635f4650ee
tests/test_scene_prescreen.py   4603a45e5068ac58bb5bf0717b752805f99aac51dee1042baa249b15ec9ed0de
```

后续有限固定规则评估需要新一轮明确授权，先冻结新有效配置和待比较的模式/CPU选项、
带宽、种子及有限时长，并确认只有注册的参数差异；不得覆盖两个旧预筛选目录。
继续复用现有 MeteredMDP/配对路径，记录模式、新参数、代码指纹、实际资源账及外生过程指纹，
同时报告任务完成和期末积压，不把短程结果称为无限期稳定或策略最优。
旧预筛选 smoke 的源码指纹已不再对应本轮实现，不能拿旧 smoke 记录跳过新检查。
本轮功能验收后停止；没有自动启动这些后续操作。
