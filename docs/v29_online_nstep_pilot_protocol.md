# v29 在线 n-step 时延 Critic 单因素 pilot 协议

> 状态：FROZEN（2026-08-06）。本文件落库后，不得根据 pilot 结果调整 n、训练轮数、
> checkpoint、探针窗口、统计门槛或失败分支。v29 只做机制判别，不启动 formal。

## 1. 目的

v28a 在冻结策略反事实数据上发现：质心区 target 排序一致率由 n=1 的 60.0% 提升到
n=5 的 92.5%、n=15 的 100%，且 H=30/H=60 真实回报符号 100% 稳定。v28 离线
绝对 Q 双臂因回归分辨率不足被判 `INVALID_OR_INCONCLUSIVE`。

v29 直接检验：在真实在线 Actor-Critic 训练中，将宏观**时延 Critic**的 backup
从一步改为五步，能否阻止 60-80 轮出现的排序反转与有害轨迹漂移。

## 2. 冻结对象与两臂

两臂均从干净状态启动，训练 seed 固定为 1，训练 100 episode；共用 v24 dual-on 的
微观 SAC checkpoint、环境、负载、用户移动、840 W 预算、lambda 状态、CPU 定价、
探索噪声、网络、优化器、更新频率、回放容量和随机种子。

| 臂 | 唯一科学变量 |
|---|---|
| v29-control-n1 | `td3_delay_n_step=1` |
| v29-treatment-n5 | `td3_delay_n_step=5` |

两臂均通过同一个新版 sequence sampler 取 replay 起点；n=1 不走旧 `sample()` 快捷路径。
这样内部对照只差 target 视野。能耗 Critic 固定使用一步 target，Actor loss、lambda 更新和
微观 SAC 更新完全不变。

## 3. n-step target 与边界

episode 含 30 个宏观 transition，索引 t=0,...,29。对任一起点：

```text
m = min(n, 30 - t)
Y_delay = sum_{k=0}^{m-1} gamma^k r_{t+k}
          + gamma^m min_j Q'_j(s_{t+m}, pi'(s_{t+m}) + epsilon)
```

- 不跨 episode，不将第 30 步伪装为终止；
- t=26-29 的 n=5 样本分别使用 m=4/3/2/1；
- bootstrap 位于最后一个实测 transition 的 `next_state`；
- 仅末端 target action 使用 v24 原有 TD3 policy smoothing；
- 中间奖励来自 replay 中实际行为动作，不做重要性修正；该离策略近似作为已知边界记录；
- sequence 必须满足相同 `episode_index` 且 `macro_index` 严格逐一递增，环形回放覆盖导致
  不连续时该起点无效并重采样。

能耗 Critic 仍按原来的一步式更新。一步与五步 target 估计的是相同折扣价值，只改变
backup 分解；保持能耗一步是有意的单因素隔离，不称为“目标视野语义不一致”。

## 4. v29-control 的历史复现门禁

新版 n=1 final 与历史 v24 dual-on final 均在 eval seed 1000-1019 上确定性重放，外生
轨迹哈希必须逐 seed 相同。定义配对差：

```text
delta_repro = D_v29_n1 - D_v24
```

历史复现采用等价性判定，不使用“CI 覆盖 0”作为充分证据：

- 95% 配对 CI 必须完全落入 `[-3 ms, +3 ms]`；
- 3 ms 约为历史 v24 均值 773.44 ms 的 0.39%，作为本轮预注册工程等价界限；
- control 的最终功率 UCB95 必须不超过 840 W；
- 不要求浮点逐位一致，运行时字段不参与比较。

任一失败则 v29 判为 `CONTROL_NOT_REPRODUCED`，不得比较 n=5 与 n=1 的算法优劣。

## 5. 固定探针与加密频率

使用 v28 测试集 seed 64000-64019 的固定状态、固定 v24 Actor 动作、联合零动作及 G30
标签。探针只做网络前向，不进入 replay 或梯度。

- checkpoint：20、40、60、70、80、90、100 episode；
- 主窗口：60、70、80、90、100；
- 主 Critic 口径：实际驱动 v24 Actor 的 Q1；Q2 和 min 仅作诊断；
- 每个 checkpoint、每个状态区先在 episode 内聚合 6 个分叉点；
- 主窗口统计时，再在同一 eval episode seed 内平均 5 个 checkpoint，最后以 20 个
  eval seed 的 n5-minus-n1 配对差计算 95% t 区间；不得把 checkpoint 当独立样本。

机制通过条件：质心区主窗口 Q1 排序一致率差
`agreement_n5 - agreement_n1` 的 95% CI 下界大于 0。另报告 Actor 区、Q2、min、
预测 delta 均值和双头分歧，但不替代主判据。

## 6. replay 陈旧性只读日志

训练时按样本起点年龄分桶：`0-4`、`5-14`、`15-29`、`>=30` episode。每桶记录：

1. 有符号 `Y5-Y1` 与绝对值 `|Y5-Y1|`；
2. `|Q-Y5|` TD residual；
3. 中间存储行为动作与当前 target Actor 动作的平均 L2 距离；
4. 样本数和有效 sequence 重采样次数。

这些量只称为 target disagreement、TD residual 和 policy mismatch，不称为“真实 target
error”。日志不参与训练，也不改变验收结果。

## 7. 最终 CRN 评价

两臂 final checkpoint 固定在 episode 100，不挑 best。使用相同 20 个 eval seed：

- 优先级加权端到端时延；
- 平均功率与 UCB95；
- lambda 轨迹方向；
- 平均/95 分位质心距离、累计路径、净位移；
- 协作比例和链路流量。

定义 `delta_delay = D_n5 - D_n1`。时延改善要求 95% 配对 CI 上界小于 0；功率要求
n=5 的 UCB95 不超过 840 W。lambda 在超预算时应上升、低于预算时允许回落至 0。

## 8. 预注册判决

仅在 §4 control 复现门禁通过后使用：

| 机制探针 | 最终时延/功率 | 判决 |
|---|---|---|
| 改善 | 时延改善且功率可行 | `FULL_PASS`：n-step 同时修复排序和行为 |
| 改善 | 时延未改善但功率可行 | `MECHANISM_ONLY`：Critic 排序已修，瓶颈移至跨层兑现/U2U 表征 |
| 未改善 | 老样本的 policy mismatch 与 `|Y5-Y1|` 明显更高 | `OFF_POLICY_STALENESS_CANDIDATE`：不能否定视野假设 |
| 未改善 | 无明显年龄效应 | `BOOTSTRAP_HYPOTHESIS_WEAKENED`：U2U/状态表征升为下一主嫌疑 |
| 任意 | n=5 功率 UCB95>840 W | `CONSTRAINT_FAIL`，不得以时延改善宣布通过 |

“年龄效应明显”仅作分支解释：`>=30` 桶对应量高于 `0-4` 桶，且 episode 级 95% CI
排除 0；否则记为无明确年龄效应。

## 9. 纪律与输出

- 不改奖励、场景、U2U 状态、SAC、能耗 Critic、lambda、Advisor 或距离项；
- 不延长 100 episode，不调整 n=5，不挑 checkpoint；
- 两臂结果出来前不开 formal；
- 任何实现修复必须先证明 n=1 门禁失败源于代码错误，并以新预注册重跑两臂。

```text
result_archive/v29_online_nstep_pilot/
|-- protocol_snapshot.json
|-- control_reproduction/
|-- fixed_probe_progression/
|-- replay_age_diagnostics/
|-- final_crn_evaluation/
|-- figures/
`-- verdict.md
```
