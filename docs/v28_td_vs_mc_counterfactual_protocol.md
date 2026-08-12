# v28 反事实 TD-vs-truncated-return Critic 判别协议

> 状态：FROZEN
>
> 冻结日期：2026-08-06
>
> 本协议在任何 v28 结果产生前冻结。数据生成、目标构造、训练预算、统计单位和判决规则均不得事后修改。

## 1. 目的与冻结对象

目的：区分宏观 Critic 的三类剩余嫌疑：

1. 一步 bootstrap 偏差传播；
2. 联合状态-动作支持与跨状态泛化不足；
3. U2U 状态表征不足。

冻结对象：

- checkpoint：`results/tcom_dynamic_u2u_sparse_v24_dual_on/pilot/seed1/joint_td3_sac/checkpoints/final.pt`；
- 部署策略：checkpoint 中的 `macro_actor`，全程冻结；
- 微观策略：同一 checkpoint 中的 SAC Actor，确定性执行并冻结；
- 环境：v24 dual-on 原配置；
- 系统乘子：checkpoint 保存的 `energy_lagrange_multiplier_current=0.038131859182497896`；
- 功率预算：840 W；
- 折扣因子：`gamma=0.99`。

本实验不更新 Actor、SAC、系统乘子或环境参数，不启动 formal。

## 2. 数据生成

### 2.1 episode 与状态采样

- 训练 episode seeds：`63000--63039`，共 40 个；
- 测试 episode seeds：`64000--64019`，共 20 个；
- 训练/测试严格按原始 episode seed 整体切分；
- 每个 seed 分别生成 Actor 参考轨迹和质心零动作参考轨迹；
- 两类轨迹均在宏观索引 `0,5,10,15,20,25` 保存环境快照；
- 同一 episode 的全部快照、分叉和延续 transition 只能属于同一数据集。

### 2.2 同状态双动作反事实

每个快照分别执行：

- `a_actor = frozen_actor(s)`；
- `a_zero = zeros(3,2)`。

首步以后，两分支均使用同一个冻结部署 Actor。训练分支从分叉点起固定延续 30 个宏观周期；测试分支固定延续 60 个宏观周期。这里的 H=30/60 均是**从分叉点开始的固定视野**，不是原 300 s rollout 的剩余长度。

### 2.3 CRN 与视野稳定性

同一快照的两个动作分支必须使用相同的用户移动、任务到达和信道随机流，并逐步校验外生轨迹哈希。

CRN 只降低外生噪声方差，不消除动作导致的末端状态差异，也不使截断回报无偏。

测试集同时计算 `G_30` 与 `G_60` 的 actor-minus-zero 配对回报差。不得删除或筛选发生符号翻转的状态。若 Actor 区或质心区任一区域的 H=30/H=60 符号一致率低于 90%，v28 只能判为 `HORIZON_SENSITIVE`，不得进入根因判决表。

### 2.4 transition schema

每条分支逐步保存：

```text
split, episode_seed, snapshot_policy, fork_macro_index, fork_state_id,
branch_action, branch_id, branch_step, state, action, reward, next_state,
external_trace_hash
```

测试分支保存 60 步；训练 Critic 只使用前 30 步。所有数组使用压缩 NPZ，另保存可审计 manifest 与逐状态回报 CSV。

## 3. v28a：冻结网络的 target 视野分解

v28a 不训练网络。对测试集每个分支，使用 v24 的冻结 `macro_delay_critic_target` 和冻结部署 Actor 计算：

\[
Y_n=\sum_{k=0}^{n-1}\gamma^k r_k+
\gamma^n\min_j Q'_{j}(s_n,\pi(s_n)),
\quad n\in\{1,5,15\}.
\]

target tail 不使用 policy smoothing，因为反事实实测延续策略是确定性部署 Actor；加入 smoothing 会改变被评估策略，形成第二个科学变量。

比较 `Delta Y_n` 与 `Delta G_30` 的排序方向。统计单位为 episode：先在每个 episode 内平均六个分叉点，再跨 20 个 episode 计算均值与 95% CI。

v28a 仅回答“现有 target 排序是否随实测视野增加而改善”，不单独触发算法修改：

- n=1 排错且 n=5/15 单调改善并转正：支持 bootstrap tail 是偏差放大通道；
- 各 n 无系统变化：bootstrap 单因子假设弱化；
- 其他结果：记录为混合或不确定，继续执行第 4 节。

## 4. 两臂离线 Critic

### 4.1 共同有限视野目标

为使一步 TD 与直接回报严格对应同一个有限视野 Q，两个离线 Critic 的状态均追加一维 `h/30`，其中 h 为当前 transition 到固定 30 步边界的剩余步数。

测试只在分叉初始状态 `h=30` 上比较两动作排序。该时间条件化仅用于诊断，不声称等同于原在线 continuing Critic。

### 4.2 唯一科学变量

| 项目 | TD/FQE 臂 | truncated-return 臂 |
|---|---|---|
| target | `r + gamma*min(Q1',Q2')(s',pi(s'),(h-1)/30)`；h=1 时 target=r | 从当前 transition 到第 30 步边界的实测折扣回报 |
| Actor | 冻结部署 Actor | 同左 |
| target smoothing | 关闭 | 关闭 |
| 数据与 batch 顺序 | 完全相同 | 完全相同 |

### 4.3 冻结超参数

| 参数 | 冻结值 |
|---|---:|
| Critic 输入 | 原 84 维宏观状态 + 6 维动作 + 1 维剩余视野 |
| 网络 | 双头 MLP，各头 `256-256-ReLU` |
| 初始化 seeds | `7101,7102,7103` |
| 优化器 | Adam，默认 betas |
| 学习率 | `3e-4` |
| batch size | `128` |
| gradient updates | `20000` |
| gamma | `0.99` |
| target update | 每次更新 Polyak，`tau=0.005` |
| loss | 两头 MSE 之和 |
| gradient clipping | global norm `10.0` |
| checkpoint | 固定使用第 20000 次更新，不早停、不择优 |

同一初始化 seed 的两臂必须从完全相同的初始权重开始，并使用完全相同的 minibatch 索引序列。truncated-return 臂虽不需要 target 网络，仍保留相同模型对象，但不将其用于标签构造。

## 5. 评估与有效性门禁

### 5.1 统计单位

三个初始化的 `min(Q1,Q2)` 预测先取算术平均形成固定 ensemble。每个 episode 内分别计算 Actor 区和质心区六个分叉点的排序一致率，再以 20 个 episode 为独立统计单位计算 95% t 区间。不得把分叉点或初始化 seed 当作独立场景样本。

### 5.2 “排对/排错”定义

- `CORRECT`：排序一致率 95% CI 下界大于 0.5，且 ensemble 平均预测差与平均 `G_30` 差同号；
- `WRONG`：排序一致率 95% CI 上界小于 0.5；
- `INCONCLUSIVE`：其余情况。

### 5.3 管线门禁

在进入根因判决前必须同时满足：

1. H=30/H=60 在 Actor 区和质心区的符号一致率均不低于 90%；
2. truncated-return 臂训练集分叉初始状态排序一致率不低于 90%；
3. truncated-return 臂测试 Actor 区结果为 `CORRECT`；
4. 所有 CRN 外生轨迹哈希一致、loss 有限、样本计数与 manifest 一致。

任一失败则本轮为 `INVALID_OR_INCONCLUSIVE`，先修复数据/回归管线，不得归因于 bootstrap 或 U2U 表征。

## 6. 冻结判决表

仅在第 5.3 节全部通过后使用：

| 质心区结果 | 判决 | 下一步 |
|---|---|---|
| truncated-return=`CORRECT`，TD=`WRONG` | 支持一步 bootstrap 偏差传播 | 另行预注册在线 n-step 训练；不得直接把离线 Critic 用作主算法 |
| truncated-return 训练集通过，但测试=`WRONG` | 跨状态泛化/联合支持问题 | 设计 step 级同状态动作支持实验 |
| truncated-return 训练集门禁失败 | 离线优化、容量或标签尺度失败 | 本轮无根因判决；修复管线后按新预注册重跑 |
| 两臂=`WRONG` | bootstrap 单因子不足；支持、表征或训练问题仍未分离 | U2U 表征作为下一单因素实验，但不得宣称其为唯一原因 |
| 两臂=`CORRECT` | 固定策略离线设置未复现在线故障 | 在线 Actor-Critic 分布反馈成为候选，需另行预注册验证 |
| 任一臂=`INCONCLUSIVE` | 统计证据不足 | 不增加当前测试种子，不做根因声明；记录限制并重新评估研究路线 |

## 7. 纪律

- 不修改奖励、场景、通信模型或功率预算；
- 不加入 BC、Advisor、距离惩罚、零动作正则或中性覆盖；
- 不更新 Actor、SAC 或 lambda；
- 不使用结果选择初始化、checkpoint、状态子集或回报视野；
- v28a 与两臂结果全部落盘后才允许写 verdict；
- 不启动 formal。

## 8. 输出

```text
result_archive/v28_td_vs_mc_counterfactual/
|-- dataset_manifest.json
|-- branch_transitions/
|-- per_state_paired_returns.csv
|-- horizon_stability.json
|-- v28a_target_horizon/
|-- hyperparams_frozen.json
|-- td_arm/
|-- truncated_return_arm/
|-- ranking_report.md
`-- verdict.md
```
