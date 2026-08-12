# v27 保守双 Critic Actor 单因素 Pilot 协议

> 状态：FROZEN FOR PILOT
>
> 日期：2026-08-05
>
> 本协议在结果产生前冻结。固定使用第 100 轮终点，不得改用最佳或中间
> checkpoint，不得延长本轮训练挽救结果。

## 1. 目的

检验标准 TD3 的 Actor 仅跟随 Q1，是否会在本项目的小信号轨迹问题中利用 Q1 的
选择性乐观偏置；并检验将 Actor 的时延价值改为 `min(Q1,Q2)` 后，能否减少有害
漂移。

原始 TD3 的裁剪双 Q 最小值用于 Critic target，而策略梯度使用 Q1；见 Fujimoto
等人的 TD3 Algorithm 1。实用 SAC 双 Q 版本将两头最小值用于策略目标，说明保守
双 Q 可以进入策略改进环节。但本实验是确定性 TD3 的诊断变体，不把 SAC 的随机
策略推导直接等同于 TD3。

参考：

- Fujimoto S, van Hoof H, Meger D. Addressing Function Approximation Error in
  Actor-Critic Methods. ICML 2018.
- Haarnoja T, et al. Soft Actor-Critic Algorithms and Applications. 2019.

## 2. 唯一科学变量

控制组为 `configs/tcom_dynamic_u2u_sparse_v24_dual_on.json`，实验组为
`configs/tcom_dynamic_u2u_sparse_v27_min_actor.json`。

除结果目录、复用 v24 微观 checkpoint 的目录声明外，唯一科学变量为：

| 参数 | v24 控制 | v27 实验 |
|---|---|---|
| Actor 时延 Critic 聚合 | `q1` | `min` |

保持不变：

- 固定负载动态用户场景、10 GHz CPU、840 W 系统长期平均功率预算；
- 纯拉格朗日约束、系统乘子进入两层状态、固定功率罚项为 0；
- v24 的同一微观 SAC checkpoint；
- 中性 episode 覆盖关闭；
- TD3 target noise、探索、回放、学习率、Actor warm-up 和所有环境参数；
- 不修改 U2U 状态表征。

## 3. 训练与评估

- 训练 seed：1；
- 联合训练：100 episodes；
- 每 20 episodes 用 20 个固定测试种子评估；
- 固定终审：第 100 轮 `final.pt`；
- 终点使用测试种子 1000--1019，与 v24 做共同随机数配对；
- 同状态反事实使用种子 62000--62019、宏观步 0/5/10/15/20/25；
- 反事实分别从 Actor 轨迹状态和质心参考状态取样，两分支首步动作不同，后续均
  使用同一 Actor，外生随机轨迹必须逐步匹配。

训练日志逐轮记录：

- `Q1-Q2` 在 Actor 动作与零动作处的有符号差；
- 双 Critic 绝对分歧；
- Actor/零动作的保守 Q 排序；
- 距质心、累计路径、平均功率、功率 UCB95 和乘子。

## 4. 预注册判据

### 4.1 行为改善

v27 相对 v24，在 20 个共同测试种子上必须同时满足：

1. 加权端到端时延配对差 `v27-v24 < 0`，95% CI 上界小于 0；
2. 平均距动态质心配对差 `v27-v24 < 0`，95% CI 上界小于 0；
3. 功率 UCB95 不超过 840 W。

不要求漂移归零。该实验的目标是显著减少有害漂移，而不是强制 Actor 输出零动作。

### 4.2 机制改善

质心状态探针的排序一致率与 v26 终点结果逐 episode 配对比较。若一致率差
`v27-v26 > 0` 且 95% CI 下界大于 0，则确认保守 Actor 目标改善了 Critic 到 Actor
的排序传导。Actor 漂移状态的 Q1、Q2 与真实回报方向同时报告，不单独作为硬门禁。

### 4.3 判决

- 4.1 与 4.2 全通过：支持“标准 TD3 单头 Actor 在小信号场景下利用乐观偏置，
  保守双头 Actor 可缓解该失效模式”；
- 仅 4.1 通过：行为改善成立，但不能把收益唯一归因于探针排序修复；
- 4.1 未通过或功率越界：本变体 FAIL，不延长、不挑 checkpoint；
- 任一外生轨迹哈希不一致、状态维度或 checkpoint 测试失败：实验无效，先修工程
  问题后按原协议重跑。

## 5. 明确禁止

- 不加入中性覆盖、step 级探针数据、BC、Advisor、距离惩罚或动作正则；
- 不修改 U2U 表征或重训微观 SAC；
- 不调整功率预算、对偶学习率或轨迹动作半径；
- 不启动三种子 formal；
- 不把本 pilot checkpoint 继承到以后 formal。

## 6. 归档

```text
results/tcom_dynamic_u2u_sparse_v27_min_actor/pilot/seed1/joint_td3_sac/
result_archive/v27_min_actor_pilot_audit/
```
