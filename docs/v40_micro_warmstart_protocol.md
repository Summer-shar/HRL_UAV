# v40 微观 SAC 暖启动协议（FROZEN）

## 目的

检验 v39 退化是否由困难场景下的冷启动策略触发“队列积压—奖励方差—Critic 误差”反馈环。
本实验只改变微观 SAC 的初始化与启动时序，不改变环境、目标函数、网络结构、学习率或评估口径。

## 冻结设置

- 环境与目标：完全继承 `tcom_v39_calibrated_soft_objective.json`。
- 训练种子：1；总轮数：300；最终只使用 `final.pt`，不按中途表现挑 checkpoint。
- Actor：只加载 v33 seed-1 微观预训练 `final.pt` 中的 `sac_actor` 权重。
- Critic、target Critic、温度参数、三个优化器与 replay：全部重新初始化；禁止加载 v33 replay。
- 第 1–20 轮：用初始化 Actor 采集新环境、新目标下的 transition，不执行任何梯度更新。
- 第 21–50 轮：只更新 Critic；Actor 与温度冻结。
- 第 51–300 轮：恢复原 SAC 更新规则并微调。
- 训练行为沿用原 SAC 随机采样，不额外切换为确定性行为策略。

## 验收

1. 最终 checkpoint 在当前 v39 场景、20 个共同随机种子下的端到端时延不劣于 v33 旧 Actor 的 684.81 ms 参考值；正式报告使用重新配对重放结果，不直接复用旧数字。
2. 最后 60 轮固定评估时延的波动范围小于 20 ms。
3. 同步报告队列、奖励标准差、Critic loss 与熵温度曲线，用于区分：
   - 暖启动稳定：支持冷启动触发反馈环；
   - 暖启动仍退化：转向奖励尺度与场景难度对 Critic 可学性的交互审计。

## 实现审计

启动时必须生成 `micro_warmstart_initialization.json`，证明 Actor 哈希与来源一致、Critic 哈希与来源不同、优化器状态为空且 replay 未加载。
