# 大时隙奖励核查（2026-09-24）

本审查沿当前 control pilot 的真实调用链，只读取源码和有效配置，不改变奖励。下述实现事实是源码静态核实；功能检查是否通过另见 validation 的执行结果。

## 物理成本与两个时间尺度

有效场景为 `industrial_hotspot_records/load24725_seed1_20260923/training/scene.json`。物理步 0.1 秒、下层分配周期 0.5 秒、上层周期 5 秒，一个下层周期含 5 个物理步，一个上层周期含 10 个下层周期。

`hrl_uav/industrial_env.py:344` 的 `micro_step` 在物理步首统计所有尚未完成父任务：

\[
H_k=0.1(10N_{H,k}+N_{L,k}),\qquad c_k=0.9H_k/10+0.1E_k/14.
\]

其中 H 的单位是加权任务秒，E 是该 0.1 秒内上传、机间转发、CPU 和完整推进能耗之和（J）。驻留覆盖上传、待分配、机间在途和计算等全部未完成阶段，并非只统计某一种队列的等待。

下层折扣 `gamma_per_routing_frame=0.99`，对应物理步折扣
\(g=0.99^{1/5}=0.9979919516614258\)。`IndustrialMDP.step_frame` 返回

\[
r_j=-\sum_{i=0}^{4}g^i c_{5j+i}.
\]

当前 control 的 `PilotRollout.run` 从同一份原始成本重算上层奖励，因上下层物理时域一致，严格等价于

\[
R_t=\sum_{j=0}^{9}0.99^j r_{t,j}
    =-\sum_{k=0}^{49}g^k c_{50t+k},\qquad
\Gamma=0.99^{10}=0.9043820750088044.
\]

所以“大时隙奖励来自小时隙”是对的，但这里是 10 个下层奖励的折扣累计，既不是平均，也不是仅取最后一帧。未来大时隙另乘 \(\Gamma\)，因此没有漏折扣或重复计算物理成本。对应折扣半衰期为 34.4838 秒。

## SAC 熵项与 PPO 的区别

`hrl_uav/learning.py:217` 的 SAC 双评价网络目标为
\(r+\gamma(\min Q_{\mathrm{target}}-\alpha\log\pi)\)，动作网络目标也有熵项；这些是 SAC 内部学习目标。上层奖励只累积实际物理成本，不累积 SAC 的熵、损失或 Q 值。

PPO 在 `industrial_deployment_learning.py:62` 使用该宏观 R、后继 V、宏观折扣 \(\Gamma\) 和有效 \(\lambda=0.95\) 构造 GAE。900 秒轮末是采样截断：切断跨轮 GAE 递推，仍以真实后继 V bootstrap；并非把统计窗口误当吸收终态。SAC 同样保存真实下层后继和 0.99 折扣。

即使 control 两层共享同一物理折扣成本，SAC 的最大熵目标和 PPO 的纯物理成本目标也不能称为完全相同的优化目标。

## 已结束 long150 对照的例外

`PilotMDP` 在 long150 中只将上层半衰期改为 150 秒：上层物理步折扣 \(2^{-0.1/150}\)，下层帧对应的上层折扣 \(2^{-0.5/150}\)，宏观折扣 \(2^{-5/150}=0.9771599684342459\)。`PilotRollout` 用原始成本同步重算大时隙内部奖励。此时不能拿 SAC 原来的 0.99 帧奖励直接按新宏观折扣拼接，否则区间内部目标不一致。SAC 的 r 和 0.99 仍保持原样。

新逐设备观测候选以 control 为参考，不同时采用 long150，也不改变物理奖励权重。

## 图表口径

历史 pilot 的 `physical_return` 是一轮 900 秒未折扣总成本的负数；`upper_discounted_return` 才是从该轮起点按宏观折扣累计的上层回报。两者应分别保存、明确标注，不把它们当作同一条奖励曲线。端到端时延另外统计已完成父任务，应同时展示未完成数量。

源码位置：`industrial_hotspot_analysis/load24725_pilot_20260923/pilot.py:99`、`:109`，`hrl_uav/industrial_mdp.py:43`，`hrl_uav/industrial_env.py:344`，`hrl_uav/industrial_rollout.py:112`，`hrl_uav/learning.py:21`、`:217`，`hrl_uav/industrial_deployment_training.py:55`。有效配置优先于 `learning.py` 内未被采用的默认参数和历史注释。

后续功能验收 `validation/attempt01/verification.json` 已通过：新、旧编码入口各执行相同四个大时隙，物理路径及下层转移逐值相同；新入口的大时隙奖励与 10 个下层奖励折扣和的最大误差为 0。该运行核对只验证奖励累计合同，不证明当前权重、折扣或奖励的学习效果最佳。
