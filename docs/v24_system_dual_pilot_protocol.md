# v24 系统功率对偶单因素 Pilot 协议

> 状态：FROZEN FOR PILOT
>
> 日期：2026-08-04
>
> 本协议仅用于 840 W 动态对偶开机验证，不是 formal 训练协议。看到 pilot
> 结果后不得修改本轮验收口径；pilot checkpoint 不得继承到 formal。

## 1. 目的

验证系统级长期功率乘子在可观测、纯拉格朗日条件下能否形成完整闭环：

`系统功率残差 -> lambda -> CPU 能耗价格 -> CPU 频率/路由 -> 系统功率`。

本轮不验证中性动作覆盖，不修改场景、奖励语义、通信模型、CPU 上限或轨迹动作。

## 2. 单因素对照

使用两个独立配置：

- `configs/tcom_dynamic_u2u_sparse_v24_dual_off.json`
- `configs/tcom_dynamic_u2u_sparse_v24_dual_on.json`

除结果目录、微观预训练目录和 `energy_dual_lr` 外，两配置完全一致：

| 项 | dual_off | dual_on |
|---|---:|---:|
| `objective_mode` | `constrained_delay` | `constrained_delay` |
| `constraint_form` | `pure_lagrangian` | `pure_lagrangian` |
| 固定功率违约罚项 | 0 | 0 |
| 系统 lambda 入两层状态 | 是 | 是 |
| `energy_dual_lr` | 0 | 1.0 |
| 功率滑窗 | 20 episodes | 20 episodes |
| 预算 | 840 W | 840 W |

`dual_lr=1.0` 在结果产生前冻结。取值依据是实现中的归一化尺度与
`priced_maxweight` 闭式响应：代表性积压下 lambda 小于约 1 时 CPU 仍接近
10 GHz 饱和，而 lambda 位于 1--3 时开始产生可辨别的频率响应。历史
`dual_lr=0.02` 在 600 轮后仅得到约 0.20 的 lambda，不能作为有效开机尺度。

## 3. 训练预算

- seed：1；
- 微观 SAC：两臂分别从干净状态预训练 300 episodes；
- 联合 TD3-SAC：两臂分别从干净状态训练 100 episodes；
- 每 20 episodes 固定评估 20 个共同随机种子；
- TD3 target noise、探索、回放容量和两层学习率沿用 v23；
- 不复用 smoke checkpoint，不跨臂复用微观 checkpoint。

## 4. 必查量

1. lambda 每轮轨迹、20 轮平滑功率和预算残差；
2. CPU 能耗价格、每机 CPU 频率/饱和率、计算功率；
3. 评估平均功率及 UCB95；
4. 加权端到端时延、训练真实代价和队列是否崩溃；
5. dual_on 与 dual_off 的共同种子差异。

## 5. 验收

同时满足以下条件才允许进入后续长训练：

1. 平滑功率高于 840 W 时 lambda 非减，低于预算时允许下降至 0；
2. lambda 变化同步进入两层状态和 CPU 能耗价格，CPU 最优频率对价格非增；
3. dual_on 最终评估的功率 UCB95 不超过 840 W；
4. 若约束在后期不紧且 UCB95 已满足，lambda 回落至 0 不判失败；
5. dual_on 相对 dual_off 不得出现超过 10% 的加权端到端时延恶化；
6. 任一状态维度、checkpoint 恢复、响应链或数值有限性测试失败，停止 pilot。

若第 3 或第 5 条失败，本轮结论为 840 W 动态对偶 pilot 未通过；不得自动把
860 W 改成主操作点，也不得继续三种子长训练。
