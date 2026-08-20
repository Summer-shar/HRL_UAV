# Joint HRL for Multi-UAV MEC

本仓库研究多 UAV 协作边缘计算中的双时间尺度控制：宏观层控制 UAV waypoint 残差，
微观层控制双优先级任务的本地处理与 U2U 转发。

## 当前口径

- [项目当前状态](CURRENT_STATUS.md)：环境、算法、历史实验和下一步边界。
- [当前结果完整分析](CURRENT_RESULT_ANALYSIS.md)：逐层效果、收敛判断和现有图片审查。
- 当前正式筛选配置：`configs/tcom_v49_support_residual_synchronous.json`。
- 当前协议：`docs/v49_support_residual_synchronous_protocol.md`。

当前 episode 为 300 s，共 30 个宏观周期；每周期包含 1 s 部署和 18 个 0.5 s 资源帧。
用户采用 RPGM 群组移动，U2U 的 3 MHz 总带宽按活跃链路动态分配。每架 UAV 的 CPU
上限为 10 GHz，采用高优先级优先的 `priority_demand` 规则。

## 当前算法

- 上层候选：集中式 TD3 或 PPO，输出三架 UAV 相对动态业务质心的联合残差 waypoint；
- 下层候选：SCA 先构造稀疏可行支撑集，SAC 在支撑集内学习小幅路由残差；
- 训练协议：每种方法独立初始化，下层预训练 200 轮，随后同步训练 800 轮；
- 当前功率口径：时延优先软目标中的能耗代价和描述性功率账本，无 280 W 聚合投影，
  无 840 W 硬预算或动态对偶。

## 结果位置

- 六方法独立训练：`result_archive/v49_independent_synchronous_seed1/`
- 上下层控制变量审计：`result_archive/v49_macro_value_audit_seed1/`
- 保守微观门控审计：`result_archive/v50_conservative_micro_gate_audit/`

`results/` 中的大型 checkpoint、replay 和逐任务日志默认不提交；配置、协议、汇总 CSV、
报告与图片保存在 Git 中。当前正式结果只有一个训练种子，20 个环境种子仅用于该模型内部
的共同随机数配对评估。
