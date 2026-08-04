# HRL-UAV 全版本实验结果归档

本目录整合了 `results/` 中现存的 **45 个顶层实验版本**。导出基于提交 `0ec6af414196`。

## 归档范围

- 已归档：全部 PNG/PDF/SVG 图、CSV/JSON 指标和文本日志。
- 未归档：`.pt` 模型检查点、`.npz/.npy` 原始数组及其他大二进制文件。
- 已归档体积约 **106.77 MB**；本地未归档大文件约 **8.66 GB**。
- `archive_manifest.csv` 记录每个归档文件的来源、大小和 SHA-256；`versions.csv` 是版本总表。

> 重要：这些版本是研发过程记录，不是同一实验协议下的正式横向对比。场景、奖励、动作编码、训练轮数和通信模型在版本间发生过变化。

## 当前结论

- 当前诊断版本为 `tcom_dynamic_u2u_sparse_v23_target_noise`。
- 微观 SAC 能正常完成协作转发；宏观 TD3 在 400 轮中出现 U2G 距离增大和时延恶化，尚未形成最终可用轨迹。
- 20-seed 位置参考审计表明：动态跟踪相对固定锚点降低约 40.5 ms，但动态质心、上行参考点和联合位置搜索之间差异不足 1 ms。
- `joint_hrl_v1` 中 UAV-3 靠近 UAV-1 的轨迹是历史上最清晰的协作轨迹，但协议较旧，不能替代当前口径下的正式验证。

## 推荐阅读顺序

1. [`tcom_dynamic_u2u_sparse_v23_target_noise`](tcom_dynamic_u2u_sparse_v23_target_noise/)：当前训练与轨迹漂移。
2. [`diagnostics`](diagnostics/)：反事实、奖励与位置扫描诊断。
3. [`joint_hrl_v1`](joint_hrl_v1/)：历史上较清晰的协作轨迹与早期收敛图。
4. [`tcom_dynamic_u2u_v20_long600`](tcom_dynamic_u2u_v20_long600/) 到 v23：当前通信模型的演化链。
5. `tcom_fixed_region_soft_power_v2` 到 v19：固定区域与功率机制的历史迭代，仅用于追踪研发过程。

## 版本索引

| 版本 | 实验族 | 状态 | 图片 | CSV/JSON | MB | 说明 |
|---|---|---:|---:|---:|---:|---|
| [diagnostics](diagnostics/) | 跨版本诊断 | 诊断数据 | 2 | 19 | 0.55 | 宏观奖励、反事实分叉、动作覆盖和动态质心位置扫描。 |
| [dynamic_load_position_audit](dynamic_load_position_audit/) | 位置价值审计 | 诊断数据 | 0 | 4 | 0.22 | 动态负载条件下的位置敏感度与参考策略审计。 |
| [joint_hrl_30step](joint_hrl_30step/) | 早期动态用户 HRL | 历史实验 | 14 | 15 | 5.16 | 早期联合 TD3-SAC 训练与宏观周期设置迭代。 |
| [joint_hrl_static_30step](joint_hrl_static_30step/) | 静态用户 HRL | 历史诊断 | 13 | 10 | 4.16 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v1](joint_hrl_static_v1/) | 静态用户 HRL | 历史诊断 | 13 | 9 | 3.75 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v2](joint_hrl_static_v2/) | 静态用户 HRL | 历史诊断 | 7 | 15 | 1.97 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v3](joint_hrl_static_v3/) | 静态用户 HRL | 历史诊断 | 7 | 15 | 1.97 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v4](joint_hrl_static_v4/) | 静态用户 HRL | 历史诊断 | 6 | 7 | 1.65 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v5](joint_hrl_static_v5/) | 静态用户 HRL | 历史诊断 | 6 | 7 | 1.64 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v6](joint_hrl_static_v6/) | 静态用户 HRL | 历史诊断 | 5 | 14 | 1.48 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v7](joint_hrl_static_v7/) | 静态用户 HRL | 历史诊断 | 5 | 7 | 1.37 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_static_v8](joint_hrl_static_v8/) | 静态用户 HRL | 历史诊断 | 2 | 4 | 0.54 | 静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。 |
| [joint_hrl_v1](joint_hrl_v1/) | 早期动态用户 HRL | 历史轨迹参考 | 19 | 24 | 6.56 | 早期曾出现 UAV-3 靠近 UAV-1 的清晰轨迹；协议较旧且缺少当前口径下的配对验证，不作为最终结论。 |
| [joint_hrl_v2](joint_hrl_v2/) | 早期动态用户 HRL | 历史实验 | 17 | 16 | 5.69 | 早期联合 TD3-SAC 训练与宏观周期设置迭代。 |
| [joint_hrl_v3](joint_hrl_v3/) | 早期动态用户 HRL | 历史实验 | 13 | 6 | 4.80 | 早期联合 TD3-SAC 训练与宏观周期设置迭代。 |
| [load_switch_gate](load_switch_gate/) | 负载切换位置门禁 | 冻结门禁结果 | 6 | 74 | 1.66 | 20 个共同随机种子的 P0/P1/P2 纯重放；时延门槛通过，但 840 W 功率门槛未通过。 |
| [load_switch_gate_power_audit](load_switch_gate_power_audit/) | 负载切换功率归因 | 只读重放审计 | 8 | 79 | 2.81 | 20 个共同随机种子的逐机推进、CPU、U2G/U2U 功率与速度分解；原门禁策略和物理模型均未修改。 |
| [oracle_smoke](oracle_smoke/) | 工具冒烟测试 | Smoke | 0 | 1 | 0.04 | 位置搜索或参考点审计工具的快速正确性测试。 |
| [position_reference_audit_smoke](position_reference_audit_smoke/) | 工具冒烟测试 | Smoke | 0 | 2 | 0.05 | 位置搜索或参考点审计工具的快速正确性测试。 |
| [tcom_communication_convergence_v1](tcom_communication_convergence_v1/) | 通信模型收敛验证 | 历史验证 | 10 | 18 | 3.32 | 切换到参考论文风格 U2U 动态带宽后的通信模型与收敛测试。 |
| [tcom_communication_convergence_v2](tcom_communication_convergence_v2/) | 通信模型收敛验证 | 历史验证 | 10 | 10 | 4.11 | 切换到参考论文风格 U2U 动态带宽后的通信模型与收敛测试。 |
| [tcom_dynamic_u2u_sparse_v21_long600](tcom_dynamic_u2u_sparse_v21_long600/) | 动态用户与 U2U 协作 | 历史基线 | 7 | 14 | 2.88 | v21：稀疏 U2U 动态场景基线及微观预训练来源。 |
| [tcom_dynamic_u2u_sparse_v22_long1000](tcom_dynamic_u2u_sparse_v22_long1000/) | 动态用户与 U2U 协作 | 历史长训练 | 15 | 6 | 6.04 | v22：1000 轮动态 U2U 训练，用于确认长期漂移并非训练轮数不足。 |
| [tcom_dynamic_u2u_sparse_v23_target_noise](tcom_dynamic_u2u_sparse_v23_target_noise/) | 动态用户与 U2U 协作 | 当前诊断版本 | 7 | 10 | 3.23 | v23：缩小 TD3 target noise 并加入动作覆盖、位置参考和反事实诊断；宏观轨迹仍出现有害漂移。 |
| [tcom_dynamic_u2u_v20_long600](tcom_dynamic_u2u_v20_long600/) | 动态用户与 U2U 协作 | 历史基线 | 8 | 14 | 3.41 | v20：动态 U2U 通信模型的首个长训练版本。 |
| [tcom_fixed_region_reward_ablation](tcom_fixed_region_reward_ablation/) | 宏观奖励诊断 | 诊断/消融 | 11 | 10 | 4.20 | 固定区域下的宏观奖励信号、惩罚与冻结策略消融。 |
| [tcom_fixed_region_reward_v1](tcom_fixed_region_reward_v1/) | 宏观奖励诊断 | 诊断/消融 | 17 | 25 | 6.18 | 固定区域下的宏观奖励信号、惩罚与冻结策略消融。 |
| [tcom_fixed_region_soft_power_v10_long_term_credit](tcom_fixed_region_soft_power_v10_long_term_credit/) | 固定区域与软功率机制 | 迭代诊断 | 2 | 4 | 1.00 | 加强部署间隙与长期队列的跨周期信用传递。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v11_full_action_coverage](tcom_fixed_region_soft_power_v11_full_action_coverage/) | 固定区域与软功率机制 | 迭代诊断 | 0 | 4 | 0.04 | 扩大宏观动作覆盖范围。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v12_clean_warmup](tcom_fixed_region_soft_power_v12_clean_warmup/) | 固定区域与软功率机制 | 迭代诊断 | 0 | 4 | 0.08 | 清理 warm-up 数据与正式训练的边界。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v13_split_exploration](tcom_fixed_region_soft_power_v13_split_exploration/) | 固定区域与软功率机制 | 迭代诊断 | 1 | 4 | 0.54 | 拆分宏观与微观探索过程。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v14_staged_joint](tcom_fixed_region_soft_power_v14_staged_joint/) | 固定区域与软功率机制 | 迭代诊断 | 0 | 4 | 0.11 | 采用分阶段联合训练。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v15_two_timescale](tcom_fixed_region_soft_power_v15_two_timescale/) | 固定区域与软功率机制 | 迭代诊断 | 3 | 4 | 1.15 | 明确大小时间尺度更新协议。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v16_delayed_actors](tcom_fixed_region_soft_power_v16_delayed_actors/) | 固定区域与软功率机制 | 迭代诊断 | 6 | 7 | 2.43 | 延迟 TD3/SAC Actor 更新以稳定 Critic。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v17_hinge_penalty](tcom_fixed_region_soft_power_v17_hinge_penalty/) | 固定区域与软功率机制 | 迭代诊断 | 4 | 4 | 1.50 | 使用功率违约 hinge penalty。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v18_conservative_joint_sac](tcom_fixed_region_soft_power_v18_conservative_joint_sac/) | 固定区域与软功率机制 | 迭代诊断 | 4 | 4 | 1.62 | 降低联合阶段 SAC 更新强度。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v19_balanced_penalty](tcom_fixed_region_soft_power_v19_balanced_penalty/) | 固定区域与软功率机制 | 迭代诊断 | 6 | 4 | 2.13 | 重新平衡时延与功率违约惩罚。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v2](tcom_fixed_region_soft_power_v2/) | 固定区域与软功率机制 | 迭代诊断 | 6 | 23 | 2.20 | 软功率固定区域基线。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v3](tcom_fixed_region_soft_power_v3/) | 固定区域与软功率机制 | 迭代诊断 | 5 | 18 | 1.96 | 动作与功率反馈的第二轮基线修订。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v4_stick](tcom_fixed_region_soft_power_v4_stick/) | 固定区域与软功率机制 | 迭代诊断 | 2 | 11 | 0.81 | 采用 stick-breaking 微观动作编码。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v5_alpha01](tcom_fixed_region_soft_power_v5_alpha01/) | 固定区域与软功率机制 | 迭代诊断 | 7 | 7 | 2.90 | 调整 SAC 温度系数到 alpha=0.1。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v6_macro_signal](tcom_fixed_region_soft_power_v6_macro_signal/) | 固定区域与软功率机制 | 迭代诊断 | 4 | 5 | 1.78 | 强化宏观层可见的跨层反馈信号。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v7_centroid_waypoint](tcom_fixed_region_soft_power_v7_centroid_waypoint/) | 固定区域与软功率机制 | 迭代诊断 | 5 | 5 | 2.09 | 将宏观动作改为相对业务质心的 waypoint。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v8_slow_dual](tcom_fixed_region_soft_power_v8_slow_dual/) | 固定区域与软功率机制 | 迭代诊断 | 8 | 4 | 3.52 | 降低能耗对偶变量更新速度。 各版本配置不同，只用于追踪设计演化。 |
| [tcom_fixed_region_soft_power_v9_frame_power_mean](tcom_fixed_region_soft_power_v9_frame_power_mean/) | 固定区域与软功率机制 | 迭代诊断 | 3 | 4 | 1.47 | 按资源帧平均功率更新约束统计。 各版本配置不同，只用于追踪设计演化。 |

## 重新生成

```powershell
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\export_result_archive.py
```

如需在另一台机器继续训练，应从 GitHub 获取代码与本归档，再单独复制本地 `.pt` 检查点；Git 仓库不承担模型权重存储。