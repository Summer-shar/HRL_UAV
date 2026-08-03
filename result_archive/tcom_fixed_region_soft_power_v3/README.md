# tcom_fixed_region_soft_power_v3

- **实验族**：固定区域与软功率机制
- **状态**：迭代诊断
- **说明**：动作与功率反馈的第二轮基线修订。 各版本配置不同，只用于追踪设计演化。
- **归档内容**：27 个文件，1.96 MB
- **未归档大文件**：30 个检查点、8 个数组，共 128.88 MB

> 不同版本的场景、奖励、时隙或通信模型可能不同。除非总 README 明确说明，否则不要直接比较不同版本的绝对数值。

## 预览

![result preview](diagnostic/seed1/micro_pretrain/trajectory_snapshots/trajectory_final.png)

## 图片

- [trajectory_ep_0020.png](diagnostic/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0020.png)
- [trajectory_final.png](diagnostic/seed1/micro_pretrain/trajectory_snapshots/trajectory_final.png)
- [trajectory_ep_0100.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_ep_0200.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0200.png)
- [trajectory_best.png](smoke/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)

## 指标数据

- [protocol_snapshot.json](diagnostic/protocol_snapshot.json)
- [eval_central_macro.csv](diagnostic/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](diagnostic/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](diagnostic/seed1/micro_pretrain/train_centralized.csv)
- [protocol_snapshot.json](pilot/protocol_snapshot.json)
- [eval_central_macro.csv](pilot/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](pilot/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](pilot/seed1/micro_pretrain/train_centralized.csv)
- [eval_central_macro.csv](pilot/seed1/micro_refine_low_actor_lr/eval_central_macro.csv)
- [scenario.json](pilot/seed1/micro_refine_low_actor_lr/meta/scenario.json)
- [train_centralized.csv](pilot/seed1/micro_refine_low_actor_lr/train_centralized.csv)
- [protocol_snapshot.json](smoke/protocol_snapshot.json)
- [eval_central_td3.csv](smoke/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](smoke/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](smoke/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](smoke/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](smoke/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](smoke/seed1/micro_pretrain/train_centralized.csv)

## 日志

- [micro_pretrain_s1.log](_logs/diagnostic/micro_pretrain_s1.log)
- [micro_pretrain_s1.log](_logs/pilot/micro_pretrain_s1.log)
- [joint_td3_sac_s1.log](_logs/smoke/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/smoke/micro_pretrain_s1.log)
