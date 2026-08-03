# joint_hrl_static_30step

- **实验族**：静态用户 HRL
- **状态**：历史诊断
- **说明**：静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。
- **归档内容**：27 个文件，4.16 MB
- **未归档大文件**：28 个检查点、13 个数组，共 457.11 MB

> 不同版本的场景、奖励、时隙或通信模型可能不同。除非总 README 明确说明，否则不要直接比较不同版本的绝对数值。

## 预览

![result preview](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)

## 图片

- [training_convergence.pdf](formal/seed1/figures/training_convergence.pdf)
- [training_convergence.png](formal/seed1/figures/training_convergence.png)
- [trajectory_best.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)
- [trajectory_ep_0100.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_ep_0200.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0200.png)
- [trajectory_ep_0300.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0300.png)
- [trajectory_ep_0400.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0400.png)
- [trajectory_ep_0500.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0500.png)
- [trajectory_ep_0600.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0600.png)
- [trajectory_final.png](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)
- [trajectory_ep_0100.png](formal/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_ep_0200.png](formal/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0200.png)
- [trajectory_final.png](formal/seed1/micro_pretrain/trajectory_snapshots/trajectory_final.png)

## 指标数据

- [protocol_snapshot.json](formal/protocol_snapshot.json)
- [eval_central_td3.csv](formal/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](formal/seed1/joint_td3_sac/meta/scenario.json)
- [position_oracle.json](formal/seed1/joint_td3_sac/position_audit/position_oracle.json)
- [position_oracle.json](formal/seed1/joint_td3_sac/position_audit_smoke/position_oracle.json)
- [position_oracle.json](formal/seed1/joint_td3_sac/position_audit_v2/position_oracle.json)
- [train_td3.csv](formal/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](formal/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](formal/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](formal/seed1/micro_pretrain/train_centralized.csv)

## 日志

- [joint_td3_sac_s1.log](_logs/formal/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/formal/micro_pretrain_s1.log)
- [run.stderr.log](formal/seed1/joint_td3_sac/position_audit/run.stderr.log)
- [run.stdout.log](formal/seed1/joint_td3_sac/position_audit/run.stdout.log)
