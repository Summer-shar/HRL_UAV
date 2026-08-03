# tcom_fixed_region_reward_v1

- **实验族**：宏观奖励诊断
- **状态**：诊断/消融
- **说明**：固定区域下的宏观奖励信号、惩罚与冻结策略消融。
- **归档内容**：48 个文件，6.18 MB
- **未归档大文件**：41 个检查点、18 个数组，共 341.56 MB

> 不同版本的场景、奖励、时隙或通信模型可能不同。除非总 README 明确说明，否则不要直接比较不同版本的绝对数值。

## 预览

![result preview](diagnostic/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)

## 图片

- [trajectory_best.png](diagnostic/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)
- [trajectory_ep_0020.png](diagnostic/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0020.png)
- [trajectory_final.png](diagnostic/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)
- [trajectory_ep_0020.png](diagnostic/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0020.png)
- [trajectory_final.png](diagnostic/seed1/micro_pretrain/trajectory_snapshots/trajectory_final.png)
- [reward_feedback.pdf](pilot/seed1/figures/reward_feedback.pdf)
- [reward_feedback.png](pilot/seed1/figures/reward_feedback.png)
- [training_convergence.pdf](pilot/seed1/figures/training_convergence.pdf)
- [training_convergence.png](pilot/seed1/figures/training_convergence.png)
- [trajectory_best.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)
- [trajectory_ep_0100.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_final.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)
- [macro_reward_signal.png](pilot/seed1/macro_reward_signal_ep140/macro_reward_signal.png)
- [trajectory_ep_0100.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_ep_0200.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0200.png)
- [trajectory_final.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_final.png)
- [trajectory_best.png](smoke/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)

## 指标数据

- [protocol_snapshot.json](diagnostic/protocol_snapshot.json)
- [eval_central_td3.csv](diagnostic/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](diagnostic/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](diagnostic/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](diagnostic/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](diagnostic/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](diagnostic/seed1/micro_pretrain/train_centralized.csv)
- [protocol_snapshot.json](pilot/protocol_snapshot.json)
- [reward_feedback_correlations.csv](pilot/seed1/figures/reward_feedback_correlations.csv)
- [reward_feedback_summary.json](pilot/seed1/figures/reward_feedback_summary.json)
- [eval_central_td3.csv](pilot/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](pilot/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](pilot/seed1/joint_td3_sac/train_td3.csv)
- [macro_reward_signal_raw.csv](pilot/seed1/macro_reward_signal_ep140/macro_reward_signal_raw.csv)
- [macro_reward_signal_summary.json](pilot/seed1/macro_reward_signal_ep140/macro_reward_signal_summary.json)
- [eval_central_macro.csv](pilot/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](pilot/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](pilot/seed1/micro_pretrain/train_centralized.csv)
- [protocol_snapshot.json](smoke/protocol_snapshot.json)
- [eval_central_td3.csv](smoke/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](smoke/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](smoke/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](smoke/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](smoke/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](smoke/seed1/micro_pretrain/train_centralized.csv)

## 日志

- [joint_td3_sac_s1.log](_logs/diagnostic/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/diagnostic/micro_pretrain_s1.log)
- [joint_td3_sac_s1.log](_logs/pilot/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/pilot/micro_pretrain_s1.log)
- [joint_td3_sac_s1.log](_logs/smoke/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/smoke/micro_pretrain_s1.log)
