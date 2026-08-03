# joint_hrl_v1

- **实验族**：早期动态用户 HRL
- **状态**：历史轨迹参考
- **说明**：早期曾出现 UAV-3 靠近 UAV-1 的清晰轨迹；协议较旧且缺少当前口径下的配对验证，不作为最终结论。
- **归档内容**：51 个文件，6.56 MB
- **未归档大文件**：52 个检查点、21 个数组，共 374.26 MB

> 不同版本的场景、奖励、时隙或通信模型可能不同。除非总 README 明确说明，否则不要直接比较不同版本的绝对数值。

## 预览

![result preview](formal/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)

## 图片

- [micro_convergence_complete.png](formal/seed1/figures/micro_convergence_complete.png)
- [training_convergence.png](formal/seed1/figures/training_convergence.png)
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
- [training_convergence.png](pilot/seed1/figures/training_convergence.png)
- [trajectory_ep_0100.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_final.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)
- [trajectory_ep_0100.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_ep_0200.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_ep_0200.png)
- [trajectory_final.png](pilot/seed1/micro_pretrain/trajectory_snapshots/trajectory_final.png)
- [training_convergence.png](smoke/seed1/figures/training_convergence.png)

## 指标数据

- [protocol_snapshot.json](formal/protocol_snapshot.json)
- [eval_central_td3.csv](formal/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](formal/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](formal/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](formal/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](formal/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](formal/seed1/micro_pretrain/train_centralized.csv)
- [verification_seed1.json](formal/summaries/verification_seed1.json)
- [protocol_snapshot.json](pilot/protocol_snapshot.json)
- [eval_central_td3.csv](pilot/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](pilot/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](pilot/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](pilot/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](pilot/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](pilot/seed1/micro_pretrain/train_centralized.csv)
- [verification_seed1.json](pilot/summaries/verification_seed1.json)
- [protocol_snapshot.json](smoke/protocol_snapshot.json)
- [eval_central_td3.csv](smoke/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](smoke/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](smoke/seed1/joint_td3_sac/train_td3.csv)
- [eval_central_macro.csv](smoke/seed1/micro_pretrain/eval_central_macro.csv)
- [scenario.json](smoke/seed1/micro_pretrain/meta/scenario.json)
- [train_centralized.csv](smoke/seed1/micro_pretrain/train_centralized.csv)
- [verification_seed1.json](smoke/summaries/verification_seed1.json)

## 日志

- [joint_td3_sac_s1.log](_logs/formal/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/formal/micro_pretrain_s1.log)
- [joint_td3_sac_s1.log](_logs/pilot/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/pilot/micro_pretrain_s1.log)
- [runner_seed1.stderr.log](_logs/pilot/runner_seed1.stderr.log)
- [runner_seed1.stdout.log](_logs/pilot/runner_seed1.stdout.log)
- [joint_td3_sac_s1.log](_logs/smoke/joint_td3_sac_s1.log)
- [micro_pretrain_s1.log](_logs/smoke/micro_pretrain_s1.log)
