# tcom_dynamic_u2u_sparse_v23_target_noise

- **实验族**：动态用户与 U2U 协作
- **状态**：当前诊断版本
- **说明**：v23：缩小 TD3 target noise 并加入动作覆盖、位置参考和反事实诊断；宏观轨迹仍出现有害漂移。
- **归档内容**：23 个文件，3.23 MB
- **未归档大文件**：28 个检查点、9 个数组，共 344.02 MB

> 不同版本的场景、奖励、时隙或通信模型可能不同。除非总 README 明确说明，否则不要直接比较不同版本的绝对数值。

## 预览

![result preview](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)

## 图片

- [trajectory_best.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)
- [trajectory_ep_0100.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0100.png)
- [trajectory_ep_0200.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0200.png)
- [trajectory_ep_0300.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0300.png)
- [trajectory_ep_0400.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_ep_0400.png)
- [trajectory_final.png](pilot/seed1/joint_td3_sac/trajectory_snapshots/trajectory_final.png)
- [trajectory_best.png](smoke/seed1/joint_td3_sac/trajectory_snapshots/trajectory_best.png)

## 指标数据

- [protocol_snapshot.json](pilot/protocol_snapshot.json)
- [eval_central_td3.csv](pilot/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](pilot/seed1/joint_td3_sac/meta/scenario.json)
- [position_reference_audit.json](pilot/seed1/joint_td3_sac/position_reference_audit/position_reference_audit.json)
- [search_progress.json](pilot/seed1/joint_td3_sac/position_reference_audit/search_progress.json)
- [train_td3.csv](pilot/seed1/joint_td3_sac/train_td3.csv)
- [protocol_snapshot.json](smoke/protocol_snapshot.json)
- [eval_central_td3.csv](smoke/seed1/joint_td3_sac/eval_central_td3.csv)
- [scenario.json](smoke/seed1/joint_td3_sac/meta/scenario.json)
- [train_td3.csv](smoke/seed1/joint_td3_sac/train_td3.csv)

## 日志

- [joint_td3_sac_s1.log](_logs/pilot/joint_td3_sac_s1.log)
- [joint_td3_sac_s1.log](_logs/smoke/joint_td3_sac_s1.log)
- [run.stderr.log](pilot/seed1/joint_td3_sac/position_reference_audit/run.stderr.log)
- [run.stdout.log](pilot/seed1/joint_td3_sac/position_reference_audit/run.stdout.log)
- [pilot_launcher.stderr.log](pilot_launcher.stderr.log)
- [pilot_launcher.stdout.log](pilot_launcher.stdout.log)
