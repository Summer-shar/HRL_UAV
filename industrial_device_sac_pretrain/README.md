# 从零开始的 SAC 悬停预训练

用户已授权重新预训练。本入口 `industrial_device_sac_pretrain.experiment` 在 2026-09-24 21:25:47 启动，运行目录为 `industrial_hotspot_records/load24725_fresh_sac_lr1e5_20260924/training`，PID 53276。完成第 3 轮的完整检查点已独立严格恢复并核对，见运行根目录 `startup_verified.json`。这是启动事实，不代表 300 轮完成或性能达标。

## 有效设置与影响范围

- 场景固定为 `industrial_hotspot_records/load24725_seed1_20260923/training/scene.json`：期望计算负载 24.725 Gcycles/s，三业务模板约 10.5171、4.1575、10.0504 Gcycles/s。旧默认场景的 `repretrain_lr1e5_20260924` 准备结果不用于本次。
- SAC 动作网络、双评价网络、目标评价网络、温度、Adam 和经验池全部重新初始化，没有读入第 75 轮或旧 600 轮权重。实际动作/评价/温度学习率为 `1e-5/3e-4/3e-5`。
- 系统模型、优化问题、任务/服务过程、6/18 维动作、奖励与折扣、86 维 SAC 观测保持。保留 523 维 PPO 网络，但悬停预训练不采样、不积累 PPO 样本、不更新 PPO。
- 时序仍是 5 秒大时隙、1 秒调整加 4 秒驻留、0.5 秒路由和 0.1 秒物理步；严格零上层动作使 UAV 保持原位，期间服务持续。
- 训练流程改为新 SAC 悬停最多 300 轮，每轮 900 秒。每 50 轮用六个固定开发场景各 900 秒做冻结评价；既有配对 Local/CVX 数据经场景、面板和任务流校验后复用。
- 在满足原单个节点 Local 可用条件的已评价节点中，以平均物理成本选最低节点，并列选较早轮次；同时保留时延和未完成统计。单点可用条件不代表原先连续两次准入、收敛或优于 CVX。
- 本入口结束后不自动接联合训练；没有冻结 SAC 单独训练 PPO 阶段。300 轮也不自动视为最佳模型。论文、图和贡献未修改，新性能尚待评价。

预算为 2700000 个成功训练物理步、324000 个冻结评价物理步，墙钟上限 4 小时，异常不自动重跑。每整轮保存完整检查点；恢复从完整流边界进行，正式入口拒绝功能测试权重。失败半轮开销须另核，不能把成功步数当成包含失败尝试的实际总开销。

## 为什么此前未启动

上轮仅有 `experiment.py` 和包初始化文件，尚缺 `evaluation.py` 与 `validate.py`，源码身份及验收也因此不完整；前期还误用了旧默认场景。现在已补齐并验证。交接摘要声称旧 `_assert_counters` 必须读取 `pretrain_seconds`，实际静态核查该函数没有这项依赖；本入口现使用明确约束零 PPO 的本地计数检查，但不把该摘要说法当成已证实缺陷。

功能 attempt01 在完成两轮加 10 秒测试后，因为误用包含临时 `.grad` 的冻结指纹比较恢复状态而报错。检查点不保存临时梯度；修正为比较全部持久化字段，并验证恢复后实际下一段更新完全一致。旧失败记录保留，其实际消耗为 18100 物理步，原 failure 中按段预留数为 18000，漏记末段 100 步，在此纠正；不计入研究训练。

attempt02 通过：18800 物理步、133 次 SAC 事件、266 次评价拟合、17 次动作/温度更新、PPO 0；用时 40.742 秒。包含两整轮、双分支恢复后 10 秒训练、六场景各 10 秒冻结采集、奖励累计核对、真实 JSON 输出、已有完整报告的准入汇总、错误配置拒绝及中间模型选择。完整六场景 900 秒新 SAC 评价将在第 50 轮首次执行，未把短功能测试当成性能结论。两次功能测试共 36900 物理步、265 次 SAC 事件，研究分支重新从零初始化。

## 状态文件

- `training/progress.json`：当前轮次、学习次数与实际学习率。
- `training/episodes.jsonl`：每轮指标及每 5 秒轨迹/队列。
- `training/learning.jsonl`：SAC 学习统计。
- `training/evaluations/episode-XXXX/report.json`：固定六场景评价。
- `training/selection.json`：中间模型选择，首次评价后产生。
- `startup_verified.json`：正式完整检查点的独立启动核验。

启动命令：

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -B -m industrial_device_sac_pretrain.experiment train --output industrial_hotspot_records/load24725_fresh_sac_lr1e5_20260924/training --acceptance industrial_device_sac_pretrain/validation/attempt02/acceptance.json
```

该命令已执行，勿重复启动。
