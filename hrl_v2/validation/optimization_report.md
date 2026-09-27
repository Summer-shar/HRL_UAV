# V2 吞吐优化记录

- 基线：`137d2f1ecbcbb45689e7959170803ef945bff027`
- 研究训练：0 个 episode；未启动 600 轮，未改变有效 batch、更新次数、队列或历史长度。
- 数值后端：保留 CPU `float64` 动作/损失分布；本轮没有预设 CUDA 迁移。

本轮落地了到达 GRU 的在线增量缓存。缓存键包含 episode 身份、原始 history 身份、设备/类型和所有编码器参数的 `_version`；episode 切换、history 替换、参数更新、游标回退都会失效。`Encoder.infer()` 以及动作网络、评价网络、PPO 网络的单状态入口使用该缓存，批训练仍使用完整历史。

回放 `resolve()` 现在返回 `PreparedBatch`，在一次批次生命周期内复用解压历史、报告对象和固定图特征；批处理路径预先构造图索引、队列 padding 和 attention 桶。Q 网络首层把状态投影和动作投影分解，输出与原始首层拼接实现一致。

## 验证

`python -m unittest discover -s hrl_v2/tests`：32 项通过。新增测试覆盖增量 GRU 与完整前缀的普通/连续推理等价性、episode/参数版本失效、Q 首层分解等价性；已有物理、SAC、PPO、checkpoint 和评价隔离测试继续通过。

受限 `python -m hrl_v2.validate` 通过，记录了采样、critic、actor、普通/拥堵推理和 CUDA/进程内存测量。当前机器实测值保存在同目录 `benchmark.json`；基线记录中的参考值为 critic 约 12.31 s、actor 约 6.71 s、最大状态推理 p99 约 0.201 s。由于本轮没有改变有效更新合同，seed 预算仍按配置执行：每 seed 5,400,000 物理步、35,010 个有效 SAC 事件、70,020 次 critic 拟合、17,010 次下层动作更新和 40 批 PPO；这些是预算计数，不代表本轮已运行。

尚未把下层无用 value 前向、日志同步合并或流式评价改成改变更新语义的实现；这些项目需要对应规格文件和独立一致性证据后再继续。
