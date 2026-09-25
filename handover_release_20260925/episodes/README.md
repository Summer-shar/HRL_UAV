# 全部逐轮训练结果

本目录保留全新SAC预训练300轮与联合400轮，共700轮；不是只保存固定评价节点。所有数据从现有日志导出，新增仿真和学习均0。

## 文件与字段

- `all_episodes.csv`：700轮全部汇总；`pretrain_001_300.csv`和`joint_001_400.csv`分别保存两阶段。
- CSV将原记录中除trajectory外的全部字典/数组展开成列，保留种子、场景指纹、成本、驻留、能耗、完成量、时延、末尾队列及累计更新次数。可用Excel或pandas打开，UTF-8 BOM。
- `physical_return`是每轮未折扣物理成本的负数，数值越大越好；`discounted_return`按原日志原样保存，不与physical_return混称同一奖励。
- `metrics.weighted_delay_ms`只包含已完成父任务；`derived.priority.0.unfinished`/`.1.unfinished`分别为高/低优到达数减完成数，已核对其和等于原日志unfinished_parents。
- `sample5s_mean.*`、`sample5s_max.*`为每轮180个5秒采样点的全机队列总量均值/最大值。它们不是0.1秒物理步积分均值。未采样到的峰值不能据此排除。
- `metrics.*pending*`等来自原metrics，表示轮末量，不能代替全过程平均积压；原日志的物理步驻留积分另保留在相应time_s字段。
- `training_curves.png/.svg`画出每轮时延与physical_return，黑线为最近最多20轮算术均值；没有隐藏开头的高时延轮次，也没有误差带或多训练种子声明。

四份压缩文件分别对应：

| 文件 | 原运行目录 | 轮次 |
|---|---|---|
| training.episodes.jsonl.gz | training | 预训练1—300 |
| joint25_run02.episodes.jsonl.gz | joint25_run02 | 联合1—25 |
| joint50.episodes.jsonl.gz | joint50 | 联合26—50 |
| joint400.episodes.jsonl.gz | joint400 | 联合51—400 |

`.jsonl.gz`保留原始JSONL全部字节，包括每轮180条轨迹/队列、原先已记录的动作和奖励字段；没有以CSV替代或丢弃原始过程信息。可以用Python `gzip.open(path, 'rt', encoding='utf-8-sig')`逐行读取，每行`json.loads`得到一轮完整记录。原文件SHA256和轮次数见`audit.json`，导出时已逐份解压比对。

## 解释边界

训练场景逐轮变化，且存在探索和轮内参数更新；冻结开发评价的六场景、固定参数是另一统计口径。CSV是训练曲线，不能冒充每轮都在固定开发场景评估。预训练250轮被选为联合起点；预训练251—300轮仍完整保存，但未被联合继承。两个阶段轮次分别从1开始。

原始记录已有每轮结果和5秒过程采样，不包含每个物理步的完整任务账本或每条实际分片路由。不能从本次导出宣称补录了训练时未记录的事件。权重及逐次梯度学习日志未包含在这些压缩包内。

复核/重新导出脚本：`../export_episodes.py`，需本机四份原始JSONL和matplotlib。`audit.json`中的50轮时延均值是逐轮均值再平均，并非合并父任务重算；不要拿单轮未完成均值直接和六场景总数比较。
