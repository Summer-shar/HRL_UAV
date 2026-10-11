# G1 策略只读审计

审计对象：`4f8198ac1e0ce391b2860cfbbfa896f5412a73ce` 对应 seed 4101 的第 300 轮冻结参数。读取 `episode-0300-evaluated.pt`（侧车所列 SHA256 `744bac16d586fe59c7c4c03b7f4cdb9c9136d6aadb76f5974c1ddb69b9cc8ec6`），仅 CPU 前向；没有调用仿真、训练、优化器、resume，也没有修改源码或配置。物理步数 0，参数更新 0。

## 可以由已有记录确认的原因

1. 失败不能归因为“整个机群容量必然不够”。每台 CPU 的物理上限为 10 GHz，900 s 上限各为 9e12 cycles，总计 27e12。场景 8 的实际外部到达共 22.9648e12 cycles；相同外部场景的两条 policy 随机流，upload/ready 时延相同，但 CPU 分配和排队显著分叉。stream 1 截断时延 0.775439 s，stream 2 为 19.158188 s（其中 CPU 阶段 18.729212 s）。stream 2 给三台 CPU 的累计分配/各自全窗口上限为 [152.725%, 26.867%, 75.452%]。这证明已执行策略把工作集中到一台超出其最大可服务量的 CPU，同时其他节点留有明显总容量余量。容量上限不是实际可服务量的等式；突发、优先级及到达时间仍会影响排队。

2. 已有累计分配记录显示，stream 2 的低优 source 2 将约 84.0% cycles 转给 UAV 0。场景 9 两个 policy stream 也都给 UAV 0 分配约 104% 的全窗口容量，时延分别为 3.472358 / 3.499326 s。详见 `dev_assigned_work.json`（索引均为源码 0-based）。这与 CPU 排队失衡一致，不只是上传链路条件改变。

3. 冻结最终 actor 在**历史已见的真实拥堵状态**上也存在方向错误的解析分布倾向。读取最终 replay 的 246–300 轮，每轮 12 个均匀时刻，加上该轮最大 CPU 积压时刻，去重后共 715 个状态；只解析计算 support softmax × Dirichlet 条件均值，不抽样。最大积压样本来自 episode 251 / frame 964，UAV 0 低优队列约 3.35609e11 cycles；最终 actor 仍把低优 source 0/1/2 的期望分配 [98.426%, 86.184%, 99.997%] 指向 UAV 0。source 2 的单目的支持 {0} 概率为 99.9973%。因此至少这个真实已见状态下，问题在策略分布本身的倾向，不能解释成只抽到一次罕见极端比例。详见 `offline_policy_summary.json` 的 `max_cpu_backlog_sample`；完整 715 状态保存在 `offline_policy_samples.json`。

4. 上述历史状态不是 dev 场景 8 的已记录触发状态，也不是训练第 251 轮当时的参数：本审计输入是其真实旧状态、参数是最终第 300 轮冻结 actor。它证明最终策略未稳健处理这个状态；不能证明 dev 8 的首次失稳事件，或把网络、critic、采样、训练目标中的某一项定为唯一算法根因。最终 actor 的参数及所算浓度均有限；这批状态的浓度范围约 [1.000053, 22.051182]。

## 已核对的语义与源码位置

- 训练 stage 1 与 G1 评价都使用 hover 上层、同一随机 Hybrid 下层，不是训练随机而评价擅自改成 deterministic。`src/hrl_v2/run.py:98`、`src/hrl_v2/evaluate.py:35`、`src/hrl_v2/policy.py:46`。
- 训练窗口中 SAC 参数会更新，评价冻结参数且没有 learning callback。故训练日志中平均表现平稳，不足以保证最后一版固定参数在 dev 的长期稳定性。`src/hrl_v2/run.py:271`、`:276`；`src/hrl_v2/evaluate.py:63`，`src/hrl_v2/collector.py:29`。
- 每个评价 scenario/stream 新建 adapter、显式 reset、新建 Policy；开始收集再次 begin_episode 清空 LiteObserver 的 EMA 和游标。`src/hrl_v2/evaluate.py:49`–`:65`、`src/hrl_v2/policy.py:19`、`src/hrl_v2/collector.py:54`。未发现跨评价场景的 EMA 残留证据。
- EMA 只累积 `[cursor:report.history_cursor]` 的已观测历史，拒绝更换 history 对象或倒退时钟。`src/hrl_v2/lite_observation.py:38`。Lite 网络没有需要训练/评价切换的 Dropout/BatchNorm 层。
- 支持集采样和连续比例采样来自隔离的 policy 随机流；两个 dev policy stream 共用外部 layout/load/arrival。`src/hrl_v2/randomness.py:10`、`:18`、`:34`；`src/hrl_v2/hybrid_simplex.py:10`。评价保存并恢复训练的全局 torch 随机状态（`evaluate.py:48`）。
- CPU 上限及逐步资源预算见 `src/hrl_v2/configs/physical_scene.json:135`、`src/hrl_uav/industrial_env.py:280`、`:287`。

## 尚不能从已有产物回答的内容

评价 JSON 保存了累计 assigned_cycles 和任务时延，但没有逐时刻 Report/观测、完整执行 action/support 或 logits。因此不从现有日志推断“某一个 Dirichlet 尾样本是触发点”；40 个 policy 窗口也不当作 40 个独立外部场景。20 个场景各两个 policy stream，样本量不足以确定灾难性轨迹发生概率。未做候选网络/超参数比较，也没有检查或建议放宽 G1。

## 复查

在仓库根目录运行 `.venv/Scripts/python.exe outputs/diagnosis_g1_20260928/policy_audit/inspect_policy.py` 只会读取现有产物、在 CPU 解析前向，并覆盖本目录三个 JSON；不会推进环境或更新模型。工具脚本、分析 JSON、本文均位于 outputs，研究源码无改动。该 CPU 前向不是 CUDA 数值等价性验收；分布极端倾向远大于正常浮点舍入量。
