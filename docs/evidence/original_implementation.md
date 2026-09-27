# HRL_UAV V2 候选实现

本目录实现附件 1.0 规格的 PPO＋SAC 主方法。状态是**已实现并完成接口验收，尚未进行完整研究训练和性能验收**。原物理代码、旧检查点、旧结果和论文均未修改。

ReportV2 采用中央同步、零延迟、遥测开销未额外计费的报告假设。完整服务报告和有限 GRU 不构成充分马尔可夫状态的证明；900 秒有限任务也不支持无限期队列稳定性结论。

## 本轮位置与来源

- 所有新增代码、配置、测试和本轮验证记录位于 `C:/Users/PC/Desktop/HRL_UAV/hrl_v2`。
- `configs/v2_1_0.json` 是用户提供配置的逐字节副本。SHA256：`bd72f535b125acee3e0019cb60b33831995db0b90593eb621392f65f8307d45a`。其中“尚未实现”字段作为原设计材料保留，本目录实际状态由本说明及验证记录表达。
- `configs/physical_scene.json` 复制自 `HANDOVER_20260925.md:77` 指定的 `industrial_hotspot_records/load24725_seed1_20260923/training/scene.json`。SHA256：`daea8771a2cb06754724c2af373704c279c080c7216d55ad990c86f0918b6136`。
- `design/reference_hybrid_simplex.py` 来自用户提供的 `D:/Chrome/未确认 268104.crdownload`，其内容是 Python 源码。原有测试输出另存为 `design/reference_tests_supplied.txt`；它不是本轮运行记录。
- 附件未提供 `physical_source_binding.json`、`test_reference_hybrid_simplex.py` 和 `IMPLEMENTATION_SCOPE_EN.md`。本地来源记录见 `design/local_source_binding.json`；没有声称它与未提供的绑定清单完成了比对。测试为本轮新增。

配置和测试统一放在新命名空间内，因此命令中的配置路径是 `hrl_v2/configs/v2_1_0.json`，测试路径是 `hrl_v2/tests`。默认输出是工程根目录下的 `v2_runs/<protocol_hash>/ppo_sac/seed_<id>`。

## 实现内容

1. `physics_adapter.py` 直接调用原 `HotspotIndustrialEnv` 的移动与物理服务；下层执行比例不经过旧 MDP 的投影或折扣包装。报告包含全部 58 条队列、完整活跃父子关系及已发生的到达历史。原始分片比例单独记账，当前与 all-active 链路速率分开保存。
2. 独立的动作网络、双评价网络、目标评价网络及 PPO 状态价值网络；完整 FIFO GRU、三层有类型图注意力、UAV 读出、几何需求注意力、两层 UAV 注意力。在线到达 GRU 按 episode 身份和参数版本增量推理；训练时每个优化器步骤共享当前网络的完整前缀并完整反向传播。`batching.py` 合并互不连通的图以批量计算，只跨步骤缓存固定特征，不截队列。
3. CPU float64 的七支持集＋条件 Dirichlet；执行数组保持 float64，评价网络输入转换检查误差和支持内正值。动作网络逐行枚举六行×七支持集，缓存每个状态的评价编码，同时保留真实目标与代理损失两种日志。
4. 两层 gamma=1；900 秒真实终止，无终点新到达、无终点 bootstrap。宏微奖励分别累计同一物理成本。下一状态由实际下一宏动作提交及到达准备完成后封存。
5. 完整 episode 无损压缩回放、上层版本隔离、精确重要性权重、两次独立 critic 拟合。PPO 每批五个完整 episode；整批优势归一化；每轮 epoch 检查 KL，超限恢复动作网络与 Adam，评价网络继续训练。
6. 固定 300 下层＋100 上层＋10×(10 上层＋10 下层) 日程。每 seed 的期望计数为 5,400,000 训练物理步、35,010 个有效 SAC 事件、70,020 次 critic 拟合、17,010 次下层动作更新、40 批 PPO。这里是配置和计数测试结果，不是实际完成的研究训练量。
7. 分流种子、封存训练/开发/测试清单、初始参数指纹、episode 边界原子检查点、待评价游标、评价断点复用及训练随机流隔离。保留初始检查点、开发节点和最近两个普通恢复点。恢复使用本次运行 `latest.json` 指向的完整点；旧节点可独立评价，不能覆盖后续训练记录。尝试编号区分中断前未提交工作；无法观察的断电丢失物理步数记为未知。
8. 物理成本、合池完成时延、每优先级完成/未完成统计、未完成年龄、逐物理步队列积分及实际分配 cycles。未完成任务最终时延记为删失。

`baselines.py` 提供原 Local/Random/CVX 路由适配及因果需求质心。**分别训练主基线的运行分支、三项消融、五训练种子的正式统计验收尚未交付或执行。** 当前训练 CLI 对应完整 V2 主方法；配对评价支持同一 SAC 的 moving/hover，也可单独评价 centroid。这些后续研究工作不能用接口测试替代。

## 可执行命令

在 `C:/Users/PC/Desktop/HRL_UAV` 的 PowerShell 中执行。默认 dry-run 不创建模型训练任务：

```powershell
& 'D:/Anaconda3/envs/pytorch-gpu/python.exe' -B -m hrl_v2.run
```

统一验收入口会运行有界的物理对照、数学测试和梯度接口检查，并测量目标机器上的计算开销。它包含短优化器测试，但不训练完整研究 episode：

```powershell
& 'D:/Anaconda3/envs/pytorch-gpu/python.exe' -B -m hrl_v2.validate
```

下列命令会真正启动完整 600 轮研究训练；**本轮没有执行**。入口要求当前代码、配置及物理来源与已通过的本地验证记录匹配，旧 schema 检查点不被接受：

```powershell
& 'D:/Anaconda3/envs/pytorch-gpu/python.exe' -B -m hrl_v2.run --seed 4101 --fresh
```

恢复时使用该运行 `latest.json` 中的相对检查点路径：

```powershell
& 'D:/Anaconda3/envs/pytorch-gpu/python.exe' -B -m hrl_v2.run --resume '<latest.json 中的 path>'
```

正式测试要求完整 600 轮末端及已清空的开发待评价状态：

```powershell
& 'D:/Anaconda3/envs/pytorch-gpu/python.exe' -B -m hrl_v2.evaluate --checkpoint '<V2 完整末端检查点>' --manifest '<该运行的 test_manifest.json>' --output '<新的评价输出目录>' --mode paired --policy stochastic
```

确定性补充使用 `--policy deterministic`，输出单独标识，不与随机主评价混合。

## 本地证据及边界

`validation/tests.txt` 和 `validation/validation.json` 记录最近一次完整有界测试，测试数量及状态以记录为准。覆盖完整 900 秒双实例逐步一致性、终点和宏边界、完整队列及父子关系、非可加 Q 下的离散梯度、随机流、独立初始化、实际 CUDA 梯度、回放与完整学习状态恢复、PPO 的 KL/Adam 回滚，以及分图批处理输出和梯度一致性。优化接入测试另覆盖实际 Policy 缓存失效、完整历史反传、独立编码器、基线优化器对照和流式评价。

`validation/benchmark.json` 保留实现阶段的历史有界测量，不作为本轮优化前后对照。以 `137d2f1` 为基线的实际补丁、同 fixture 耗时、一致性结果及完整 seed 条件预算见 [throughput_report.md](validation/throughput_report.md)。测量使用临时模型和脚本轨迹，不代表完成了研究训练或策略性能验收。

用于测试的脚本路由 episode 压缩报告及历史约 16.60 MB；55 个同等大小 episode 的这一部分约 0.91 GB，检查点还包含模型、优化器及其他字段。真实训练状态分布和完整训练墙钟时间尚未测得，不沿用旧版本的训练时间估计。

尚无 V2 的 600 轮 pilot、正式测试或性能提升成绩。3% 改善、五种子成功及 600 ms 挑战指标仍是待实测目标。本轮没有修改论文、提交 Git 或推送。
