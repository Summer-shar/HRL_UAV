# 第二个研究点：最终实验流水线（paper2）

依据代码：`research/speedup/src`（v2.4.0，源 commit `25d59ff`）。物理场景、奖励、观测、PPO / SAC / 混合动作分布的实现**全部原样复用，没有修改源码**。本目录只重新组织训练阶段、评估、扫参和画图；以前的结果和存档一律不用。

## 一条命令跑完（Windows PowerShell）

```powershell
cd research\paper2
python tests\test_identity.py                 # 种子派生与 v2.4.0 完全一致（需要 torch）
python tests\test_perf_equivalence.py 120     # 加速补丁逐位等价（只需 numpy，约 1 分钟）
python run.py bench                           # 比较 CPU / GPU 哪个快，按提示设置 P2_DEVICE
$env:P2_DEVICE="cpu"                          # 或 "cuda:0"，以 bench 结果为准
$env:P2_THREADS="2"                           # 每个进程的 torch 线程数
python run.py pipeline --smoke --workers 4 --out D:\p2_smoke   # 冒烟：几分钟跑通全流程
python run.py pipeline --workers 6 --out D:\p2_runs            # 正式实验
```

- 中断后**原样重跑同一条命令**即可续跑：已完成的任务会跳过，训练从最近的快照（每 10 回合）继续。
- `--workers` 是同时运行的进程数，建议取「CPU 逻辑核数 ÷ P2_THREADS」。开始时每个 seed 有 4 个训练可以并行（M1、TFOO、TOO、NC）。
- 每个任务的日志在 `<out>\_jobs\<任务名>.log`；`python run.py plan --out D:\p2_runs` 可以只列出任务、不运行。
- 结果在 `<out>\results\`：`summary.md`（主表 + 配对检验 + 扫参表）、`main_table.csv`、`sweep_*.csv`、`stats.json`，图在 `results\figures\`（PNG + PDF）。
- 只重画表和图：`python run.py report --out D:\p2_runs`。

## 实验设计（运行前就固定）

| 算法 | 轨迹（上层） | 计算分配（下层） | 说明什么 | 训练 |
| --- | --- | --- | --- | --- |
| Proposed（HRL） | PPO | SAC | 本文方法 | M1 + M2 + M3 |
| TFOO（[11]：PPO + CVX） | PPO | 现有 CVX | 与参考文献比较 | PPO 400 回合 |
| TOO | PPO | 三机均分 | 智能分配的作用 | PPO 400 回合 |
| NC | PPO | 全部本地计算 | 协作的作用 | PPO 400 回合 |
| Hover-SAC | 悬停在初始位置 | SAC | 轨迹优化的作用 | M1 + 悬停微调 100 回合 |

训练阶段（每个 seed；1 回合 = 900 s 仿真）：

1. **M1**：SAC 在通用随机部署上预训练 200 回合。三分之二的回合 UAV 每 5 s 随机移动，三分之一的回合飞向全图随机航点。不使用任何任务先验。α 前 100 回合 0.005，后 100 回合 0.0025。
2. **M2**：冻结 M1 的 SAC，下层用确定性动作（与评估一致）；PPO 上层训练 400 回合，每 100 回合存 1 份候选。
3. **选上层**：4 份 M2 候选在 8 个 dev 场景上比较，选平均加权成本最低的。
4. **M3**：冻结选出的上层（确定性），SAC 微调 100 回合。α、actor 学习率和只更新 critic 的回合数由输出目录的微调配方决定（见下节）：v1 为 α 0.0005、lr 1e-4、3 回合；v2 为 α 0.0025、lr 3e-5、10 回合。
5. **基线**：TFOO / TOO / NC 各自独立训练 PPO 上层 400 回合。PPO 设置、初始权重、训练场景、探索噪声流都与 M2 完全相同，候选也同样在 dev 上选。
6. **Hover-SAC**：从 M1 出发在悬停位置微调 100 回合，微调配方与 M3 相同；候选同样在 dev 上选，第 0 份候选就是 M1 本身。

**统一设置**：所有 PPO 上层初始探索 std 0.5（v2.4.0 是 0.15，导致上层几乎学不动），其余 PPO 超参数不变。所有学习型方法都用确定性动作评估，选档指标一律是加权成本 `cost_per_s`（两层共同优化的目标）。

**评价**：
- **seed 与场景**：3 个 seed（4101–4103）；最终 test 用 20 个新场景（test split 21–40，旧的 1–20 已经用过）。
- **扫参**：只取稳定工作点，即平均计算需求 / 三机 CPU 总容量 < 0.95（标称负载下平均需求 24.725 Gcycles/s，各场景相同）。CPU 频率 9–14 GHz，任务量 0.6–1.1 倍；每个点 8 个 test 场景 × 3 个 seed。只在标称参数下训练，扫参时直接评估。第一版设计里的 8 GHz（需求/容量 1.03，需求超过容量）和 1.2 倍任务量（0.99）属于饱和区，结果保留，单独列在 `summary.md` 的饱和区表和 `saturated_*.csv` 里，不进主扫参图。
- **检验与图**：主表报告 seed 均值 ± seed 间标准差；配对 Wilcoxon 以场景为单位（先对 seed 取平均）。轨迹图画 test 场景 21。

## 微调配方修订（v1 → v2）

v1（流水线 `d458a73` 的结果）的训练日志显示：进入 M3 时 α 从 0.0025 骤降到 0.0005，三个 seed 的策略熵都持续下降；seed 4102 两次发散，低优先级任务在计算队列积压到上百秒，U2U 转发变多而总 CPU 功率下降，critic 残差升高约 20 倍；dev 上它的全部微调候选都不如 M1。v2 只改 M3 和 Hover-SAC 的微调参数（`p2/common.py: FINETUNE_RECIPES`），依据是训练诊断，按 dev 选档，不看 test：

| 配方 | α | actor 学习率 | 只更新 critic 的回合 |
| --- | --- | --- | --- |
| v1 | 0.0005 | 1e-4 | 3 |
| v2 | 0.0025（保持 M1 末值） | 3e-5 | 10 |

配方属于输出目录（`<out>/recipe.json`，没有该文件即 v1），同一目录不会混用两种配方。从已完成的 v1 目录派生 v2 目录：

```powershell
python run.py revise --from <v1 输出目录> --out <新目录>
python run.py pipeline --out <新目录>
```

`revise` 只读旧目录，把与微调无关的部分（M1、M2、三条 PPO 基线的训练；它们和 M1 候选的 dev 结果；TFOO / TOO / NC / CA 的 test 与扫参记录；上层与基线的选档）用硬链接（不支持时复制）带到新目录，并标记对应任务已完成。随后的 pipeline 只重跑 M3、Hover-SAC 微调、它们的 dev 候选与选档、Proposed 与 Hover-SAC 的 test / 扫参 / 饱和点，以及报表。`summary.md` 和 `stats.json` 记录所用配方；`stats.json` 另含每个 seed 的配对数值，`results/selection/` 存各 seed 的选档记录。

## 速度：做了什么、为什么

在本机 profile 了一个 900 s 回合，纯仿真约 11 s。
- **仿真热点**：`_publish` 约占 33%，U2U 资源分配约占 24%。`p2/perf.py` 把它们改成逐位等价的写法（同样顺序的浮点加法，加上纯函数缓存），由 `tests/test_perf_equivalence.py` 逐个报告、逐步核对。实测标称场景快 1.2–1.6 倍。
- **训练专用 adapter**：训练回合跳过两项诊断统计（队列积分、逐步 CPU 服务量）。它们不进入奖励，也不进入 Lite 观测；物理过程、奖励、观测逐位一致。这两项每 0.1 s 都要遍历全部积压，NC 这类过载回合原来要 50–85 s，现在快 1.15–1.3 倍。评估仍用完整 adapter。
- **更大的微批**：SAC 用 128，PPO 用 150（原来是 16）。损失是同一个 minibatch 上的同一个和，只是少了大量小 kernel 调用。
- **CPU 还是 GPU**：网络都是 110→256→256 级别的小 MLP，每 0.5 s 做一次单样本推理，GPU 的启动和拷贝开销可能比计算还大。用 `python run.py bench` 实测后再定 `P2_DEVICE`。真正的瓶颈是 CPU 上的仿真，靠多进程并行提速。

## 文件

- `p2/common.py`：实验设计常量、种子派生、场景缩放（CPU / 任务量）、结果汇总（不需要 torch）。
- `p2/learn.py`：M1 / M2 / M3 / Hover-SAC / 三条 PPO 基线的训练，以及评估用策略。
- `p2/evaluate.py`：单回合闭环评估、dev 选档、test、扫参、轨迹记录。
- `p2/report.py`、`p2/plots.py`：表格、配对检验、图。
- `p2/pipeline.py`：带依赖的多进程任务调度（可续跑）。
- `p2/revise.py`：从已完成的目录派生新微调配方的目录（`run.py revise`）。
- `p2/perf.py`：逐位等价的仿真加速与训练专用 adapter。
- `tests/`：等价性、种子一致性、评估与报表的测试。
