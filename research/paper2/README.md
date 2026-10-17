# 第二个研究点：最终实验流水线（paper2）

依据代码：`research/speedup/src`。主实验（`D:\p2_runs_v2`）用 v2.4.0（源 commit `25d59ff`）跑出；2026-10-17 起为 v2.5.0 = v2.4.0 + 每区终端数参数化（每区 13 个时与 v2.4.0 逐位一致，由 `tests/test_perf_equivalence.py` 与 `run.py verify` 检查）。奖励、观测（上层按终端行数自适应）、PPO / SAC / 混合动作分布不变。本目录组织训练阶段、评估、扫参和画图。

**扫参设计 v3（逐点从零重训，CPU / 负载系数 / 用户数）见 [PROTOCOL.md](PROTOCOL.md)，运行步骤见下文“设计 v3”一节。**

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
- **重训扫参**（`p2/retrain.py`，见下节）：每个非标称点把所有学习型方法从头重新训练，再在同样 8 个扫参场景上评估；与上一条“标称训练、直接评估”的结果分开存放、分开报表。
- **突发业务实验**（`p2/burst.py`）：在 8 个扫参场景、标称参数下，令期望计算需求最大的区域在 300–400 s 内全部终端处于繁忙强度（3 倍到达率），其余时间不变；任务到达与策略无关，各方法配对比较。记录按任务到达时刻分箱（10 s）的加权时延与高、低优先级时延，报表输出 `burst_bins.csv`、`burst_windows.csv` 和 `summary.md` 末尾的前/中/后三段对比。
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

## 设计 v3：逐点从零重训（当前方案，单机）

方案、判据和论文呈现见 [PROTOCOL.md](PROTOCOL.md)。CPU 9/11/12/14 GHz、负载系数 0.6/0.8/0.9/1.1、每区终端数 7/10/16/19（保持总负载不变）、U2U 总带宽 1.5/2/2.5/4 MHz，共 16 个点；每个点所有学习算法按主实验流程从零训练，在 20 个测试场景上评估；标称点即主实验。

```powershell
python tests\test_perf_equivalence.py 150         # 加速补丁逐位等价（含用户数场景与 Lite 观测）
python tests\test_users.py                        # 用户数场景
python tests\test_retrain.py                      # 扫参管道（不需要 torch）
python tests\test_checks.py                       # 试点检查与耗时估计（不需要 torch）
python run.py verify --main D:\p2_runs_v2 --work D:\p2_verify      # 新代码复现主实验记录？
python run.py retrain --out-base D:\p2_smoke_v3 --points users=7,u2u_mhz=1.5 --smoke --seeds 4101 --workers 4
python run.py timing --main D:\p2_runs_v2 --workers 8              # 预计耗时
python run.py retrain --out-base D:\p2_sweeps_v3 --main D:\p2_runs_v2 --workers 8   # 全部点，可续跑
python run.py pilot-check --out-base D:\p2_sweeps_v3 --main D:\p2_runs_v2         # 两个试点的检查（任务图里会自动做）
python run.py retrain-report --out-base D:\p2_sweeps_v3 --main D:\p2_runs_v2      # 随时汇总
```

- `retrain-report` 输出 `results\retrain_summary.md`（各点完成情况、扫参表、配对检验、训练曲线检查）、`retrain_sweep_<参数>.csv`、`retrain_paired.csv`、`retrain_convergence.csv`、`retrain_stats.json`。
- 优先级：两个试点 > CPU > 负载系数 > 用户数 > U2U，整条曲线依次完成；空闲进程会提前开始后面的点。
- 试点门控（PROTOCOL.md 第 5 节）：CPU 9 GHz 三个种子的 M1 完成后，`pilot_gate_cpu_ghz` 自动检查，放行其余 CPU、负载系数和用户数点；U2U 1.5 MHz 的 M1 完成后，`pilot_gate_u2u_mhz` 放行 U2U 2/2.5/4 MHz。不通过时对应的点挂起（日志 `_jobs\pilot_gate_<参数>.log`），其余照常。

## 重训扫参（设计 v2，已停用）：每个参数点重新训练

场景参数一变，策略就应在新场景下重新训练。`retrain` 为每个非标称点建一个独立输出目录 `<out-base>\<参数>=<取值>\`，其中 `scene.json` 记录该点的参数（`common.load_scene(root)` 据此缩放场景，v2.4.0 源码不动），`recipe.json` 固定为 v2 配方。seed、训练场景、初始权重、探索噪声流、训练阶段和选档规则都与主实验相同，唯一不同的是这一个场景参数。每个点在扫参场景 21–28 上 test（不在点目录里再做嵌套扫参或突发实验）；标称点直接取主实验在同样 8 个场景上的 test 结果。

```powershell
python tests\test_retrain.py                                        # 不需要 torch
python run.py retrain --smoke --out-base D:\p2_retrain_smoke --points u2u_mhz=2 --workers 4
python run.py retrain --out-base D:\p2_retrain --first u2u_mhz=1.5,u2u_mhz=2 --workers N
python run.py retrain-report --out-base D:\p2_retrain --main D:\p2_runs_v2
```

- 默认重训全部 14 个稳定点（CPU 9/11/12/13/14 GHz，任务量 0.6/0.7/0.8/0.9/1.1，U2U 1.5/2/4.5/6 MHz），共 1736 个任务，可用 `--points` 只跑一部分。
- 所有点放在**同一个任务图**里调度：`--first` 指定的点整体优先（优先级 +1000）；它们的任务因依赖暂时不能开始时，空闲进程自动去跑其他点，进程不会闲着。中断后原样重跑同一条命令即可续跑。
- 每个点内部的关键路径是 M1 → M2 → 选上层 → M3 → 选下层 → test，与主实验相同；增加 `--workers` 只能让更多点同时推进，不能缩短单个点的关键路径。
- 训练阶段结束后删除该阶段的续跑快照 `state.pt`（含经验池），只保留 `final.pt` 和候选权重，以节省磁盘。
- **多台电脑**：各点互相独立，可以用 `--points` 分给几台电脑，每台各用自己的 `--out-base`；同一个点的所有方法必须在同一台电脑上训练。每台电脑用 `python run.py retrain-export --out-base <目录> --points <本机的点> --dest <zip> --machine <标签>` 导出已完成的点（只含记录：scene/recipe/design、选档、test JSON、train.jsonl、dev summary，不含权重、轨迹和日志；附 `manifest_<标签>.json`，记录软件版本、P2_DEVICE / P2_THREADS、代码指纹和每个文件的 SHA-256）。各 zip 解压到同一个文件夹后，用它作 `--out-base` 运行 `retrain-report`；`--main` 可以指向主实验目录，或论文仓库里 `results_v2/raw` 这样只含 test 记录的导出。
- `retrain-report` 只汇总已经完成的点，可以随时运行：`<out-base>\results\retrain_summary.md`（每个点是否完成 + 各参数的重训扫参表）、`retrain_sweep_<参数>.csv`（标称点 + 已完成的点）和 `results\points\<点>.csv`。每个点目录下也有自己的 `results\summary.md`。

## 速度：做了什么、为什么

v3 补充（2026-10-17）：积压大时，每 0.1 s 和每 0.5 s 都要把系统内全部任务过一遍（U2U 1.5 MHz 的悬停均分回合结束时有 6324 个任务，900 s 仿真 87 s，八成在 `PhysicsAdapter.report`）。`p2/perf.py` 新增四处逐位等价的改写：报告里未变化任务的记录对象复用（按属性对象的同一性判断）、`view` 用纯 Python 浮点按原顺序累加、驻留代价用存活计数精确求和、CPU 频率分配的排队量求和在前缀已超过可用频率时提前结束。重积压的训练回合约快 2.2 倍，标称附近约 1.1–1.5 倍；复制的方法带源代码 SHA-256 保护。

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
- `p2/burst.py`：受控突发业务实验（只包装单次评估的场景实例，不改 v2.4.0 源码）。
- `p2/revise.py`：从已完成的目录派生新微调配方的目录（`run.py revise`）。
- `p2/retrain.py`：重训扫参的点目录（`run.py retrain` / `retrain-report`）。
- `p2/perf.py`：逐位等价的仿真加速与训练专用 adapter。
- `tests/`：等价性、种子一致性、评估与报表的测试。
