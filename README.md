# MH-TPCO：多无人机移动边缘计算的多时间尺度分层强化学习

第二篇论文的完整实验代码。上层 PPO 每 5 s 规划 3 架无人机的位移，下层 SAC 每 0.5 s 按任务优先级和源无人机拆分计算任务（本地计算或经 U2U 链路转发），共同降低优先级加权任务时延和能耗。代码包括仿真器、两层算法、5 个对比算法、主实验流水线和逐点重训扫参。

装好 Python 环境就能运行，不需要其他文件。所有命令都在仓库根目录（本文件所在目录）下执行，示例用 Windows PowerShell 写法。

## 目录

```
├── run.py                  唯一入口：python run.py <命令>，python run.py -h 看全部命令
├── requirements.txt        依赖（版本范围）
├── requirements-lock.txt   历史验证环境的确切版本
├── environment.yml         conda 环境
├── pyproject.toml          包信息（可选：pip install -e .）
├── src/
│   ├── hrl_uav/            仿真器：场景生成、终端移动、上传 / U2U / CPU 排队与能耗
│   ├── hrl_v2/             强化学习组件：观测、混合支撑集卸载分布、SAC、PPO、指标、TFOO 的 CVX 卸载
│   │   └── configs/        场景（physical_scene.json）与超参数（v2_4_0_lite.json）
│   └── p2/                 实验流水线：训练阶段、选档、测试、扫参、突发实验、报表、作图、任务调度
├── tests/                  自检脚本（见第 2 节）
├── docs/
│   ├── EXPERIMENTS.md      主实验设计：算法、训练阶段、选档与评价、加速措施
│   ├── PROTOCOL.md         逐点重训扫参的实验方案（运行前写定）
│   ├── CODE_MAP.md         代码地图：每个文件做什么、按什么顺序读
│   └── HISTORY.md          版本来源、已有主实验结果的来历、本代码包与原仓库的一致性
├── CLAUDE.md               给 AI 助手的阅读入口
└── outputs/                默认输出目录（不入库）
```

## 1 安装环境

**已经跑过主实验的电脑**：直接用当时的 Python 环境（同样的 torch、numpy、cvxpy 版本），不要升级任何包，跳到第 2 节。同一环境才能逐位复现已有结果。

**新电脑**（Windows 或 Linux，NVIDIA 显卡可选）：

```powershell
conda env create -f environment.yml        # 创建环境 mhtpco（Python 3.9 + numpy/scipy/cvxpy/matplotlib）
conda activate mhtpco
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128    # RTX 50 系等新显卡需要 CUDA 12.8 版
# 没有 NVIDIA 显卡：pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

不用 conda 时：先装 Python 3.9（3.10–3.12 也可），按上面的命令先装 torch，再 `pip install -r requirements.txt`（顺序不能反，否则 pip 可能装上不带 CUDA 的 torch）。各包版本见 `requirements-lock.txt`；换了版本也能运行，但 torch、cvxpy / clarabel 版本不同时数值可能在末几位不同，因此不能与已有结果逐位比较（第 2 节的 `verify` 会指出）。

## 2 自检

```powershell
python run.py check-env              # 版本、显卡、仿真器，以及 M1 和 TFOO 的冒烟训练（几分钟），最后应为 ALL CHECKS PASSED
python tests\test_users.py           # 用户数场景的生成与观测维度
python tests\test_retrain.py         # 扫参任务图、门控、报表（不需要 torch）
python tests\test_checks.py          # 试点检查、耗时估计、进度（不需要 torch）
python tests\test_eval_report.py     # 评估与报表（不需要 torch）
python tests\test_verify_compare.py  # verify 的记录比较（不需要 torch）
python tests\test_feasibility.py     # 可行性 / 扫参命令的流程与状态文件（不需要 torch）
python tests\test_publish.py         # 状态推送到 git 分支（需要 git）
python tests\test_perf_equivalence.py 150   # 加速补丁与原始仿真逐位一致，每行都应是 identical
python tests\test_identity.py        # 随机数派生与原实现一致（需要 torch）
```

已有主实验结果目录（例如 `D:\p2_runs_v2`）时，再检查本代码能否复现它：

```powershell
python run.py verify --main D:\p2_runs_v2 --work D:\p2_verify     # 半小时到一小时；最后一行应为 ALL IDENTICAL
```

它用本代码重跑各训练阶段的前 3–5 个回合和各算法的 2 个测试场景，与主实验记录逐项比较（除耗时外，每个字段都比，列出全部不同的值）。主实验目录只读。主实验里有些记录由更早版本的流水线写成，缺少后来新增的字段（`actor_lr`、`u2u_mhz`），这些字段会以 `note:` 列出，不算差异。

**设备与进程**：`check-env` 会显示 `P2_DEVICE`（不设时有显卡用 `cuda:0`，否则 `cpu`）和 `P2_THREADS`（不设时 2）。要和已有结果逐位一致，用主实验时的设备：它写在 `<主实验目录>\seed_4101\train\M1\contract.json` 的 `"device"` 字段。`python run.py bench` 可以实测 CPU 与 GPU 哪个快。设置方法：`$env:P2_DEVICE="cuda:0"`、`$env:P2_THREADS="2"`。

## 3 主实验（标称参数）

```powershell
python run.py pipeline --smoke --workers 4 --out D:\p2_smoke     # 冒烟：极小训练量，几分钟跑通全流程，数值无意义
python run.py pipeline --workers 8 --out D:\p2_main              # 正式：3 个种子 × 6 种算法，可断点续跑
python run.py report --out D:\p2_main                            # 只重做表和图
```

- 内容：MH-TPCO（M1 → M2 → 选上层 → M3 → 选下层）、TFOO / TOO / NC（各 PPO 400 回合）、HoverSAC、CA；20 个测试场景；固定策略扫参与突发业务实验。设计见 `docs/EXPERIMENTS.md`。不需要固定策略扫参时加 `--no-sweeps`（论文的扫参改用第 4 节的逐点重训）。
- 新建的输出目录自动使用论文的微调配方 v2。
- 耗时：旧版流水线（151 个任务）曾在你的电脑上用约 7 小时；现在任务更多（含 CA、突发实验），会更久。
- 已有的主实验结果 `D:\p2_runs_v2` 就是这一步的产物（来历见 `docs/HISTORY.md`），不需要重跑。
- 结果在 `<out>\results\`：`summary.md`（主表、配对检验）、`main_table.csv`、`stats.json`、`burst_*.csv`，图在 `results\figures\`。

## 4 扫参

**主结果：零样本泛化扫参（方案 v4）**。只用主实验在标称参数下训练好的模型，不再训练，直接在 16 个参数点上测试（20 个测试场景 × 3 个种子，配对检验），用来衡量部署后参数变化时各方法的鲁棒性：

```powershell
python run.py zeroshot --main D:\p2_runs_v2 --out-base D:\p2_zeroshot_v4
```

- 参数：CPU 频率 9/11/12/14 GHz，负载系数 0.6/0.8/0.9/1.1，每区终端数 7/9/11/15（`terminals`：每个终端负载不变，终端越多总负载越大），U2U 总带宽 1.5/2/2.5/4 MHz；标称值即主实验。
- 步骤：管道冒烟 → 逐位一致性检查（标称点对主实验，新代码重算已有点，`p2/consistency.py`；任一不同即停止）→ 全部点的测试（已完成的点跳过）→ 汇总。
- 状态写在 `ZEROSHOT.md`，并推送到 GitHub。
- 结果在 `results\zeroshot_summary.md`、`zeroshot_sweep_<参数>.csv`（含种子间最小/最大值）、`zeroshot_per_seed.csv`、`zeroshot_paired.csv`；一致性检查结果在 `_zeroshot\consistency.json`。

下面的逐点从零重训（方案 v3）只完成了两个试点，原因见 `docs/PROTOCOL.md` 的“方案 v4”一节。

## 4b 逐点重训扫参（方案 v3，已停在试点）

方案见 `docs/PROTOCOL.md`。4 个参数各 5 个取值，标称值即主实验：

| 参数 | 取值（加粗为标称） |
|---|---|
| 无人机 CPU 频率 / GHz | 9, **10**, 11, 12, 14 |
| 负载系数 | 0.6, 0.8, 0.9, **1.0**, 1.1 |
| 每区终端数（共 3 倍） | 7, 10, **13**, 16, 19 |
| U2U 总带宽 / MHz | 1.5, 2, 2.5, **3**, 4 |

共 16 个非标称点。每个点上所有学习算法都按主实验流程、同样的训练量从零训练，在 20 个测试场景上评估。

**第一步：可行性（一条命令）**

```powershell
python run.py feasibility --main D:\p2_runs_v2 --out-base D:\p2_sweeps_v3
```

依次运行：

1. 环境检查；
2. 全部自检脚本；
3. 与主实验记录比较（`verify`）；
4. 用户数 7 与 U2U 1.5 MHz 的扫参冒烟；
5. 两个最难的点（CPU 9 GHz 计算最紧，U2U 1.5 MHz 通信最紧），3 个种子跑完整流程；
6. 试点检查、按实测耗时估计剩余时间、汇总表。

**状态始终写在 `D:\p2_sweeps_v3\FEASIBILITY.md`**：

- 内容包括每一步是否通过、开始与结束时间；第 5 步运行时每 10 分钟更新一次进度；
- 任何一步失败就停在那一步，文件开头写明哪一步、哪条命令、日志在哪里；
- 修正后原样重跑同一条命令，已通过的检查自动跳过，第 5 步从断点继续。

其他设置：

- 不设 `P2_DEVICE` 时自动取主实验 M1 的设备；
- 进程数默认取“逻辑核数 ÷ P2_THREADS”（最多 8），可用 `--workers` 改；
- 第 5 步推测约半天到一天。

试点检查的判据：每个种子的 M1 最后 20 回合比最初 20 回合好，且不超过主实验 M1 的 3 倍。也可以手动分步运行：`retrain --points "cpu_ghz=9,u2u_mhz=1.5"`、`pilot-check`、`progress`、`retrain-report`。

**第二步：其余点**（第一步通过后）

```powershell
python run.py sweep --main D:\p2_runs_v2 --out-base D:\p2_sweeps_v3
```

- 用同一个输出目录，试点已完成的任务自动跳过。第一步的两个试点和试点检查没有跑完时，这条命令拒绝运行；某个试点检查未通过时，只挂起它门控的点，其余照常（PROTOCOL 第 5 节）。
- 任务图里有两个门控：`pilot_gate_cpu_ghz` 放行其余 CPU、负载、用户数点，`pilot_gate_u2u_mhz` 放行其余 U2U 点。日志在 `D:\p2_sweeps_v3\_jobs\pilot_gate_*.log`；门控不通过时对应的点挂起，其余照常。
- 状态写在 `SWEEP.md`，汇总表每 30 分钟更新一次。全部完成后自动做最终汇总，并打包只含记录的 `records.zip`。

**汇总与出图**

```powershell
python run.py retrain-report --out-base D:\p2_sweeps_v3 --main D:\p2_runs_v2   # 随时可运行，只读
python run.py retrain-export --out-base D:\p2_sweeps_v3 --dest D:\p2_sweeps_v3_records.zip   # 只含记录的压缩包
```

`results\` 下有 `retrain_summary.md`（完成情况、各点表格、配对检验、训练曲线检查）、`retrain_sweep_<参数>.csv`、`retrain_paired.csv`、`retrain_convergence.csv`、`retrain_stats.json`。论文的扫参图由论文仓库的 `figures_src/build_paper2_figures.py` 读取这些 csv 生成。

## 5 通过 GitHub 跟踪运行

`feasibility` 和 `sweep` 会把状态推送到本仓库的 `paper2-runs` 分支，路径为 `<输出目录名>/`（例如 `p2_sweeps_v3/`）：

- 推送内容：状态页、各步骤日志的末尾、verify 结果、`results\` 下的汇总表、门控日志、未完成任务日志的末尾；不推送权重、轨迹和完整日志。
- 推送时机：每一步开始和结束时，长步骤运行中每 30 分钟一次。
- 前提：代码文件夹必须是 `git clone` 得到的，推送用本机已有的 GitHub 登录。推送失败不影响实验，状态页上会显示失败原因。
- 手动推送一次：`python run.py publish --out-base D:\p2_sweeps_v3`；不想推送时加 `--no-publish`。

同一分支根目录下的 `NEXT.md` 是给本机的下一步说明。

**无人值守**：`tools/watch.py` 每 3 分钟查看本分支（`paper2-release`）上的 `RUN_REQUEST.json`。出现新的请求编号、且它启动的命令都已结束时，它先拉取代码，再运行请求里指定的固定命令（`zeroshot`、`feasibility` 或 `sweep`，参数都来自 watch.py 自己的命令行）。这样可以在没人操作电脑时开始下一步，或修复问题后重新开始。

## 6 运行注意事项

- **续跑**：中断（断电、重启、报错）后原样重跑同一条命令。已完成的任务跳过，训练从最近的快照继续。
- **同一个输出目录同时只能有一个命令在跑**。`report`、`retrain-report`、`progress`、`pilot-check` 只读，可以随时运行。
- **进程数** `--workers`：约为 CPU 逻辑核数 ÷ `P2_THREADS`，例如 16 核、`P2_THREADS=2` 时用 8。进程数不影响结果。
- **日志**：每个任务在 `<输出目录>\_jobs\<任务>.log`。命令结束时若出现 `Failed or skipped jobs`，看对应日志末尾。
- **睡眠与磁盘**：长时间运行前把系统睡眠设为“从不”。训练阶段结束后会删除含经验池的续跑快照，只保留最终权重和候选权重。
- **不要修改** `src/` 下的代码后接着续跑同一个输出目录：已完成的任务不会重跑，结果会混用两版代码。

## 7 版本

仿真器 v2.5.0 = v2.4.0（每区终端数成为场景参数，每区 13 个时逐位一致）。本代码包由原仓库分支 `claude/paper2-sweeps-v3` 的 commit `50868ac` 整理而来，仿真器与算法文件逐字节相同。详情、主实验结果的来历和一致性检查见 `docs/HISTORY.md`。
