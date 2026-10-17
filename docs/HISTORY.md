# 版本与来源

## 本代码包

- 分支 `paper2-release`，2026-10-17 由原仓库 `Summer-shar/HRL_UAV` 分支 `claude/paper2-sweeps-v3` 的 commit `50868ac` 整理而来。
- 只带入论文 2 实验实际用到的代码：
  - `src/hrl_uav/`（16 个文件，原 `research/speedup/src/hrl_uav/`）；
  - `src/hrl_v2/`（实验实际导入的 26 个模块、3 个配置文件、`design/` 参考实现，原 `research/speedup/src/hrl_v2/`）；
  - `src/p2/`、`run.py`、`tests/`（原 `research/paper2/`）。
- **逐字节比较**（整理时做的检查）：
  - 相同的文件：仿真器与算法的全部 46 个文件；`src/p2/` 中的 `burst`、`evaluate`、`learn`、`perf`、`pipeline`、`plots`、`positioning`、`revise`。
  - 有改动的文件：
    - `common.py`、`retrain.py`：只改路径常量和注释；
    - `report.py`：只改方案文件的路径文字；
    - `checks.py`、`run.py`：新增只读命令 `check-env`、`progress`；`retrain-export` 不再需要机器标签；新建的主实验目录默认使用微调配方 v2；
    - `tests/`：只改 `sys.path`，另新增 `progress` 的测试。
- **运行比较**：CA 在测试场景 21、22 上运行完整 900 s，场景取标称、每区 7 个终端、U2U 1.5 MHz 三种，原目录与本代码包得到的评估记录和轨迹完全相同（除耗时外）。
- 需要 torch 的训练部分没有在整理环境中运行。与已有主实验记录的一致性，在你的电脑上用 `python run.py verify` 检查。

## 仿真器版本

| 版本 | 位置（原仓库） | 内容 |
|---|---|---|
| v2.3.0 | 根目录 `src/`，commit `a9953dc` | 物理候选版本 |
| v2.4.0 | `research/speedup/src/`，commit `25d59ff` | 终端 Gauss-Markov 移动；U2U 发射功率按每架无人机的总预算分配；存档优化 |
| v2.5.0 | 同上，2026-10-17 | 每区终端数成为场景参数；每区 13 个时与 v2.4.0 逐位一致（`tests/test_perf_equivalence.py`、`run.py verify`） |

本代码包是 v2.5.0。

## 已有主实验结果 `D:\p2_runs_v2` 的来历

1. **v1 运行**：流水线 commit `d458a73`，在你的电脑上完成，输出目录 `D:\p2_runs`。据当时的运行记录：151 个任务，约 7 小时 15 分。
2. **微调配方修订**（commit `cfdcf3f`）。v1 的训练日志显示：
   - 进入 M3 时 α 从 0.0025 骤降到 0.0005，三个种子的策略熵持续下降；
   - 种子 4102 两次发散：低优先级任务在计算队列积压到上百秒，critic 残差升高约 20 倍；
   - dev 上它的全部微调候选都不如 M1。

   v2 只改 M3 和 HoverSAC 的微调参数。依据是训练诊断，按 dev 选档，不看测试结果：

   | 配方 | α | actor 学习率 | 只更新 critic 的回合 |
   |---|---|---|---|
   | v1 | 0.0005 | 1e-4 | 3 |
   | v2 | 0.0025（保持 M1 末值） | 3e-5 | 10 |

3. **派生 v2 目录**：`python run.py revise --from D:\p2_runs --out D:\p2_runs_v2`。
   - 与微调无关的部分从 v1 目录带入：M1、M2、TFOO / TOO / NC 的训练，它们的 dev、测试与扫参记录，以及上层与基线的选档。
   - M3 与 HoverSAC 用配方 v2 重新微调、选档、测试。
   - 配方记录在输出目录的 `recipe.json` 中；没有这个文件的旧目录按 v1 处理。

因此，在同一软件环境中用本代码包从零运行 `pipeline`，应得到与 `D:\p2_runs_v2` 相同的结果（同一代码、同一种子，训练是确定性的）。这是推断，`verify` 检查了它的一部分：各阶段的前几个回合和各算法的 2 个测试场景。换了 torch 或 cvxpy 版本，同一设计的结果可能在末几位不同。

## 扫参设计的演变

| 设计 | 做法 | 状态 |
|---|---|---|
| v1 | 标称参数下训练好的策略直接在 CPU 9–14 GHz、负载 0.6–1.1、U2U 1.5–6 MHz 下评估，场景 21–28 | 已完成，结果在论文仓库 `results_v2/sweep_*.csv`；流水线仍保留（`--no-sweeps` 可关闭） |
| v2 | 逐点重训，场景 21–28，14 个点 | 未运行，已停用 |
| v3 | 逐点从零重训，20 个测试场景，CPU / 负载 / 用户数 / U2U 共 16 个点，两个试点门控 | 当前方案，见 `PROTOCOL.md` |

## 没有带入本代码包的内容

以下内容保留在原仓库的其他分支，需要时去那里查看：

- 原仓库根目录的 v2.3.0 源码；
- `research/experiments/` 的早期训练驱动与诊断；
- `docs/history/` 的历史报告；
- `hrl_v2` 中实验没有导入的模块（旧训练入口、评估与基准脚本等）。
