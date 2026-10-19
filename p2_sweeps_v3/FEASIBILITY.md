# 可行性阶段：状态

**运行中：两个试点（3 个种子，完整流程）**

最后更新：2026-10-19 11:00:22（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

- 主实验目录：`D:\p2_runs_v2`
- 输出目录：`D:\p2_sweeps_v3`
- 代码指纹：`0455a40d9866`；P2_DEVICE=cpu；P2_THREADS=未设置（2）；进程数 8
- GitHub 推送：成功（分支 paper2-runs，文件夹 p2_sweeps_v3/）

| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |
|---|---|---|---|---|---|
| 1 | 环境检查 | 通过 | 2026-10-18 17:15:11 | 2026-10-18 17:15:36 | 0 分钟 |
| 2 | 自检脚本 | 通过 | 2026-10-18 17:15:43 | 2026-10-18 17:17:01 | 1 分钟 |
| 3 | 与主实验记录比较 | 通过 | 2026-10-18 17:17:06 | 2026-10-18 17:23:26 | 6 分钟 |
| 4 | 扫参管道冒烟 | 通过 | 2026-10-18 17:23:31 | 2026-10-18 17:24:39 | 1 分钟 |
| 5 | 两个试点（3 个种子，完整流程） | 运行中 | 2026-10-18 17:24:46 |  |  |
| 6 | 试点检查、耗时、汇总 | 未开始 |  |  |  |

输出目录所在磁盘剩余 390 GB。

## 进度（run.py progress）

```
point           jobs done  job-hours  state
cpu_ghz=9       124/124         41.4  finished
u2u_mhz=1.5      98/124         87.4  running

finished points: 1, mean 41.4 job-hours per point (41.4-41.4)
remaining: 1 points, about 0 job-hours; with 8 workers about 0 h (0.0 days) if every worker stays busy
Low-U2U and high-load points run slower than the average; read this as an estimate.
```

## 环境检查：输出摘录

```
  torch      2.7.0.dev20250227+cu128
  cvxpy      1.7.1
  clarabel   0.11.1
  matplotlib 3.9.4
torch CUDA available: True (NVIDIA GeForce RTX 5060, CUDA 12.8)
P2_DEVICE=cpu  P2_THREADS=(not set: 2)
cvxpy solvers: CLARABEL, OSQP, SCIPY, SCS
simulator: 20 s of the nominal scene in 0.1 s, 39 terminals
smoke training M1: ok in 6 s
smoke training TFOO: ok in 9 s

ALL CHECKS PASSED
```

## 自检脚本：输出摘录

```
OK random scenario 22 scene 0: identical; 1.44s -> 1.13s (1.28x)
OK local  scenario 21 scene 0: identical; 1.60s -> 1.11s (1.45x)
OK random scenario 23 scene 1: identical; 3.52s -> 2.26s (1.55x)
OK hover  scenario 24 scene 2: identical; 3.12s -> 1.76s (1.77x)
OK local  scenario 25 scene 3: identical; 0.96s -> 0.86s (1.12x)
OK random scenario 26 scene 4: identical; 2.10s -> 1.85s (1.13x)
OK hover  scenario 27 scene 4: identical; 1.64s -> 1.38s (1.18x)
OK training adapter local scenario 21: identical except diagnostics; 1.24s -> 1.22s (1.02x)
OK training adapter random scenario 22: identical except diagnostics; 1.24s -> 1.27s (0.98x)

=== 2026-10-18 17:16:59  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\tests\test_identity.py
derive identical
```

## 与主实验记录比较：输出摘录

```
train M1       3 episodes: identical (fields added after the main run, not compared: actor_lr)
train M2       5 episodes: identical
train M3       3 episodes: identical
train HoverSAC 3 episodes: identical
train TFOO     5 episodes: identical
train TOO      5 episodes: identical
train NC       5 episodes: identical
test  Proposed scenarios [21, 22]: identical
test  TFOO     scenarios [21, 22]: identical (fields added after the main run, not compared: u2u_mhz)
test  TOO      scenarios [21, 22]: identical (fields added after the main run, not compared: u2u_mhz)
test  NC       scenarios [21, 22]: identical (fields added after the main run, not compared: u2u_mhz)
test  HoverSAC scenarios [21, 22]: identical
test  CA       scenarios [21, 22]: identical
note: actor_lr exists only in the new records (SAC training rows, added in commit cfdcf3f (fine-tuning recipe v2))
note: u2u_mhz exists only in the new records (evaluation records, added in commit d458a73 (U2U sweep); the main run's TFOO / TOO / NC test records were written before it)
ALL IDENTICAL -> D:\p2_sweeps_v3\_feasibility\verify\verify_db8c14e65adc.json
```

## 扫参管道冒烟：输出摘录

```
START users=7/report
DONE  u2u_mhz=1.5/s4101_test_TOO (0.1 min) [55/60]
DONE  u2u_mhz=1.5/s4101_test_CA (0.1 min) [56/60]
DONE  users=7/report (0.0 min) [57/60]
DONE  u2u_mhz=1.5/s4101_test_NC (0.1 min) [58/60]
DONE  u2u_mhz=1.5/s4101_test_HoverSAC (0.1 min) [59/60]
START u2u_mhz=1.5/report
DONE  u2u_mhz=1.5/report (0.0 min) [60/60]
All jobs finished.

=== 2026-10-18 17:24:38  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py retrain-report --out-base=D:\p2_sweeps_v3\_feasibility\smoke --main=D:\p2_runs_v2 --points=users=7,u2u_mhz=1.5 --smoke --seeds=4101
D:\p2_sweeps_v3\_feasibility\smoke\results
```
