# 可行性阶段：状态

**运行中：两个试点（3 个种子，完整流程）**

最后更新：2026-10-19 13:42:48（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

- 主实验目录：`D:\p2_runs_v2`
- 输出目录：`D:\p2_sweeps_v3`
- 代码指纹：`9734a775ee4e`；P2_DEVICE=cpu；P2_THREADS=未设置（2）；进程数 8
- GitHub 推送：成功（分支 paper2-runs，文件夹 p2_sweeps_v3/）

| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |
|---|---|---|---|---|---|
| 1 | 环境检查 | 通过 | 2026-10-19 13:27:16 | 2026-10-19 13:27:43 | 0 分钟 |
| 2 | 自检脚本 | 通过 | 2026-10-19 13:27:47 | 2026-10-19 13:29:20 | 1 分钟 |
| 3 | 与主实验记录比较 | 通过 | 2026-10-19 13:29:25 | 2026-10-19 13:40:26 | 11 分钟 |
| 4 | 扫参管道冒烟 | 通过 | 2026-10-19 13:40:38 | 2026-10-19 13:42:39 | 2 分钟 |
| 5 | 两个试点（3 个种子，完整流程） | 运行中 | 2026-10-19 13:42:48 |  |  |
| 6 | 试点检查、耗时、汇总 | 未开始 |  |  |  |

输出目录所在磁盘剩余 388 GB。

## 进度（run.py progress）

```
point           jobs done  job-hours  estimate  state
cpu_ghz=9       124/124         41.4      41.4  finished
u2u_mhz=1.5     110/124        100.5     113.3  running

finished points: 1 (41.4 job-hours each on average)
remaining: 1 running and 0 waiting points, about 13 job-hours; with 8 workers about 2 h (0.1 days) if every worker stays busy
Estimates from job counts: long training jobs make early estimates of a running point too low.
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
simulator: 20 s of the nominal scene in 0.2 s, 39 terminals
smoke training M1: ok in 7 s
smoke training TFOO: ok in 11 s

ALL CHECKS PASSED
```

## 自检脚本：输出摘录

```
OK random scenario 22 scene 0: identical; 1.72s -> 1.30s (1.32x)
OK local  scenario 21 scene 0: identical; 1.94s -> 1.34s (1.45x)
OK random scenario 23 scene 1: identical; 4.32s -> 2.82s (1.53x)
OK hover  scenario 24 scene 2: identical; 3.77s -> 2.26s (1.67x)
OK local  scenario 25 scene 3: identical; 1.10s -> 0.95s (1.16x)
OK random scenario 26 scene 4: identical; 1.78s -> 1.57s (1.14x)
OK hover  scenario 27 scene 4: identical; 1.74s -> 1.51s (1.15x)
OK training adapter local scenario 21: identical except diagnostics; 1.45s -> 1.36s (1.06x)
OK training adapter random scenario 22: identical except diagnostics; 1.37s -> 1.41s (0.97x)

=== 2026-10-19 13:29:18  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\tests\test_identity.py
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
ALL IDENTICAL -> D:\p2_sweeps_v3\_feasibility\verify\verify_9bcb1b2fbd6c.json
```

## 扫参管道冒烟：输出摘录

```
DONE  u2u_mhz=1.5/s4101_test_Proposed (0.2 min) [54/60]
DONE  u2u_mhz=1.5/s4101_test_TFOO (0.2 min) [55/60]
DONE  u2u_mhz=1.5/s4101_test_TOO (0.2 min) [56/60]
DONE  u2u_mhz=1.5/s4101_test_CA (0.1 min) [57/60]
DONE  u2u_mhz=1.5/s4101_test_NC (0.1 min) [58/60]
DONE  u2u_mhz=1.5/s4101_test_HoverSAC (0.1 min) [59/60]
START u2u_mhz=1.5/report
DONE  u2u_mhz=1.5/report (0.0 min) [60/60]
All jobs finished.

=== 2026-10-19 13:42:38  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py retrain-report --out-base=D:\p2_sweeps_v3\_feasibility\smoke --main=D:\p2_runs_v2 --points=users=7,u2u_mhz=1.5 --smoke --seeds=4101
D:\p2_sweeps_v3\_feasibility\smoke\results
```
