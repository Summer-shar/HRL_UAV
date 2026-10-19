# 可行性阶段：状态

**已停止：试点检查未通过**

最后更新：2026-10-19 18:59:09（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

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
| 5 | 两个试点（3 个种子，完整流程） | 通过 | 2026-10-19 13:42:48 | 2026-10-19 18:58:52 | 5.3 小时 |
| 6 | 试点检查、耗时、汇总 | 失败 | 2026-10-19 18:58:57 | 2026-10-19 18:59:09 | 0 分钟 |

## 停止原因

试点检查未通过（判据见 docs/PROTOCOL.md 第 5 节），按方案先讨论，不继续其余点。日志：`D:\p2_sweeps_v3\_feasibility\summary.log`，汇总表：`D:\p2_sweeps_v3\results\retrain_summary.md`。

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

## 两个试点（3 个种子，完整流程）：输出摘录

```
START u2u_mhz=1.5/s4103_test_NC
START u2u_mhz=1.5/s4103_test_HoverSAC
START u2u_mhz=1.5/s4103_test_CA
DONE  u2u_mhz=1.5/s4103_test_TFOO (14.9 min) [242/248]
DONE  u2u_mhz=1.5/s4103_test_NC (17.8 min) [243/248]
DONE  u2u_mhz=1.5/s4103_test_HoverSAC (51.2 min) [244/248]
DONE  u2u_mhz=1.5/s4103_test_Proposed (58.1 min) [245/248]
DONE  u2u_mhz=1.5/s4103_test_TOO (77.0 min) [246/248]
DONE  u2u_mhz=1.5/s4103_test_CA (88.1 min) [247/248]
START u2u_mhz=1.5/report
DONE  u2u_mhz=1.5/report (0.0 min) [248/248]
All jobs finished.
```

## 试点检查、耗时、汇总：输出摘录

```

=== 2026-10-19 18:59:07  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py pilot-check --out-base=D:\p2_sweeps_v3 --main=D:\p2_runs_v2 --seeds=4101,4102,4103
Pilot point cpu_ghz=9 (gates: cpu_ghz, load, users): M1 trainability check (last 20 episodes; threshold 3 x nominal)
  seed 4101: first 22.463 s -> last 8.600 s (nominal last 3.172 s); improves: yes; within 3x nominal: yes
  seed 4102: first 13.551 s -> last 2.883 s (nominal last 2.433 s); improves: yes; within 3x nominal: yes
  seed 4103: first 8.509 s -> last 2.554 s (nominal last 2.203 s); improves: yes; within 3x nominal: yes
Pilot point u2u_mhz=1.5 (gates: u2u_mhz): M1 trainability check (last 20 episodes; threshold 3 x nominal)
  seed 4101: first 34.143 s -> last 24.685 s (nominal last 3.172 s); improves: yes; within 3x nominal: NO
  seed 4102: first 56.235 s -> last 29.367 s (nominal last 2.433 s); improves: yes; within 3x nominal: NO
  seed 4103: first 57.611 s -> last 59.522 s (nominal last 2.203 s); improves: NO; within 3x nominal: NO
FAIL: the gated points wait; stop and discuss (do not change the protocol silently)
=== exit code 1

=== 2026-10-19 18:59:08  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py progress --out-base=D:\p2_sweeps_v3 --seeds=4101,4102,4103 --workers=8
point           jobs done  job-hours  estimate  state
cpu_ghz=9       124/124         41.4      41.4  finished
u2u_mhz=1.5     124/124        111.0     111.0  finished
cpu_ghz=11        0/124          0.0      76.2  not started
cpu_ghz=12        0/124          0.0      76.2  not started
cpu_ghz=14        0/124          0.0      76.2  not started
load=0.6          0/124          0.0      76.2  not started
load=0.8          0/124          0.0      76.2  not started
load=0.9          0/124          0.0      76.2  not started
load=1.1          0/124          0.0      76.2  not started
users=7           0/124          0.0      76.2  not started
users=10          0/124          0.0      76.2  not started
users=16          0/124          0.0      76.2  not started
users=19          0/124          0.0      76.2  not started
u2u_mhz=2         0/124          0.0      76.2  not started
u2u_mhz=2.5       0/124          0.0      76.2  not started
u2u_mhz=4         0/124          0.0      76.2  not started

finished points: 2 (76.2 job-hours each on average)
remaining: 0 running and 14 waiting points, about 1067 job-hours; with 8 workers about 133 h (5.6 days) if every worker stays busy
Estimates from job counts: long training jobs make early estimates of a running point too low.

=== 2026-10-19 18:59:08  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py retrain-report --out-base=D:\p2_sweeps_v3 --main=D:\p2_runs_v2 --points=cpu_ghz=9,u2u_mhz=1.5 --seeds=4101,4102,4103
D:\p2_sweeps_v3\results
```
