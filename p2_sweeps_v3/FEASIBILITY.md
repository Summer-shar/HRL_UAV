# 可行性阶段：状态

**运行中：自检脚本完成**

最后更新：2026-10-19 13:29:20（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

- 主实验目录：`D:\p2_runs_v2`
- 输出目录：`D:\p2_sweeps_v3`
- 代码指纹：`9734a775ee4e`；P2_DEVICE=cpu；P2_THREADS=未设置（2）；进程数 8
- GitHub 推送：成功（分支 paper2-runs，文件夹 p2_sweeps_v3/）

| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |
|---|---|---|---|---|---|
| 1 | 环境检查 | 通过 | 2026-10-19 13:27:16 | 2026-10-19 13:27:43 | 0 分钟 |
| 2 | 自检脚本 | 通过 | 2026-10-19 13:27:47 | 2026-10-19 13:29:20 | 1 分钟 |
| 3 | 与主实验记录比较 | 未开始 |  |  |  |
| 4 | 扫参管道冒烟 | 未开始 |  |  |  |
| 5 | 两个试点（3 个种子，完整流程） | 未开始 |  |  |  |
| 6 | 试点检查、耗时、汇总 | 未开始 |  |  |  |

输出目录所在磁盘剩余 388 GB。

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
