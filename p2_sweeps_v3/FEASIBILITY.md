# 可行性阶段：状态

**运行中：环境检查完成**

最后更新：2026-10-18 17:15:36（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

- 主实验目录：`D:\p2_runs_v2`
- 输出目录：`D:\p2_sweeps_v3`
- 代码指纹：`0455a40d9866`；P2_DEVICE=cpu；P2_THREADS=未设置（2）；进程数 8
- GitHub 推送：成功（分支 paper2-runs，文件夹 p2_sweeps_v3/）

| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |
|---|---|---|---|---|---|
| 1 | 环境检查 | 通过 | 2026-10-18 17:15:11 | 2026-10-18 17:15:36 | 0 分钟 |
| 2 | 自检脚本 | 未开始 |  |  |  |
| 3 | 与主实验记录比较 | 未开始 |  |  |  |
| 4 | 扫参管道冒烟 | 未开始 |  |  |  |
| 5 | 两个试点（3 个种子，完整流程） | 未开始 |  |  |  |
| 6 | 试点检查、耗时、汇总 | 未开始 |  |  |  |

输出目录所在磁盘剩余 390 GB。

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
