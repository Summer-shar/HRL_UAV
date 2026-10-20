# 零样本泛化扫参（方案 v4）：状态

**运行中：全部 16 个点的测试**

最后更新：2026-10-20 13:31:35（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

- 主实验目录：`D:\p2_runs_v2`
- 输出目录：`D:\p2_zeroshot_v4`
- 代码指纹：`f3aab1118cdf`；P2_DEVICE=cpu；P2_THREADS=未设置（2）；进程数 8
- GitHub 推送：成功（分支 paper2-runs，文件夹 p2_zeroshot_v4/）

| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |
|---|---|---|---|---|---|
| 1 | 管道冒烟（terminals=7、u2u_mhz=1.5，1 个种子） | 通过 | 2026-10-20 13:07:41 | 2026-10-20 13:07:56 | 0 分钟 |
| 2 | 逐位一致性检查（标称点对主实验；新代码重算已有点） | 通过 | 2026-10-20 13:08:01 | 2026-10-20 13:31:30 | 23 分钟 |
| 3 | 全部 16 个点的测试 | 运行中 | 2026-10-20 13:31:35 |  |  |
| 4 | 汇总表与配对检验 | 未开始 |  |  |  |

输出目录所在磁盘剩余 388 GB。

## 进度（run.py progress）

```
point               records
cpu_ghz=9        360/360   
cpu_ghz=11       360/360   
cpu_ghz=12       360/360   
cpu_ghz=14       360/360   
load=0.6         360/360   
load=0.8         360/360   
load=0.9         360/360   
load=1.1         360/360   
terminals=7        0/360   
terminals=9        0/360   
terminals=11       0/360   
terminals=15       0/360   
u2u_mhz=1.5      360/360   
u2u_mhz=2        360/360   
u2u_mhz=2.5      360/360   
u2u_mhz=4        360/360   

records: 4320/5760
about 0.0 h left at the rate so far (slow jobs run first, so this estimate is pessimistic early on)
```

## 管道冒烟（terminals=7、u2u_mhz=1.5，1 个种子）：输出摘录

```
START terminals=7/s4101_zeroshot_HoverSAC
START terminals=7/s4101_zeroshot_CA
DONE  terminals=7/s4101_zeroshot_CA (0.1 min) [7/12]
DONE  terminals=7/s4101_zeroshot_NC (0.1 min) [8/12]
DONE  terminals=7/s4101_zeroshot_Proposed (0.1 min) [9/12]
DONE  terminals=7/s4101_zeroshot_TFOO (0.1 min) [10/12]
DONE  terminals=7/s4101_zeroshot_TOO (0.1 min) [11/12]
DONE  terminals=7/s4101_zeroshot_HoverSAC (0.1 min) [12/12]
All jobs finished.

=== 2026-10-20 13:07:55  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py zeroshot-report --out-base=D:\p2_zeroshot_v4\_zeroshot\smoke --main=D:\p2_runs_v2 --points=terminals=7,u2u_mhz=1.5 --smoke --seeds=4101
D:\p2_zeroshot_v4\_zeroshot\smoke\results
```

## 逐位一致性检查（标称点对主实验；新代码重算已有点）：输出摘录

```
nominal    nominal         360 records vs D:\p2_runs_v2: IDENTICAL
regression cpu_ghz=11       40 records vs D:\p2_zeroshot_v4\cpu_ghz=11: IDENTICAL
regression load=0.8         40 records vs D:\p2_zeroshot_v4\load=0.8: IDENTICAL
regression users=10         40 records vs D:\p2_zeroshot_v4\users=10: IDENTICAL
regression u2u_mhz=2.5      40 records vs D:\p2_zeroshot_v4\u2u_mhz=2.5: IDENTICAL
note: u2u_mhz exists only in the new records (evaluation records, added in commit d458a73 (U2U sweep); the main run's TFOO / TOO / NC test records were written before it)
ALL IDENTICAL (520 records) -> D:\p2_zeroshot_v4\_zeroshot\consistency.json
```
