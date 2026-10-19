# 零样本泛化扫参（方案 v4）：状态

**运行中：全部 16 个点的测试**

最后更新：2026-10-20 06:41:10（运行中每 10 分钟更新一次；如果这个时间停在很久以前，说明程序已经退出）

- 主实验目录：`D:\p2_runs_v2`
- 输出目录：`D:\p2_zeroshot_v4`
- 代码指纹：`4cdb31407064`；P2_DEVICE=cpu；P2_THREADS=未设置（2）；进程数 8
- **GitHub 推送：失败（实验不受影响，继续运行）：git push: fatal: unable to access 'https://github.com/Summer-shar/HRL_UAV.git/': TLS connect error: error:0A000126:SSL routines::unexpected eof while reading**

| 步骤 | 内容 | 状态 | 开始 | 结束 | 用时 |
|---|---|---|---|---|---|
| 1 | 管道冒烟（users=7、u2u_mhz=1.5，1 个种子） | 通过 | 2026-10-20 02:38:24 | 2026-10-20 02:38:47 | 0 分钟 |
| 2 | 全部 16 个点的测试 | 运行中 | 2026-10-20 02:38:52 |  |  |
| 3 | 汇总表与配对检验 | 未开始 |  |  |  |

输出目录所在磁盘剩余 388 GB。

## 进度（run.py progress）

```
point               records
cpu_ghz=9        360/360   
u2u_mhz=1.5      180/360   
cpu_ghz=11       360/360   
cpu_ghz=12       360/360   
cpu_ghz=14       360/360   
load=0.6         360/360   
load=0.8         360/360   
load=0.9         360/360   
load=1.1         360/360   
users=7          360/360   
users=10         264/360   
users=16          60/360   
users=19          60/360   
u2u_mhz=2        180/360   
u2u_mhz=2.5       60/360   
u2u_mhz=4         60/360   

records: 4104/5760
about 1.6 h left at the rate so far (slow jobs run first, so this estimate is pessimistic early on)
```

## 管道冒烟（users=7、u2u_mhz=1.5，1 个种子）：输出摘录

```
START u2u_mhz=1.5/s4101_zeroshot_TFOO
START u2u_mhz=1.5/s4101_zeroshot_HoverSAC
DONE  u2u_mhz=1.5/s4101_zeroshot_TOO (0.2 min) [7/12]
DONE  users=7/s4101_zeroshot_TFOO (0.2 min) [8/12]
DONE  users=7/s4101_zeroshot_CA (0.1 min) [9/12]
DONE  u2u_mhz=1.5/s4101_zeroshot_Proposed (0.1 min) [10/12]
DONE  u2u_mhz=1.5/s4101_zeroshot_TFOO (0.1 min) [11/12]
DONE  u2u_mhz=1.5/s4101_zeroshot_HoverSAC (0.1 min) [12/12]
All jobs finished.

=== 2026-10-20 02:38:47  D:\HRL_UAV_V2\.venv\Scripts\python.exe D:\p2_code\run.py zeroshot-report --out-base=D:\p2_zeroshot_v4\_zeroshot\smoke --main=D:\p2_runs_v2 --points=users=7,u2u_mhz=1.5 --smoke --seeds=4101
D:\p2_zeroshot_v4\_zeroshot\smoke\results
```
