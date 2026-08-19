# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 676.139 | 7.461 | 7.697 | 822.975 |
| 动态质心跟踪 | 673.986 | 7.619 | 7.852 | 831.879 |
| 协作感知移动 | 670.391 | 7.292 | 7.888 | 850.442 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 3.594 ms，差值 95% CI [-5.667, -1.522] ms。
- 协作感知相对固定站位：平均收益 5.748 ms，差值 95% CI [-8.640, -2.856] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
