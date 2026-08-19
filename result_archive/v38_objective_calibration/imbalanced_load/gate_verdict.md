# v34 协作移动零训练门禁

**判决：PASS**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 695.232 | 6.795 | 12.830 | 823.745 |
| 动态质心跟踪 | 692.870 | 6.980 | 13.301 | 832.203 |
| 协作感知移动 | 682.848 | 6.442 | 12.027 | 852.201 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 10.022 ms，差值 95% CI [-13.446, -6.597] ms。
- 协作感知相对固定站位：平均收益 12.384 ms，差值 95% CI [-14.764, -10.004] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
