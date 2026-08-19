# v34 协作移动零训练门禁

**判决：PASS**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 1067.325 | 26.895 | 52.290 | 821.668 |
| 动态质心跟踪 | 1052.420 | 26.684 | 53.693 | 830.550 |
| 协作感知移动 | 927.795 | 18.679 | 19.882 | 850.459 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 124.625 ms，差值 95% CI [-161.793, -87.458] ms。
- 协作感知相对固定站位：平均收益 139.530 ms，差值 95% CI [-188.931, -90.129] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
