# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 1027.232 | 29.191 | 53.300 | 819.599 |
| 动态质心跟踪 | 1087.659 | 35.977 | 77.412 | 825.809 |
| 协作感知移动 | 954.225 | 24.409 | 40.720 | 846.184 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 133.433 ms，差值 95% CI [-162.318, -104.549] ms。
- 协作感知相对固定站位：平均收益 73.006 ms，差值 95% CI [-152.582, 6.570] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [ ] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
