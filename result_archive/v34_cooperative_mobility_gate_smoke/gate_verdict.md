# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 586.828 | 3.285 | 3.544 | 776.529 |
| 动态质心跟踪 | 578.260 | 3.594 | 4.329 | 781.885 |
| 协作感知移动 | 577.893 | 3.499 | 4.475 | 801.794 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 0.367 ms，差值 95% CI [-2.985, 2.251] ms。
- 协作感知相对固定站位：平均收益 8.935 ms，差值 95% CI [-10.728, -7.142] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [ ] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
