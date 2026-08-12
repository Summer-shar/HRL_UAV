# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 662.619 | 5.640 | 7.145 | 823.583 |
| 动态质心跟踪 | 672.525 | 6.811 | 9.553 | 826.947 |
| 协作感知移动 | 661.757 | 6.202 | 7.771 | 847.212 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 10.768 ms，差值 95% CI [-28.450, 6.913] ms。
- 协作感知相对固定站位：平均收益 0.862 ms，差值 95% CI [-18.144, 16.419] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [ ] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [ ] `cooperative_beats_fixed_statistically`
- [ ] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
