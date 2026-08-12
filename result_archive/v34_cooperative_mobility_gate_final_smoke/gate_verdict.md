# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 687.718 | 6.030 | 6.873 | 826.811 |
| 动态质心跟踪 | 675.058 | 6.683 | 8.550 | 841.196 |
| 协作感知移动 | 669.542 | 6.140 | 7.363 | 870.807 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 5.516 ms，差值 95% CI [-13.655, 2.623] ms。
- 协作感知相对固定站位：平均收益 18.175 ms，差值 95% CI [-34.093, -2.257] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [ ] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
