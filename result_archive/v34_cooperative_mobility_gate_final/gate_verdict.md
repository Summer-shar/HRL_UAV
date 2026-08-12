# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 681.452 | 5.895 | 5.727 | 824.415 |
| 动态质心跟踪 | 663.153 | 6.519 | 7.412 | 838.545 |
| 协作感知移动 | 659.758 | 6.057 | 6.966 | 866.116 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 3.396 ms，差值 95% CI [-5.055, -1.736] ms。
- 协作感知相对固定站位：平均收益 21.694 ms，差值 95% CI [-24.630, -18.758] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
