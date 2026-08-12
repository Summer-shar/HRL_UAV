# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 666.928 | 6.418 | 6.792 | 824.495 |
| 动态质心跟踪 | 664.880 | 6.528 | 7.345 | 831.658 |
| 协作感知移动 | 660.953 | 6.065 | 7.011 | 861.088 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 3.927 ms，差值 95% CI [-5.473, -2.380] ms。
- 协作感知相对固定站位：平均收益 5.975 ms，差值 95% CI [-7.963, -3.986] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
