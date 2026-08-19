# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 659.022 | 6.310 | 6.967 | 823.342 |
| 动态质心跟踪 | 655.599 | 6.430 | 7.256 | 831.977 |
| 协作感知移动 | 651.962 | 6.153 | 7.095 | 850.182 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 3.637 ms，差值 95% CI [-5.123, -2.151] ms。
- 协作感知相对固定站位：平均收益 7.060 ms，差值 95% CI [-10.156, -3.965] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
