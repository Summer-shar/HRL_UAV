# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 665.158 | 6.447 | 8.418 | 824.672 |
| 动态质心跟踪 | 664.031 | 6.540 | 8.324 | 832.441 |
| 协作感知移动 | 659.391 | 6.220 | 7.963 | 852.290 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 4.640 ms，差值 95% CI [-6.328, -2.952] ms。
- 协作感知相对固定站位：平均收益 5.767 ms，差值 95% CI [-7.796, -3.739] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
