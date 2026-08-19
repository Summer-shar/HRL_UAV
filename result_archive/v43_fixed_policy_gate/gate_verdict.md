# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 994.841 | 25.184 | 32.768 | 821.520 |
| 动态质心跟踪 | 1095.227 | 32.850 | 57.275 | 827.786 |
| 协作感知移动 | 932.740 | 21.724 | 27.537 | 847.622 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 162.488 ms，差值 95% CI [-204.210, -120.765] ms。
- 协作感知相对固定站位：平均收益 62.102 ms，差值 95% CI [-147.870, 23.667] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [ ] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
