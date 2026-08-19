# v34 协作移动零训练门禁

**判决：PASS**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 805.298 | 12.620 | 17.253 | 824.170 |
| 动态质心跟踪 | 816.566 | 13.941 | 22.880 | 832.088 |
| 协作感知移动 | 763.722 | 10.618 | 12.432 | 851.248 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 52.844 ms，差值 95% CI [-69.144, -36.545] ms。
- 协作感知相对固定站位：平均收益 41.576 ms，差值 95% CI [-55.126, -28.025] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
