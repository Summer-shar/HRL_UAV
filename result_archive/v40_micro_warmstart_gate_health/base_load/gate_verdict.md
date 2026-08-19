# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 645.830 | 7.007 | 7.432 | 824.539 |
| 动态质心跟踪 | 642.987 | 7.160 | 7.547 | 832.450 |
| 协作感知移动 | 642.165 | 6.949 | 7.717 | 851.487 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 0.821 ms，差值 95% CI [-1.917, 0.274] ms。
- 协作感知相对固定站位：平均收益 3.665 ms，差值 95% CI [-5.529, -1.800] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [ ] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [ ] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
