# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 658.091 | 6.387 | 9.662 | 821.224 |
| 动态质心跟踪 | 657.167 | 6.566 | 9.770 | 827.686 |
| 协作感知移动 | 654.469 | 5.960 | 9.709 | 869.618 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 2.698 ms，差值 95% CI [-9.641, 4.245] ms。
- 协作感知相对固定站位：平均收益 3.621 ms，差值 95% CI [-11.806, 4.563] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [ ] `cooperative_beats_centroid_statistically`
- [ ] `cooperative_beats_centroid_practically`
- [ ] `cooperative_beats_fixed_statistically`
- [ ] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
