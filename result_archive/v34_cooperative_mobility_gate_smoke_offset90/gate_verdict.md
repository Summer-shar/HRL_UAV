# v34 协作移动零训练门禁

**判决：FAIL**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 663.172 | 5.693 | 6.754 | 823.388 |
| 动态质心跟踪 | 668.520 | 6.622 | 8.829 | 833.907 |
| 协作感知移动 | 660.301 | 5.876 | 7.539 | 862.323 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 8.219 ms，差值 95% CI [-28.242, 11.805] ms。
- 协作感知相对固定站位：平均收益 2.871 ms，差值 95% CI [-8.480, 2.738] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [ ] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [ ] `cooperative_beats_fixed_statistically`
- [ ] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
