# v34 协作移动零训练门禁

**判决：PASS**

## 三策略结果

| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |
|---|---:|---:|---:|---:|
| 固定初始站位 | 701.112 | 5.244 | 7.047 | 831.751 |
| 动态质心跟踪 | 699.811 | 5.585 | 10.536 | 839.244 |
| 协作感知移动 | 692.590 | 5.058 | 7.322 | 868.415 |

## 配对判据

- 协作感知相对质心跟踪：平均收益 7.220 ms，差值 95% CI [-7.220, -7.220] ms。
- 协作感知相对固定站位：平均收益 8.522 ms，差值 95% CI [-8.522, -8.522] ms。
- 功率仅作描述性指标，不设硬阈值。

## 判据逐项

- [x] `cooperative_beats_centroid_statistically`
- [x] `cooperative_beats_centroid_practically`
- [x] `cooperative_beats_fixed_statistically`
- [x] `cooperative_beats_fixed_practically`
- [x] `queue_noninferior_to_centroid`
