# v37 负载不均衡零训练门禁结果

**判决：FAIL**

增强不均衡负载下，60 m 协作相对质心的时延收益为 10.022 ms，差值 95% CI [-13.446, -6.597] ms。
软目标 J 差值为 0.003549，95% CI [0.000116, 0.006983]；负值表示协作更优。
相对匹配基准负载，协作时延收益增加 5.381 ms，95% CI [1.834, 8.928] ms。

## 判据

- [x] `delay_gain_gt_5ms`
- [x] `delay_ci_excludes_zero`
- [ ] `soft_j_mean_improves`
- [ ] `soft_j_ci_excludes_zero`
- [x] `queue_noninferior`
