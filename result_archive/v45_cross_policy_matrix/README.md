# v45 宏观-微观交叉重放

> 零训练；3种宏观策略 × 2个SAC微观策略；20个共同随机评估种子。
> 差值统一为“左侧组合减右侧组合”，负值表示左侧时延更低。

| 宏观策略 | 微观策略 | 端到端时延/ms | 95% CI | 队列/Mbit | 功率/W |
|---|---|---:|---:|---:|---:|
| TD3宏观 | TD3训练出的SAC | 665.91 | [656.44, 675.37] | 10.93 | 852.20 |
| TD3宏观 | PPO训练出的SAC | 651.61 | [643.89, 659.34] | 10.66 | 851.74 |
| PPO宏观 | TD3训练出的SAC | 726.02 | [694.51, 757.53] | 14.41 | 826.63 |
| PPO宏观 | PPO训练出的SAC | 654.02 | [644.55, 663.48] | 11.19 | 827.86 |
| 动态质心 | TD3训练出的SAC | 731.66 | [696.14, 767.18] | 14.68 | 826.53 |
| 动态质心 | PPO训练出的SAC | 655.19 | [645.49, 664.88] | 11.25 | 827.78 |

## 配对差值

- `macro_td3_minus_ppo_with_td3_micro`：-60.12 ms，95% CI [-87.32, -32.92]。
- `macro_td3_minus_ppo_with_ppo_micro`：-2.41 ms，95% CI [-5.53, +0.72]。
- `micro_ppo_minus_td3_with_td3_macro`：-14.29 ms，95% CI [-19.36, -9.22]。
- `micro_ppo_minus_td3_with_ppo_macro`：-72.00 ms，95% CI [-100.45, -43.55]。
- `td3_macro_minus_centroid_with_td3_micro`：-65.75 ms，95% CI [-97.23, -34.27]。
- `ppo_macro_minus_centroid_with_td3_micro`：-5.63 ms，95% CI [-13.48, +2.21]。
- `td3_macro_minus_centroid_with_ppo_micro`：-3.58 ms，95% CI [-7.41, +0.26]。
- `ppo_macro_minus_centroid_with_ppo_micro`：-1.17 ms，95% CI [-2.34, +0.00]。
