# v32b matched-reference replay: training seed 1

- Selected checkpoint: `C:\Users\PC\Desktop\HRL_UAV\results\tcom_dynamic_load_switch_v32b_formal\formal\seed1\joint_td3_sac\checkpoints\best.pt`
- Frozen system multiplier: `0.000000000`
- TD3 power UCB95: `835.259 W` (feasible: `True`)
- P1-prime power UCB95: `842.488 W` (feasible: `False`)
- TD3 minus P0 delay: `5.674 ms`, 95% CI `[-3.873, 15.220] ms`
- P1-prime minus P0 delay: `-8.652 ms`, 95% CI `[-21.665, 4.361] ms`
- Descriptive capture fraction: `-0.655767`
- Seed-level TD3 direction beneficial: `False`

P1-prime participates in feasible-policy comparison only if its matched power UCB95 is at most 840 W.
