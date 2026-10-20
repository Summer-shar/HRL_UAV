# Zero-shot sweep (design v4, docs/PROTOCOL.md)

Every method is the policy trained once at the nominal point (main run, dev-selected checkpoints), evaluated unchanged at each value; evaluated on the test scenarios 21, 22. Nominal value of each parameter: the main run D:\p2_runs_v2. Values: mean over seeds of scenario means (std, min and max over seeds in the CSV files; every seed in zeroshot_per_seed.csv). terminals = terminals per region, nominal load per terminal (the total load grows with the number of terminals; demand/capacity column).

| point | finished |
|---|---|
| terminals=7 | yes |
| u2u_mhz=1.5 | yes |

## terminals (zero-shot; cells: delay / compute+comm power / joint cost)

| terminals | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.46 | 0.3692 s / 132.1 W / 2.294 | 0.3696 s / 121.9 W / 2.223 | 0.4264 s / 106.5 W / 2.32 | 0.3795 s / 139.7 W / 2.386 | 0.3803 s / 137 W / 2.37 | 0.4285 s / 106.7 W / 2.329 |
| 13 | 0.82 | 0.6109 s / 229.4 W / 5.821 | 0.6509 s / 224.3 W / 6.053 | 0.6661 s / 215.4 W / 6.093 | 4.49 s / 228.4 W / 32.35 | 1.029 s / 230.9 W / 8.683 | 1.138 s / 212.7 W / 9.3 |

Proposed vs others (terminals): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| terminals | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 7 | -0.1% / +3.2%, p=1/0.5, 1/1 | -13.4% / -1.1%, p=0.5/0.5, 1/1 | -2.7% / -3.8%, p=0.5/0.5, 1/1 | -2.9% / -3.2%, p=0.5/0.5, 1/1 | -13.8% / -1.5%, p=0.5/0.5, 1/1 |
| 13 | -6.1% / -3.8%, p=1/1, 1/1 | -8.3% / -4.5%, p=0.5/1, 1/1 | -86.4% / -82.0%, p=0.5/0.5, 1/1 | -40.6% / -33.0%, p=0.5/0.5, 1/1 | -46.3% / -37.4%, p=0.5/0.5, 1/1 |

Weighted delay per training seed (terminals; seeds 4101)

| terminals | Proposed (HRL) | TFOO (PPO+CVX) |
|---:|---:|---:|
| 7 | 0.3692 | 0.3696 |
| 13 | 0.6109 | 0.6509 |

## u2u_mhz (zero-shot; cells: delay / compute+comm power / joint cost)

| u2u_mhz | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.82 | 0.5778 s / 220.1 W / 5.467 | 0.6263 s / 218.2 W / 5.777 | 2.615 s / 137.2 W / 18.57 | 0.7763 s / 214.7 W / 6.755 | 0.5672 s / 223.6 W / 5.42 | 2.8 s / 128.7 W / 19.75 |
| 3 | 0.82 | 0.6109 s / 229.4 W / 5.821 | 0.6509 s / 224.3 W / 6.053 | 0.6661 s / 215.4 W / 6.093 | 4.49 s / 228.4 W / 32.35 | 1.029 s / 230.9 W / 8.683 | 1.138 s / 212.7 W / 9.3 |

Proposed vs others (u2u_mhz): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| u2u_mhz | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 1.5 | -7.7% / -5.4%, p=0.5/0.5, 1/1 | -77.9% / -70.6%, p=0.5/0.5, 1/1 | -25.6% / -19.1%, p=0.5/0.5, 1/1 | +1.9% / +0.9%, p=0.5/0.5, 0/1 | -79.4% / -72.3%, p=0.5/0.5, 1/1 |
| 3 | -6.1% / -3.8%, p=1/1, 1/1 | -8.3% / -4.5%, p=0.5/1, 1/1 | -86.4% / -82.0%, p=0.5/0.5, 1/1 | -40.6% / -33.0%, p=0.5/0.5, 1/1 | -46.3% / -37.4%, p=0.5/0.5, 1/1 |

Weighted delay per training seed (u2u_mhz; seeds 4101)

| u2u_mhz | Proposed (HRL) | TFOO (PPO+CVX) |
|---:|---:|---:|
| 1.5 | 0.5778 | 0.6263 |
| 3 | 0.6109 | 0.6509 |
