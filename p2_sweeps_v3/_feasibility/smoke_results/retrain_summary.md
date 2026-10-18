# Retrained sweep (design v3, docs/PROTOCOL.md)

Every learned method trained from scratch at each non-nominal value with the main pipeline (same stages, budgets, hyper-parameters, seeds, training scenarios, initial weights, fine-tuning recipe and dev selection); evaluated on the test scenarios 21, 22. Nominal value of each parameter: the main run D:\p2_runs_v2. Values: mean over seeds of scenario means (std over seeds in the CSV files). users = terminals per region, nominal computation load.

| point | finished |
|---|---|
| users=7 | yes |
| u2u_mhz=1.5 | yes |

## users (retrained; cells: delay / compute+comm power / joint cost)

| users | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.82 | 1.944 s / 216.3 W / 14.24 | 0.7784 s / 241 W / 6.727 | 0.635 s / 245.5 W / 5.874 | 1.294 s / 218.9 W / 10.02 | 1.944 s / 216 W / 14.23 | 0.6356 s / 245.7 W / 5.88 |
| 13 | 0.82 | 0.6109 s / 229.4 W / 5.821 | 0.6509 s / 224.3 W / 6.053 | 0.6661 s / 215.4 W / 6.093 | 4.49 s / 228.4 W / 32.35 | 1.029 s / 230.9 W / 8.683 | 1.138 s / 212.7 W / 9.3 |

Proposed vs others (users): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| users | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 7 | +149.8% / +111.7%, p=0.5/0.5, 0/1 | +206.2% / +142.4%, p=0.5/0.5, 0/1 | +50.3% / +42.2%, p=0.5/0.5, 0/1 | +0.0% / +0.0%, p=1/1, 0/1 | +205.9% / +142.2%, p=0.5/0.5, 0/1 |
| 13 | -6.1% / -3.8%, p=1/1, 1/1 | -8.3% / -4.5%, p=0.5/1, 1/1 | -86.4% / -82.0%, p=0.5/0.5, 1/1 | -40.6% / -33.0%, p=0.5/0.5, 1/1 | -46.3% / -37.4%, p=0.5/0.5, 1/1 |

## u2u_mhz (retrained; cells: delay / compute+comm power / joint cost)

| u2u_mhz | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.82 | 2.217 s / 176.9 W / 16.17 | 0.5886 s / 219.5 W / 5.534 | 2.799 s / 128.9 W / 19.75 | 0.774 s / 214.4 W / 6.738 | 2.191 s / 177.4 W / 16 | 2.8 s / 128.7 W / 19.75 |
| 3 | 0.82 | 0.6109 s / 229.4 W / 5.821 | 0.6509 s / 224.3 W / 6.053 | 0.6661 s / 215.4 W / 6.093 | 4.49 s / 228.4 W / 32.35 | 1.029 s / 230.9 W / 8.683 | 1.138 s / 212.7 W / 9.3 |

Proposed vs others (u2u_mhz): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| u2u_mhz | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 1.5 | +276.6% / +192.2%, p=0.5/0.5, 0/1 | -20.8% / -18.1%, p=0.5/0.5, 1/1 | +186.4% / +140.0%, p=0.5/0.5, 0/1 | +1.2% / +1.1%, p=0.5/0.5, 0/1 | -20.8% / -18.1%, p=0.5/0.5, 1/1 |
| 3 | -6.1% / -3.8%, p=1/1, 1/1 | -8.3% / -4.5%, p=0.5/1, 1/1 | -86.4% / -82.0%, p=0.5/0.5, 1/1 | -40.6% / -33.0%, p=0.5/0.5, 1/1 | -46.3% / -37.4%, p=0.5/0.5, 1/1 |
