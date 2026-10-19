# Retrained sweep (design v3, docs/PROTOCOL.md)

Every learned method trained from scratch at each non-nominal value with the main pipeline (same stages, budgets, hyper-parameters, seeds, training scenarios, initial weights, fine-tuning recipe and dev selection); evaluated on the test scenarios 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40. Nominal value of each parameter: the main run D:\p2_runs_v2. Values: mean over seeds of scenario means (std over seeds in the CSV files). users = terminals per region, nominal computation load.

| point | finished |
|---|---|
| cpu_ghz=9 | yes |
| u2u_mhz=1.5 | yes |

## cpu_ghz (retrained, finished points only; cells: delay / compute+comm power / joint cost)

| cpu_ghz | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.92 | 3.091 s / 193.2 W / 23.24 | 1.058 s / 194.9 W / 8.866 | 1.042 s / 192.9 W / 8.736 | 12.62 s / 172.4 W / 90.24 | 4.068 s / 190.5 W / 30.11 | 1.527 s / 190.6 W / 12.14 |
| 10 | 0.82 | 0.6126 s / 234.4 W / 5.993 | 0.6843 s / 228 W / 6.452 | 0.7319 s / 220.9 W / 6.737 | 5.881 s / 227.1 W / 43.2 | 1.256 s / 232.4 W / 10.52 | 1.337 s / 216.4 W / 10.98 |

Proposed vs others (cpu_ghz): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| cpu_ghz | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 9 | +192.0% / +162.1%, p=1.9e-06/1.9e-06, 0/3 | +196.7% / +166.0%, p=1.9e-06/1.9e-06, 0/3 | -75.5% / -74.2%, p=1.9e-06/1.9e-06, 3/3 | -24.0% / -22.8%, p=0.0012/0.0012, 2/3 | +102.5% / +91.4%, p=1.9e-06/1.9e-06, 1/3 |
| 10 | -10.5% / -7.1%, p=1.9e-06/1.9e-06, 3/3 | -16.3% / -11.0%, p=1.9e-06/1.9e-06, 3/3 | -89.6% / -86.1%, p=1.9e-06/1.9e-06, 3/3 | -51.2% / -43.0%, p=1.9e-06/1.9e-06, 3/3 | -54.2% / -45.4%, p=1.9e-06/1.9e-06, 3/3 |

## u2u_mhz (retrained, finished points only; cells: delay / compute+comm power / joint cost)

| u2u_mhz | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.82 | 19.1 s / 196.2 W / 136.1 | 0.815 s / 230 W / 7.388 | 34.45 s / 157.1 W / 244.1 | 5.944 s / 227.2 W / 43.64 | 19.44 s / 194.7 W / 138.5 | 46.56 s / 130.2 W / 328.8 |
| 3 | 0.82 | 0.6126 s / 234.4 W / 5.993 | 0.6843 s / 228 W / 6.452 | 0.7319 s / 220.9 W / 6.737 | 5.881 s / 227.1 W / 43.2 | 1.256 s / 232.4 W / 10.52 | 1.337 s / 216.4 W / 10.98 |

Proposed vs others (u2u_mhz): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| u2u_mhz | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 1.5 | +2244.2% / +1742.2%, p=1.9e-06/1.9e-06, 0/3 | -44.5% / -44.2%, p=1.9e-06/1.9e-06, 3/3 | +221.4% / +211.8%, p=1.9e-06/1.9e-06, 1/3 | -1.7% / -1.8%, p=0.13/0.12, 2/3 | -59.0% / -58.6%, p=1.9e-06/1.9e-06, 3/3 |
| 3 | -10.5% / -7.1%, p=1.9e-06/1.9e-06, 3/3 | -16.3% / -11.0%, p=1.9e-06/1.9e-06, 3/3 | -89.6% / -86.1%, p=1.9e-06/1.9e-06, 3/3 | -51.2% / -43.0%, p=1.9e-06/1.9e-06, 3/3 | -54.2% / -45.4%, p=1.9e-06/1.9e-06, 3/3 |

## Training curves: 42 stage runs checked, 22 with a cost change above 10 % between the last two 20-episode windows (reported, nothing dropped or extended)

- cpu_ghz=9 seed 4101 M1: cost 70.63 -> 61.96 (-12.3%)
- cpu_ghz=9 seed 4101 M2: cost 68 -> 57.15 (-16.0%)
- cpu_ghz=9 seed 4101 M3: cost 51.79 -> 38.04 (-26.6%)
- cpu_ghz=9 seed 4101 HoverSAC: cost 56.19 -> 70.72 (+25.9%)
- cpu_ghz=9 seed 4101 TOO: cost 11.09 -> 9.758 (-12.0%)
- cpu_ghz=9 seed 4102 M2: cost 13.72 -> 18.17 (+32.4%)
- cpu_ghz=9 seed 4102 M3: cost 14.26 -> 20.96 (+47.0%)
- cpu_ghz=9 seed 4102 HoverSAC: cost 31.45 -> 21.55 (-31.5%)
- cpu_ghz=9 seed 4102 TFOO: cost 9.134 -> 12.6 (+38.0%)
- cpu_ghz=9 seed 4102 TOO: cost 8.887 -> 11.35 (+27.8%)
- cpu_ghz=9 seed 4103 M1: cost 24.08 -> 19.46 (-19.2%)
- cpu_ghz=9 seed 4103 M3: cost 13.4 -> 11.03 (-17.7%)
- u2u_mhz=1.5 seed 4101 M1: cost 218 -> 174.5 (-19.9%)
- u2u_mhz=1.5 seed 4101 HoverSAC: cost 43.68 -> 65.87 (+50.8%)
- u2u_mhz=1.5 seed 4101 NC: cost 42.33 -> 34.03 (-19.6%)
- u2u_mhz=1.5 seed 4102 M1: cost 392.5 -> 208 (-47.0%)
- u2u_mhz=1.5 seed 4102 M2: cost 141 -> 168.8 (+19.7%)
- u2u_mhz=1.5 seed 4102 M3: cost 128.2 -> 163.3 (+27.4%)
- u2u_mhz=1.5 seed 4102 HoverSAC: cost 159.6 -> 181.3 (+13.5%)
- u2u_mhz=1.5 seed 4102 TOO: cost 230.3 -> 202.3 (-12.2%)
- u2u_mhz=1.5 seed 4102 NC: cost 40.81 -> 45.12 (+10.6%)
- u2u_mhz=1.5 seed 4103 HoverSAC: cost 388.3 -> 318.3 (-18.0%)
