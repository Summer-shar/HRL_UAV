# Zero-shot sweep (design v4, docs/PROTOCOL.md)

Every method is the policy trained once at the nominal point (main run, dev-selected checkpoints), evaluated unchanged at each value; evaluated on the test scenarios 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40. Nominal value of each parameter: the main run D:\p2_runs_v2. Values: mean over seeds of scenario means (std over seeds in the CSV files). users = terminals per region, nominal computation load.

| point | finished |
|---|---|
| cpu_ghz=9 | yes |
| u2u_mhz=1.5 | no |
| cpu_ghz=11 | yes |
| cpu_ghz=12 | yes |
| cpu_ghz=14 | yes |
| load=0.6 | yes |
| load=0.8 | yes |
| load=0.9 | no |
| load=1.1 | no |
| users=7 | no |
| users=10 | no |
| users=16 | no |
| users=19 | no |
| u2u_mhz=2 | no |
| u2u_mhz=2.5 | no |
| u2u_mhz=4 | no |

## cpu_ghz (zero-shot; cells: delay / compute+comm power / joint cost)

| cpu_ghz | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.92 | 1.001 s / 196.5 W / 8.475 | 1.133 s / 194.7 W / 9.392 | 1.337 s / 191.6 W / 10.82 | 12.78 s / 172.3 W / 91.43 | 1.572 s / 194.8 W / 12.49 | 1.527 s / 190.6 W / 12.14 |
| 10 | 0.82 | 0.6126 s / 234.4 W / 5.993 | 0.6843 s / 228 W / 6.452 | 0.7319 s / 220.9 W / 6.737 | 5.881 s / 227.1 W / 43.2 | 1.256 s / 232.4 W / 10.52 | 1.337 s / 216.4 W / 10.98 |
| 11 | 0.75 | 0.5533 s / 273.6 W / 5.854 | 0.6224 s / 260.8 W / 6.249 | 0.6956 s / 246.6 W / 6.662 | 2.213 s / 278.8 W / 17.67 | 1.197 s / 271.7 W / 10.38 | 1.322 s / 240.2 W / 11.04 |
| 12 | 0.69 | 0.5373 s / 313.4 W / 6.025 | 0.6051 s / 293.4 W / 6.36 | 0.7247 s / 270.1 W / 7.035 | 1.159 s / 323.9 W / 10.5 | 1.174 s / 312.3 W / 10.51 | 1.317 s / 262.6 W / 11.17 |
| 14 | 0.59 | 0.5312 s / 395.7 W / 6.569 | 0.5948 s / 361.4 W / 6.772 | 0.755 s / 314.8 W / 7.568 | 0.8782 s / 416.9 W / 9.169 | 1.154 s / 398.3 W / 10.98 | 1.314 s / 304.9 W / 11.44 |

Proposed vs others (cpu_ghz): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| cpu_ghz | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 9 | -11.7% / -9.8%, p=1.9e-06/1.9e-06, 3/3 | -25.1% / -21.7%, p=1.9e-06/1.9e-06, 3/3 | -92.2% / -90.7%, p=1.9e-06/1.9e-06, 3/3 | -36.3% / -32.1%, p=1.9e-06/1.9e-06, 3/3 | -34.4% / -30.2%, p=1.9e-06/1.9e-06, 3/3 |
| 10 | -10.5% / -7.1%, p=1.9e-06/1.9e-06, 3/3 | -16.3% / -11.0%, p=1.9e-06/1.9e-06, 3/3 | -89.6% / -86.1%, p=1.9e-06/1.9e-06, 3/3 | -51.2% / -43.0%, p=1.9e-06/1.9e-06, 3/3 | -54.2% / -45.4%, p=1.9e-06/1.9e-06, 3/3 |
| 11 | -11.1% / -6.3%, p=1.9e-06/1.9e-06, 3/3 | -20.5% / -12.1%, p=1.9e-06/1.9e-06, 3/3 | -75.0% / -66.9%, p=1.9e-06/1.9e-06, 3/3 | -53.8% / -43.6%, p=1.9e-06/1.9e-06, 3/3 | -58.1% / -47.0%, p=1.9e-06/1.9e-06, 3/3 |
| 12 | -11.2% / -5.3%, p=1.9e-06/3.8e-06, 3/3 | -25.9% / -14.4%, p=1.9e-06/1.9e-06, 3/3 | -53.6% / -42.6%, p=1.9e-06/1.9e-06, 3/3 | -54.2% / -42.7%, p=1.9e-06/1.9e-06, 3/3 | -59.2% / -46.1%, p=1.9e-06/1.9e-06, 3/3 |
| 14 | -10.7% / -3.0%, p=1.9e-06/0.00059, 3/3 | -29.6% / -13.2%, p=1.9e-06/5.7e-06, 3/3 | -39.5% / -28.4%, p=1.9e-06/1.9e-06, 3/3 | -54.0% / -40.2%, p=1.9e-06/1.9e-06, 3/3 | -59.6% / -42.6%, p=1.9e-06/1.9e-06, 3/3 |

## load (zero-shot, finished points only; cells: delay / compute+comm power / joint cost)

| load | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.49 | 0.4798 s / 123.2 W / 2.917 | 0.5378 s / 108 W / 3.054 | 0.5573 s / 99.87 W / 3.079 | 0.6315 s / 130.4 W / 3.613 | 0.6589 s / 126.3 W / 3.699 | 0.6934 s / 99.98 W / 3.657 |
| 0.8 | 0.66 | 0.515 s / 176.6 W / 4.167 | 0.5734 s / 164.2 W / 4.407 | 0.5905 s / 155.3 W / 4.44 | 0.9386 s / 183 W / 6.606 | 0.85 s / 178 W / 6.066 | 0.8769 s / 154.1 W / 6.047 |
| 1 | 0.82 | 0.6126 s / 234.4 W / 5.993 | 0.6843 s / 228 W / 6.452 | 0.7319 s / 220.9 W / 6.737 | 5.881 s / 227.1 W / 43.2 | 1.256 s / 232.4 W / 10.52 | 1.337 s / 216.4 W / 10.98 |

Proposed vs others (load): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| load | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 0.6 | -10.8% / -4.5%, p=1.9e-06/1.9e-06, 3/3 | -13.9% / -5.3%, p=1.9e-06/1.9e-06, 3/3 | -24.0% / -19.2%, p=1.9e-06/1.9e-06, 3/3 | -27.2% / -21.1%, p=1.9e-06/1.9e-06, 3/3 | -30.8% / -20.2%, p=1.9e-06/1.9e-06, 3/3 |
| 0.8 | -10.2% / -5.5%, p=1.9e-06/1.9e-06, 3/3 | -12.8% / -6.2%, p=1.9e-06/1.9e-06, 3/3 | -45.1% / -36.9%, p=1.9e-06/1.9e-06, 3/3 | -39.4% / -31.3%, p=1.9e-06/1.9e-06, 3/3 | -41.3% / -31.1%, p=1.9e-06/1.9e-06, 3/3 |
| 1 | -10.5% / -7.1%, p=1.9e-06/1.9e-06, 3/3 | -16.3% / -11.0%, p=1.9e-06/1.9e-06, 3/3 | -89.6% / -86.1%, p=1.9e-06/1.9e-06, 3/3 | -51.2% / -43.0%, p=1.9e-06/1.9e-06, 3/3 | -54.2% / -45.4%, p=1.9e-06/1.9e-06, 3/3 |
