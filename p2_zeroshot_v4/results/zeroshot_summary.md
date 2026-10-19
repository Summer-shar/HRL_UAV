# Zero-shot sweep (design v4, docs/PROTOCOL.md)

Every method is the policy trained once at the nominal point (main run, dev-selected checkpoints), evaluated unchanged at each value; evaluated on the test scenarios 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40. Nominal value of each parameter: the main run D:\p2_runs_v2. Values: mean over seeds of scenario means (std over seeds in the CSV files). users = terminals per region, nominal computation load.

| point | finished |
|---|---|
| cpu_ghz=9 | yes |
| u2u_mhz=1.5 | no |
| cpu_ghz=11 | yes |
| cpu_ghz=12 | no |
| cpu_ghz=14 | no |
| load=0.6 | no |
| load=0.8 | no |
| load=0.9 | no |
| load=1.1 | no |
| users=7 | no |
| users=10 | no |
| users=16 | no |
| users=19 | no |
| u2u_mhz=2 | no |
| u2u_mhz=2.5 | no |
| u2u_mhz=4 | no |

## cpu_ghz (zero-shot, finished points only; cells: delay / compute+comm power / joint cost)

| cpu_ghz | demand/capacity | Proposed (HRL) | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.92 | 1.001 s / 196.5 W / 8.475 | 1.133 s / 194.7 W / 9.392 | 1.337 s / 191.6 W / 10.82 | 12.78 s / 172.3 W / 91.43 | 1.572 s / 194.8 W / 12.49 | 1.527 s / 190.6 W / 12.14 |
| 10 | 0.82 | 0.6126 s / 234.4 W / 5.993 | 0.6843 s / 228 W / 6.452 | 0.7319 s / 220.9 W / 6.737 | 5.881 s / 227.1 W / 43.2 | 1.256 s / 232.4 W / 10.52 | 1.337 s / 216.4 W / 10.98 |
| 11 | 0.75 | 0.5533 s / 273.6 W / 5.854 | 0.6224 s / 260.8 W / 6.249 | 0.6956 s / 246.6 W / 6.662 | 2.213 s / 278.8 W / 17.67 | 1.197 s / 271.7 W / 10.38 | 1.322 s / 240.2 W / 11.04 |

Proposed vs others (cpu_ghz): relative change of the delay / cost, Wilcoxon p (20 scenarios), better seeds

| cpu_ghz | TFOO (PPO+CVX) | TOO (PPO+Equal) | NC (no cooperation) | Hover-SAC | CA (hover+equal) |
|---:|---:|---:|---:|---:|---:|
| 9 | -11.7% / -9.8%, p=1.9e-06/1.9e-06, 3/3 | -25.1% / -21.7%, p=1.9e-06/1.9e-06, 3/3 | -92.2% / -90.7%, p=1.9e-06/1.9e-06, 3/3 | -36.3% / -32.1%, p=1.9e-06/1.9e-06, 3/3 | -34.4% / -30.2%, p=1.9e-06/1.9e-06, 3/3 |
| 10 | -10.5% / -7.1%, p=1.9e-06/1.9e-06, 3/3 | -16.3% / -11.0%, p=1.9e-06/1.9e-06, 3/3 | -89.6% / -86.1%, p=1.9e-06/1.9e-06, 3/3 | -51.2% / -43.0%, p=1.9e-06/1.9e-06, 3/3 | -54.2% / -45.4%, p=1.9e-06/1.9e-06, 3/3 |
| 11 | -11.1% / -6.3%, p=1.9e-06/1.9e-06, 3/3 | -20.5% / -12.1%, p=1.9e-06/1.9e-06, 3/3 | -75.0% / -66.9%, p=1.9e-06/1.9e-06, 3/3 | -53.8% / -43.6%, p=1.9e-06/1.9e-06, 3/3 | -58.1% / -47.0%, p=1.9e-06/1.9e-06, 3/3 |
