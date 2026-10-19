# Zero-shot sweep (design v4, docs/PROTOCOL.md)

Every method is the policy trained once at the nominal point (main run, dev-selected checkpoints), evaluated unchanged at each value; evaluated on the test scenarios 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40. Nominal value of each parameter: the main run D:\p2_runs_v2. Values: mean over seeds of scenario means (std over seeds in the CSV files). users = terminals per region, nominal computation load.

| point | finished |
|---|---|
| cpu_ghz=9 | no |
| u2u_mhz=1.5 | no |
| cpu_ghz=11 | no |
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
