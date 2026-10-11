# Frozen scene9 numerical tables

| Anchor | time /s | critic descending preference | replay fraction with at least this low backlog |
|---|---:|---|---:|
| low_cpu_1s | 52.5 | equal_low > avoid_low > sac > equal > inverse | 0.212808 |
| fixed_450s | 450.0 | sac > inverse > equal > avoid_low > equal_low | 0.374818 |
| low_cpu_5s | 465.5 | sac > equal > avoid_low > equal_low > inverse | 0.049717 |
| low_cpu_15s | 477.5 | sac > equal > avoid_low > inverse > equal_low | 0.006980 |

Delta is alternative minus SAC; lower holding/cost is better, higher Q/return is better.

| Anchor | regime | alternative | ΔminQ | ΔH30 mean | ΔH60 mean [min,max] | H60 better pairs | Δfinite soft return60 | Δonline-bootstrap return |
|---|---|---|---:|---:|---|---:|---:|---:|
| low_cpu_1s | one_action_then_stochastic | equal | -0.003167 | 10.850 | 8.200 [-2.700,21.900] | 1/4 | -0.011400 | -0.011090 |
| low_cpu_1s | one_action_then_stochastic | avoid_low | 0.000778 | 2.475 | 1.900 [-5.100,5.800] | 1/4 | -0.002442 | -0.002442 |
| low_cpu_1s | one_action_then_stochastic | inverse | -0.004554 | 8.575 | 7.475 [-0.900,12.500] | 1/4 | -0.008243 | -0.008243 |
| low_cpu_1s | one_action_then_stochastic | equal_low | 0.001472 | 10.150 | 10.525 [7.600,13.600] | 0/4 | -0.010453 | -0.010262 |
| low_cpu_1s | sustained | equal | -0.003167 | 107.700 | 219.100 [219.100,219.100] | 0/1 | -0.100599 | -0.101627 |
| low_cpu_1s | sustained | avoid_low | 0.000778 | 105.900 | 178.000 [178.000,178.000] | 0/1 | -0.097509 | -0.101338 |
| low_cpu_1s | sustained | inverse | -0.004554 | 97.300 | 224.200 [224.200,224.200] | 0/1 | -0.096546 | -0.097734 |
| low_cpu_1s | sustained | equal_low | 0.001472 | 67.000 | 113.200 [113.200,113.200] | 0/1 | -0.053684 | -0.052922 |
| fixed_450s | one_action_then_stochastic | equal | -0.004969 | 41.425 | 170.300 [-66.900,640.800] | 2/4 | -0.075493 | -0.097075 |
| fixed_450s | one_action_then_stochastic | avoid_low | -0.006041 | 5.425 | -22.350 [-98.600,28.700] | 2/4 | 0.005584 | -0.004707 |
| fixed_450s | one_action_then_stochastic | inverse | -0.004801 | 7.925 | -43.750 [-192.500,47.600] | 3/4 | 0.010285 | 0.006808 |
| fixed_450s | one_action_then_stochastic | equal_low | -0.008420 | 44.075 | 144.625 [20.500,256.200] | 0/4 | -0.064023 | -0.077442 |
| fixed_450s | sustained | equal | -0.004969 | -102.900 | -1454.900 [-1454.900,-1454.900] | 1/1 | 0.448399 | 0.673857 |
| fixed_450s | sustained | avoid_low | -0.006041 | 59.300 | -955.900 [-955.900,-955.900] | 1/1 | 0.231589 | 0.470927 |
| fixed_450s | sustained | inverse | -0.004801 | -185.200 | -1635.700 [-1635.700,-1635.700] | 1/1 | 0.537124 | 0.772051 |
| fixed_450s | sustained | equal_low | -0.008420 | 259.300 | 532.200 [532.200,532.200] | 0/1 | -0.316742 | -0.274728 |
| low_cpu_5s | one_action_then_stochastic | equal | -0.001292 | -6.475 | -229.375 [-524.400,187.900] | 3/4 | 0.052321 | 0.082093 |
| low_cpu_5s | one_action_then_stochastic | avoid_low | -0.002630 | -51.275 | -164.475 [-523.000,26.700] | 3/4 | 0.066335 | 0.082085 |
| low_cpu_5s | one_action_then_stochastic | inverse | -0.013364 | 35.925 | -157.300 [-514.700,248.300] | 2/4 | 0.023380 | 0.040329 |
| low_cpu_5s | one_action_then_stochastic | equal_low | -0.010105 | 63.050 | 133.050 [-125.700,470.100] | 2/4 | -0.080450 | -0.085075 |
| low_cpu_5s | sustained | equal | -0.001292 | 1602.300 | 1246.800 [1246.800,1246.800] | 0/1 | -1.078518 | -0.800745 |
| low_cpu_5s | sustained | avoid_low | -0.002630 | -77.700 | -2986.200 [-2986.200,-2986.200] | 1/1 | 0.939846 | 1.443552 |
| low_cpu_5s | sustained | inverse | -0.013364 | -63.500 | -3175.400 [-3175.400,-3175.400] | 1/1 | 0.968593 | 1.473705 |
| low_cpu_5s | sustained | equal_low | -0.010105 | 2501.000 | 5456.900 [5456.900,5456.900] | 0/1 | -2.789638 | -2.816794 |
| low_cpu_5s | one_action_then_deterministic | equal | -0.001292 | 193.400 | 122.600 [122.600,122.600] | 0/1 | -0.114438 | -0.096209 |
| low_cpu_5s | one_action_then_deterministic | avoid_low | -0.002630 | 24.800 | -232.600 [-232.600,-232.600] | 1/1 | 0.056454 | 0.095924 |
| low_cpu_5s | one_action_then_deterministic | inverse | -0.013364 | 251.700 | 265.000 [265.000,265.000] | 0/1 | -0.181455 | -0.177887 |
| low_cpu_5s | one_action_then_deterministic | equal_low | -0.010105 | 168.200 | -66.500 [-66.500,-66.500] | 1/1 | -0.046735 | -0.011939 |
| low_cpu_15s | one_action_then_stochastic | equal | -0.071552 | 193.925 | 339.725 [-18.600,752.000] | 1/4 | -0.193526 | -0.198772 |
| low_cpu_15s | one_action_then_stochastic | avoid_low | -0.075784 | 121.775 | 241.500 [102.700,585.600] | 0/4 | -0.126254 | -0.139618 |
| low_cpu_15s | one_action_then_stochastic | inverse | -0.079918 | 320.350 | 488.900 [-467.200,1035.400] | 1/4 | -0.294429 | -0.299489 |
| low_cpu_15s | one_action_then_stochastic | equal_low | -0.081618 | 273.550 | 683.100 [367.100,1229.800] | 0/4 | -0.330472 | -0.347839 |
| low_cpu_15s | sustained | equal | -0.071552 | 3548.300 | 6530.100 [6530.100,6530.100] | 0/1 | -3.445044 | -3.378441 |
| low_cpu_15s | sustained | avoid_low | -0.075784 | -457.100 | -4744.200 [-4744.200,-4744.200] | 1/1 | 1.646198 | 2.267458 |
| low_cpu_15s | sustained | inverse | -0.079918 | 1129.000 | -3191.900 [-3191.900,-3191.900] | 1/1 | 0.569340 | 1.191368 |
| low_cpu_15s | sustained | equal_low | -0.081618 | 4242.500 | 9757.300 [9757.300,9757.300] | 0/1 | -4.739408 | -4.844061 |
| low_cpu_15s | one_action_then_deterministic | equal | -0.071552 | 192.200 | 369.900 [369.900,369.900] | 0/1 | -0.194345 | -0.190640 |
| low_cpu_15s | one_action_then_deterministic | avoid_low | -0.075784 | 134.500 | 371.200 [371.200,371.200] | 0/1 | -0.171330 | -0.173551 |
| low_cpu_15s | one_action_then_deterministic | inverse | -0.079918 | 310.300 | 528.900 [528.900,528.900] | 0/1 | -0.297319 | -0.292197 |
| low_cpu_15s | one_action_then_deterministic | equal_low | -0.081618 | 158.800 | 161.500 [161.500,161.500] | 0/1 | -0.117598 | -0.075511 |

| Scene | win equal | low-row switch rates | cycle-weighted local high/low |
|---:|---|---|---|
| 1 | True | 0.786, 0.511, 0.740 | 0.866/0.635 |
| 2 | True | 0.592, 0.666, 0.512 | 0.891/0.723 |
| 3 | False | 0.790, 0.543, 0.770 | 0.778/0.601 |
| 4 | False | 0.782, 0.514, 0.765 | 0.807/0.642 |
| 5 | True | 0.246, 0.581, 0.385 | 0.950/0.701 |
| 6 | True | 0.180, 0.588, 0.400 | 0.960/0.698 |
| 7 | True | 0.210, 0.518, 0.355 | 0.942/0.698 |
| 8 | False | 0.775, 0.440, 0.723 | 0.811/0.615 |
| 9 | False | 0.399, 0.599, 0.336 | 0.634/0.523 |
| 10 | True | 0.703, 0.717, 0.629 | 0.930/0.705 |
| 11 | True | 0.634, 0.684, 0.611 | 0.926/0.726 |
| 12 | True | 0.663, 0.668, 0.580 | 0.867/0.709 |
| 13 | True | 0.644, 0.676, 0.588 | 0.916/0.719 |
| 14 | True | 0.629, 0.683, 0.566 | 0.896/0.714 |
| 15 | True | 0.251, 0.580, 0.373 | 0.965/0.681 |
| 16 | True | 0.211, 0.540, 0.285 | 0.974/0.708 |
| 17 | True | 0.271, 0.534, 0.486 | 0.930/0.699 |
| 18 | True | 0.635, 0.685, 0.568 | 0.926/0.712 |
| 19 | True | 0.792, 0.558, 0.739 | 0.859/0.628 |
| 20 | True | 0.640, 0.708, 0.561 | 0.929/0.711 |
