# Fixed inverse CPU backlog diagnostic

Only the existing scene and frozen 20-scene dev manifest were used. No model was instantiated, no learning update or replay write was performed, and no research source/configuration was changed.

`B_d = (current high-priority CPU cycles + current low-priority CPU cycles) / cpu_hz`
`w_d = 1/(B_d + 0.5 s) / sum_j 1/(B_j + 0.5 s)`
`route[priority, source, destination] = w_destination`

All sources and priorities use the same weights, recomputed every existing 0.5 s routing interval. The upper action is hover. The 0.5 s regularizer was fixed before execution; no formula or parameter was selected from results. The rule ignores ready work, in-transit work, communication delay and future arrivals. It uses only Report.uavs.compute and cpu_hz.

One deterministic 900 s window per scene: 20 windows, 180,000 physical steps. Each frozen baseline has two identical policy streams per scene. Rates and delays are pooled by the existing PooledMetrics; unfinished counts below are means per window. Scenario comparisons average the two frozen streams first, so their doubled window count is not treated as doubled congestion.

## Overall metrics

| Policy | Truncated delay (weighted s) | High s | Low s | Completed weighted mean s | High p95 s | Low p95 s | Mean unfinished/window | Oldest unfinished s | Cost/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inverse_cpu_backlog | 0.583973 | 0.475706 | 0.974637 | 0.583898 | 0.800000 | 2.100000 | 14.300 | 3.000 | 9.872825 |
| hover_equal | 0.580338 | 0.472992 | 0.967681 | 0.580259 | 0.800000 | 2.100000 | 14.250 | 3.000 | 9.844955 |
| hover_local | 5.460339 | 0.391833 | 23.749373 | 5.166323 | 0.700000 | 79.300000 | 704.650 | 114.500 | 44.154346 |
| hover_cvx | 0.638502 | 0.481676 | 1.204390 | 0.638405 | 0.800000 | 2.900000 | 17.650 | 3.500 | 10.301826 |
| centroid_equal | 0.565260 | 0.457339 | 0.954680 | 0.565196 | 0.700000 | 2.100000 | 13.850 | 2.500 | 9.595147 |

Stage residence uses the existing last-finishing-branch definition, including queue waiting; it is not pure resource service time.

| Policy | Upload s | Ready s | U2U s | CPU s | Propulsion J/window | CPU J/window | Upload J/window | U2U J/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inverse_cpu_backlog | 0.258186 | 0.049765 | 0.107300 | 0.168722 | 525861.302 | 199690.206 | 541.300 | 1078.431 |
| hover_equal | 0.258186 | 0.049765 | 0.107856 | 0.164531 | 525861.302 | 199394.188 | 541.300 | 1079.267 |
| hover_local | 0.258186 | 0.049765 | 0.000000 | 5.152388 | 525861.302 | 204748.207 | 541.300 | 0.000 |
| hover_cvx | 0.258186 | 0.049765 | 0.095088 | 0.235464 | 525861.302 | 205802.750 | 541.300 | 762.285 |
| centroid_equal | 0.237651 | 0.051020 | 0.110769 | 0.165820 | 506998.967 | 200121.517 | 507.337 | 1115.839 |

## Paired scenario comparison

Arithmetic mean across 20 scenes: inverse backlog 0.583671926 s; equal split 0.580034889 s. Relative improvement = -0.6270%; wins = 0/20.

Sum of one cutoff per scene: heuristic unfinished 286; equal split (two streams averaged within scene) 285.0.

| Scene | Heuristic s | Equal s | Improvement % | Heuristic unfinished | Equal mean unfinished |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.567552 | 0.563838 | -0.659 | 20 | 20.0 |
| 2 | 0.622603 | 0.618772 | -0.619 | 53 | 54.0 |
| 3 | 0.610802 | 0.606809 | -0.658 | 3 | 3.0 |
| 4 | 0.608634 | 0.605700 | -0.484 | 12 | 13.0 |
| 5 | 0.554559 | 0.551187 | -0.612 | 7 | 7.0 |
| 6 | 0.572045 | 0.568530 | -0.618 | 6 | 5.0 |
| 7 | 0.563722 | 0.559606 | -0.736 | 18 | 16.0 |
| 8 | 0.642518 | 0.639779 | -0.428 | 12 | 12.0 |
| 9 | 0.621713 | 0.617232 | -0.726 | 8 | 9.0 |
| 10 | 0.562108 | 0.557575 | -0.813 | 9 | 7.0 |
| 11 | 0.583819 | 0.579781 | -0.696 | 18 | 18.0 |
| 12 | 0.627751 | 0.624444 | -0.530 | 9 | 9.0 |
| 13 | 0.608075 | 0.605583 | -0.411 | 31 | 32.0 |
| 14 | 0.596024 | 0.593952 | -0.349 | 12 | 11.0 |
| 15 | 0.545356 | 0.540890 | -0.826 | 10 | 11.0 |
| 16 | 0.542912 | 0.539778 | -0.581 | 13 | 13.0 |
| 17 | 0.554167 | 0.548896 | -0.960 | 10 | 10.0 |
| 18 | 0.578423 | 0.573365 | -0.882 | 6 | 6.0 |
| 19 | 0.535396 | 0.531110 | -0.807 | 22 | 22.0 |
| 20 | 0.575261 | 0.573871 | -0.242 | 7 | 7.0 |

## Verification and scope

- Every scene matched all four frozen baselines, both policy streams, on realization SHA256, parent ID/birth/priority SHA256, and per-region arrived bits/cycles/count.
- Each scene completed exactly 9,000 physical steps and 1,800 scripted lower actions.
- Across 36000 actions, all six rows were finite, positive and full-support. Maximum absolute row-sum error: 2.2204460492503131e-16.
- The existing collector also checked support consistency, physical reward/cost identity and holding/censored-residence identity.
- Total wall-clock for this tool invocation: 230.805 s. Source/config/scene identity remained unchanged.
- Full per-parent physical metrics and action digests are in each hover_inverse_backlog_XX.json. The manifest fixes formula, observation fields, scene seeds and source/config hashes.
- This simple heuristic provides one achievable reference. Success could show room above equal split; failure does not demonstrate equal split is optimal or that such room is absent. No G1 threshold was changed.
