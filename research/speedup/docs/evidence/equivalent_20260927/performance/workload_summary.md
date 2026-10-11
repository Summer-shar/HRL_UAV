# Baseline workload and memory summary

Counts below use explicit episode metadata. Report records are not independent episodes. Sources 0 and 1 are the two actual 900 s trajectories; copying a source does not create another trajectory.

| Case | Episode objects / true trajectories | History upper bound | Single-graph nodes | Single-graph edges | Max FIFO upload / ready / transfer / CPU |
| --- | ---: | ---: | ---: | ---: | --- |
| critic.e2.h120.normal | 2 / 2 | 120 | 65–176 | 271–1032 | 4 / 4 / 4 / 58 |
| critic.e2.h120.backlog | 2 / 2 | 120 | 65–213 | 271–1295 | 4 / 5 / 4 / 58 |
| critic.e2.h600.normal | 2 / 2 | 600 | 71–542 | 297–3600 | 6 / 5 / 3 / 174 |
| critic.e2.h600.backlog | 2 / 2 | 600 | 71–609 | 307–4063 | 6 / 4 / 4 / 182 |
| critic.e2.h1800.normal | 2 / 2 | 1800 | 64–660 | 256–4424 | 7 / 5 / 4 / 171 |
| critic.e2.h1800.backlog | 2 / 2 | 1800 | 64–660 | 256–4424 | 7 / 4 / 4 / 181 |

These are fixed stratified samples with one maximum-prefix anchor per episode object. The history bound includes the successor; current report cursors can be earlier. Source mappings and sample counts are preserved in JSON.

Memory values below are maxima over recorded repetitions in MiB. RSS was sampled; CUDA peaks include the resident model and allocator cache. They are not memory deltas or full-training bounds.

| Case | Repeats | Max RSS MiB | Max allocated MiB | Max reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| actor.e1.h1800.backlog | 3 | 2366.88 | 553.39 | 1128.00 |
| actor.e1.h1800.normal | 3 | 2310.04 | 428.80 | 864.00 |
| actor.e2.h1800.backlog | 3 | 2417.66 | 590.67 | 1128.00 |
| actor.e2.h1800.normal | 3 | 2427.02 | 470.36 | 1252.00 |
| actor.e4.h1800.backlog | 3 | 2431.91 | 682.65 | 1384.00 |
| actor.e4.h1800.normal | 3 | 2432.01 | 563.94 | 1180.00 |
| checkpoint.digest2 | 3 | 2651.25 | 149.53 | 1410.00 |
| checkpoint.load2 | 3 | 2801.56 | 149.53 | 1410.00 |
| checkpoint.save0 | 3 | 2789.45 | 306.64 | 1410.00 |
| checkpoint.save2 | 3 | 2840.90 | 201.90 | 1410.00 |
| critic.e1.h1800.backlog | 3 | 2365.81 | 916.63 | 1128.00 |
| critic.e1.h1800.normal | 3 | 2311.59 | 686.61 | 864.00 |
| critic.e2.h120.backlog | 3 | 2314.40 | 385.25 | 470.00 |
| critic.e2.h120.normal | 3 | 2295.99 | 327.64 | 352.00 |
| critic.e2.h1800.backlog | 3 | 2447.86 | 987.30 | 1252.00 |
| critic.e2.h1800.normal | 3 | 2423.10 | 764.54 | 1252.00 |
| critic.e2.h600.backlog | 3 | 2403.06 | 871.98 | 1134.00 |
| critic.e2.h600.normal | 3 | 2359.89 | 662.35 | 754.00 |
| critic.e4.h1800.backlog | 3 | 2435.84 | 1167.12 | 1384.00 |
| critic.e4.h1800.normal | 3 | 2433.82 | 943.09 | 1180.00 |
| evaluation.replay900s | 3 | 3362.55 | 163.59 | 1410.00 |
| inference.lower.h1.rebuild | 3 | 2421.05 | 157.94 | 1384.00 |
| inference.lower.h120.backlog.cached | 3 | 2412.67 | 158.06 | 1384.00 |
| inference.lower.h120.backlog.rebuild | 3 | 2428.96 | 163.07 | 1384.00 |
| inference.lower.h120.cached | 3 | 2408.00 | 157.84 | 1384.00 |
| inference.lower.h120.normal.cached | 3 | 2412.67 | 157.91 | 1384.00 |
| inference.lower.h120.normal.rebuild | 3 | 2428.97 | 164.71 | 1384.00 |
| inference.lower.h120.rebuild | 3 | 2420.28 | 165.93 | 1384.00 |
| inference.lower.h1800.after_actor_update | 3 | 2351.75 | 275.11 | 1384.00 |
| inference.lower.h1800.backlog.cached | 3 | 2418.20 | 158.05 | 1384.00 |
| inference.lower.h1800.backlog.rebuild | 3 | 2430.61 | 268.30 | 1384.00 |
| inference.lower.h1800.cached | 3 | 2405.11 | 157.84 | 1384.00 |
| inference.lower.h1800.normal.cached | 3 | 2414.30 | 157.87 | 1384.00 |
| inference.lower.h1800.normal.rebuild | 3 | 2428.99 | 269.55 | 1384.00 |
| inference.lower.h1800.rebuild | 3 | 2419.81 | 269.79 | 1384.00 |
| inference.lower.h600.backlog.cached | 3 | 2412.79 | 157.98 | 1384.00 |
| inference.lower.h600.backlog.rebuild | 3 | 2429.02 | 194.19 | 1384.00 |
| inference.lower.h600.cached | 3 | 2404.11 | 158.01 | 1384.00 |
| inference.lower.h600.normal.cached | 3 | 2416.68 | 158.01 | 1384.00 |
| inference.lower.h600.normal.rebuild | 3 | 2428.98 | 194.87 | 1384.00 |
| inference.lower.h600.rebuild | 3 | 2419.81 | 194.87 | 1384.00 |
| inference.upper.h0.rebuild | 3 | 2429.46 | 149.73 | 1384.00 |
| inference.upper.h120.cached | 3 | 2413.26 | 158.18 | 1384.00 |
| inference.upper.h120.rebuild | 3 | 2429.23 | 165.93 | 1384.00 |
| inference.upper.h1800.cached | 3 | 2414.52 | 158.18 | 1384.00 |
| inference.upper.h1800.rebuild | 3 | 2429.21 | 269.79 | 1384.00 |
| inference.upper.h600.cached | 3 | 2416.91 | 158.18 | 1384.00 |
| inference.upper.h600.rebuild | 3 | 2429.24 | 194.87 | 1384.00 |
| inference.upper_value.h0.rebuild | 3 | 2429.45 | 149.74 | 1384.00 |
| inference.upper_value.h120.cached | 3 | 2413.39 | 158.19 | 1384.00 |
| inference.upper_value.h120.rebuild | 3 | 2429.21 | 165.94 | 1384.00 |
| inference.upper_value.h1800.cached | 3 | 2418.17 | 158.19 | 1384.00 |
| inference.upper_value.h1800.rebuild | 3 | 2428.96 | 269.79 | 1384.00 |
| inference.upper_value.h600.cached | 3 | 2413.01 | 158.19 | 1384.00 |
| inference.upper_value.h600.rebuild | 3 | 2429.22 | 194.88 | 1384.00 |
| io.learning_jsonl | 3 | 2654.42 | 149.53 | 1410.00 |
| io.metrics_json | 3 | 2654.42 | 149.53 | 1410.00 |
| ppo.rows900 | 3 | 2727.16 | 715.83 | 1410.00 |
| replay.sample128 | 3 | 2649.07 | 149.53 | 1410.00 |
| source.identity | 3 | 2648.50 | 149.53 | 1410.00 |
| target.polyak | 3 | 2654.41 | 149.53 | 1410.00 |

Global observed peaks across these cases:

- rss_peak: 3362.55 MiB (evaluation.replay900s).
- cuda_peak_allocated: 1167.12 MiB (critic.e4.h1800.backlog).
- cuda_peak_reserved: 1410.00 MiB (checkpoint.digest2, checkpoint.load2, checkpoint.save0, checkpoint.save2, evaluation.replay900s, io.learning_jsonl, io.metrics_json, ppo.rows900, replay.sample128, source.identity, target.polyak).

Copied episode objects in the measured workload set:

- `actor.e4.h1800.backlog`: 4 objects from source IDs [0, 1, 0, 1]; only 2 actual trajectories.
- `actor.e4.h1800.normal`: 4 objects from source IDs [0, 1, 0, 1]; only 2 actual trajectories.
- `critic.e4.h1800.backlog`: 4 objects from source IDs [0, 1, 0, 1]; only 2 actual trajectories.
- `critic.e4.h1800.normal`: 4 objects from source IDs [0, 1, 0, 1]; only 2 actual trajectories.
- `ppo.rows900`: 5 objects from source IDs [0, 1, 0, 1, 0]; only 2 actual trajectories.
