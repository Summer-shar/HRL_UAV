# Task 2 acceptance

Code commit: `a74002c0c649819ec2cf890537611ed6ed77c617` (exactly the eight accepted Task 2 files).
Final test commit: `e914db2707581c456f5943c0aa7de3e202e48c8f` (one additional test).

## Tests

Previous result: 159 V2 tests = 158 passed + 1 skipped; root suite = 17 passed (configuration 8, packaging 5, performance tools 4). Thus 176 tests total, 175 passed. The apparent difference of 16 is +17 root tests minus the one skipped V2 test.
Current validation: 160 tests = 159 passed + the same one conditional skip; 0 failures, 0 errors; 83.2022158 s. Root tests were unchanged and passed in the preceding Task 2 run.
The added test runs actual deterministic paper panels on six dev scenarios with five controllers and two streams; every metric has identical serialized bytes across the two streams. Synthetic weights only; no learning updates.

## References and exact comparison

Old reference: `../baselines/summary.json`, verified byte-identical to the blob in commit `049c0da`.
Old SHA256: `475477f47a126fb47127065701a6752eb87094df58e798d388affc651dd7154c`
New reference: `../baselines_task2/summary.json`.
New SHA256: `0217baec1d3397e2bbedde92880e17701520c9372f9591bafe5d32f25544ab7c`

Only hover_local, hover_equal, hover_cvx were recomputed: 20 dev scenarios x 2 policy streams x 3 baselines = 120 windows, 1,080,000 physical steps. No old reference cache was reused.
All 60 baseline/scenario metric dictionaries, all 120 raw window metric dictionaries (including parents, costs, energy and diagnostic metrics), and all 3 by-mode summaries are exactly equal. Float leaves were compared as IEEE-754 bytes, with no tolerance. Non-metric source identity and wall times are reported separately and necessarily differ.
All old-directory file hashes were unchanged. See comparison.json, comparison.csv and old_directory_snapshot.json.

## Wall time

| Run | Workload | Wall seconds |
|---|---|---:|
| Old complete run | 5 groups / 200 windows | 771.311547 |
| New complete computation + freeze | 3 groups / 120 windows | 467.128818 |
| New computation including worker startup | 3 groups / 120 windows | 451.059095 |
| New cache aggregation + summary write | 3 groups | 16.065961 |
| Exact comparison and old-file integrity check | 120 raw + 60 pooled + 3 totals | 47.195528 |

The complete old and new runs have different workloads; these numbers are not a speedup claim. The summary.json wall_seconds fields only time cached aggregation: old 27.411310 s, new 15.984092 s.

Per-case cumulative times (parallel durations summed; not elapsed wall time):

| Baseline | Old summed case seconds | New summed case seconds |
|---|---:|---:|
| hover_local | 1607.427404 | 1321.201791 |
| hover_equal | 602.892205 | 518.712375 |
| hover_cvx | 806.174684 | 700.838165 |

## State

Validation and new reference identities match the final committed implementation/config/scene. Git working tree is clean. No configuration, G1, checkpoint identity rules, existing checkpoint or seed 4101 output was modified. No training, resume or push was performed.
