# V2.2.1 Lite observation and gated run

Parent: `b03e4985fc9eeb4e8b2759c93e8cf87ad9155872`, branch `codex/v2-lite-candidate`.
Configuration: `src/hrl_v2/configs/v2_2_1_lite.json`. The previous files are unchanged.
Only `encoder=lite` and `observation_version=lite_110_729_v2_2_1` differ from
`v2_1_0.json`; a test asserts every other field remains identical. Old Lite
configuration files are retained as historical artifacts and are rejected by
this training entrypoint rather than silently acquiring the new observation.

## Observation contract

| Layer / segment | Dimensions | Contents |
|---|---:|---|
| Lower demand | 36 | Each priority × source: ready bits, cycles, count; upload bits; cycles EMA at 5/60 s |
| Lower CPU | 18 | Backlog cycles, count, oldest queue age |
| Lower U2U | 24 | Backlog bits and fragment count per priority/directed link |
| Lower links | 12 | Six current rates followed by six all-active potential rates |
| Lower parents | 6 | Unfinished parent counts |
| Lower geometry | 12 | UAV positions and committed targets |
| Lower clocks | 2 | Remaining macro and motion adjustment time |
| **Lower total** | **110** | Existing lower network topology, updated input width |
| Upper terminals | 39 × 17 | Own-UAV relative dx/dy; uplink rate; each priority's upload bits/count/oldest age; bits/cycles EMA at 5/60 s |
| Upper global | 66 | UAV positions 6; CPU 18; ready bits/cycles/count 18; U2U backlog bits 12; potential rates 6; unfinished parents 6 |
| **Upper raw total** | **729** | No clocks or targets |

Upper actor and value each own an independent 17→64→64 terminal MLP. Mean and
max over the 13 terminals of each source produce 384 features, concatenated
with 66 global features, then 450→256→256 and the existing output heads.
Pooling does not mix source groups. Geometry is divided by map size; positive
quantities use the existing `log1p(value/scale)` transform. Bits/rates use 1e6,
cycles use 1e9, queue ages use 10 s. The EMA is per terminal/priority/workload
with decay `exp(-0.5/tau)`, consuming only frames before `history_cursor`.

Upper reports are taken before committing the next motion action. Tests check
both remaining clocks are zero and targets equal current positions at those
reports, including the true upper bootstrap successor. Lower retains both
clocks and targets. Independent reset scenarios and pre-reset true successors
retain their existing semantics.

## Statistics

`truncated_delay_s` is priority-weighted parent residence within the window,
including unfinished parents through the cutoff, divided by weighted arrivals.
For adjacent continuing subwindows, carry-in parents are explicitly included
in the service-population denominator; formal reset episodes have no carry-in.
Priority weights remain 10 and 1. Completed delay and per-priority p95 are
reported separately; p95 is computed on completed parent samples, not on means.

`stage_delay_s` partitions this residence into upload, ready, U2U and CPU along
the last-finishing fragment of each parent. All stages include their queue
waiting time. At censoring, unfinished fragments tie at the cutoff; the smallest
fragment ID is chosen. This is a reproducible attribution of censored parent
time, not a measurement of pure transmission/CPU execution time. Parallel
fragment durations are never summed. Stage totals equal censored parent
residence, and actual upload finish timestamps preserve sub-tick precision.

These statistics only read existing timestamps. Physical holding, energy,
cost, reward, queue service, arrival streams and update schedules are unchanged.
`R_physical` / `R_learning` retain the existing separate fields and definitions.
Gamma 0.99 per lower step, macro gamma 0.99^10 and lambda 0.95 remain candidate
baseline values, not claims of optimality or performance acceptance.

`train_metrics.jsonl` records one normal completed episode per line: physical
metrics, phase, alpha, mean critic residual MSE, actual SAC counter deltas,
memory, work/checkpoint/evaluation wall times. Actor relative entropy is the
negative log density of already executed actions against the existing Hybrid
reference measure, including the real lower bootstrap action. No extra random
draw is made. Interrupted log appends can be recovered from the episode record;
unknown checkpoint/evaluation times are recorded as null, never fabricated.

## Baselines and gates

The existing development manifest has **6 scenes × 2 policy streams**. The
existing **20-scene manifest is test data**. The requested G1 threshold is
12/20, so the final gate protocol requires an explicit choice: extend dev to
20 independent dev-seeded scenes, or retain dev6 and approve 4/6. Neither option
changes the physical or 600-episode training configuration. Test data must not
be repurposed for development. The evaluator supports both sealed dev manifests;
it never counts policy streams as separate scenarios.

`baseline_reference` uses `Policy.lower_override`, no model or optimizer, for
hover+Local, hover+equal, hover+CVX, and centroid+equal. Each scene/stream is saved
separately. CVX fallback counts are exposed. Final `summary.json` binds the
source/config/scene identity, seed, manifest and exact win threshold. Existing
six-scene workloads can be prepared before the gate choice, without freezing
an unapproved threshold:

```powershell
.venv/Scripts/python.exe -m hrl_v2.baseline_reference --config src/hrl_v2/configs/v2_2_1_lite.json --output outputs/lite_v2_2_1/baselines --scenarios 6 --prepare-only
```

After the gate choice, rerun with `--scenarios 20 --required-wins 12` or the
explicitly approved `--scenarios 6 --required-wins 4`, without `--prepare-only`.
Matching cached workloads are reused. Final baselines must be frozen before
fresh training; the training entrypoint enforces identity and seed matching.

G1 evaluates means of paired scenario metrics (streams pooled within scenario),
strictly lower than equal and Local, at least the approved number of wins, and
no greater total unfinished count than equal. Failure persists a checkpoint and
returns `paused_G1` before episode 301. Resume retains this block; no implicit
threshold override exists. G2 uses moving/hover with the same SAC and paired
seeds; improvement below 1% or win fraction below 60% records weak upper signal
and continues the fixed schedule. Weak gates create `diagnosis_required.json`.
Subsequent diagnosis/candidate scenes require no change to the running code.

## Commands and monitoring

```powershell
.venv/Scripts/hrl-v2.exe --workspace outputs/lite_v2_2_1 validate --config D:/HRL_UAV_V2/src/hrl_v2/configs/v2_2_1_lite.json --skip-benchmark
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/hrl-v2.exe --workspace outputs/lite_v2_2_1 train --config D:/HRL_UAV_V2/src/hrl_v2/configs/v2_2_1_lite.json --seed 4101 --fresh --output seed_4101 --baseline-reference baselines/summary.json
.venv/Scripts/python.exe tools/monitor_lite_run.py outputs/lite_v2_2_1/seed_4101
```

The launcher must redirect stdout/stderr and save its PID. Resume uses the path
in the run's `latest.json`, with the same workspace/config. Do not load an older
checkpoint to overwrite later work. Source/config identity is checked at every
episode; any requested implementation change requires stopping and a fresh
version/run. Baselines, tests and scripts are not formal training results.

First-10 reporting separates observed layers; all first ten are lower episodes,
so upper/PPO timings are explicitly unmeasured until that phase occurs. The
monitor budget distinguishes empty replay, critic-only warmup and actor updates,
including phase changes that clear replay; it does not assume 400 full-update
lower windows. Development panels and unknown upper timings remain explicit
unknowns until measured. Every 50 episodes report the 10-episode moving delay,
critic residual, alpha and relative actor entropy. G1 stops at 300 on failure;
G2 begins at 400. Candidate scenario training is never automatic.

## Verified implementation state

Full package validation: 137 tests, 0 failures, 0 errors, 1 pre-existing opt-in
64/128 critic audit skipped; 67.323 s. Entry/configuration/packaging tests: 17
passed. `validation.json` is passed and its source/config identity matches this
V2.2.1 candidate. Evidence: `docs/V2_2_1_VALIDATION.json`; full test log:
`outputs/lite_v2_2_1/validation/tests.txt`. Formal training has not started at
this implementation checkpoint; the dev gate choice remains pending.
