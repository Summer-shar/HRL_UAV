# Current single-seed conditional budget

No research training or physical steps are executed by this tool.

| Exact current count | Value |
|---|---:|
| episodes | 600 |
| physics_steps | 5,400,000 |
| sac_triggers | 36,000 |
| skipped_events | 990 |
| sac_events | 35,010 |
| critic_fits | 70,020 |
| sac_actor_updates | 17,010 |
| ppo_batches | 40 |
| ppo_rows_per_batch | 900 |
| lower_inference | 1,080,000 |
| upper_inference | 54,300 |
| value_inference | 36,200 |
| interior_actor_cache_rebuilds | 16,632 |
| final_actor_updates_without_online_rebuild | 378 |
| dev_episodes | 156 |
| dev_physics_steps | 1,404,000 |
| dev_upper_inference | 13,032 |
| checkpoint_saves | 608 |

| Scope / component | Conditional hours |
|---|---:|
| training / sac_critic | 120.656 – 150.774 |
| training / sac_actor | 15.726 – 19.710 |
| training / replay_sampling | 0.013 – 0.014 |
| training / target_polyak | 0.062 – 0.074 |
| training / ppo_batches_fixture_KL | 0.916 – 0.937 |
| training / physics_reports_compression_metrics | 2.287 – 2.287 |
| training / lower_cold_inference | 0.004 – 0.004 |
| training / lower_cached_inference | 7.901 – 10.862 |
| training / lower_post_update_rebuild | 0.139 – 0.184 |
| training / upper_without_value_cold | 0.001 – 0.001 |
| training / upper_without_value_cached | 0.125 – 0.164 |
| training / upper_with_value_cold | 0.002 – 0.002 |
| training / upper_with_value_cached | 0.426 – 0.588 |
| training / learning_and_attempt_logs | 0.004 – 0.007 |
| training / atomic_json_outside_checkpoint_save | 0.092 – 0.095 |
| training / source_identity_hashing | 0.001 – 0.001 |
| training / initial_parameter_hashing | unknown |
| training / checkpoint_save | 2.384 – 2.567 |
| training / additional_checkpoint_digests | 0.228 – 0.251 |
| **training: known subtotal** | **150.97 – 188.52** |
| development / physics_reports_metrics | 0.402 – 0.402 |
| development / moving_complete_report_replay | 1.984 – 2.009 |
| development / hover_lower_cold_inference | 0.001 – 0.001 |
| development / hover_lower_cached_inference | 1.123 – 1.544 |
| development / uncovered_window_and_panel_json | 0.005 – 0.005 |
| **development: known subtotal** | **3.52 – 3.96** |
| optional_primary_test / physics_reports_metrics | 0.413 – 0.413 |
| optional_primary_test / moving_complete_report_replay | 2.204 – 2.232 |
| optional_primary_test / hover_lower_cold_inference | 0.001 – 0.001 |
| optional_primary_test / hover_lower_cached_inference | 1.070 – 1.471 |
| optional_primary_test / uncovered_window_and_panel_json | 0.004 – 0.004 |
| **optional_primary_test: known subtotal** | **3.69 – 4.12** |

Unknown components are not zero. These are measured-workload sensitivity estimates, not confidence bounds or a completed seed runtime.

| Scope | Primary conditional hours | Separate hover-as-moving scenario hours |
|---|---:|---:|
| development | 3.52 – 3.96 | 4.70 – 4.75 |
| optional_primary_test | 3.69 – 4.12 | 4.82 – 4.88 |
| training_plus_development | 154.48 – 192.48 | 155.67 – 193.28 |
| training_development_primary_test | 158.17 – 196.60 | 160.49 – 198.15 |

The separate scenario prices hover using the moving replay, including additional upper work. It is neither a measured pure-hover runtime nor a strict upper bound. Moving replay already contains all its inference, metric pooling and one JSON write; these are not added again.

Inference cross-check: complete source-0 report replay 99.20 – 100.43 s versus fixed source-1 section model 52.89 – 72.34 s per moving window (1.37 – 1.90 times). The difference includes trajectory, sequencing and pooling/JSON effects. It is not added to training or applied as a training multiplier.

PPO's empirical timing range holds its measured fixture KL behavior fixed. It does not bound arbitrary early stopping: actor steps attempted are 240–1,440, accepted steps can be 0; value steps remain 1,440. Startup/reset/GC, unmeasured hashing and workload-growth residuals remain unpriced; total wall-time bounds are unknown.

Limitations:

- Counts describe a fresh uninterrupted 600-window run under the current code/config; resumed or failed work adds unknown discarded work and I/O.
- Warmup is two lower windows per block. Empty replay skips each lower block's first window; upper-version changes flush replay. Event parity is global across nonempty events.
- Each moving window includes one extra committed upper action at cutoff; upper-training windows also infer its value. That successor is created before the final SAC event and the world is then discarded by run.train.
- The two measured scripted physical trajectories are not an empirical distribution of trained policies. Queue growth, graph size, history lengths, replay uniqueness and hardware variability can move runtime outside every shown range.
- Profile operation times include their documented boundaries only. Moving evaluation uses full evaluation.replay900s in place of its standalone lower/upper inference and per-window JSON; physical/report construction remains separate. Standalone report/graph/GRU diagnostics are not added again.
- No complete pure-lower hover replay was measured. The primary estimate retains hover component prices; a separate scenario charges it the full moving replay with extra upper work, without claiming a strict upper bound. The whole-replay versus fixed-section discrepancy is reported but not used to rescale training.
- Checkpoint occupancy is enumerated exactly; 0/2 to 55 episode-record time/bytes are affine extrapolations. Pending PPO records use replay-record size as a proxy.
- PPO fixture KL behavior does not determine future rollback frequency. Actor optimizer steps attempted range from 240 to 1440; accepted steps can be zero after rollback. Value steps remain 1440. Empirical fixture timing ranges are not theoretical bounds over arbitrary early-stop outcomes.
- Unpriced residuals include scene/reset/startup cost outside measured collection, three scene integrity digests, initial module construction, report-history retention/GC, filesystem contention, checkpoint pruning, and future workload drift. No arbitrary reserve converts these unknowns into measured time.
- Optional final-checkpoint paired stochastic testing is shown separately; secondary checkpoints, deterministic panels, baselines and five-seed multiplication are outside the one-seed training-plus-development scope.
