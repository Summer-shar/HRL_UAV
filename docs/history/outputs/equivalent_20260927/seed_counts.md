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

Limitations:

- Counts describe a fresh uninterrupted 600-window run under the current code/config; resumed or failed work adds unknown discarded work and I/O.
- Warmup is two lower windows per block. Empty replay skips each lower block's first window; upper-version changes flush replay. Event parity is global across nonempty events.
- Each moving window includes one extra committed upper action at cutoff; upper-training windows also infer its value. That successor is created before the final SAC event and the world is then discarded by run.train.
- The two measured scripted physical trajectories are not an empirical distribution of trained policies. Queue growth, graph size, history lengths, replay uniqueness and hardware variability can move runtime outside every shown range.
- Profile operation times include their documented boundaries only. Standalone report/graph/GRU diagnostics and evaluation.replay900s are not added to avoid double counting.
- Checkpoint occupancy is enumerated exactly; 0/2 to 55 episode-record time/bytes are affine extrapolations. Pending PPO records use replay-record size as a proxy.
- PPO fixture KL behavior does not determine future rollback frequency. Actor optimizer steps can range from the first epoch through all six; value steps remain fixed.
- Unpriced residuals include scene/reset/startup cost outside measured collection, three scene integrity digests, initial module construction, report-history retention/GC, filesystem contention, checkpoint pruning, and future workload drift. No arbitrary reserve converts these unknowns into measured time.
- Optional final-checkpoint paired stochastic testing is shown separately; secondary checkpoints, deterministic panels, baselines and five-seed multiplication are outside the one-seed training-plus-development scope.
