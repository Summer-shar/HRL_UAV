# Chrome trace CPU summary

Trace: D:\HRL_UAV_V2\outputs\equivalent_20260927\current\critic.e2.h1800.backlog.trace.json

All durations below are **nested inclusive CPU spans**. They overlap and cannot be summed or interpreted as wall-clock shares.

## CUDA activity

Kernel events: 0; memcpy events: 0; memset events: 0.

No actual CUDA kernel/memcpy events were captured. CUPTI activity data is unavailable in this trace; do not attribute time to CUDA kernels or transfers.

## User annotations

| Name | Calls | Inclusive CPU seconds | Instant events |
| --- | ---: | ---: | ---: |
| critic.e2.h1800.backlog | 1 | 7.224074700 | 0 |
| sac/critic_fit | 1 | 7.223958400 | 0 |
| autograd/backward | 10 | 2.520483500 | 0 |
| sac/target_construction | 1 | 2.415798500 | 0 |
| data/batch_graph_and_transfer | 40 | 2.341822500 | 0 |
| sac/actor/batch_forward | 8 | 1.781791400 | 0 |
| data/fixed_graph_features | 201 | 1.452557600 | 0 |
| sac/q1/batch_forward | 8 | 1.352515400 | 0 |
| network/typed_attention | 120 | 1.023984000 | 0 |
| data/replay_unpack | 224 | 0.349212000 | 0 |
| sac/q2/batch_forward | 8 | 0.310651900 | 0 |
| sac/target1/batch_forward | 8 | 0.285110000 | 0 |
| sac/target2/batch_forward | 8 | 0.284418300 | 0 |
| data/attention_indices | 16 | 0.263796900 | 0 |
| history/prefix_forward | 5 | 0.095357400 | 0 |
| sac/q1/arrival_gru | 2 | 0.074261300 | 0 |
| optimizer/adam_step | 1 | 0.045919300 | 0 |
| Optimizer.step#Adam.step | 1 | 0.045867300 | 0 |
| sac/hybrid_sampling | 8 | 0.026665400 | 0 |
| data/arrival_normalize_and_transfer | 10 | 0.014840300 | 0 |
| sac/actor/queue_upload_gru | 8 | 0.011661900 | 0 |
| sac/action_cast | 16 | 0.007884700 | 0 |
| history/full_prefix_backward | 2 | 0.006934500 | 0 |
| sac/q1/q_value_head | 8 | 0.005405300 | 0 |
| sac/target1/q_value_head | 8 | 0.004494500 | 0 |
| sac/q1/queue_upload_gru | 8 | 0.004411200 | 0 |
| sac/actor/queue_ready_gru | 8 | 0.004318700 | 0 |
| sac/q2/q_value_head | 8 | 0.004306800 | 0 |
| sac/target2/q_value_head | 8 | 0.004156100 | 0 |
| sac/target1/queue_upload_gru | 8 | 0.003824900 | 0 |
| sac/actor/queue_cpu_gru | 8 | 0.003787200 | 0 |
| sac/q2/queue_ready_gru | 8 | 0.003652400 | 0 |
| sac/target2/queue_cpu_gru | 8 | 0.003577000 | 0 |
| sac/q1/queue_cpu_gru | 8 | 0.003520600 | 0 |
| sac/q1/queue_ready_gru | 8 | 0.003425100 | 0 |
| sac/q2/queue_upload_gru | 8 | 0.003400200 | 0 |
| sac/target1/queue_ready_gru | 8 | 0.003313500 | 0 |
| sac/target2/queue_upload_gru | 8 | 0.003096400 | 0 |
| sac/target1/queue_cpu_gru | 8 | 0.003072900 | 0 |
| sac/target2/queue_ready_gru | 8 | 0.002950700 | 0 |
| sac/q2/queue_cpu_gru | 8 | 0.002757700 | 0 |
| sac/actor/queue_transfer_gru | 4 | 0.001671000 | 0 |
| sac/target1/queue_transfer_gru | 4 | 0.001482400 | 0 |
| sac/target2/queue_transfer_gru | 4 | 0.001448100 | 0 |
| sac/q2/queue_transfer_gru | 4 | 0.001362400 | 0 |
| sac/q2/arrival_gru | 2 | 0.001357300 | 0 |
| sac/target1/arrival_gru | 2 | 0.001342600 | 0 |
| sac/q1/queue_transfer_gru | 4 | 0.001310800 | 0 |
| sac/actor/arrival_gru | 2 | 0.001161800 | 0 |
| sac/target2/arrival_gru | 2 | 0.000752900 | 0 |
| Optimizer.zero_grad#Adam.zero_grad | 1 | 0.000224100 | 0 |

## Transfer-related CPU calls

| Name | Calls | Inclusive CPU seconds | Instant events |
| --- | ---: | ---: | ---: |
| aten::copy_ | 14815 | 0.196824700 | 0 |
| aten::to | 5585 | 0.082600800 | 0 |
| aten::_to_copy | 1686 | 0.076416200 | 0 |

## Interpretation and coverage

- All reported CPU durations are inclusive nested spans. Rows overlap and must not be added together.
- These tables are not a wall-clock decomposition and contain no wall-clock percentage attribution.
- Transfer-related CPU calls measure host dispatch/conversion/copy work, not actual CUDA transfer duration; aten::to may only change dtype or return its input.
- CUDA counts come only from actual kernel/gpu_memcpy/gpu_memset categories, never cudaLaunchKernel or other CPU runtime API calls.
- Chrome trace ts/dur fields are interpreted as microseconds; displayTimeUnit does not change their storage unit.
- Only completed X or matched B/E spans contribute duration. Instant annotation events are counted with zero duration; incomplete/unsupported annotation events are reported separately.
- This offline parser loads the trace into host memory. Run it after performance measurements, not alongside them.

Events read: 834311; invalid durations: 0.

Ignored/incomplete events: `{}`.
