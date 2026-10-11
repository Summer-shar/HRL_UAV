# Chrome trace CPU summary

Trace: D:\HRL_UAV_V2\outputs\equivalent_20260927\baseline\critic.e2.h1800.backlog.trace.json

All durations below are **nested inclusive CPU spans**. They overlap and cannot be summed or interpreted as wall-clock shares.

## CUDA activity

Kernel events: 0; memcpy events: 0; memset events: 0.

No actual CUDA kernel/memcpy events were captured. CUPTI activity data is unavailable in this trace; do not attribute time to CUDA kernels or transfers.

## User annotations

| Name | Calls | Inclusive CPU seconds | Instant events |
| --- | ---: | ---: | ---: |
| critic.e2.h1800.backlog | 1 | 8.881389800 | 0 |
| sac/critic_fit | 1 | 8.881292600 | 0 |
| data/batch_graph_and_transfer | 40 | 4.360825200 | 0 |
| data/fixed_graph_features | 201 | 3.780929900 | 0 |
| sac/q1/batch_forward | 8 | 2.591575800 | 0 |
| autograd/backward | 10 | 2.540981800 | 0 |
| sac/actor/batch_forward | 8 | 2.383606900 | 0 |
| network/typed_attention | 120 | 0.966574500 | 0 |
| sac/q2/batch_forward | 8 | 0.327006900 | 0 |
| data/replay_unpack | 224 | 0.300388900 | 0 |
| sac/target1/batch_forward | 8 | 0.289779200 | 0 |
| sac/target2/batch_forward | 8 | 0.275251500 | 0 |
| data/attention_indices | 16 | 0.226336300 | 0 |
| sac/hybrid_sampling | 8 | 0.029155500 | 0 |
| history/prefix_forward | 5 | 0.023206900 | 0 |
| optimizer/adam_step | 1 | 0.020378900 | 0 |
| Optimizer.step#Adam.step | 1 | 0.020326500 | 0 |
| data/arrival_normalize_and_transfer | 10 | 0.015384600 | 0 |
| sac/action_cast | 16 | 0.006964000 | 0 |
| history/full_prefix_backward | 2 | 0.005003000 | 0 |
| sac/q1/q_value_head | 8 | 0.004856800 | 0 |
| sac/q2/q_value_head | 8 | 0.004575400 | 0 |
| sac/target1/q_value_head | 8 | 0.004308200 | 0 |
| sac/actor/queue_upload_gru | 8 | 0.004117900 | 0 |
| sac/actor/queue_ready_gru | 8 | 0.003962300 | 0 |
| sac/q1/queue_upload_gru | 8 | 0.003911600 | 0 |
| sac/target2/q_value_head | 8 | 0.003872500 | 0 |
| sac/target1/queue_upload_gru | 8 | 0.003733000 | 0 |
| sac/q1/queue_ready_gru | 8 | 0.003671000 | 0 |
| sac/q2/queue_upload_gru | 8 | 0.003512100 | 0 |
| sac/target2/queue_upload_gru | 8 | 0.003434100 | 0 |
| sac/q2/queue_ready_gru | 8 | 0.003399300 | 0 |
| sac/target2/queue_ready_gru | 8 | 0.003278900 | 0 |
| sac/q1/arrival_gru | 2 | 0.003155300 | 0 |
| sac/target2/queue_cpu_gru | 8 | 0.003078100 | 0 |
| sac/q1/queue_cpu_gru | 8 | 0.002918300 | 0 |
| sac/target1/queue_ready_gru | 8 | 0.002883800 | 0 |
| sac/q2/queue_cpu_gru | 8 | 0.002862800 | 0 |
| sac/target1/queue_cpu_gru | 8 | 0.002854100 | 0 |
| sac/actor/queue_cpu_gru | 8 | 0.002785100 | 0 |
| sac/q2/queue_transfer_gru | 4 | 0.001851400 | 0 |
| sac/q1/queue_transfer_gru | 4 | 0.001812600 | 0 |
| sac/actor/queue_transfer_gru | 4 | 0.001381100 | 0 |
| sac/target2/queue_transfer_gru | 4 | 0.001325700 | 0 |
| sac/target1/queue_transfer_gru | 4 | 0.001171300 | 0 |
| sac/q2/arrival_gru | 2 | 0.000897800 | 0 |
| sac/actor/arrival_gru | 2 | 0.000834700 | 0 |
| sac/target1/arrival_gru | 2 | 0.000764800 | 0 |
| sac/target2/arrival_gru | 2 | 0.000733600 | 0 |
| Optimizer.zero_grad#Adam.zero_grad | 1 | 0.000113700 | 0 |

## Transfer-related CPU calls

| Name | Calls | Inclusive CPU seconds | Instant events |
| --- | ---: | ---: | ---: |
| aten::copy_ | 14815 | 0.195181700 | 0 |
| aten::to | 5585 | 0.084144900 | 0 |
| aten::_to_copy | 1686 | 0.078242100 | 0 |

## Interpretation and coverage

- All reported CPU durations are inclusive nested spans. Rows overlap and must not be added together.
- These tables are not a wall-clock decomposition and contain no wall-clock percentage attribution.
- Transfer-related CPU calls measure host dispatch/conversion/copy work, not actual CUDA transfer duration; aten::to may only change dtype or return its input.
- CUDA counts come only from actual kernel/gpu_memcpy/gpu_memset categories, never cudaLaunchKernel or other CPU runtime API calls.
- Chrome trace ts/dur fields are interpreted as microseconds; displayTimeUnit does not change their storage unit.
- Only completed X or matched B/E spans contribute duration. Instant annotation events are counted with zero duration; incomplete/unsupported annotation events are reported separately.
- This offline parser loads the trace into host memory. Run it after performance measurements, not alongside them.

Events read: 834292; invalid durations: 0.

Ignored/incomplete events: `{}`.
