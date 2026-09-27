# Chrome trace CPU summary

Trace: D:\HRL_UAV_V2\outputs\equivalent_20260927\current\inference.lower.h1800.cached.trace.json

All durations below are **nested inclusive CPU spans**. They overlap and cannot be summed or interpreted as wall-clock shares.

## CUDA activity

Kernel events: 0; memcpy events: 0; memset events: 0.

No actual CUDA kernel/memcpy events were captured. CUPTI activity data is unavailable in this trace; do not attribute time to CUDA kernels or transfers.

## User annotations

| Name | Calls | Inclusive CPU seconds | Instant events |
| --- | ---: | ---: | ---: |
| inference.lower.h1800.cached | 1 | 0.039216100 | 0 |
| network/typed_attention | 3 | 0.017015700 | 0 |
| data/batch_graph_and_transfer | 1 | 0.006858100 | 0 |
| data/attention_indices | 1 | 0.002420800 | 0 |
| data/fixed_graph_features | 1 | 0.002051400 | 0 |
| history/inference_cache | 1 | 0.001337900 | 0 |
| sac/actor/queue_upload_gru | 1 | 0.000607700 | 0 |
| sac/actor/queue_cpu_gru | 1 | 0.000410600 | 0 |
| sac/actor/arrival_gru | 1 | 0.000350900 | 0 |

## Transfer-related CPU calls

| Name | Calls | Inclusive CPU seconds | Instant events |
| --- | ---: | ---: | ---: |
| aten::to | 101 | 0.002502900 | 0 |
| aten::_to_copy | 90 | 0.002395600 | 0 |
| aten::copy_ | 122 | 0.001274000 | 0 |

## Interpretation and coverage

- All reported CPU durations are inclusive nested spans. Rows overlap and must not be added together.
- These tables are not a wall-clock decomposition and contain no wall-clock percentage attribution.
- Transfer-related CPU calls measure host dispatch/conversion/copy work, not actual CUDA transfer duration; aten::to may only change dtype or return its input.
- CUDA counts come only from actual kernel/gpu_memcpy/gpu_memset categories, never cudaLaunchKernel or other CPU runtime API calls.
- Chrome trace ts/dur fields are interpreted as microseconds; displayTimeUnit does not change their storage unit.
- Only completed X or matched B/E spans contribute duration. Instant annotation events are counted with zero duration; incomplete/unsupported annotation events are reported separately.
- This offline parser loads the trace into host memory. Run it after performance measurements, not alongside them.

Events read: 7016; invalid durations: 0.

Ignored/incomplete events: `{}`.
