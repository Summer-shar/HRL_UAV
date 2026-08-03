# Latest Result Figures

This folder contains the latest **v23 seed=1 pilot and diagnostic figures** exported on 2026-08-03.
It is intentionally separate from `results/`: checkpoints, old experiments, and large intermediate files are not committed.

## Current status

These are diagnostic results, not final paper results.

- Configuration: 3 UAVs, 39 users, 10 GHz CPU, load scale 1.15, 840 W system power budget.
- The current TD3 macro policy becomes worse during the 400-episode pilot: weighted delay rises from about 739.1 ms to 765.5 ms while mean U2G distance rises from 0.7 m to 55.9 m.
- A 20-seed paired position audit shows that dynamic tracking is valuable: fixed anchors require 761.6 ms, while dynamic-centroid tracking requires 721.0 ms.
- Traffic-centroid tracking (721.0 ms), the U2G-optimal reference (721.7 ms), and joint position search (722.0 ms) are statistically equivalent within the predefined 5 ms practical margin.
- Therefore, dynamic tracking has been established, but a material collaborative residual-position gain and a successful learned TD3 trajectory have not yet been established.

## Figures

1. `01_macro_training_diagnostic`: weighted delay, U2G distance, and power feasibility across the v23 pilot.
2. `02_position_reference_comparison`: 20-seed comparison of fixed anchors, dynamic centroid tracking, U2G reference tracking, and joint position search.
3. `03_trajectory_best`: trajectory saved at the selected best checkpoint.
4. `04_trajectory_ep0100` to `07_trajectory_ep0400`: trajectory evolution during training.
5. `08_trajectory_final`: final v23 trajectory.

The two newly generated summary plots are provided as both PNG and vector PDF files.

## Reproducibility data

The `data/` directory includes:

- `eval_central_td3.csv`: checkpoint evaluation records used by Figure 1.
- `position_reference_audit.json`: complete 20-seed audit data used by Figure 2.
- `tcom_dynamic_u2u_sparse_v23_target_noise.json`: source experiment configuration.

Regenerate the folder with:

```powershell
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\export_latest_results.py
```
