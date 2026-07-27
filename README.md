# Joint HRL for Multi-UAV MEC

This directory is the clean experiment workspace for the two-timescale
hierarchical controller:

- the macro TD3 policy selects the joint displacement of three UAVs;
- the micro SAC policy selects forwarding ratios every resource frame;
- SAC is first warm-started with a fixed macro hover policy;
- SAC and TD3 then continue learning in the same joint-training run;
- realized micro-frame costs are aggregated into each macro transition.

The active experiment parameters are defined only in
`configs/main_experiment.json`.

The current protocol uses 20 macro periods per episode. A
`user_zone_radius_m` value of `0` disables the artificial circular zone
boundary; users remain inside the 1000 m by 1000 m map through boundary
reflection.

## Commands

Run the minimal smoke test:

```powershell
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\run_experiment.py `
  --stage smoke --method joint_td3_sac
```

Run the seed-1 pilot after smoke succeeds:

```powershell
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\run_experiment.py `
  --stage pilot --method joint_td3_sac --seed 1
```

Plot raw training returns and moving averages:

```powershell
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\plot_convergence.py `
  --stage smoke --seed 1
```

## Results

Every run is stored under:

```text
results/joint_hrl_v2/<stage>/seed<seed>/<method>/
```

The micro pretraining and joint training phases have separate CSV files and
checkpoints. Old experiment outputs are intentionally not mixed into this
workspace.
