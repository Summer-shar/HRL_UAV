# HRL_UAV

UAV-MEC hierarchical reinforcement-learning simulation code.

## Directory Layout

```text
HRL_UAV/
  hrl_uav/              Core simulator, wrappers, plotting helpers
    agents/             SAC, PPO, and micro-PPO agents
  scripts/              Training, baseline comparison, and CSV export entrypoints
  matlab/               MATLAB plotting scripts
  results/              Training outputs, checkpoints, CSVs, figures
  docs/                 Run notes
```

## Quick Start

The easiest way is to open [run.py](run.py), change the config block at the
top, and click Run in VSCode.

Set `MODE = "run_all_one_cpu"` to run SAC, PPO, NoCoop, Random, and SCA for
one CPU frequency in a single click.

For a command-line smoke test:

```powershell
cd C:\Users\PC\Desktop\HRL_UAV
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\compare_random.py --episodes 1 --macro_steps 1 --micro_steps 2 --eval_episodes 1 --eval_interval 0 --results_subdir _smoke\random --cpu_ghz 8
```

More complete commands are in [docs/RUNNING.md](docs/RUNNING.md).
