@echo off
echo Starting HRL training at cpu=7.5 (verification run)...
echo Results will be saved to results/sac_cpu75_verify/
echo.
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts\train_hrl.py --episodes 1500 --cpu_ghz 7.5 --traffic_scale 1.0 --results_subdir sac_cpu75_verify
echo.
echo Training complete!
pause
