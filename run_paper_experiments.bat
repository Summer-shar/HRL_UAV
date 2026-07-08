@echo off
echo ============================================================
echo  UAV-MEC HRL Paper Experiments - Batch Runner
echo  5 methods x 3 seeds = 15 experiments
echo  Scenario: zone125_55_35, cpu=8GHz, priority_demand
echo  300 episodes each, eval every 50 episodes
echo ============================================================
echo.
echo Estimated total time: 8-12 hours (depends on GPU)
echo Results will be saved in: results\paper_batch\
echo.
echo Press Ctrl+C to cancel, or
pause

set PYTHON=D:\Anaconda3\envs\pytorch-gpu\python.exe

echo.
echo Starting at %date% %time%
echo.

%PYTHON% scripts\run_paper_experiments.py --python "%PYTHON%"

echo.
echo ============================================================
echo  All experiments completed at %date% %time%
echo ============================================================
echo.
echo To generate convergence plots:
echo   %PYTHON% scripts\plot_paper_curves.py --bar
echo.
pause
