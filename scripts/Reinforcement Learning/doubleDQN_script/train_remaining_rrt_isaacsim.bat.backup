@echo off
REM Resume train_all_ddqn_wandb.bat from experiment 8/9
REM Completes the remaining 2 RRT Isaac Sim experiments
REM These will appear in the same WandB project: ddqn-object-selection

echo ============================================================
echo RESUMING BATCH TRAINING - EXPERIMENTS 8-9 OF 9
echo ============================================================
echo Completed: 7/9 (A* x3, RRT Viz x3, RRT IsaacSim x1)
echo Remaining: 2/9 (RRT IsaacSim x2)
echo.
echo Remaining experiments:
echo   [8/9] RRT IsaacSim - 4x4 grid, 6 cubes
echo   [9/9] RRT IsaacSim - 4x4 grid, 9 cubes
echo.
echo WandB Project: ddqn-object-selection
echo New runs will appear alongside existing 7 runs
echo ============================================================
echo.
echo Start time: %date% %time%
echo.

cd /d "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"

REM ============================================================
REM RRT ISAAC SIM - 4x4 grid, 6 cubes
REM ============================================================
echo ============================================================
echo [8/9] RRT IsaacSim - 4x4 grid, 6 cubes (50K steps, ~2 hours)
echo ============================================================
echo.

C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 6 --timesteps 50000 --use_wandb

REM Note: Isaac Sim may not return proper exit codes, so we continue regardless
echo Training for 4x4/6 cubes completed (or stopped)
echo.

REM ============================================================
REM RRT ISAAC SIM - 4x4 grid, 9 cubes
REM ============================================================
echo ============================================================
echo [9/9] RRT IsaacSim - 4x4 grid, 9 cubes (50K steps, ~2 hours)
echo ============================================================
echo.

C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --use_wandb

REM Note: Isaac Sim may not return proper exit codes, so we continue regardless
echo Training for 4x4/9 cubes completed (or stopped)
echo.

echo ============================================================
echo ALL 9 TRAINING RUNS COMPLETE
echo ============================================================
echo End time: %date% %time%
echo.
echo Completed in this session: 2/9 (experiments 8-9)
echo Total completed: 9/9 (all experiments)
echo.
echo Models saved to: models\
echo Logs saved to: logs\
echo WandB dashboard: https://wandb.ai/falgunsinha/ddqn-object-selection
echo.
echo All 9 runs are now visible in the same WandB project!
echo ============================================================
pause

