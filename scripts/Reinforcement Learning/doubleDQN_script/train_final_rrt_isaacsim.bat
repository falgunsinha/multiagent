@echo off
REM Train the final RRT Isaac Sim experiment (9/9)
REM This completes all 9 DDQN experiments

echo ============================================================
echo FINAL EXPERIMENT - 9/9
echo ============================================================
echo Completed: 8/9
echo Remaining: 1/9 (RRT IsaacSim - 4x4 grid, 9 cubes)
echo.
echo WandB Project: ddqn-object-selection
echo This run will appear alongside your existing 8 runs
echo ============================================================
echo.
echo Start time: %date% %time%
echo.

cd /d "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"

echo ============================================================
echo [9/9] RRT IsaacSim - 4x4 grid, 9 cubes (50K steps, ~2 hours)
echo ============================================================
echo.

C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --use_wandb

echo.
echo ============================================================
echo ALL 9 TRAINING RUNS COMPLETE!
echo ============================================================
echo End time: %date% %time%
echo.
echo Total completed: 9/9 (all experiments)
echo.
echo Models saved to: models\
echo Logs saved to: logs\
echo WandB dashboard: https://wandb.ai/falgunsinha/ddqn-object-selection
echo.
echo All 9 runs are now visible in the same WandB project!
echo ============================================================
pause

