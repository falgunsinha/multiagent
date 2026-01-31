@echo off
REM Resume DDQN RRT IsaacSim Training - Grid 4x4, 9 Cubes
REM This script resumes training from the latest checkpoint (step 45,000)
REM and continues to 50,000 steps

echo ============================================================
echo RESUME DDQN TRAINING - RRT ISAAC SIM (GRID 4x4, 9 CUBES)
echo ============================================================
echo Configuration:
echo   Grid Size: 4x4
echo   Number of Cubes: 9
echo   Resume from: Step 45,000
echo   Target Timesteps: 50,000
echo   Remaining Steps: 5,000
echo   Execute Picks: TRUE (actual pick-and-place)
echo   WandB Logging: DISABLED
echo ============================================================
echo.
echo IMPORTANT: Make sure you have restarted your computer to clear GPU memory!
echo.
echo This will resume training from the checkpoint:
echo   ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_step_45000.pt
echo.
echo Estimated time: 10-20 minutes for remaining 5,000 steps
echo.
pause
echo.
echo Start time: %date% %time%
echo.

cd /d "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"

echo ============================================================
echo Resuming Training...
echo ============================================================
echo.

C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --execute_picks --resume "models\ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_step_45000.pt"

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Training failed or was interrupted!
    echo ============================================================
    echo Exit code: %errorlevel%
    echo End time: %date% %time%
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo TRAINING COMPLETE
echo ============================================================
echo End time: %date% %time%
echo.
echo Models saved to: models\
echo Logs saved to: logs\
echo.
echo Check the logs directory for updated training and episode logs.
echo ============================================================
pause

