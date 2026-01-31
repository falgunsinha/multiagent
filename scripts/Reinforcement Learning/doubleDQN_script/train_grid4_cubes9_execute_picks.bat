@echo off
REM Train DDQN RRT IsaacSim - Grid 4x4, 9 Cubes with Execute Picks
REM Configuration:
REM   - Grid: 4x4
REM   - Cubes: 9
REM   - Timesteps: 50,000
REM   - Execute Picks: TRUE (actual pick-and-place during training)
REM   - WandB: DISABLED (no logging to Weights & Biases)
REM   - Logs: Saved to logs\ directory with timestamp

echo ============================================================
echo DDQN TRAINING - RRT ISAAC SIM (GRID 4x4, 9 CUBES)
echo ============================================================
echo Configuration:
echo   Grid Size: 4x4
echo   Number of Cubes: 9
echo   Total Timesteps: 50,000
echo   Execute Picks: TRUE (actual pick-and-place)
echo   WandB Logging: DISABLED
echo   Logs Directory: logs\
echo ============================================================
echo.
echo This training will take approximately 30-60 minutes.
echo.
echo Logs will be saved with timestamp in:
echo   C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs\
echo.
echo Format: ddqn_rrt_isaacsim_grid4_cubes9_YYYYMMDD_HHMMSS_training.csv
echo         ddqn_rrt_isaacsim_grid4_cubes9_YYYYMMDD_HHMMSS_episodes.csv
echo.
pause
echo.
echo Start time: %date% %time%
echo.

cd /d "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"

echo ============================================================
echo Starting Training...
echo ============================================================
echo.

C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --execute_picks

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
echo Check the logs directory for training and episode logs.
echo ============================================================
pause

