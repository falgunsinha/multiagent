@echo off
REM Clean logs and resume DDQN training from step 45,000

echo ============================================================
echo CLEAN LOGS AND RESUME DDQN TRAINING
echo ============================================================
echo.
echo This script will:
echo   1. Backup your current log files
echo   2. Remove log entries after step 45,000 / episode 5,219
echo   3. Resume training from step 45,000 to 50,000
echo.
echo Note: Episode 5220 was in progress at step 45,000 and will
echo       be replayed from the beginning to avoid duplicates.
echo.
echo This ensures clean, continuous logs without duplicates.
echo.
echo ============================================================
echo STEP 1: CLEAN LOGS
echo ============================================================
echo.

cd /d "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"

C:\isaacsim\python.bat clean_logs_for_resume.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to clean logs!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 2: RESUME TRAINING
echo ============================================================
echo.
echo IMPORTANT: Make sure you have restarted your computer to clear GPU memory!
echo.
pause
echo.
echo Starting resume training from step 45,000...
echo Target: 50,000 steps (5,000 remaining)
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
echo TRAINING COMPLETE!
echo ============================================================
echo End time: %date% %time%
echo.
echo Your logs are now clean and continuous from step 1 to 50,000!
echo ============================================================
pause

