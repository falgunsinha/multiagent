@echo off
REM Quick test to verify WandB is working before running full training
REM Runs 3 short tests (100 steps each, ~3 minutes total)

echo ============================================================
echo WANDB VERIFICATION TEST
echo ============================================================
echo This will run 3 quick tests (100 steps each) to verify WandB
echo Total time: ~3 minutes
echo ============================================================
echo.
echo NOTE: WandB installation will be verified during the actual tests
echo If WandB is not installed or configured, the tests will fail with clear error messages
echo.
pause
echo.

REM Test A*
echo [1/3] Testing A* with WandB (100 steps, ~1 min)...
py -3.11 train_astar_ddqn.py --grid_size 3 --num_cubes 4 --timesteps 100 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: A* WandB test failed!
    pause
    exit /b 1
)
echo [OK] A* WandB test passed
echo.

REM Test RRT Viz
echo [2/3] Testing RRT Viz with WandB (100 steps, ~1 min)...
py -3.11 train_rrt_viz_ddqn.py --grid_size 3 --num_cubes 4 --timesteps 100 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT Viz WandB test failed!
    pause
    exit /b 1
)
echo [OK] RRT Viz WandB test passed
echo.

REM Test RRT Isaac Sim
echo [3/3] Testing RRT Isaac Sim with WandB (100 steps, ~1 min)...
C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 3 --num_cubes 4 --timesteps 100 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT Isaac Sim WandB test failed!
    pause
    exit /b 1
)
echo [OK] RRT Isaac Sim WandB test passed
echo.

echo ============================================================
echo ALL WANDB TESTS PASSED!
echo ============================================================
echo You can now run the full training with:
echo   train_all_ddqn_wandb.bat
echo.
echo Check your WandB dashboard at:
echo   https://wandb.ai/
echo   Project: ddqn-object-selection
echo ============================================================
pause

