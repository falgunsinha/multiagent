@echo off
REM Quick test with 2 episodes per model to verify everything works

echo ========================================
echo Quick Test - All Models (2 Episodes Each)
echo ========================================
echo.

set ISAAC_SIM_PATH=C:\isaacsim
set PYTHON_EXE=%ISAAC_SIM_PATH%\python.bat
set SCRIPT_PATH=%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\test_rrt_isaacsim_ddqn.py

echo Cleaning up old WandB processes...
taskkill /F /IM wandb-core.exe 2>nul
timeout /t 2 /nobreak >nul

REM Delete old wandb folder
echo Deleting old WandB folder...
if exist "%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\wandb" (
    rmdir /s /q "%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\wandb" 2>nul
    if exist "%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\wandb" (
        echo WARNING: Could not delete wandb folder, renaming instead...
        set timestamp=%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
        set timestamp=%timestamp::=%
        ren "%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\wandb" wandb_old_%timestamp%
    )
)
timeout /t 2 /nobreak >nul

echo.
echo [1/2] Running quick test - Discrete Models (exp1)
echo   - 7 discrete models
echo   - Episodes: 2 (just for testing)
echo   - Grid: 3x3
echo   - Cubes: 4
echo   - W&B: Enabled
echo.

REM Run exp1 (Isaac Sim doesn't return proper exit codes, so we continue regardless)
call "%PYTHON_EXE%" "%SCRIPT_PATH%" --experiment exp1 --headless --use_wandb --episodes 2 --grid_size 3 --num_cubes 4

echo.
echo ========================================
echo Discrete Test Completed!
echo ========================================
echo.
echo NOTE: Check output above for any errors
echo.

REM Wait between experiments
echo Waiting 5 seconds before testing continuous models...
timeout /t 5 /nobreak >nul
echo.

echo [2/2] Running quick test - Continuous Models (exp2)
echo   - 4 continuous models + 1 custom
echo   - Episodes: 2 (just for testing)
echo   - Grid: 3x3
echo   - Cubes: 4
echo   - W&B: Enabled
echo.

REM Run exp2 (Isaac Sim doesn't return proper exit codes, so we continue regardless)
call "%PYTHON_EXE%" "%SCRIPT_PATH%" --experiment exp2 --headless --use_wandb --episodes 2 --grid_size 3 --num_cubes 4

echo.
echo ========================================
echo Quick Test Completed!
echo ========================================
echo.
echo Both discrete and continuous models tested!
echo.
echo Visualizations generated:
echo   - Line charts with 95%% CI bands
echo   - Faceted relplot charts
echo   - Interactive 3D scatter plots
echo   - And many more...
echo.
echo NOTE: Check output above for any errors
echo.
echo If this worked, you can run the full experiments with:
echo   run_all_experiments_headless.bat
echo.

pause

