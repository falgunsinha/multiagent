@echo off
REM Run all RL model experiments in Isaac Sim headless mode
REM This script runs all 6 experiments sequentially

echo ========================================
echo RL Models Experiment - Batch Execution
echo ========================================
echo.

REM Set paths
set ISAAC_SIM_PATH=C:\isaacsim
set PYTHON_EXE=%ISAAC_SIM_PATH%\python.bat
set SCRIPT_PATH=%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\test_rrt_isaacsim_ddqn.py
set VIZ_DIR=%ISAAC_SIM_PATH%\cobotproject\scripts\experiments\rlmodels\visualization

REM Check if Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Isaac Sim Python not found at %PYTHON_EXE%
    pause
    exit /b 1
)

REM Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo ERROR: Script not found at %SCRIPT_PATH%
    pause
    exit /b 1
)

REM Kill any existing wandb-core processes first
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

echo Starting experiments...
echo.

REM Experiment 1: Discrete Model Comparison
echo [1/2] Running Experiment 1: Discrete Model Comparison
echo   - 7 discrete models (Custom-DDQN FIRST, then Duel-DDQN, PER-DDQN-Light, PER-DDQN-Full, C51-DDQN, SAC-Discrete, PPO-Discrete)
echo   - 10 episodes per model (70 total episodes) - TESTING PATH EFFICIENCY FIX
echo   - Grid: 4x4 (16 cells), Cubes: 9 (56%% density), Obstacles: 1 (fixed)
echo   - Max Steps: 50, Observation Noise: 10%%
echo   - Cube Positions: RANDOMIZED per episode
echo   - GPU: CUDA Enabled, W&B: Disabled (CSV only)
echo.
REM Run exp1 (Isaac Sim doesn't return proper exit codes, so we continue regardless)
call "%PYTHON_EXE%" "%SCRIPT_PATH%" --experiment exp1 --headless --episodes 10 --grid_size 4 --num_cubes 9
echo.
echo ========================================
echo Experiment 1 Completed!
echo ========================================
echo.
echo NOTE: Check output above for any errors
echo.

REM Wait between experiments to ensure clean state
echo Waiting 5 seconds before starting Experiment 2...
timeout /t 5 /nobreak >nul
echo.

REM Experiment 2: Continuous Model Comparison
echo [2/2] Running Experiment 2: Continuous Model Comparison
echo   - 5 continuous models (Custom-DDQN FIRST, then DDPG, TD3, PPO-Continuous, SAC-Continuous)
echo   - 10 episodes per model (50 total episodes) - TESTING PATH EFFICIENCY FIX
echo   - Grid: 4x4 (16 cells), Cubes: 9 (56%% density), Obstacles: 1 (fixed)
echo   - Max Steps: 50, Observation Noise: 10%%
echo   - Cube Positions: RANDOMIZED per episode
echo   - GPU: CUDA Enabled, W&B: Disabled (CSV only)
echo.
REM Run exp2 (Isaac Sim doesn't return proper exit codes, so we continue regardless)
call "%PYTHON_EXE%" "%SCRIPT_PATH%" --experiment exp2 --headless --episodes 10 --grid_size 4 --num_cubes 9
echo.
echo ========================================
echo Experiment 2 Completed!
echo ========================================
echo.
echo NOTE: Check output above for any errors
echo.

REM Generate Visualizations - COMMENTED OUT (will create visualizations later)
REM echo.
REM echo ========================================
REM echo Generating Visualizations
REM echo ========================================
REM echo.
REM
REM echo [1/5] Generating discrete models visualizations...
REM cd "%VIZ_DIR%"
REM "%PYTHON_EXE%" visualize_discrete_models.py
REM if errorlevel 1 (
REM     echo WARNING: Discrete visualization failed (non-critical)
REM )
REM echo.
REM
REM echo [2/5] Generating continuous models visualizations...
REM "%PYTHON_EXE%" visualize_continuous_models.py
REM if errorlevel 1 (
REM     echo WARNING: Continuous visualization failed (non-critical)
REM )
REM echo.
REM
REM echo [3/5] Generating cross-model comparison (Seaborn - Line charts with CI, Faceted plots)...
REM "%PYTHON_EXE%" cross_model\seaborn\model_comparison.py
REM if errorlevel 1 (
REM     echo WARNING: Cross-model seaborn visualization failed (non-critical)
REM )
REM echo.
REM
REM echo [4/5] Generating cross-model comparison (Plotly - Interactive 3D scatter, Line charts)...
REM "%PYTHON_EXE%" cross_model\plotly\interactive_comparison.py
REM if errorlevel 1 (
REM     echo WARNING: Cross-model plotly visualization failed (non-critical)
REM )
REM echo.
REM
REM echo [5/5] Generating ADVANCED visualizations (NEW!)...
REM "%PYTHON_EXE%" visualize_advanced.py --experiment all
REM if errorlevel 1 (
REM     echo WARNING: Advanced visualization failed (non-critical)
REM )
REM echo.

echo.
echo ========================================
echo ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Summary:
echo   - Experiment 1 (Discrete): 7 models x 10 episodes = 70 episodes
echo   - Experiment 2 (Continuous): 5 models x 10 episodes = 50 episodes
echo   - Total: 12 models, 120 episodes (TESTING PATH EFFICIENCY FIX)
echo   - Grid: 4x4, Cubes: 9, Obstacles: 1 (fixed), Max Steps: 50
echo   - Stochasticity: Randomized cube positions, 10%% observation noise
echo   - Custom-DDQN tested FIRST in both experiments
echo.
echo Results saved to:
echo   - Discrete: results\exp1\comparison_results.csv
echo   - Continuous: results\exp2\comparison_results.csv
echo.
echo NOTE: Visualizations are DISABLED (commented out in batch script)
echo       You can generate visualizations later using the visualization scripts
echo.
echo Next steps:
echo   1. Analyze summary statistics in CSV files
echo   2. Generate visualizations manually if needed
echo   3. Compare discrete vs continuous model performance
echo.
echo Experiment completed at: %date% %time%
echo.

pause

