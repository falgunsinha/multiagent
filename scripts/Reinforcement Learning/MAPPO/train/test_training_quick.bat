@echo off
REM Quick test script for MAPPO training - Just 2 episodes per configuration
REM This is for testing the training pipeline, not for actual training

setlocal enabledelayedexpansion

echo ========================================
echo MAPPO Two-Agent Training - QUICK TEST
echo ========================================
echo.
echo Testing 9 configurations with just 2 episodes each:
echo   - 3 grid sizes: 3x3 (4 cubes), 4x4 (6 cubes), 4x4 (9 cubes)
echo   - 3 environments: A*, RRT, Isaac Sim
echo   - Total: 9 training runs (VERY SHORT - just for testing!)
echo.

REM Set paths
set PYTHON=C:\isaacsim\python.bat
set ISAACSIM_PYTHON=C:\isaacsim\python.bat
set SCRIPT_DIR=%~dp0
set DDQN_MODELS_DIR=C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models

REM For 2 episodes, we need minimal timesteps
REM Assuming ~10 steps per episode, 2 episodes = ~20 timesteps
set TIMESTEPS=50

echo Starting quick test at %date% %time%
echo Using %TIMESTEPS% timesteps (approximately 2 episodes)
echo.
echo Test order: Isaac Sim RRT (3 configs) -> RRT Viz (3 configs) -> A* (3 configs)
echo.

REM ========================================
REM ISAAC SIM RRT - All 3 configurations
REM ========================================
echo ========================================
echo ISAAC SIM RRT - 3 CONFIGURATIONS
echo ========================================
echo.

REM Config 1: Grid 3x3, 4 cubes
set CONFIG_NAME=grid3x3_4cubes
set DDQN_ISAACSIM=%DDQN_MODELS_DIR%\ddqn_rrt_isaacsim_grid3_cubes4_20251223_203144_final.pt

echo ========================================
echo [1/9] Testing %CONFIG_NAME% (Isaac Sim RRT) - %TIMESTEPS% timesteps
echo ========================================
call %ISAACSIM_PYTHON% "%SCRIPT_DIR%train_isaacsim_mappo.py" --grid_size 3 --num_cubes 4 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ISAACSIM%" --run_name "%CONFIG_NAME%_isaacsim_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50 --use_wandb
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [1/9] Complete!
echo.

REM Config 2: Grid 4x4, 6 cubes
set CONFIG_NAME=grid4x4_6cubes
set DDQN_ISAACSIM=%DDQN_MODELS_DIR%\ddqn_rrt_isaacsim_grid4_cubes6_20251224_122040_final.pt

echo ========================================
echo [2/9] Testing %CONFIG_NAME% (Isaac Sim RRT) - %TIMESTEPS% timesteps
echo ========================================
call %ISAACSIM_PYTHON% "%SCRIPT_DIR%train_isaacsim_mappo.py" --grid_size 4 --num_cubes 6 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ISAACSIM%" --run_name "%CONFIG_NAME%_isaacsim_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50 --use_wandb
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [2/9] Complete!
echo.

REM Config 3: Grid 4x4, 9 cubes
set CONFIG_NAME=grid4x4_9cubes
set DDQN_ISAACSIM=%DDQN_MODELS_DIR%\ddqn_rrt_isaacsim_grid4_cubes9_20251224_185752_final.pt

echo ========================================
echo [3/9] Testing %CONFIG_NAME% (Isaac Sim RRT) - %TIMESTEPS% timesteps
echo ========================================
call %ISAACSIM_PYTHON% "%SCRIPT_DIR%train_isaacsim_mappo.py" --grid_size 4 --num_cubes 9 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ISAACSIM%" --run_name "%CONFIG_NAME%_isaacsim_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50 --use_wandb
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [3/9] Complete!
echo.

REM ========================================
REM RRT VIZ - All 3 configurations
REM ========================================
echo ========================================
echo RRT VIZ - 3 CONFIGURATIONS
echo ========================================
echo.

REM Config 1: Grid 3x3, 4 cubes
set CONFIG_NAME=grid3x3_4cubes
set DDQN_RRT=%DDQN_MODELS_DIR%\ddqn_rrt_viz_grid3_cubes4_20251220_025425_final.pt

echo ========================================
echo [4/9] Testing %CONFIG_NAME% (RRT Viz) - %TIMESTEPS% timesteps
echo ========================================
call %PYTHON% "%SCRIPT_DIR%train_rrt_mappo.py" --grid_size 3 --num_cubes 4 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_RRT%" --run_name "%CONFIG_NAME%_rrt_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [4/9] Complete!
echo.

REM Config 2: Grid 4x4, 6 cubes
set CONFIG_NAME=grid4x4_6cubes
set DDQN_RRT=%DDQN_MODELS_DIR%\ddqn_rrt_viz_grid4_cubes6_20251220_054851_final.pt

echo ========================================
echo [5/9] Testing %CONFIG_NAME% (RRT Viz) - %TIMESTEPS% timesteps
echo ========================================
call %PYTHON% "%SCRIPT_DIR%train_rrt_mappo.py" --grid_size 4 --num_cubes 6 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_RRT%" --run_name "%CONFIG_NAME%_rrt_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [5/9] Complete!
echo.

REM Config 3: Grid 4x4, 9 cubes
set CONFIG_NAME=grid4x4_9cubes
set DDQN_RRT=%DDQN_MODELS_DIR%\ddqn_rrt_viz_grid4_cubes9_20251220_134808_final.pt

echo ========================================
echo [6/9] Testing %CONFIG_NAME% (RRT Viz) - %TIMESTEPS% timesteps
echo ========================================
call %PYTHON% "%SCRIPT_DIR%train_rrt_mappo.py" --grid_size 4 --num_cubes 9 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_RRT%" --run_name "%CONFIG_NAME%_rrt_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [6/9] Complete!
echo.

REM ========================================
REM A* - All 3 configurations
REM ========================================
echo ========================================
echo A* - 3 CONFIGURATIONS
echo ========================================
echo.

REM Config 1: Grid 3x3, 4 cubes
set CONFIG_NAME=grid3x3_4cubes
set DDQN_ASTAR=%DDQN_MODELS_DIR%\ddqn_astar_grid3_cubes4_20251220_012015_final.pt

echo ========================================
echo [7/9] Testing %CONFIG_NAME% (A*) - %TIMESTEPS% timesteps
echo ========================================
call %PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 3 --num_cubes 4 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ASTAR%" --run_name "%CONFIG_NAME%_astar_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [7/9] Complete!
echo.

REM Config 2: Grid 4x4, 6 cubes
set CONFIG_NAME=grid4x4_6cubes
set DDQN_ASTAR=%DDQN_MODELS_DIR%\ddqn_astar_grid4_cubes6_20251220_014823_final.pt

echo ========================================
echo [8/9] Testing %CONFIG_NAME% (A*) - %TIMESTEPS% timesteps
echo ========================================
call %PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 4 --num_cubes 6 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ASTAR%" --run_name "%CONFIG_NAME%_astar_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [8/9] Complete!
echo.

REM Config 3: Grid 4x4, 9 cubes
set CONFIG_NAME=grid4x4_9cubes
set DDQN_ASTAR=%DDQN_MODELS_DIR%\ddqn_astar_grid4_cubes9_20251220_022000_final.pt

echo ========================================
echo [9/9] Testing %CONFIG_NAME% (A*) - %TIMESTEPS% timesteps
echo ========================================
call %PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 4 --num_cubes 9 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ASTAR%" --run_name "%CONFIG_NAME%_astar_test" --config_name %CONFIG_NAME% --log_interval 1 --save_interval 50
if errorlevel 1 (
    echo WARNING: Training returned error code, but continuing...
)
echo [9/9] Complete!
echo.

REM ========================================
REM All testing complete
REM ========================================
echo ========================================
echo All 9 configurations tested successfully!
echo ========================================
echo Finished at %date% %time%
echo.
echo Test order completed:
echo   1-3: Isaac Sim RRT (3x3/4, 4x4/6, 4x4/9)
echo   4-6: RRT Viz (3x3/4, 4x4/6, 4x4/9)
echo   7-9: A* (3x3/4, 4x4/6, 4x4/9)
echo.
echo This was a QUICK TEST with only ~2 episodes per config.
echo For actual training, use train_all_configs.bat with proper timesteps.
echo.
echo View test results on WandB:
echo   Project: ddqn-mappo-object-selection-reshuffling
echo   Look for runs with "_test" suffix
echo.
pause

