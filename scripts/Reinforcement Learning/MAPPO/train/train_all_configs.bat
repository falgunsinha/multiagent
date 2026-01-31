@echo off
REM Batch training script for all MAPPO configurations
REM Trains on Grid 3x3 (4 cubes), Grid 4x4 (6 cubes), and Grid 4x4 (9 cubes)
REM Uses A*, RRT, and Isaac Sim environments (9 total configurations)
REM
REM Usage:
REM   train_all_configs.bat [timesteps]
REM
REM Examples:
REM   train_all_configs.bat           (uses default timesteps per config)
REM   train_all_configs.bat 50000     (trains all configs for 50000 timesteps)

setlocal enabledelayedexpansion

REM Parse timesteps argument
set CUSTOM_TIMESTEPS=%1

echo ========================================
echo MAPPO Two-Agent Training - All Configs
echo ========================================
echo.
echo Training 9 configurations:
echo   - 3 grid sizes: 3x3 (4 cubes), 4x4 (6 cubes), 4x4 (9 cubes)
echo   - 3 environments: A*, RRT, Isaac Sim
echo   - Total: 9 training runs
echo.

if "%CUSTOM_TIMESTEPS%"=="" (
    echo Using default timesteps per configuration
) else (
    echo Using custom timesteps: %CUSTOM_TIMESTEPS% for all configs
)
echo.

REM Set paths
set PYTHON=C:\isaacsim\python.bat
set ISAACSIM_PYTHON=C:\isaacsim\python.bat
set SCRIPT_DIR=%~dp0
set DDQN_MODELS_DIR=C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models

echo Starting training at %date% %time%
echo.

REM ========================================
REM Configuration 1: Grid 3x3, 4 cubes
REM ========================================
if "%CUSTOM_TIMESTEPS%"=="" (set TIMESTEPS=10000) else (set TIMESTEPS=%CUSTOM_TIMESTEPS%)
set CONFIG_NAME=grid3x3_4cubes
set DDQN_ASTAR=%DDQN_MODELS_DIR%\ddqn_astar_grid3_cubes4_20251220_012015_final.pt
set DDQN_RRT=%DDQN_MODELS_DIR%\ddqn_rrt_viz_grid3_cubes4_20251220_025425_final.pt
set DDQN_ISAACSIM=%DDQN_MODELS_DIR%\ddqn_rrt_isaacsim_grid3_cubes4_20251223_203144_final.pt

echo ========================================
echo [1/9] Training %CONFIG_NAME% (A*) - %TIMESTEPS% timesteps
echo ========================================
%PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 3 --num_cubes 4 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ASTAR%" --run_name "%CONFIG_NAME%_astar" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: A* training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================
echo [2/9] Training %CONFIG_NAME% (RRT) - %TIMESTEPS% timesteps
echo ========================================
%PYTHON% "%SCRIPT_DIR%train_rrt_mappo.py" --grid_size 3 --num_cubes 4 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_RRT%" --run_name "%CONFIG_NAME%_rrt" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: RRT training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================
echo [3/9] Training %CONFIG_NAME% (RRT Isaac Sim) - %TIMESTEPS% timesteps
echo ========================================
%ISAACSIM_PYTHON% "%SCRIPT_DIR%train_isaacsim_mappo.py" --grid_size 3 --num_cubes 4 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ISAACSIM%" --run_name "%CONFIG_NAME%_rrt_isaacsim" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Isaac Sim training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

REM ========================================
REM Configuration 2: Grid 4x4, 6 cubes
REM ========================================
if "%CUSTOM_TIMESTEPS%"=="" (set TIMESTEPS=15000) else (set TIMESTEPS=%CUSTOM_TIMESTEPS%)
set CONFIG_NAME=grid4x4_6cubes
set DDQN_ASTAR=%DDQN_MODELS_DIR%\ddqn_astar_grid4_cubes6_20251220_014823_final.pt
set DDQN_RRT=%DDQN_MODELS_DIR%\ddqn_rrt_viz_grid4_cubes6_20251220_054851_final.pt
set DDQN_ISAACSIM=%DDQN_MODELS_DIR%\ddqn_rrt_isaacsim_grid4_cubes6_20251224_122040_final.pt

echo ========================================
echo [4/9] Training %CONFIG_NAME% (A*) - %TIMESTEPS% timesteps
echo ========================================
%PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 4 --num_cubes 6 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ASTAR%" --run_name "%CONFIG_NAME%_astar" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: A* training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================
echo [5/9] Training %CONFIG_NAME% (RRT) - %TIMESTEPS% timesteps
echo ========================================
%PYTHON% "%SCRIPT_DIR%train_rrt_mappo.py" --grid_size 4 --num_cubes 6 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_RRT%" --run_name "%CONFIG_NAME%_rrt" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: RRT training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================
echo [6/9] Training %CONFIG_NAME% (RRT Isaac Sim) - %TIMESTEPS% timesteps
echo ========================================
%ISAACSIM_PYTHON% "%SCRIPT_DIR%train_isaacsim_mappo.py" --grid_size 4 --num_cubes 6 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ISAACSIM%" --run_name "%CONFIG_NAME%_rrt_isaacsim" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Isaac Sim training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

REM ========================================
REM Configuration 3: Grid 4x4, 9 cubes
REM ========================================
if "%CUSTOM_TIMESTEPS%"=="" (set TIMESTEPS=20000) else (set TIMESTEPS=%CUSTOM_TIMESTEPS%)
set CONFIG_NAME=grid4x4_9cubes
set DDQN_ASTAR=%DDQN_MODELS_DIR%\ddqn_astar_grid4_cubes9_20251220_022000_final.pt
set DDQN_RRT=%DDQN_MODELS_DIR%\ddqn_rrt_viz_grid4_cubes9_20251220_134808_final.pt
set DDQN_ISAACSIM=%DDQN_MODELS_DIR%\ddqn_rrt_isaacsim_grid4_cubes9_20251224_185752_final.pt

echo ========================================
echo [7/9] Training %CONFIG_NAME% (A*) - %TIMESTEPS% timesteps
echo ========================================
%PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 4 --num_cubes 9 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ASTAR%" --run_name "%CONFIG_NAME%_astar" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: A* training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================
echo [8/9] Training %CONFIG_NAME% (RRT) - %TIMESTEPS% timesteps
echo ========================================
%PYTHON% "%SCRIPT_DIR%train_rrt_mappo.py" --grid_size 4 --num_cubes 9 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_RRT%" --run_name "%CONFIG_NAME%_rrt" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: RRT training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================
echo [9/9] Training %CONFIG_NAME% (RRT Isaac Sim) - %TIMESTEPS% timesteps
echo ========================================
%ISAACSIM_PYTHON% "%SCRIPT_DIR%train_isaacsim_mappo.py" --grid_size 4 --num_cubes 9 --timesteps %TIMESTEPS% --ddqn_model_path "%DDQN_ISAACSIM%" --run_name "%CONFIG_NAME%_rrt_isaacsim" --config_name %CONFIG_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Isaac Sim training failed for %CONFIG_NAME%
    pause
    exit /b %errorlevel%
)
echo.

REM ========================================
REM All training complete
REM ========================================
echo ========================================
echo All 9 configurations trained successfully!
echo ========================================
echo Finished at %date% %time%
echo.
echo Trained configurations:
echo   - Grid 3x3, 4 cubes (A* + RRT Viz + RRT Isaac Sim)
echo   - Grid 4x4, 6 cubes (A* + RRT Viz + RRT Isaac Sim)
echo   - Grid 4x4, 9 cubes (A* + RRT Viz + RRT Isaac Sim)
echo.
echo Models saved to: cobotproject\scripts\Reinforcement Learning\MAPPO\models\
echo Logs saved to: cobotproject\scripts\Reinforcement Learning\MAPPO\logs\
echo.
echo View results on WandB:
echo   Project: ddqn-mappo-object-selection-reshuffling
echo   Groups: grid3x3_4cubes, grid4x4_6cubes, grid4x4_9cubes
echo.
pause

