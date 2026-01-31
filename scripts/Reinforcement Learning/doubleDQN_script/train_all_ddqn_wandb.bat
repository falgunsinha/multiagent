@echo off
REM Train all 9 Double DQN configurations WITH WANDB LOGGING
REM 3 methods x 3 configurations = 9 total experiments
REM Estimated total time: 3-5 hours (with WandB)
REM STRICT MODE: Training stops if WandB fails

echo ============================================================
echo TRAINING ALL 9 DOUBLE DQN CONFIGURATIONS (WITH WANDB)
echo ============================================================
echo Methods: A*, RRT Viz, RRT IsaacSim
echo Configurations: 3x3/4cubes, 4x4/6cubes, 4x4/9cubes
echo Total experiments: 9
echo Timesteps per experiment: 50,000
echo WandB Logging: ENABLED (STRICT MODE - stops on WandB failure)
echo Estimated time: 8-10 hours
echo ============================================================
echo.
echo NOTE: WandB must be installed and configured in both environments:
echo   - Python 3.11: py -3.11 -m pip install wandb
echo   - Isaac Sim: C:\isaacsim\python.bat -m pip install wandb
echo   - Login: py -3.11 -m wandb login
echo   - Login: C:\isaacsim\python.bat -m wandb login
echo.
echo If WandB is not configured, training will stop with an error.
echo.
pause
echo.
echo Start time: %date% %time%
echo.

REM ============================================================
REM A* METHOD (3 experiments)
REM ============================================================
echo ============================================================
echo A* METHOD - 3 EXPERIMENTS
echo ============================================================
echo.

echo [1/9] A* - 3x3 grid, 4 cubes (50K steps, ~45 min)
py -3.11 train_astar_ddqn.py --grid_size 3 --num_cubes 4 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: A* 3x3/4 training failed!
    pause
    exit /b 1
)
echo.

echo [2/9] A* - 4x4 grid, 6 cubes (50K steps, ~45 min)
py -3.11 train_astar_ddqn.py --grid_size 4 --num_cubes 6 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: A* 4x4/6 training failed!
    pause
    exit /b 1
)
echo.

echo [3/9] A* - 4x4 grid, 9 cubes (50K steps, ~45 min)
py -3.11 train_astar_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: A* 4x4/9 training failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM RRT VIZ METHOD (3 experiments)
REM ============================================================
echo ============================================================
echo RRT VIZ METHOD - 3 EXPERIMENTS
echo ============================================================
echo.

echo [4/9] RRT Viz - 3x3 grid, 4 cubes (50K steps, ~45 min)
py -3.11 train_rrt_viz_ddqn.py --grid_size 3 --num_cubes 4 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT Viz 3x3/4 training failed!
    pause
    exit /b 1
)
echo.

echo [5/9] RRT Viz - 4x4 grid, 6 cubes (50K steps, ~45 min)
py -3.11 train_rrt_viz_ddqn.py --grid_size 4 --num_cubes 6 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT Viz 4x4/6 training failed!
    pause
    exit /b 1
)
echo.

echo [6/9] RRT Viz - 4x4 grid, 9 cubes (50K steps, ~45 min)
py -3.11 train_rrt_viz_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT Viz 4x4/9 training failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM RRT ISAAC SIM METHOD (3 experiments)
REM ============================================================
echo ============================================================
echo RRT ISAAC SIM METHOD - 3 EXPERIMENTS
echo ============================================================
echo.

echo [7/9] RRT IsaacSim - 3x3 grid, 4 cubes (50K steps, ~2 hours)
C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 3 --num_cubes 4 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT IsaacSim 3x3/4 training failed!
    pause
    exit /b 1
)
echo.

echo [8/9] RRT IsaacSim - 4x4 grid, 6 cubes (50K steps, ~2 hours)
C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 6 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT IsaacSim 4x4/6 training failed!
    pause
    exit /b 1
)
echo.

echo [9/9] RRT IsaacSim - 4x4 grid, 9 cubes (50K steps, ~2 hours)
C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --grid_size 4 --num_cubes 9 --timesteps 50000 --use_wandb
if %errorlevel% neq 0 (
    echo ERROR: RRT IsaacSim 4x4/9 training failed!
    pause
    exit /b 1
)
echo.

echo ============================================================
echo ALL 9 TRAINING RUNS COMPLETE
echo ============================================================
echo End time: %date% %time%
echo.
echo Models saved to: models\
echo Logs saved to: logs\
echo W&B Dashboard: https://wandb.ai/
echo ============================================================
pause

