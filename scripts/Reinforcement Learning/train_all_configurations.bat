@echo off
REM Batch Training Script for All Configurations with Action Masking
REM Trains models with different grid sizes and cube counts
REM
REM NEW: All models now use MaskablePPO with action masking
REM      - Invalid actions (already-picked cubes) are prevented, not penalized
REM      - Faster training and better convergence expected
REM
REM Recommended configurations:
REM - 4 cubes: 3x3 grid (44%% filled)
REM - 6 cubes: 3x3 or 4x4 grid (67%% or 38%% filled)
REM - 9 cubes: 4x4 grid (56%% filled)

echo ============================================================
echo BATCH TRAINING - ALL CONFIGURATIONS (WITH ACTION MASKING)
echo ============================================================
echo.
echo This will train 12 models (4 methods x 3 configurations)
echo All methods will use 50,000 timesteps for fair comparison.
echo Strategic obstacles will be placed in the grid during training.
echo.
echo NEW: Action masking enabled - invalid picks prevented!
echo NEW: RRT-based reachability in observations (Fix 2)!
echo.
echo Methods:
echo   1. Heuristic - Distance-based reachability
echo   2. A* - A*-based reachability
echo   3. RRT Isaac Sim - Isaac Sim RRT-based reachability
echo   4. RRT Viz - PythonRobotics RRT-based reachability
echo.
echo Estimated time:
echo   - Heuristic: ~5 min per config = 15 min total
echo   - A*: ~6 min per config = 18 min total
echo   - RRT Viz: ~8 min per config = 24 min total
echo   - RRT Isaac Sim: ~10-15 hours per config = 30-45 hours total
echo.
echo Total estimated time: 31-46 hours
echo ============================================================
echo.

set TIMESTEPS_HEURISTIC=50000
set TIMESTEPS_ASTAR=50000
set TIMESTEPS_RRT=50000
set TIMESTEPS_RRT_VIZ=50000

REM Configuration 1: 4 cubes, 3x3 grid
echo.
echo ============================================================
echo CONFIG 1: 4 cubes, 3x3 grid (44%% filled)
echo ============================================================
echo.
echo [1/12] Training Heuristic method...
call py -3.11 train_object_selection.py --method heuristic --timesteps %TIMESTEPS_HEURISTIC% --training_grid_size 3 --num_cubes 4
echo.
echo [2/12] Training A* method...
call py -3.11 train_object_selection.py --method astar --timesteps %TIMESTEPS_ASTAR% --training_grid_size 3 --num_cubes 4
echo.
echo [3/12] Training RRT Viz method...
call py -3.11 train_object_selection_rrt_viz.py --timesteps %TIMESTEPS_RRT_VIZ% --grid_size 3 --num_cubes 4
echo.
echo [4/12] Training RRT Isaac Sim method...
call C:\isaacsim\python.bat train_object_selection_rrt.py --timesteps %TIMESTEPS_RRT% --training_grid_size 3 --num_cubes 4

REM Configuration 2: 6 cubes, 4x4 grid
echo.
echo ============================================================
echo CONFIG 2: 6 cubes, 4x4 grid (38%% filled)
echo ============================================================
echo.
echo [5/12] Training Heuristic method...
call py -3.11 train_object_selection.py --method heuristic --timesteps %TIMESTEPS_HEURISTIC% --training_grid_size 4 --num_cubes 6
echo.
echo [6/12] Training A* method...
call py -3.11 train_object_selection.py --method astar --timesteps %TIMESTEPS_ASTAR% --training_grid_size 4 --num_cubes 6
echo.
echo [7/12] Training RRT Viz method...
call py -3.11 train_object_selection_rrt_viz.py --timesteps %TIMESTEPS_RRT_VIZ% --grid_size 4 --num_cubes 6
echo.
echo [8/12] Training RRT Isaac Sim method...
call C:\isaacsim\python.bat train_object_selection_rrt.py --timesteps %TIMESTEPS_RRT% --training_grid_size 4 --num_cubes 6

REM Configuration 3: 9 cubes, 4x4 grid
echo.
echo ============================================================
echo CONFIG 3: 9 cubes, 4x4 grid (56%% filled)
echo ============================================================
echo.
echo [9/12] Training Heuristic method...
call py -3.11 train_object_selection.py --method heuristic --timesteps %TIMESTEPS_HEURISTIC% --training_grid_size 4 --num_cubes 9
echo.
echo [10/12] Training A* method...
call py -3.11 train_object_selection.py --method astar --timesteps %TIMESTEPS_ASTAR% --training_grid_size 4 --num_cubes 9
echo.
echo [11/12] Training RRT Viz method...
call py -3.11 train_object_selection_rrt_viz.py --timesteps %TIMESTEPS_RRT_VIZ% --grid_size 4 --num_cubes 9
echo.
echo [12/12] Training RRT Isaac Sim method...
call C:\isaacsim\python.bat train_object_selection_rrt.py --timesteps %TIMESTEPS_RRT% --training_grid_size 4 --num_cubes 9

echo.
echo ============================================================
echo BATCH TRAINING COMPLETE!
echo ============================================================
echo.
echo All 12 models have been trained:
echo   - 3 Heuristic models (distance-based reachability)
echo   - 3 A* models (A*-based reachability)
echo   - 3 RRT Viz models (PythonRobotics RRT-based reachability)
echo   - 3 RRT Isaac Sim models (Isaac Sim RRT-based reachability)
echo.
echo Check models/object_selection/ for saved models.
echo.
echo To compare results, run:
echo   py -3.11 compare_training_results.py
echo.
pause

