@echo off
REM Quick Training Script - Train Single Configuration
REM Usage: quick_train.bat <method> <grid_size> <num_cubes>
REM Example: quick_train.bat heuristic 3 4

if "%1"=="" (
    echo Usage: quick_train.bat ^<method^> ^<grid_size^> ^<num_cubes^>
    echo.
    echo Methods: heuristic, astar, rrt
    echo Grid sizes: 3, 4, 5, 6
    echo Num cubes: 4, 6, 9, etc.
    echo.
    echo Examples:
    echo   quick_train.bat heuristic 3 4
    echo   quick_train.bat astar 4 6
    echo   quick_train.bat rrt 4 9
    exit /b 1
)

set METHOD=%1
set GRID_SIZE=%2
set NUM_CUBES=%3

REM Validate method
if not "%METHOD%"=="heuristic" if not "%METHOD%"=="astar" if not "%METHOD%"=="rrt" (
    echo Error: Invalid method '%METHOD%'
    echo Valid methods: heuristic, astar, rrt
    exit /b 1
)

REM Set timesteps based on method
if "%METHOD%"=="rrt" (
    set TIMESTEPS=10000
) else (
    set TIMESTEPS=100000
)

echo ============================================================
echo QUICK TRAINING
echo ============================================================
echo Method: %METHOD%
echo Grid: %GRID_SIZE%x%GRID_SIZE%
echo Cubes: %NUM_CUBES%
echo Timesteps: %TIMESTEPS%
echo ============================================================
echo.

if "%METHOD%"=="rrt" (
    call C:\isaacsim\python.bat train_object_selection_rrt.py --timesteps %TIMESTEPS% --training_grid_size %GRID_SIZE% --num_cubes %NUM_CUBES%
) else (
    call C:\isaacsim\python.bat train_object_selection.py --method %METHOD% --timesteps %TIMESTEPS% --training_grid_size %GRID_SIZE% --num_cubes %NUM_CUBES%
)

echo.
echo ============================================================
echo TRAINING COMPLETE
echo ============================================================
echo.
pause

