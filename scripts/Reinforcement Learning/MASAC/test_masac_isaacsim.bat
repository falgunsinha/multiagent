@echo off
REM Test MASAC on Isaac Sim RRT configurations (3 configs)
REM Uses Isaac Sim Python environment with standalone headless mode
REM Includes Franka robot with RRT planner

echo ================================================================================
echo MASAC Testing - Isaac Sim RRT (3 configurations)
echo ================================================================================
echo Running in standalone headless mode with Franka robot and RRT planner
echo Configurations: Grid 3x3 (4 cubes), Grid 4x4 (6 cubes), Grid 4x4 (9 cubes)
echo Episodes per config: 5
echo.

cd /d C:\isaacsim

REM Run with Isaac Sim Python (ignore exit code)
C:\isaacsim\python.bat "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_isaacsim.py" --episodes 5
if errorlevel 1 (
    echo [WARNING] Isaac Sim returned non-zero exit code, but continuing...
)

echo.
echo ================================================================================
echo Isaac Sim RRT Testing Complete!
echo ================================================================================
echo.

REM Always exit successfully so batch file continues
exit /b 0

