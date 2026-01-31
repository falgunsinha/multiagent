@echo off
REM Test MASAC on all 9 configurations
REM Runs Isaac Sim RRT (3 configs) with Isaac Sim Python in standalone headless mode
REM Runs RRT Viz & A* (6 configs) with native Python

echo ================================================================================
echo MASAC Testing - ALL 9 CONFIGURATIONS
echo ================================================================================
echo.
echo This will run in order:
echo   1. A* (3 configs) - Native Python
echo      - Grid 3x3, 4 cubes
echo      - Grid 4x4, 6 cubes
echo      - Grid 4x4, 9 cubes
echo.
echo   2. RRT Viz (3 configs) - Native Python
echo      - Grid 3x3, 4 cubes
echo      - Grid 4x4, 6 cubes
echo      - Grid 4x4, 9 cubes
echo.
echo   3. Isaac Sim RRT (3 configs) - Isaac Sim Python with Franka + RRT
echo      - Grid 3x3, 4 cubes
echo      - Grid 4x4, 6 cubes
echo      - Grid 4x4, 9 cubes
echo.
echo Total: 9 configurations x 5 episodes each = 45 test episodes
echo.
echo Press any key to start testing...
pause > nul

REM Test A* and RRT Viz configurations (Native Python)
echo.
echo ================================================================================
echo PART 1/2: Testing A* ^& RRT Viz (6 configurations)
echo ================================================================================
echo.
echo [BATCH] This part uses native Python (not Isaac Sim)
echo.

cd /d C:\isaacsim

REM Run native Python tests directly
echo [BATCH] Starting Part 1/2...
echo Running A* ^& RRT Viz tests...
echo [BATCH] Calling: python test_masac_native.py --episodes 5
echo.

REM Use call to ensure batch file continues
call python "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_native.py" --episodes 5

set NATIVE_EXIT_CODE=%errorlevel%
echo.
echo [BATCH] Native tests finished with exit code: %NATIVE_EXIT_CODE%
if %NATIVE_EXIT_CODE% neq 0 (
    echo [WARNING] Python returned non-zero exit code %NATIVE_EXIT_CODE%, but continuing...
) else (
    echo [BATCH] Native tests completed successfully!
)
echo.
echo [BATCH] ========================================
echo [BATCH] A* ^& RRT Viz Testing Complete!
echo [BATCH] Moving to Part 2/2 (Isaac Sim RRT)...
echo [BATCH] ========================================
echo.

REM Wait a moment before starting Isaac Sim
echo [BATCH] Waiting 2 seconds before starting Isaac Sim...
timeout /t 2 /nobreak > nul
echo [BATCH] Continuing...
echo.

REM Test Isaac Sim RRT configurations
echo.
echo ================================================================================
echo PART 2/2: Testing Isaac Sim RRT (3 configurations)
echo ================================================================================
echo.

REM Run Isaac Sim tests directly
echo Running Isaac Sim RRT tests...
echo [BATCH] Calling: C:\isaacsim\python.bat test_masac_isaacsim.py --episodes 5
echo.

REM Use call to ensure batch file continues after Python exits
call C:\isaacsim\python.bat "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_isaacsim.py" --episodes 5

set ISAAC_EXIT_CODE=%errorlevel%
echo.
echo [BATCH] Isaac Sim tests finished with exit code: %ISAAC_EXIT_CODE%
if %ISAAC_EXIT_CODE% neq 0 (
    echo [WARNING] Isaac Sim returned non-zero exit code %ISAAC_EXIT_CODE%, but continuing...
) else (
    echo [BATCH] Isaac Sim tests completed successfully!
)
echo.
echo [BATCH] ========================================
echo [BATCH] Isaac Sim RRT Testing Complete!
echo [BATCH] ========================================
echo.

REM Wait a moment to ensure Isaac Sim has fully closed
echo [BATCH] Waiting 3 seconds for Isaac Sim to fully close...
timeout /t 3 /nobreak > nul
echo [BATCH] Continuing...
echo.

echo.
echo ================================================================================
echo ALL TESTING COMPLETE!
echo ================================================================================
echo Results saved to: cobotproject\scripts\Reinforcement Learning\MASAC\logs
echo.
echo Check the logs directory for detailed results from all 9 configurations.
echo.
pause

