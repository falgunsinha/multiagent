@echo off
REM Test MASAC on RRT Viz and A* configurations (6 configs)
REM Uses native Python environment (no Isaac Sim)

echo ================================================================================
echo MASAC Testing - RRT Viz ^& A* (6 configurations)
echo ================================================================================
echo RRT Viz: Grid 3x3 (4 cubes), Grid 4x4 (6 cubes), Grid 4x4 (9 cubes)
echo A*:      Grid 3x3 (4 cubes), Grid 4x4 (6 cubes), Grid 4x4 (9 cubes)
echo Episodes per config: 5
echo.

cd /d C:\isaacsim

REM Run with native Python
python "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_native.py" --episodes 5
if errorlevel 1 (
    echo [WARNING] Python returned non-zero exit code, but continuing...
)

echo.
echo ================================================================================
echo RRT Viz ^& A* Testing Complete!
echo ================================================================================
echo.

REM Always exit successfully
exit /b 0

