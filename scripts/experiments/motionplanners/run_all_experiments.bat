@echo off
REM ===========================================================================
REM Motion Planner Experiments Complete Run Script
REM ===========================================================================
REM
REM This script runs ALL motion planning experiments with Isaac Sim in HEADLESS mode
REM Based on LLM-A* paper methodology with 30 trials per configuration
REM
REM EXPERIMENT CONFIGURATION
REM   Obstacle Densities: 3 levels (10%, 25%, 40%)
REM   Obstacle Types: 3 types (cube, bar, giant_bar)
REM   Grid Sizes: 7 sizes (3x3, 4x4, 5x5, 6x6, 7x7, 8x8, 9x9)
REM   Planners: 8 algorithms (isaac_rrt, astar, rrt, rrtstar, prm, rrtstar_rs, lqr_rrtstar, lqr)
REM   Trials per config: 30 (following LLM-A* paper)
REM
REM TOTAL EXPERIMENTS: 3 x 3 x 7 x 8 x 30 = 15,120 planning cycles
REM
REM ESTIMATED TIME: 20-30 hours (depending on hardware)
REM ===========================================================================

echo.
echo ============================================================================
echo MOTION PLANNER EXPERIMENTS - COMPREHENSIVE RUN
echo ============================================================================
echo.
echo Configuration (Based on LLM-A* Paper Methodology):
echo   - Obstacle Densities: 0.10, 0.25, 0.40 (3 levels)
echo   - Obstacle Types: cube, bar, giant_bar (3 types)
echo   - Grid Sizes: 3x3, 4x4, 5x5, 6x6, 7x7, 8x8, 9x9 (7 sizes)
echo   - Planners: isaac_rrt, astar, rrt, rrtstar, prm, rrtstar_rs, lqr_rrtstar, lqr (8 planners)
echo   - Trials per config: 30 (following LLM-A* paper)
echo.
echo TOTAL EXPERIMENTS: 15,120 planning cycles
echo.
echo Mode: HEADLESS (no GUI - optimized for performance)
echo.
echo ============================================================================
echo.

REM Change to Isaac Sim directory
cd /d C:\isaacsim

REM Run experiments with headless mode enabled
echo Starting comprehensive experiments...
echo This will take approximately 20-30 hours to complete.
echo.

python.bat cobotproject\scripts\experiments\motionplanners\run_pick_place_experiments.py ^
    --planners isaac_rrt astar rrt rrtstar prm rrtstar_rs lqr_rrtstar lqr ^
    --num_cubes 1 ^
    --num_trials 30 ^
    --grid_sizes 3 4 5 6 7 8 9 ^
    --obstacle_densities 0.10 0.25 0.40 ^
    --obstacle_types cube bar giant_bar ^
    --headless ^
    --output_dir C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results

echo.
echo ============================================================================
echo EXPERIMENTS COMPLETE!
echo ============================================================================
echo.
echo Results saved to:
echo   C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results
echo.
echo Output files:
echo   - pick_place_comparison_TIMESTAMP.csv (raw data)
echo   - pick_place_comparison_TIMESTAMP.json (structured data)
echo   - waypoint_selection_TIMESTAMP.csv (waypoint selection experiment data)
echo   - all_experiments_TIMESTAMP.tex (LaTeX tables for ALL 6 experiments)
echo.
echo LaTeX file includes:
echo   - Table 1: Map Size Scalability
echo   - Table 2: Obstacle Density Study
echo   - Table 3: Waypoint Selection (LLM-A* Style)
echo   - Table 4: Path Quality Distribution
echo   - Table 5: Bar-Shaped Obstacles
echo   - Table 6: Scalability Analysis
echo.
echo ============================================================================

pause

