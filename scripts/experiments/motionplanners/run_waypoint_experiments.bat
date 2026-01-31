@echo off
REM ===========================================================================
REM Waypoint Selection Experiments - LLM-A* Paper Table 4 Replication
REM ===========================================================================
REM
REM This script runs waypoint selection experiments to replicate Table 4
REM from the LLM-A* paper using Isaac-RRT and RRT planners
REM
REM EXPERIMENT CONFIGURATION
REM   Planners: Isaac-RRT, RRT (analogous to LLM-A* and A* in paper)
REM   Selection Methods: Start, Uniform, Random, Goal (4 methods)
REM   Waypoints: 1, 2, 3, 4 (4 counts)
REM   Trials per config: 30 (following LLM-A* paper)
REM
REM TOTAL EXPERIMENTS: 2 planners x 4 methods x 4 waypoint counts x 30 trials = 960 trials
REM
REM ESTIMATED TIME: 2-4 hours (depending on hardware)
REM ===========================================================================

echo.
echo ===========================================================================
echo WAYPOINT SELECTION EXPERIMENTS - LLM-A* Paper Table 4 Replication
echo ===========================================================================
echo.
echo Configuration:
echo   Planners: Isaac-RRT, RRT
echo   Selection Methods: Start, Uniform, Random, Goal (4 methods)
echo   Waypoints: 1, 2, 3, 4 (4 counts)
echo   Trials per config: 30 (following LLM-A* paper)
echo.
echo TOTAL EXPERIMENTS: 960 trials
echo.
echo Mode: HEADLESS (no GUI - optimized for performance)
echo.
echo ===========================================================================
echo.

REM Change to Isaac Sim directory
cd /d C:\isaacsim

REM Run waypoint selection experiments
echo Starting waypoint selection experiments...
echo This will take approximately 2-4 hours to complete.
echo.

python.bat cobotproject\scripts\experiments\motionplanners\run_waypoint_selection_experiments.py ^
    --planners isaac_rrt rrt ^
    --num_trials 30 ^
    --grid_size 5 ^
    --obstacle_density 0.25 ^
    --obstacle_type cube ^
    --num_waypoints_range 1 4 ^
    --headless ^
    --output_dir C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results

echo.
echo ===========================================================================
echo EXPERIMENTS COMPLETE!
echo ===========================================================================
echo.
echo Results saved to:
echo   C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results
echo.
echo Output files:
echo   - waypoint_selection_results_TIMESTAMP.csv (raw data)
echo   - waypoint_selection_results_TIMESTAMP.json (structured data)
echo.
echo To analyze results and generate Table 4:
echo   python cobotproject\scripts\experiments\motionplanners\analyze_waypoint_results.py --input results\waypoint_selection_results_TIMESTAMP.csv
echo.
echo ===========================================================================

pause

