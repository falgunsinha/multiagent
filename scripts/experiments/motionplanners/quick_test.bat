@echo off
REM Quick test script - runs 1 trial per configuration with all 7 planners
REM Shows compact one-line results for each test, comprehensive summary at the end

echo.
echo ================================================================================
echo QUICK PLANNER TEST - 1 TRIAL PER CONFIGURATION
echo ================================================================================
echo.
echo Testing 7 planners: isaac_rrt, rrt, astar, prm, rrtstar, rrtstar_rs, lqr_rrtstar
echo Densities: 10%%, 25%%, 40%%
echo Obstacle types: cube, bar, giant_bar
echo.
echo Format: [OK/FAIL] [obstacle_type] [density] [planner] [time/FAILED]
echo Comprehensive summary will be displayed at the end.
echo.
echo ================================================================================
echo.

C:\isaacsim\python.bat run_pick_place_experiments.py ^
    --planners isaac_rrt rrt astar prm rrtstar rrtstar_rs lqr_rrtstar ^
    --grid_sizes 4 ^
    --obstacle_densities 0.10 0.25 0.40 ^
    --obstacle_types cube bar giant_bar ^
    --num_trials 1 ^
    --quick-test ^
    --headless

echo.
echo ================================================================================
echo QUICK TEST COMPLETE
echo ================================================================================
pause

