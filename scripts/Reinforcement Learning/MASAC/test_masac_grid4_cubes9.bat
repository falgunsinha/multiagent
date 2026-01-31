@echo off
REM Test MASAC on Grid 4x4, 9 cubes ONLY (all 3 planners with 2 seeds)
REM Runs Isaac Sim RRT, RRT Viz, and A* with seeds 42 and 123 in continuation

echo ================================================================================
echo MASAC Multi-Seed Testing - Grid 4x4, 9 Cubes (3 Planners x 2 Seeds)
echo ================================================================================
echo.
echo This will test Grid 4x4 with 9 cubes on:
echo   1. Isaac Sim RRT planner (Isaac Sim Python) - 2 seeds
echo   2. RRT Viz planner (native Python) - 2 seeds
echo   3. A* planner (native Python) - 2 seeds
echo.
echo Seeds: 42, 123
echo Episodes per seed: 500
echo Total: 3 planners x 2 seeds x 500 episodes = 3000 test episodes
echo.
echo Estimated time:
echo   - Isaac Sim RRT: ~25 hours (2 seeds x 12.5 hours)
echo   - RRT Viz: ~6 minutes (2 seeds x 3 minutes)
echo   - A*: ~6 minutes (2 seeds x 3 minutes)
echo.
echo Press any key to start testing...
pause > nul

cd /d C:\isaacsim

REM ============================================================================
REM PART 1: Isaac Sim RRT - 2 Seeds
REM ============================================================================

echo.
echo ================================================================================
echo PART 1/2: Testing Isaac Sim RRT - Grid 4x4, 9 Cubes (2 Seeds)
echo ================================================================================
echo.

REM Seed 1: 42
echo.
echo --------------------------------------------------------------------------------
echo Isaac Sim RRT - Seed 42 (Run 1/2)
echo --------------------------------------------------------------------------------
echo.
echo [BATCH] Starting Isaac Sim RRT with seed=42...
echo [BATCH] Calling: C:\isaacsim\python.bat test_masac_grid4_cubes9_isaacsim.py --episodes 500 --seed 42 --run_id 1
echo.

call C:\isaacsim\python.bat "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_grid4_cubes9_isaacsim.py" --episodes 500 --seed 42 --run_id 1

set ISAAC_SEED1_EXIT_CODE=%errorlevel%
echo.
echo [BATCH] Isaac Sim RRT (seed=42) finished with exit code: %ISAAC_SEED1_EXIT_CODE%
if %ISAAC_SEED1_EXIT_CODE% neq 0 echo [WARNING] Isaac Sim returned non-zero exit code, but continuing...
if %ISAAC_SEED1_EXIT_CODE% equ 0 echo [BATCH] Isaac Sim RRT (seed=42) completed successfully!
echo.

REM Wait for Isaac Sim to fully close
echo [BATCH] Waiting 5 seconds for Isaac Sim to fully close...
timeout /t 5 /nobreak > nul
echo [BATCH] Continuing to seed 123...
echo.

REM Seed 2: 123
echo.
echo --------------------------------------------------------------------------------
echo Isaac Sim RRT - Seed 123 (Run 2/2)
echo --------------------------------------------------------------------------------
echo.
echo [BATCH] Starting Isaac Sim RRT with seed=123...
echo [BATCH] Calling: C:\isaacsim\python.bat test_masac_grid4_cubes9_isaacsim.py --episodes 500 --seed 123 --run_id 2
echo.

call C:\isaacsim\python.bat "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_grid4_cubes9_isaacsim.py" --episodes 500 --seed 123 --run_id 2

set ISAAC_SEED2_EXIT_CODE=%errorlevel%
echo.
echo [BATCH] Isaac Sim RRT (seed=123) finished with exit code: %ISAAC_SEED2_EXIT_CODE%
if %ISAAC_SEED2_EXIT_CODE% neq 0 echo [WARNING] Isaac Sim returned non-zero exit code, but continuing...
if %ISAAC_SEED2_EXIT_CODE% equ 0 echo [BATCH] Isaac Sim RRT (seed=123) completed successfully!
echo.
echo [BATCH] ========================================
echo [BATCH] Isaac Sim RRT Testing Complete (2 seeds)!
echo [BATCH] Moving to Part 2/2 (RRT Viz ^& A*)...
echo [BATCH] ========================================
echo.

REM Wait for Isaac Sim to fully close
echo [BATCH] Waiting 5 seconds for Isaac Sim to fully close...
timeout /t 5 /nobreak > nul
echo [BATCH] Continuing...
echo.

REM ============================================================================
REM PART 2: RRT Viz & A* - 2 Seeds (Native Python)
REM ============================================================================

echo.
echo ================================================================================
echo PART 2/2: Testing RRT Viz ^& A* - Grid 4x4, 9 Cubes (2 Seeds)
echo ================================================================================
echo.
echo [BATCH] This part uses native Python (not Isaac Sim)
echo [BATCH] Note: The native script tests BOTH RRT Viz and A* together
echo.

REM Seed 1: 42
echo.
echo --------------------------------------------------------------------------------
echo RRT Viz ^& A* - Seed 42 (Run 1/2)
echo --------------------------------------------------------------------------------
echo.
echo [BATCH] Starting RRT Viz ^& A* with seed=42...
echo [BATCH] Calling: python test_masac_grid4_cubes9_native.py --episodes 500 --seed 42 --run_id 1
echo.

call python "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_grid4_cubes9_native.py" --episodes 500 --seed 42 --run_id 1

set NATIVE_SEED1_EXIT_CODE=%errorlevel%
echo.
echo [BATCH] RRT Viz ^& A* (seed=42) finished with exit code: %NATIVE_SEED1_EXIT_CODE%
if %NATIVE_SEED1_EXIT_CODE% neq 0 echo [WARNING] Python returned non-zero exit code, but continuing...
if %NATIVE_SEED1_EXIT_CODE% equ 0 echo [BATCH] RRT Viz ^& A* (seed=42) completed successfully!
echo.

REM Seed 2: 123
echo.
echo --------------------------------------------------------------------------------
echo RRT Viz ^& A* - Seed 123 (Run 2/2)
echo --------------------------------------------------------------------------------
echo.
echo [BATCH] Starting RRT Viz ^& A* with seed=123...
echo [BATCH] Calling: python test_masac_grid4_cubes9_native.py --episodes 500 --seed 123 --run_id 2
echo.

call python "cobotproject\scripts\Reinforcement Learning\MASAC\test_masac_grid4_cubes9_native.py" --episodes 500 --seed 123 --run_id 2

set NATIVE_SEED2_EXIT_CODE=%errorlevel%
echo.
echo [BATCH] RRT Viz ^& A* (seed=123) finished with exit code: %NATIVE_SEED2_EXIT_CODE%
if %NATIVE_SEED2_EXIT_CODE% neq 0 echo [WARNING] Python returned non-zero exit code, but continuing...
if %NATIVE_SEED2_EXIT_CODE% equ 0 echo [BATCH] RRT Viz ^& A* (seed=123) completed successfully!
echo.
echo [BATCH] ========================================
echo [BATCH] RRT Viz ^& A* Testing Complete (2 seeds)!
echo [BATCH] ========================================
echo.

echo.
echo ================================================================================
echo ALL MULTI-SEED TESTING COMPLETE - Grid 4x4, 9 Cubes
echo ================================================================================
echo.
echo Test Summary:
echo   1. Isaac Sim RRT planner - 2 seeds (42, 123) - COMPLETE
echo   2. RRT Viz planner - 2 seeds (42, 123) - COMPLETE
echo   3. A* planner - 2 seeds (42, 123) - COMPLETE
echo.
echo Total runs: 6 (3 planners x 2 seeds)
echo Total episodes: 3000 (6 runs x 500 episodes)
echo.
echo Results saved to: cobotproject\scripts\Reinforcement Learning\MASAC\logs
echo.
echo Check the logs directory for detailed results:
echo   Isaac Sim RRT:
echo     - masac_rrt_isaacsim_grid4_cubes9_*_seed42_run1_*.csv/json
echo     - masac_rrt_isaacsim_grid4_cubes9_*_seed123_run2_*.csv/json
echo   RRT Viz:
echo     - masac_rrt_viz_grid4_cubes9_*_seed42_run1_*.csv/json
echo     - masac_rrt_viz_grid4_cubes9_*_seed123_run2_*.csv/json
echo   A*:
echo     - masac_astar_grid4_cubes9_*_seed42_run1_*.csv/json
echo     - masac_astar_grid4_cubes9_*_seed123_run2_*.csv/json
echo.
echo ================================================================================
echo NEXT STEP: Aggregate Multi-Seed Results
echo ================================================================================
echo.
echo To calculate mean +- std across seeds, run:
echo   python aggregate_multi_seed_results.py --log_dir "cobotproject\scripts\Reinforcement Learning\MASAC\logs"
echo.
echo This will create:
echo   - aggregated_multi_seed_TIMESTAMP.csv
echo   - aggregated_multi_seed_TIMESTAMP.json
echo.
echo With results in MAPPO-style format: mean(std)
echo Example: Success Rate: 90.5(2.5) = 90.5%% +- 2.5%%
echo.
echo To visualize results, run:
echo   python plot_timesteps_reward.py
echo   python plot_episodes_reward.py
echo.
pause

