@echo off
REM Resume MASAC Multi-Seed Testing - Grid 4x4, 9 Cubes
REM Starts from Isaac Sim RRT (seed=123) and continues with remaining tests
REM Assumes Isaac Sim RRT (seed=42) has already been completed

echo ================================================================================
echo MASAC Multi-Seed Testing - RESUME from Isaac Sim RRT (seed=123)
echo ================================================================================
echo.
echo This will resume testing from where it left off:
echo   1. Isaac Sim RRT (seed=123) - STARTING HERE
echo   2. RRT Viz ^& A* (seed=42)
echo   3. RRT Viz ^& A* (seed=123)
echo.
echo Assumes already completed:
echo   - Isaac Sim RRT (seed=42) - SKIPPED
echo.
echo Seeds: 123, 42, 123
echo Episodes per seed: 500
echo Total remaining: 3 runs x 500 episodes = 1500 test episodes
echo.
echo Estimated time:
echo   - Isaac Sim RRT (seed=123): ~12.5 hours
echo   - RRT Viz (seed=42): ~3 minutes
echo   - A* (seed=42): ~3 minutes
echo   - RRT Viz (seed=123): ~3 minutes
echo   - A* (seed=123): ~3 minutes
echo   Total: ~12.5 hours
echo.
echo Press any key to start testing...
pause > nul

cd /d C:\isaacsim

REM ============================================================================
REM PART 1: Isaac Sim RRT - Seed 123 ONLY
REM ============================================================================

echo.
echo ================================================================================
echo PART 1/2: Testing Isaac Sim RRT - Grid 4x4, 9 Cubes (Seed 123 ONLY)
echo ================================================================================
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
echo [BATCH] Isaac Sim RRT Testing Complete!
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
echo   1. Isaac Sim RRT planner - seed 42 (SKIPPED - already completed)
echo   2. Isaac Sim RRT planner - seed 123 - COMPLETE
echo   3. RRT Viz planner - seeds 42, 123 - COMPLETE
echo   4. A* planner - seeds 42, 123 - COMPLETE
echo.
echo Total runs completed in this session: 3
echo Total episodes completed in this session: 1500 (3 runs x 500 episodes)
echo.
echo Combined with previous run:
echo   Total runs: 4 (Isaac Sim RRT x2, RRT Viz x2, A* x2)
echo   Total episodes: 2000 (4 runs x 500 episodes)
echo.
echo Results saved to: cobotproject\scripts\Reinforcement Learning\MASAC\logs
echo.
echo Check the logs directory for detailed results:
echo   Isaac Sim RRT:
echo     - masac_rrt_isaacsim_grid4_cubes9_*_seed42_run1_*.csv/json (from previous run)
echo     - masac_rrt_isaacsim_grid4_cubes9_*_seed123_run2_*.csv/json (NEW)
echo   RRT Viz:
echo     - masac_rrt_viz_grid4_cubes9_*_seed42_run1_*.csv/json (NEW)
echo     - masac_rrt_viz_grid4_cubes9_*_seed123_run2_*.csv/json (NEW)
echo   A*:
echo     - masac_astar_grid4_cubes9_*_seed42_run1_*.csv/json (NEW)
echo     - masac_astar_grid4_cubes9_*_seed123_run2_*.csv/json (NEW)
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


