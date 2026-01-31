@echo off
REM Automated Testing Script for All Trained Models
REM Tests each model and records results

echo ============================================================
echo AUTOMATED MODEL TESTING
echo ============================================================
echo.
echo This script will test all trained models automatically.
echo Each test runs for 1 episode to measure:
echo   - Success rate
echo   - Total time
echo   - Pick order decisions
echo.
echo Results will be saved to: test_results.txt
echo ============================================================
echo.

set RESULTS_FILE=test_results.txt
echo Model Testing Results - %date% %time% > %RESULTS_FILE%
echo ============================================================ >> %RESULTS_FILE%
echo. >> %RESULTS_FILE%

REM Find all trained models
echo Searching for trained models...
echo.

REM Test pure heuristic baseline first
echo ============================================================
echo TESTING: Pure Heuristic Baseline (No RL)
echo ============================================================
echo.
echo [BASELINE] Testing pure heuristic with 4 cubes, 3x3 grid...
echo. >> %RESULTS_FILE%
echo ---------------------------------------- >> %RESULTS_FILE%
echo Pure Heuristic Baseline - 4 cubes, 3x3 grid >> %RESULTS_FILE%
echo ---------------------------------------- >> %RESULTS_FILE%
C:\isaacsim\python.bat franka_pure_heuristic_baseline.py --num_cubes 4 --grid_size 3 --headless >> %RESULTS_FILE% 2>&1
echo. >> %RESULTS_FILE%

echo [BASELINE] Testing pure heuristic with 6 cubes, 4x4 grid...
echo. >> %RESULTS_FILE%
echo ---------------------------------------- >> %RESULTS_FILE%
echo Pure Heuristic Baseline - 6 cubes, 4x4 grid >> %RESULTS_FILE%
echo ---------------------------------------- >> %RESULTS_FILE%
C:\isaacsim\python.bat franka_pure_heuristic_baseline.py --num_cubes 6 --grid_size 4 --headless >> %RESULTS_FILE% 2>&1
echo. >> %RESULTS_FILE%

echo [BASELINE] Testing pure heuristic with 9 cubes, 4x4 grid...
echo. >> %RESULTS_FILE%
echo ---------------------------------------- >> %RESULTS_FILE%
echo Pure Heuristic Baseline - 9 cubes, 4x4 grid >> %RESULTS_FILE%
echo ---------------------------------------- >> %RESULTS_FILE%
C:\isaacsim\python.bat franka_pure_heuristic_baseline.py --num_cubes 9 --grid_size 4 --headless >> %RESULTS_FILE% 2>&1
echo. >> %RESULTS_FILE%

REM Now test RL models
echo.
echo ============================================================
echo TESTING: RL-Trained Models
echo ============================================================
echo.

REM You'll need to manually add the model paths after training
REM Example format:
REM echo [RL] Testing heuristic model - 4 cubes, 3x3 grid...
REM C:\isaacsim\python.bat franka_rrt_physXLidar_depth_camera_rl_standalone_v1.9.py --rl_model "models/object_selection/object_selection_heuristic_grid3x3_cubes4_TIMESTAMP_final.zip" --headless >> %RESULTS_FILE% 2>&1

echo.
echo NOTE: To test RL models, you need to manually add their paths to this script.
echo After training completes, edit test_all_models.bat and add the model paths.
echo.
echo Example:
echo   C:\isaacsim\python.bat franka_rrt_physXLidar_depth_camera_rl_standalone_v1.9.py ^
echo     --rl_model "models/object_selection/MODEL_NAME_final.zip" --headless
echo.

echo ============================================================
echo TESTING COMPLETE
echo ============================================================
echo.
echo Results saved to: %RESULTS_FILE%
echo.
echo To view results:
echo   type %RESULTS_FILE%
echo.
echo To compare all models:
echo   py -3.11 compare_training_results.py
echo.
pause

