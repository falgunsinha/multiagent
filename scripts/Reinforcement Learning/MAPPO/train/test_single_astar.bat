@echo off
REM Single A* test to verify WandB logging

echo ========================================
echo MAPPO A* Training - Single Test
echo ========================================
echo.

set PYTHON=C:\isaacsim\python.bat
set SCRIPT_DIR=%~dp0
set DDQN_MODEL=C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models\ddqn_astar_grid3_cubes4_20251220_012015_final.pt

echo Running A* training with 50 timesteps...
echo.

%PYTHON% "%SCRIPT_DIR%train_astar_mappo.py" --grid_size 3 --num_cubes 4 --timesteps 50 --ddqn_model_path "%DDQN_MODEL%" --run_name "astar_grid3_cubes4_test" --config_name "grid3_cubes4" --log_interval 1 --save_interval 50

echo.
echo ========================================
echo Training complete!
echo ========================================
echo.
echo Check WandB dashboard:
echo   1. Go to: https://wandb.ai/falgunsinha/ddqn-mappo-object-selection-reshuffling
echo   2. Charts should now show data automatically!
echo.
echo Metrics logged:
echo   - episode/total_reward (matches DDQN)
echo   - episode/total_length (matches DDQN)
echo   - episode/cubes_picked
echo   - episode/reshuffles_performed
echo.
pause

