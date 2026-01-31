@echo off
REM Batch script to train Double DQN with Isaac Sim RRT
REM Fixed: Added missing _update_rrt_grid method to ObjectSelectionEnvRRT

echo ============================================================
echo Training Double DQN with Isaac Sim RRT
echo ============================================================
echo.
echo Configuration:
echo - Grid Size: 4x4
echo - Number of Cubes: 9
echo - Timesteps: 50000
echo - Path Planning: Isaac Sim RRT (Full)
echo.
echo Fixed Issue: AttributeError '_update_rrt_grid' not found
echo Solution: Added _update_rrt_grid method to object_selection_env_rrt.py
echo.
echo ============================================================
echo.

cd /d C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script

echo Starting training...
C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --timesteps 50000 --grid_size 4 --num_cubes 9 --save_freq 5000

echo.
echo ============================================================
echo Training completed!
echo Check the logs and models in doubleDQN_script folder
echo ============================================================
pause

