@echo off
echo ========================================
echo Resume RRT Isaac Sim DDQN Training
echo ========================================
echo.
echo This will resume training from the latest checkpoint to 50,000 steps
echo Estimated time: 1-2 hours
echo.
echo IMPORTANT: Make sure you have restarted your computer to clear GPU memory!
echo.
pause

cd /d "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"

echo.
echo Starting resume training...
echo The script will automatically find the latest checkpoint.
echo.

C:\isaacsim\python.bat resume_rrt_isaacsim_ddqn.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause

