@echo off
REM Quick test for PPO-Discrete+MASAC only
REM Tests 1 seed, 1 episode for fast error detection

echo ================================================================================
echo Testing PPO-Discrete+MASAC ONLY (Quick Test)
echo ================================================================================
echo Seed: 42
echo Episodes: 1
echo ================================================================================
echo.

cd C:\isaacsim
C:\isaacsim\python.bat C:\isaacsim\cobotproject\scripts\experiments\rlmodels\mutliagent\test_single_model.py --model "PPO-Discrete+MASAC" --seed 42 --episodes 1

pause

