@echo off
REM Run seed 123 only with 20 episodes for all 12 models
REM This is to resume after seed 42 completed successfully

echo ================================================================================
echo TWO-AGENT SYSTEM TESTING - SEED 123 ONLY
echo ================================================================================
echo Seed: 123
echo Episodes per model: 20
echo Models: 12 (4 continuous + 8 discrete)
echo Total episodes: 240
echo ================================================================================
echo.

cd C:\isaacsim
C:\isaacsim\python.bat C:\isaacsim\cobotproject\scripts\experiments\rlmodels\mutliagent\test_two_agent_system.py --seeds 123 --episodes 20

pause

