@echo off
REM ============================================================================
REM Two-Agent System Testing - Run Both Discrete and Continuous in Single Run
REM ============================================================================
REM This batch file runs both discrete and continuous tests in a SINGLE run
REM with seed 42 and 3 episodes each.
REM
REM Usage: Simply double-click this file or run from command line
REM ============================================================================

echo.
echo ================================================================================
echo TWO-AGENT SYSTEM TESTING - DISCRETE + CONTINUOUS (SINGLE RUN)
echo ================================================================================
echo Seed: 42
echo Episodes: 3
echo Models: 8 discrete + 4 continuous = 12 total
echo ================================================================================
echo.

REM Get start time
echo [%date% %time%] Starting tests...
echo.

REM ============================================================================
REM RUN BOTH DISCRETE AND CONTINUOUS IN SINGLE RUN
REM ============================================================================
echo.
echo ################################################################################
echo # TESTING ALL MODELS (CONTINUOUS FIRST, THEN DISCRETE)
echo ################################################################################
echo NOTE: Continuous models tested first to catch errors early
echo.

echo [%date% %time%] Starting test with --action_space both...
cd /d C:\isaacsim
C:\isaacsim\python.bat C:\isaacsim\cobotproject\scripts\experiments\rlmodels\mutliagent\test_two_agent_system.py --action_space both --seeds 42 --episodes 3

REM Don't check exit code, just continue
echo.
echo [%date% %time%] All tests completed (or encountered error)
echo.

REM ============================================================================
REM SUMMARY
REM ============================================================================
echo.
echo ================================================================================
echo ALL TESTS COMPLETED
echo ================================================================================
echo [%date% %time%] Both discrete and continuous tests have finished
echo.
echo Results saved to:
echo   - C:\isaacsim\cobotproject\scripts\experiments\rlmodels\mutliagent\two_agent_results\discrete\seed_42\
echo   - C:\isaacsim\cobotproject\scripts\experiments\rlmodels\mutliagent\two_agent_results\continuous\seed_42\
echo.
echo To analyze results, run:
echo   cd C:\isaacsim\cobotproject\scripts\experiments\rlmodels\mutliagent
echo   py -3.11 analyze_results.py --action_space both
echo.
echo ================================================================================
echo.

REM Keep window open
pause

