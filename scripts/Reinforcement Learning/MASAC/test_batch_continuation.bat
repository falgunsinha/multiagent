@echo off
REM Simple test to verify batch file continuation works

echo ================================================================================
echo Testing Batch File Continuation
echo ================================================================================
echo.

echo [TEST] Part 1: Simulating Isaac Sim test...
echo [TEST] This will exit with code 0
call python -c "import sys; print('Isaac Sim test running...'); sys.exit(0)"
set EXIT_CODE_1=%errorlevel%
echo [TEST] Part 1 finished with exit code: %EXIT_CODE_1%
echo.

echo [TEST] Waiting 2 seconds...
timeout /t 2 /nobreak > nul
echo.

echo ================================================================================
echo [TEST] Part 2: Simulating Native Python test...
echo ================================================================================
echo [TEST] This will also exit with code 0
call python -c "import sys; print('Native Python test running...'); sys.exit(0)"
set EXIT_CODE_2=%errorlevel%
echo [TEST] Part 2 finished with exit code: %EXIT_CODE_2%
echo.

echo ================================================================================
echo [TEST] Both parts completed!
echo ================================================================================
echo Part 1 exit code: %EXIT_CODE_1%
echo Part 2 exit code: %EXIT_CODE_2%
echo.
echo If you see this message, batch continuation is working correctly!
pause

