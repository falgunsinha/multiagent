@echo off
REM Analysis Script for GAT+CVD vs DDQN Comparison
REM 
REM Usage:
REM   1. Edit the paths below to point to your actual log files
REM   2. Run: run_analysis.bat

echo ========================================
echo GAT+CVD vs DDQN Analysis
echo ========================================
echo.

REM Set paths to log files (EDIT THESE!)
set DDQN_LOG=C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs\ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_episodes.csv
set GAT_CVD_LOG=C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd\logs\gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_episodes.csv

REM Output directory
set OUTPUT_DIR=analysis_results

echo DDQN Log: %DDQN_LOG%
echo GAT+CVD Log: %GAT_CVD_LOG%
echo Output Directory: %OUTPUT_DIR%
echo.

REM Check if files exist
if not exist "%DDQN_LOG%" (
    echo ERROR: DDQN log file not found!
    echo Please edit run_analysis.bat and set the correct path.
    pause
    exit /b 1
)

if not exist "%GAT_CVD_LOG%" (
    echo ERROR: GAT+CVD log file not found!
    echo Please edit run_analysis.bat and set the correct path.
    echo Note: Replace TIMESTAMP with your actual timestamp.
    pause
    exit /b 1
)

REM Install required packages (if not already installed)
echo Installing required packages...
C:\isaacsim\python.bat -m pip install pandas matplotlib seaborn scipy numpy --quiet

REM Run analysis
echo.
echo Running analysis...
C:\isaacsim\python.bat analyze_results.py --ddqn_log "%DDQN_LOG%" --gat_cvd_log "%GAT_CVD_LOG%" --output_dir "%OUTPUT_DIR%"

echo.
echo ========================================
echo Analysis complete!
echo Results saved to: %OUTPUT_DIR%
echo ========================================
pause

