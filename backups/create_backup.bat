@echo off
REM Create backup of files before training changes

set TIMESTAMP=%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_DIR=C:\isaacsim\cobotproject\backups\pre_training_changes_%TIMESTAMP%

echo === Creating Backups ===
echo Backup directory: %BACKUP_DIR%
echo.

REM Create backup directory structure
mkdir "%BACKUP_DIR%\src\rl\doubleDQN" 2>nul
mkdir "%BACKUP_DIR%\scripts\Reinforcement Learning\doubleDQN_script" 2>nul

REM Backup files
echo Backing up files...
copy "C:\isaacsim\cobotproject\src\rl\object_selection_env.py" "%BACKUP_DIR%\src\rl\" >nul
copy "C:\isaacsim\cobotproject\src\rl\object_selection_env_rrt.py" "%BACKUP_DIR%\src\rl\" >nul
copy "C:\isaacsim\cobotproject\src\rl\doubleDQN\double_dqn_agent.py" "%BACKUP_DIR%\src\rl\doubleDQN\" >nul
copy "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\train_astar_ddqn.py" "%BACKUP_DIR%\scripts\Reinforcement Learning\doubleDQN_script\" >nul
copy "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\train_rrt_viz_ddqn.py" "%BACKUP_DIR%\scripts\Reinforcement Learning\doubleDQN_script\" >nul
copy "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\train_rrt_isaacsim_ddqn.py" "%BACKUP_DIR%\scripts\Reinforcement Learning\doubleDQN_script\" >nul

echo.
echo === Backup Complete ===
echo Backup location: %BACKUP_DIR%
echo.

