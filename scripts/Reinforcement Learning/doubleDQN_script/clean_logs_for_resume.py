"""
Clean training logs to prepare for resume from step 45,000

This script removes all log entries after step 45,000 and episode 5219
so that when training resumes, the logs will be continuous and clean.

At step 45,000:
- Episode 5219 was the last COMPLETED episode (finished at step 44,992)
- Episode 5220 was IN PROGRESS (started at step 44,993, at step 7/9 when checkpoint saved)

When resuming from step 45,000, episode 5220 will be replayed from the beginning,
so we must remove episode 5220 and all subsequent episodes to avoid duplicates.
"""

import pandas as pd
from pathlib import Path

# Paths
log_dir = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs")
training_log = log_dir / "ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_training.csv"
episode_log = log_dir / "ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_episodes.csv"

# Backup original files
training_backup = log_dir / "ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_training_BACKUP.csv"
episode_backup = log_dir / "ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_episodes_BACKUP.csv"

print("="*60)
print("CLEANING LOGS FOR RESUME FROM STEP 45,000")
print("="*60)
print("\nAnalysis:")
print("  At step 45,000:")
print("    - Last completed episode: 5219 (finished at step 44,992)")
print("    - Episode in progress: 5220 (step 7/9)")
print("  When resuming:")
print("    - Episode 5220 will replay from beginning")
print("    - Must remove episodes 5220+ to avoid duplicates")
print("="*60)

# Backup training log
print(f"\n1. Creating backup of training log...")
with open(training_log, 'r') as src, open(training_backup, 'w') as dst:
    dst.write(src.read())
print(f"   Backup saved: {training_backup.name}")

# Backup episode log
print(f"\n2. Creating backup of episode log...")
with open(episode_log, 'r') as src, open(episode_backup, 'w') as dst:
    dst.write(src.read())
print(f"   Backup saved: {episode_backup.name}")

# Clean training log (keep only steps <= 45000)
print(f"\n3. Cleaning training log...")
df_training = pd.read_csv(training_log)
print(f"   Original entries: {len(df_training)}")
df_training_clean = df_training[df_training['step'] <= 45000]
print(f"   Entries after step 45,000: {len(df_training_clean)}")
print(f"   Removed entries: {len(df_training) - len(df_training_clean)}")
df_training_clean.to_csv(training_log, index=False)
print(f"   ✓ Training log cleaned")

# Clean episode log (keep only episodes <= 5219 - last completed episode)
print(f"\n4. Cleaning episode log...")
df_episode = pd.read_csv(episode_log)
print(f"   Original entries: {len(df_episode)}")
print(f"   Last completed episode at checkpoint: 5219")
df_episode_clean = df_episode[df_episode['episode'] <= 5219]
print(f"   Entries after episode 5,219: {len(df_episode_clean)}")
print(f"   Removed entries: {len(df_episode) - len(df_episode_clean)}")
print(f"   Removed episodes: 5220 to {df_episode['episode'].max()}")
df_episode_clean.to_csv(episode_log, index=False)
print(f"   ✓ Episode log cleaned")

print("\n" + "="*60)
print("LOGS CLEANED SUCCESSFULLY!")
print("="*60)
print("\nSummary:")
print(f"  Training log: {len(df_training_clean)} entries (steps 1-45,000)")
print(f"  Episode log: {len(df_episode_clean)} entries (episodes 0-5,219)")
print(f"\nBackups saved in case you need to restore:")
print(f"  - {training_backup.name}")
print(f"  - {episode_backup.name}")
print("\nWhen training resumes from step 45,000:")
print("  - Episode 5220 will start fresh (no duplicates)")
print("  - Logs will be continuous and clean!")
print("="*60)

