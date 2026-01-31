"""Remove DDQN entries from CSV files"""
import pandas as pd
from pathlib import Path

# Remove from episode_results.csv
episode_file = Path("two_agent_results/discrete/seed_42/episode_results.csv")
if episode_file.exists():
    df = pd.read_csv(episode_file)
    original_count = len(df)
    df_filtered = df[df['model'] != 'DDQN']
    df_filtered.to_csv(episode_file, index=False)
    print(f"✅ Episode CSV: Removed {original_count - len(df_filtered)} DDQN entries ({len(df_filtered)} remaining)")
else:
    print(f"❌ Episode CSV not found: {episode_file}")

# Remove from timestep_results.csv
timestep_file = Path("two_agent_results/discrete/seed_42/timestep_results.csv")
if timestep_file.exists():
    df = pd.read_csv(timestep_file)
    original_count = len(df)
    df_filtered = df[df['model'] != 'DDQN']
    df_filtered.to_csv(timestep_file, index=False)
    print(f"✅ Timestep CSV: Removed {original_count - len(df_filtered)} DDQN entries ({len(df_filtered)} remaining)")
else:
    print(f"❌ Timestep CSV not found: {timestep_file}")

