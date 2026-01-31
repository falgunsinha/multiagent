import pandas as pd

# Check discrete timestep data
df_d42_ts = pd.read_csv('two_agent_results/discrete/seed_42/timestep_results.csv')
df_d123_ts = pd.read_csv('two_agent_results/discrete/seed_123/timestep_results.csv')

print("DISCRETE TIMESTEP DATA:")
print(f"Seed 42: {len(df_d42_ts)} timesteps, {df_d42_ts['model'].nunique()} models")
print(f"Timesteps per model (seed 42):")
print(df_d42_ts.groupby('model').size())
print(f"\nSeed 123: {len(df_d123_ts)} timesteps")

# Check continuous timestep data
df_c42_ts = pd.read_csv('two_agent_results/continuous/seed_42/timestep_results.csv')
df_c123_ts = pd.read_csv('two_agent_results/continuous/seed_123/timestep_results.csv')

print("\n\nCONTINUOUS TIMESTEP DATA:")
print(f"Seed 42: {len(df_c42_ts)} timesteps, {df_c42_ts['model'].nunique()} models")
print(f"Timesteps per model (seed 42):")
print(df_c42_ts.groupby('model').size())
print(f"\nSeed 123: {len(df_c123_ts)} timesteps")

# Check columns
print("\n\nCOLUMNS IN TIMESTEP DATA:")
print(df_d42_ts.columns.tolist())

# Check global_timestep range
print("\n\nGLOBAL TIMESTEP RANGE (discrete seed 42):")
for model in df_d42_ts['model'].unique():
    model_df = df_d42_ts[df_d42_ts['model'] == model]
    print(f"{model}: timesteps {model_df['global_timestep'].min()} to {model_df['global_timestep'].max()} ({len(model_df)} total)")

# Check episode range in timestep data
print("\n\nEPISODE RANGE IN TIMESTEP DATA (discrete seed 42):")
for model in df_d42_ts['model'].unique():
    model_df = df_d42_ts[df_d42_ts['model'] == model]
    print(f"{model}: episodes {model_df['episode'].min()} to {model_df['episode'].max()}")

