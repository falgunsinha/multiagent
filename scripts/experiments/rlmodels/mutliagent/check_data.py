import pandas as pd

# Check discrete data
df_d42 = pd.read_csv('two_agent_results/discrete/seed_42/episode_results.csv')
df_d123 = pd.read_csv('two_agent_results/discrete/seed_123/episode_results.csv')

print("DISCRETE DATA:")
print(f"Seed 42: {len(df_d42)} episodes, {df_d42['model'].nunique()} models")
print(f"Episodes per model (seed 42):")
print(df_d42.groupby('model').size())
print(f"\nSeed 123: {len(df_d123)} episodes, {df_d123['model'].nunique()} models")

# Check continuous data
df_c42 = pd.read_csv('two_agent_results/continuous/seed_42/episode_results.csv')
df_c123 = pd.read_csv('two_agent_results/continuous/seed_123/episode_results.csv')

print("\n\nCONTINUOUS DATA:")
print(f"Seed 42: {len(df_c42)} episodes, {df_c42['model'].nunique()} models")
print(f"Episodes per model (seed 42):")
print(df_c42.groupby('model').size())
print(f"\nSeed 123: {len(df_c123)} episodes, {df_c123['model'].nunique()} models")

# Check what columns are available
print("\n\nCOLUMNS IN EPISODE DATA:")
print(df_d42.columns.tolist())

# Check episode numbers
print("\n\nEPISODE RANGE (discrete seed 42):")
for model in df_d42['model'].unique():
    model_df = df_d42[df_d42['model'] == model]
    print(f"{model}: episodes {model_df['episode'].min()} to {model_df['episode'].max()}")

