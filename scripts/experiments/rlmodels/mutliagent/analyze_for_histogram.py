import pandas as pd
import numpy as np

# Load both seeds
df_42 = pd.read_csv('two_agent_results/discrete/seed_42/episode_results.csv')
df_123 = pd.read_csv('two_agent_results/discrete/seed_123/episode_results.csv')
df_combined = pd.concat([df_42, df_123])

# Filter models
models = ['DDQN+MASAC', 'Heuristic', 'PER-DDQN-Light+MASAC']
df_filtered = df_combined[df_combined['model'].isin(models)]

# Episodes to analyze
episodes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
df_ep = df_filtered[df_filtered['episode'].isin(episodes)]

print("="*100)
print("HISTOGRAM/KDE SUITABILITY ANALYSIS")
print("="*100)

# Calculate success rate
df_ep['success_rate'] = (df_ep['cubes_picked'] / df_ep['num_cubes']) * 100

metrics = {
    'success_rate': 'Pick Success (%)',
    'total_distance_reduced': 'Distance Reduced (m)',
    'total_time_saved': 'Time Saved (s)'
}

for metric_col, metric_name in metrics.items():
    print(f"\n{metric_name}:")
    print("-"*100)
    
    for model in models:
        model_data = df_ep[df_ep['model'] == model][metric_col]
        
        print(f"\n  {model}:")
        print(f"    Sample size: {len(model_data)}")
        print(f"    Mean: {model_data.mean():.3f}")
        print(f"    Std: {model_data.std():.3f}")
        print(f"    Min: {model_data.min():.3f}")
        print(f"    Max: {model_data.max():.3f}")
        print(f"    Range: {model_data.max() - model_data.min():.3f}")
        print(f"    Unique values: {len(model_data.unique())}")
        print(f"    Coefficient of Variation (CV): {(model_data.std() / model_data.mean() * 100):.2f}%")
        
        # Check distribution shape
        quartiles = model_data.quantile([0.25, 0.5, 0.75])
        print(f"    Q1: {quartiles[0.25]:.3f}, Median: {quartiles[0.5]:.3f}, Q3: {quartiles[0.75]:.3f}")
        
        # Skewness indicator
        from scipy import stats
        skewness = stats.skew(model_data)
        print(f"    Skewness: {skewness:.3f}")

print("\n" + "="*100)
print("RECOMMENDATIONS:")
print("="*100)
print("\nHistogram with KDE is suitable when:")
print("  - Large sample size (>30 per model)")
print("  - Continuous data with good spread")
print("  - Want to show distribution shape")
print("  - High coefficient of variation (CV > 10-15%)")
print("\nNOT suitable when:")
print("  - Small sample size")
print("  - Very low variance (most values clustered)")
print("  - Discrete/categorical data")
print("  - CV < 5% (too little variation)")

