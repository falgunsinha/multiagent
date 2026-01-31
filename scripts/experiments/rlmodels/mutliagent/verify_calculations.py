"""
Verify calculations by separating results for each seed
"""
import pandas as pd
from pathlib import Path

def analyze_by_seed(seed, data_type):
    """Analyze results for a specific seed and data type"""
    csv_path = Path(f"two_agent_results/{data_type}/seed_{seed}/episode_results.csv")
    
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"\n{'='*100}")
    print(f"{data_type.upper()} - SEED {seed}")
    print(f"{'='*100}")
    print(f"Total episodes: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"\nResults per model:")
    print(f"{'Model':<30} {'Episodes':>8} {'Avg Reward':>12} {'Success Rate':>12} {'Reshuffles':>12} {'Distance':>12} {'Time Saved':>12}")
    print("-" * 100)
    
    results = []
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        num_episodes = len(model_df)
        
        # Calculate metrics
        if model == 'Heuristic':
            avg_reward = "N/A"
        else:
            avg_reward = f"{model_df['total_reward'].mean():.2f}"
        
        success_rate = (model_df['success'].sum() / num_episodes) * 100
        avg_reshuffles = model_df['reshuffles_performed'].mean()
        avg_distance = model_df['total_distance_reduced'].mean()
        avg_time = model_df['total_time_saved'].mean()
        
        print(f"{model:<30} {num_episodes:>8} {avg_reward:>12} {success_rate:>11.1f}% {avg_reshuffles:>12.2f} {avg_distance:>12.4f} {avg_time:>12.4f}")
        
        results.append({
            'model': model,
            'episodes': num_episodes,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_reshuffles': avg_reshuffles,
            'avg_distance': avg_distance,
            'avg_time': avg_time
        })
    
    return results

# Analyze discrete for both seeds
print("\n" + "="*100)
print("DISCRETE ALGORITHMS - SEPARATED BY SEED")
print("="*100)

discrete_42 = analyze_by_seed(42, 'discrete')
discrete_123 = analyze_by_seed(123, 'discrete')

# Analyze continuous for both seeds
print("\n" + "="*100)
print("CONTINUOUS ALGORITHMS - SEPARATED BY SEED")
print("="*100)

continuous_42 = analyze_by_seed(42, 'continuous')
continuous_123 = analyze_by_seed(123, 'continuous')

# Now combine both seeds and compare with analyze_results_custom.py
print("\n" + "="*100)
print("COMBINED RESULTS (Both Seeds 42 + 123)")
print("="*100)

for data_type in ['discrete', 'continuous']:
    csv_42 = Path(f"two_agent_results/{data_type}/seed_42/episode_results.csv")
    csv_123 = Path(f"two_agent_results/{data_type}/seed_123/episode_results.csv")
    
    df_42 = pd.read_csv(csv_42)
    df_123 = pd.read_csv(csv_123)
    
    combined_df = pd.concat([df_42, df_123], ignore_index=True)
    
    print(f"\n{data_type.upper()}:")
    print(f"{'Model':<30} {'Episodes':>8} {'Avg Reward':>12} {'Success Rate':>12} {'Reshuffles':>12} {'Distance':>12} {'Time Saved':>12}")
    print("-" * 100)
    
    for model in sorted(combined_df['model'].unique()):
        model_df = combined_df[combined_df['model'] == model]
        num_episodes = len(model_df)
        
        if model == 'Heuristic':
            avg_reward = "N/A"
        else:
            avg_reward = f"{model_df['total_reward'].mean():.2f}"
        
        success_rate = (model_df['success'].sum() / num_episodes) * 100
        avg_reshuffles = model_df['reshuffles_performed'].mean()
        avg_distance = model_df['total_distance_reduced'].mean()
        avg_time = model_df['total_time_saved'].mean()
        
        print(f"{model:<30} {num_episodes:>8} {avg_reward:>12} {success_rate:>11.1f}% {avg_reshuffles:>12.2f} {avg_distance:>12.4f} {avg_time:>12.4f}")

print("\n" + "="*100)
print("✅ VERIFICATION COMPLETE")
print("="*100)

