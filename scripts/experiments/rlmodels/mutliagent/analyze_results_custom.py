"""
Analyze two-agent system test results
Shows: avg total reward, avg success rate, avg reshuffles, avg distance reduced, avg time saved
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results(seeds=[42, 123]):
    """Analyze results from both continuous and discrete models across multiple seeds"""

    all_results = []

    # Load results for each seed
    for seed in seeds:
        # Define paths for this seed
        continuous_csv = Path(f"two_agent_results_3/continuous/seed_{seed}/episode_results.csv")
        discrete_csv = Path(f"two_agent_results_3/discrete/seed_{seed}/episode_results.csv")

        # Load continuous results
        if continuous_csv.exists():
            df_cont = pd.read_csv(continuous_csv)
            print(f"✅ Loaded {len(df_cont)} continuous episodes from seed {seed}")
            all_results.append(df_cont)
        else:
            print(f"⚠️  Continuous results not found for seed {seed}: {continuous_csv}")

        # Load discrete results
        if discrete_csv.exists():
            df_disc = pd.read_csv(discrete_csv)
            print(f"✅ Loaded {len(df_disc)} discrete episodes from seed {seed}")
            all_results.append(df_disc)
        else:
            print(f"⚠️  Discrete results not found for seed {seed}: {discrete_csv}")
    
    if not all_results:
        print("❌ No results found!")
        return
    
    # Combine all results
    df = pd.concat(all_results, ignore_index=True)
    
    print(f"\n{'='*120}")
    print(f"TWO-AGENT SYSTEM RESULTS ANALYSIS")
    print(f"{'='*120}")
    print(f"Total episodes: {len(df)}")
    print(f"Models tested: {df['model'].nunique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Episodes per model per seed: {len(df) // (df['model'].nunique() * len(df['seed'].unique()))}")
    print(f"{'='*120}\n")
    
    # Group by model and calculate metrics
    results = []
    
    for model_name in sorted(df['model'].unique()):
        model_df = df[df['model'] == model_name]
        num_episodes = len(model_df)
        
        # Calculate metrics
        # Handle Heuristic model (has empty/NaN for rewards)
        if model_name == "Heuristic":
            avg_total_reward = "N/A"
        else:
            avg_total_reward = model_df['total_reward'].mean()
        
        # Success rate: count True values and convert to percentage
        avg_success_rate = (model_df['success'].sum() / num_episodes) * 100
        
        # Other metrics
        avg_reshuffles = model_df['reshuffles_performed'].mean()
        avg_distance_reduced = model_df['total_distance_reduced'].mean()
        avg_time_saved = model_df['total_time_saved'].mean()
        
        results.append({
            'Model': model_name,
            'Episodes': num_episodes,
            'Avg Total Reward': avg_total_reward,
            'Avg Success Rate (%)': avg_success_rate,
            'Avg Reshuffles': avg_reshuffles,
            'Avg Distance Reduced': avg_distance_reduced,
            'Avg Time Saved': avg_time_saved
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by success rate (descending), then by total reward (descending)
    # For sorting, convert "N/A" to -inf
    results_df['_sort_reward'] = results_df['Avg Total Reward'].apply(
        lambda x: x if isinstance(x, (int, float)) else -np.inf
    )
    results_df = results_df.sort_values(['Avg Success Rate (%)', '_sort_reward'], 
                                        ascending=[False, False])
    results_df = results_df.drop('_sort_reward', axis=1)
    
    # Display results
    print("\n" + "="*120)
    print("SUMMARY RESULTS (Model-wise from Episode Files)")
    print("="*120)
    print()
    
    # Format the display
    print(f"{'Model':<30} {'Episodes':>8} {'Avg Total Reward':>18} {'Avg Success Rate':>18} {'Avg Reshuffles':>15} {'Avg Distance Reduced':>22} {'Avg Time Saved':>15}")
    print("-" * 120)
    
    for _, row in results_df.iterrows():
        model = row['Model']
        episodes = row['Episodes']
        
        if row['Avg Total Reward'] == "N/A":
            reward_str = "N/A"
        else:
            reward_str = f"{row['Avg Total Reward']:.2f}"
        
        success_str = f"{row['Avg Success Rate (%)']:.1f}%"
        reshuffles_str = f"{row['Avg Reshuffles']:.2f}"
        distance_str = f"{row['Avg Distance Reduced']:.4f}"
        time_str = f"{row['Avg Time Saved']:.4f}"
        
        print(f"{model:<30} {episodes:>8} {reward_str:>18} {success_str:>18} {reshuffles_str:>15} {distance_str:>22} {time_str:>15}")
    
    print("=" * 120)
    
    # Save to CSV
    output_file = Path("two_agent_results_3/analysis_summary_custom.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved to: {output_file}")
    
    # Print top performers
    print("\n" + "="*120)
    print("TOP PERFORMERS")
    print("="*120)
    
    # Best by success rate
    best_success_idx = results_df['Avg Success Rate (%)'].idxmax()
    best_success = results_df.loc[best_success_idx]
    print(f"\n🏆 Best Success Rate: {best_success['Model']} ({best_success['Avg Success Rate (%)']:.1f}%)")
    
    # Best by total reward (excluding Heuristic/N/A)
    reward_df = results_df[results_df['Avg Total Reward'] != "N/A"].copy()
    if not reward_df.empty:
        best_reward_idx = reward_df['Avg Total Reward'].idxmax()
        best_reward = reward_df.loc[best_reward_idx]
        print(f"🏆 Best Total Reward: {best_reward['Model']} ({best_reward['Avg Total Reward']:.2f})")
    
    # Best by time saved
    best_time_idx = results_df['Avg Time Saved'].idxmax()
    best_time = results_df.loc[best_time_idx]
    print(f"🏆 Best Time Saved: {best_time['Model']} ({best_time['Avg Time Saved']:.4f}s)")
    
    # Best by distance reduced
    best_dist_idx = results_df['Avg Distance Reduced'].idxmax()
    best_dist = results_df.loc[best_dist_idx]
    print(f"🏆 Best Distance Reduced: {best_dist['Model']} ({best_dist['Avg Distance Reduced']:.4f})")
    
    print("\n" + "="*120)

if __name__ == "__main__":
    import sys

    # Check if seeds are provided as arguments
    if len(sys.argv) > 1:
        seeds = [int(s) for s in sys.argv[1:]]
        print(f"Analyzing seeds: {seeds}")
        analyze_results(seeds)
    else:
        # Default: try both seeds, but continue if one is missing
        print("Analyzing default seeds: [42, 123]")
        analyze_results([42, 123])

