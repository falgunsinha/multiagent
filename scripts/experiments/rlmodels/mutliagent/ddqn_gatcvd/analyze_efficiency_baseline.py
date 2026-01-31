"""
Analyze what baseline distance and time should be used for efficiency calculations
"""
import pandas as pd
import numpy as np

# Load filtered data
df = pd.read_csv('cobotproject/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/filtered_combined_results.csv')

print("="*120)
print("EFFICIENCY BASELINE ANALYSIS")
print("="*120)

# Check if baseline columns exist
print("\nAvailable columns:")
print(df.columns.tolist())

# Check for baseline-related columns
baseline_cols = [col for col in df.columns if 'baseline' in col.lower() or 'without' in col.lower() or 'initial' in col.lower()]
print(f"\nBaseline-related columns: {baseline_cols}")

print("\n" + "="*120)
print("APPROACH 1: Use Heuristic as Baseline (Common in RL)")
print("="*120)

heuristic = df[df['model'] == 'Heuristic']
print(f"\nHeuristic Performance:")
print(f"  Episodes: {len(heuristic)}")
print(f"  Avg Distance Traveled: {heuristic['total_distance_traveled'].mean():.4f} m")
print(f"  Avg Time Taken: {heuristic['total_time_taken'].mean():.4f} s")
print(f"  Avg Cubes Picked: {heuristic['cubes_picked'].mean():.2f}")

print("\nEfficiency Formula (if using Heuristic as baseline):")
print("  Distance Efficiency = (Heuristic_Distance - Model_Distance) / Heuristic_Distance * 100")
print("  Time Efficiency = (Heuristic_Time - Model_Time) / Heuristic_Time * 100")
print("\nInterpretation:")
print("  Positive % = Model is MORE efficient (uses less distance/time)")
print("  Negative % = Model is LESS efficient (uses more distance/time)")

print("\n" + "="*120)
print("APPROACH 2: Check if 'distance_reduced' and 'time_saved' columns exist")
print("="*120)

if 'total_distance_reduced' in df.columns and 'total_time_saved' in df.columns:
    print("\n✓ Found 'total_distance_reduced' and 'total_time_saved' columns!")
    print("\nThese columns likely represent:")
    print("  distance_reduced = baseline_distance - actual_distance")
    print("  time_saved = baseline_time - actual_time")
    
    print("\nSample data for DDQN+GAT:")
    ddqn_gat = df[df['model'] == 'DDQN+GAT'].head(5)
    print(ddqn_gat[['episode', 'total_distance_traveled', 'total_distance_reduced', 'total_time_taken', 'total_time_saved']])
    
    print("\nTo find baseline:")
    print("  baseline_distance = total_distance_traveled + total_distance_reduced")
    print("  baseline_time = total_time_taken + total_time_saved")
    
    # Calculate baseline
    ddqn_gat_sample = df[df['model'] == 'DDQN+GAT'].iloc[0]
    baseline_dist = ddqn_gat_sample['total_distance_traveled'] + ddqn_gat_sample['total_distance_reduced']
    baseline_time = ddqn_gat_sample['total_time_taken'] + ddqn_gat_sample['total_time_saved']
    
    print(f"\nExample calculation (Episode {ddqn_gat_sample['episode']}):")
    print(f"  Baseline Distance = {ddqn_gat_sample['total_distance_traveled']:.4f} + {ddqn_gat_sample['total_distance_reduced']:.4f} = {baseline_dist:.4f} m")
    print(f"  Baseline Time = {ddqn_gat_sample['total_time_taken']:.4f} + {ddqn_gat_sample['total_time_saved']:.4f} = {baseline_time:.4f} s")
    
    # Check if baseline is consistent across episodes
    df['baseline_distance'] = df['total_distance_traveled'] + df['total_distance_reduced']
    df['baseline_time'] = df['total_time_taken'] + df['total_time_saved']
    
    print("\n" + "="*120)
    print("BASELINE CONSISTENCY CHECK")
    print("="*120)
    
    for model in ['DDQN+GAT', 'Heuristic', 'C51-DDQN+SAC']:
        model_df = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  Baseline Distance: Mean={model_df['baseline_distance'].mean():.4f}, Std={model_df['baseline_distance'].std():.4f}")
        print(f"  Baseline Time: Mean={model_df['baseline_time'].mean():.4f}, Std={model_df['baseline_time'].std():.4f}")
    
    print("\n" + "="*120)
    print("RECOMMENDATION")
    print("="*120)
    print("\nIf baseline is consistent across all models and episodes:")
    print("  → Use the common baseline values")
    print("\nIf baseline varies by episode:")
    print("  → Baseline represents 'no reshuffling' scenario for each specific episode")
    print("  → Each episode has its own baseline based on initial cube positions")
    
else:
    print("\n✗ 'total_distance_reduced' and 'total_time_saved' columns NOT found")
    print("\nRecommendation: Use Heuristic as baseline")

print("\n" + "="*120)
print("APPROACH 3: Theoretical Minimum (Best Case)")
print("="*120)

print("\nFind the model with:")
print("  - Minimum distance traveled (most efficient path)")
print("  - Minimum time taken (fastest completion)")

models_summary = []
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    models_summary.append({
        'Model': model,
        'Avg_Distance': model_df['total_distance_traveled'].mean(),
        'Avg_Time': model_df['total_time_taken'].mean(),
        'Avg_Cubes': model_df['cubes_picked'].mean()
    })

summary_df = pd.DataFrame(models_summary).sort_values('Avg_Distance')
print("\nModels sorted by Average Distance:")
print(summary_df[['Model', 'Avg_Distance', 'Avg_Time', 'Avg_Cubes']].to_string(index=False))

print("\n" + "="*120)
print("FINAL RECOMMENDATIONS")
print("="*120)
print("""
Option 1: Use existing 'distance_reduced' and 'time_saved' columns
  - If these columns exist, they already contain the efficiency metrics
  - Efficiency % = (reduced/saved) / baseline * 100
  - Need to determine what the baseline is (check if it's consistent)

Option 2: Use Heuristic as baseline (Common in RL research)
  - Distance Efficiency = (Heuristic_Dist - Model_Dist) / Heuristic_Dist * 100
  - Time Efficiency = (Heuristic_Time - Model_Time) / Heuristic_Time * 100
  - Pros: Heuristic is a well-defined, deterministic baseline
  - Cons: Heuristic may not be optimal

Option 3: Use theoretical minimum (Best performing model)
  - Find model with minimum distance/time among successful episodes
  - Use as baseline for comparison
  - Pros: Represents achievable best performance
  - Cons: May not be fair if that model has low success rate

RECOMMENDED: Check if 'distance_reduced' and 'time_saved' represent savings
compared to a 'no reshuffling' baseline. If yes, use those directly.
Otherwise, use Heuristic as baseline for fair comparison.
""")

