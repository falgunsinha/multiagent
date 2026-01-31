import pandas as pd
import os

folders = ['two_agent_results', 'two_agent_results_1', 'two_agent_results_2', 'two_agent_results_3']
planners = set()

for folder in folders:
    for subtype in ['discrete', 'continuous']:
        for seed in [42, 123]:
            path = f'{folder}/{subtype}/seed_{seed}/episode_results.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                planners.update(df['planner'].unique())
                print(f"{folder}/{subtype}/seed_{seed}: {df['planner'].unique()}")

print('\nAll planners found:', sorted(planners))

