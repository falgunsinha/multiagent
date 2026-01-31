# GAT+CVD Test Results Summary

## Test Configuration
- **Seeds**: 42, 123
- **Episodes per model per seed**: 20
- **Total models tested**: 8
- **Grid size**: 4x4
- **Number of cubes**: 9
- **Planner**: RRT

## Models Tested
1. **DDQN + mGAT** (DDQN+GAT+CVD) - Primary contribution
2. **Heuristic** - Baseline
3. **Duel-DDQN + SAC** - Pretrained on LunarLander-v2
4. **PER-DDQN-Full + SAC** - Pretrained on LunarLander-v2
5. **PER-DDQN-Light + SAC** - Pretrained on LunarLander-v2
6. **C51-DDQN + SAC** - Pretrained on LunarLander-v2
7. **PPO-Discrete + SAC** - Pretrained on CartPole-v1
8. **SAC-Discrete + SAC** - Pretrained on LunarLander-v2

## Metrics Comparison Table (Mean ± Std)

| Model | Reward | Pick Success (%) | Distance Reduced (m) | Time Saved (s) | Distance Efficiency (%) | Time Efficiency (%) |
|-------|--------|------------------|----------------------|----------------|-------------------------|---------------------|
| DDQN + mGAT | -289.46 ± 130.18 | 68.89 ± 8.64 | 0.0903 ± 0.0539 | 0.1807 ± 0.1077 | 1.19 ± 0.36 | 0.98 ± 0.59 |
| Heuristic | N/A | **98.06 ± 0.39** | 0.2149 ± 0.0138 | 0.4297 ± 0.0275 | **5.85 ± 0.35** | **9.62 ± 5.01** |
| Duel-DDQN + SAC | -96.63 ± 86.09 | 73.33 ± 3.93 | 0.0609 ± 0.0234 | 0.1217 ± 0.0468 | 1.25 ± 0.81 | 3.09 ± 2.50 |
| PER-DDQN-Full + SAC | 74.89 ± 129.35 | 80.56 ± 8.64 | 0.0311 ± 0.1329 | 0.0622 ± 0.2659 | 0.57 ± 1.96 | 1.69 ± 0.27 |
| PER-DDQN-Light + SAC | **136.32 ± 44.88** | 85.28 ± 2.75 | 0.0486 ± 0.0170 | 0.0971 ± 0.0340 | 1.25 ± 0.08 | 2.81 ± 2.11 |
| C51-DDQN + SAC | -107.72 ± 201.06 | 79.72 ± 5.89 | **0.2712 ± 0.0310** | **0.5424 ± 0.0620** | 4.12 ± 0.07 | 7.18 ± 0.97 |
| PPO-Discrete + SAC | -105.24 ± 253.10 | 75.00 ± 12.57 | 0.0425 ± 0.0972 | 0.0850 ± 0.1944 | -0.03 ± 0.19 | 2.03 ± 1.14 |
| SAC-Discrete + SAC | -66.42 ± 60.05 | 70.56 ± 7.07 | -0.0469 ± 0.1375 | -0.0938 ± 0.2750 | -0.95 ± 1.41 | 1.26 ± 0.37 |

**Bold** values indicate the best performance for each metric (excluding Heuristic for Reward).

## Key Findings

### Best Performers
- **Best Reward**: PER-DDQN-Light + SAC (136.32 ± 44.88)
- **Best Pick Success**: Heuristic (98.06 ± 0.39%)
- **Best Distance Reduced**: C51-DDQN + SAC (0.2712 ± 0.0310 m)
- **Best Time Saved**: C51-DDQN + SAC (0.5424 ± 0.0620 s)
- **Best Distance Efficiency**: Heuristic (5.85 ± 0.35%)
- **Best Time Efficiency**: Heuristic (9.62 ± 5.01%)

### DDQN + mGAT Performance
- Pick Success: 68.89% (lower than expected, needs investigation)
- Reward: -289.46 (negative, indicating issues with task completion)
- Distance/Time Efficiency: Lower than Heuristic baseline

### Observations
1. **Heuristic baseline** performs exceptionally well in pick success and efficiency metrics
2. **PER-DDQN-Light + SAC** achieves the best reward among learned models
3. **C51-DDQN + SAC** shows strong performance in distance and time savings
4. **DDQN + mGAT** underperforms compared to expectations - may need:
   - More training episodes
   - Hyperparameter tuning
   - Investigation of failure modes (timeout episodes)

## Generated Graphs

All graphs are saved in: `cobotproject/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/`

1. **reward_by_episode.png** - Bar graph comparing DDQN+mGAT, Duel-DDQN+SAC, PER-DDQN-Full+SAC
2. **pick_success_by_episode.png** - Line graph: DDQN+mGAT vs Heuristic
3. **distance_reduced_by_episode.png** - Line graph: DDQN+mGAT vs Heuristic
4. **distance_efficiency_by_episode.png** - Line graph: DDQN+mGAT vs Heuristic
5. **time_saved_by_episode.png** - Line graph: DDQN+mGAT vs Heuristic
6. **time_efficiency_by_episode.png** - Line graph: DDQN+mGAT vs Heuristic

## LaTeX Table Code

```latex
\begin{table}[h]
\centering
\caption{Discrete Algorithm Performance - Mean ± Std across Seeds 42 and 123}
\label{tab:gat_cvd_discrete_results}
\begin{tabular}{lcccccc}
\toprule
Model & Reward & Pick Success (\%) & Distance Reduced (m) & Time Saved (s) & Distance Efficiency (\%) & Time Efficiency (\%) \\
\midrule
DDQN + mGAT & -289.46 $\pm$ 130.18 & 68.89 $\pm$ 8.64 & 0.0903 $\pm$ 0.0539 & 0.1807 $\pm$ 0.1077 & 1.19 $\pm$ 0.36 & 0.98 $\pm$ 0.59 \\
Heuristic & N/A & \textbf{98.06 $\pm$ 0.39} & 0.2149 $\pm$ 0.0138 & 0.4297 $\pm$ 0.0275 & \textbf{5.85 $\pm$ 0.35} & \textbf{9.62 $\pm$ 5.01} \\
Duel-DDQN + SAC & -96.63 $\pm$ 86.09 & 73.33 $\pm$ 3.93 & 0.0609 $\pm$ 0.0234 & 0.1217 $\pm$ 0.0468 & 1.25 $\pm$ 0.81 & 3.09 $\pm$ 2.50 \\
PER-DDQN-Full + SAC & 74.89 $\pm$ 129.35 & 80.56 $\pm$ 8.64 & 0.0311 $\pm$ 0.1329 & 0.0622 $\pm$ 0.2659 & 0.57 $\pm$ 1.96 & 1.69 $\pm$ 0.27 \\
PER-DDQN-Light + SAC & \textbf{136.32 $\pm$ 44.88} & 85.28 $\pm$ 2.75 & 0.0486 $\pm$ 0.0170 & 0.0971 $\pm$ 0.0340 & 1.25 $\pm$ 0.08 & 2.81 $\pm$ 2.11 \\
C51-DDQN + SAC & -107.72 $\pm$ 201.06 & 79.72 $\pm$ 5.89 & \textbf{0.2712 $\pm$ 0.0310} & \textbf{0.5424 $\pm$ 0.0620} & 4.12 $\pm$ 0.07 & 7.18 $\pm$ 0.97 \\
PPO-Discrete + SAC & -105.24 $\pm$ 253.10 & 75.00 $\pm$ 12.57 & 0.0425 $\pm$ 0.0972 & 0.0850 $\pm$ 0.1944 & -0.03 $\pm$ 0.19 & 2.03 $\pm$ 1.14 \\
SAC-Discrete + SAC & -66.42 $\pm$ 60.05 & 70.56 $\pm$ 7.07 & -0.0469 $\pm$ 0.1375 & -0.0938 $\pm$ 0.2750 & -0.95 $\pm$ 1.41 & 1.26 $\pm$ 0.37 \\
\bottomrule
\end{tabular}
\end{table}
```

## Scripts Used
- `analyze_gat_cvd_results.py` - Generates metrics comparison table
- `plot_gat_cvd_graphs.py` - Generates all graphs

## Next Steps
1. Review the graphs to understand performance trends
2. Investigate why DDQN+mGAT has lower pick success than expected
3. Analyze timeout episodes to identify failure patterns
4. Consider which 2 additional models to include in the reward bar graph (currently using Duel-DDQN+SAC and PER-DDQN-Full+SAC)

