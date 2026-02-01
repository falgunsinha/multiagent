# Training Results and Analysis Guide

## 📊 **How Training Results Are Saved**

### **Directory Structure:**

```
cobotproject/scripts/Reinforcement Learning/MARL/src/gat_cvd/
├── logs/                                    ← Training logs (CSV files)
│   ├── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_training.csv
│   └── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_episodes.csv
├── models/                                  ← Model checkpoints
│   ├── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_step_5000.pt
│   ├── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_step_10000.pt
│   ├── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_final.pt
│   └── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_metadata.json
└── wandb/                                   ← W&B logs (if enabled)
    └── run-TIMESTAMP-RUNID/
```

---

## 📁 **File Types and Contents**

### **1. Training Log (Per-Step Metrics)**

**File:** `logs/gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_training.csv`

**Columns:**
```csv
step,episode,loss_ddqn,loss_masac,loss_cvd,step_reward,epsilon_ddqn,epsilon_masac,episode_reward,episode_length,avg_reward_100,success_rate
```

**Example:**
```csv
1,0,0.123456,0.056789,0.023456,8.569594,1.0,1.0,8.569594,1,0.0,0.0
2,0,0.134567,0.067890,0.034567,15.078273,0.999,0.999,23.647867,2,0.0,0.0
...
```

**What each column means:**
- `step`: Global training step (1 to 20,000)
- `episode`: Episode number
- `loss_ddqn`: DDQN training loss
- `loss_masac`: MASAC training loss
- `loss_cvd`: CVD training loss
- `step_reward`: Reward received at this step
- `epsilon_ddqn`: Exploration rate for DDQN
- `epsilon_masac`: Exploration rate for MASAC
- `episode_reward`: Cumulative reward in current episode
- `episode_length`: Steps taken in current episode
- `avg_reward_100`: Average reward over last 100 episodes
- `success_rate`: Success rate over last 100 episodes

**Use for:**
- Step-by-step training progress
- Loss curves
- Epsilon decay visualization
- Detailed debugging

---

### **2. Episode Log (Per-Episode Metrics)**

**File:** `logs/gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_episodes.csv`

**Columns:**
```csv
episode,total_reward,length,success,avg_reward_100,success_rate_100
```

**Example:**
```csv
0,153.003556,9,1.0,153.003556,1.0
1,153.054556,9,1.0,153.029056,1.0
2,143.773527,8,0.888889,149.94388,0.962963
...
```

**What each column means:**
- `episode`: Episode number
- `total_reward`: Total reward in this episode
- `length`: Number of steps in this episode
- `success`: Success rate (cubes_picked / total_cubes)
- `avg_reward_100`: Rolling average reward (last 100 episodes)
- `success_rate_100`: Rolling success rate (last 100 episodes)

**Use for:**
- Episode-level analysis
- Learning curves
- Convergence analysis
- Performance comparison

---

### **3. Model Checkpoints**

**Files:** `models/gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_step_XXXXX.pt`

**Saved every 5000 steps (configurable with `--save_freq`)**

**Contents:**
```python
{
    'ddqn_q_network': state_dict,
    'ddqn_target_network': state_dict,
    'masac_policy': state_dict,
    'masac_q1': state_dict,
    'masac_q2': state_dict,
    'masac_target_q1': state_dict,
    'masac_target_q2': state_dict,
    'cvd_module': state_dict,
    'gat_encoder': state_dict,
    'optimizers': {...},
    'total_steps': int,
    'episodes': int,
    'config': dict
}
```

**Use for:**
- Resume training
- Evaluation
- Model deployment
- Ablation studies

---

### **4. Metadata File**

**File:** `models/gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_metadata.json`

**Contents:**
```json
{
  "method": "gat_cvd_isaacsim",
  "algorithm": "gat_cvd",
  "training_grid_size": 4,
  "num_cubes": 9,
  "max_objects": 16,
  "max_steps": 50,
  "execute_picks": false,
  "timestamp": "20260123_120000",
  "total_timesteps": 20000,
  "config": {...},
  "total_episodes": 2500,
  "avg_reward_100": 145.23,
  "success_rate_100": 0.95
}
```

**Use for:**
- Experiment tracking
- Hyperparameter comparison
- Reproducibility

---

### **5. W&B Logs (Optional)**

**Directory:** `wandb/run-TIMESTAMP-RUNID/`

**Contents:**
- Real-time metrics
- Interactive plots
- System metrics (GPU, CPU, memory)
- Hyperparameter tracking
- Model artifacts

**Access:**
- Online: https://wandb.ai/your-username/gat-cvd-object-selection
- Offline: `wandb offline` mode

---

## 🔍 **What to Compare With**

### **Baseline: DDQN with RRT (Isaac Sim)**

**Your existing DDQN results are in:**
```
cobotproject/scripts/Reinforcement Learning/doubleDQN_script/
├── logs/
│   ├── ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_training.csv
│   └── ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_episodes.csv
└── models/
    ├── ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_final.pt
    └── ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_metadata.json
```

**DDQN Log Format:**
```csv
step,episode,loss,step_reward,epsilon,q_value,episode_reward,episode_length,avg_reward_100,success_rate
```

**DDQN Episode Format:**
```csv
episode,total_reward,length,success,avg_reward_100,success_rate_100
```

---

## 📈 **Comparison Metrics**

### **1. Primary Metrics (Most Important)**

| Metric | Description | Better is... | DDQN Baseline | GAT+CVD Target |
|--------|-------------|--------------|---------------|----------------|
| **Success Rate** | % of cubes successfully picked | Higher | ~95% | >95% |
| **Avg Reward (100 ep)** | Average reward over last 100 episodes | Higher | ~145 | >145 |
| **Sample Efficiency** | Steps to reach 90% success | Lower | ~10,000 | <10,000 |
| **Final Performance** | Success rate at end of training | Higher | ~95% | >95% |

### **2. Secondary Metrics**

| Metric | Description | Better is... |
|--------|-------------|--------------|
| **Episode Length** | Steps per episode | Lower (more efficient) |
| **Training Time** | Wall-clock time | Lower (but GAT+CVD will be slower) |
| **Convergence Speed** | Episodes to converge | Lower |
| **Stability** | Variance in performance | Lower |

### **3. GAT+CVD Specific Metrics**

| Metric | Description | Why Important |
|--------|-------------|---------------|
| **Loss DDQN** | DDQN training loss | Should decrease over time |
| **Loss MASAC** | MASAC training loss | Should decrease over time |
| **Loss CVD** | CVD training loss | Should decrease over time |
| **Credit Assignment** | How well CVD assigns credit | Novel contribution |

---

## 📊 **Analysis Scripts to Create**

### **1. Basic Comparison Script**

Create: `compare_ddqn_vs_gat_cvd.py`

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load DDQN results
ddqn_episodes = pd.read_csv('path/to/ddqn_episodes.csv')

# Load GAT+CVD results
gat_cvd_episodes = pd.read_csv('path/to/gat_cvd_episodes.csv')

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(ddqn_episodes['episode'], ddqn_episodes['avg_reward_100'], label='DDQN')
plt.plot(gat_cvd_episodes['episode'], gat_cvd_episodes['avg_reward_100'], label='GAT+CVD')
plt.xlabel('Episode')
plt.ylabel('Average Reward (100 ep)')
plt.legend()
plt.title('DDQN vs GAT+CVD: Learning Curves')
plt.savefig('comparison_learning_curves.png')
plt.show()
```

### **2. Multi-Metric Comparison**

Create: `comprehensive_comparison.py`

**Metrics to compare:**
- Success rate over time
- Average reward over time
- Episode length over time
- Sample efficiency (steps to 90% success)
- Final performance (last 100 episodes)

### **3. Statistical Significance Test**

Create: `statistical_test.py`

```python
from scipy import stats

# Compare final 100 episodes
ddqn_final = ddqn_episodes['total_reward'].tail(100)
gat_cvd_final = gat_cvd_episodes['total_reward'].tail(100)

# T-test
t_stat, p_value = stats.ttest_ind(ddqn_final, gat_cvd_final)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Statistically significant difference!")
else:
    print("❌ No significant difference")
```

---

## 🎯 **Recommended Comparisons for Thesis**

### **Experiment 1: Main Comparison**

**Setup:**
- DDQN (baseline): Grid 4x4, 9 cubes, 15,000 steps
- GAT+CVD (proposed): Grid 4x4, 9 cubes, 20,000 steps

**Metrics:**
- Success rate
- Average reward
- Sample efficiency
- Training time

**Expected Result:** GAT+CVD should match or exceed DDQN performance

---

### **Experiment 2: Ablation Studies**

**A. GAT vs RNN (Encoder Comparison)**
- DDQN + RNN (baseline)
- DDQN + GAT (ablation)
- GAT+CVD (full)

**B. CVD vs QMIX (Value Decomposition Comparison)**
- GAT + QMIX (ablation)
- GAT + CVD (full)

**Metrics:**
- Success rate
- Credit assignment accuracy
- Convergence speed

---

### **Experiment 3: Scalability**

**Setup:**
- Small: Grid 3x3, 4 cubes
- Medium: Grid 4x4, 6 cubes
- Large: Grid 4x4, 9 cubes

**Metrics:**
- Success rate vs problem size
- Training time vs problem size
- Sample efficiency vs problem size

**Expected Result:** GAT+CVD should scale better due to graph structure

---

## 📝 **Thesis Figures to Create**

### **Figure 1: Learning Curves**
- X-axis: Episodes
- Y-axis: Average Reward (100 ep)
- Lines: DDQN, GAT+CVD

### **Figure 2: Success Rate Over Time**
- X-axis: Episodes
- Y-axis: Success Rate (100 ep)
- Lines: DDQN, GAT+CVD

### **Figure 3: Sample Efficiency**
- X-axis: Training Steps
- Y-axis: Success Rate
- Lines: DDQN, GAT+CVD
- Highlight: Steps to 90% success

### **Figure 4: Loss Curves (GAT+CVD only)**
- X-axis: Training Steps
- Y-axis: Loss
- Lines: Loss DDQN, Loss MASAC, Loss CVD

### **Figure 5: Ablation Study**
- Bar chart comparing:
  - DDQN (baseline)
  - GAT + QMIX
  - RNN + CVD
  - GAT + CVD (full)

### **Figure 6: Scalability**
- X-axis: Problem Size (num_cubes)
- Y-axis: Success Rate
- Lines: DDQN, GAT+CVD

---

## 🔧 **Tools for Analysis**

### **Python Libraries:**
```bash
C:\isaacsim\python.bat -m pip install pandas matplotlib seaborn scipy numpy
```

### **Jupyter Notebook:**
```bash
C:\isaacsim\python.bat -m pip install jupyter
C:\isaacsim\python.bat -m jupyter notebook
```

### **W&B (Optional):**
```bash
C:\isaacsim\python.bat -m pip install wandb
```

---

## 📌 **Summary**

### **Training Results Saved:**
1. ✅ **Training log** (per-step): `logs/*_training.csv`
2. ✅ **Episode log** (per-episode): `logs/*_episodes.csv`
3. ✅ **Model checkpoints**: `models/*_step_*.pt`
4. ✅ **Metadata**: `models/*_metadata.json`
5. ✅ **W&B logs** (optional): `wandb/run-*/`

### **Compare With:**
1. ✅ **DDQN (Isaac Sim RRT)** - Main baseline
2. ✅ **Ablations** - GAT+QMIX, RNN+CVD
3. ✅ **Scalability** - Different grid sizes

### **Key Metrics:**
1. ✅ Success rate
2. ✅ Average reward
3. ✅ Sample efficiency
4. ✅ Training time
5. ✅ Credit assignment (CVD-specific)

---

## 🔧 **Analysis Scripts Created**

### **1. Main Analysis Script**

**File:** `analyze_results.py`

**Usage:**
```bash
cd "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd"

# Install dependencies
C:\isaacsim\python.bat -m pip install pandas matplotlib seaborn scipy numpy

# Run analysis
C:\isaacsim\python.bat analyze_results.py \
    --ddqn_log "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs\ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_episodes.csv" \
    --gat_cvd_log "logs\gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_episodes.csv" \
    --output_dir "analysis_results"
```

**What it does:**
- ✅ Loads DDQN and GAT+CVD episode logs
- ✅ Generates 4 comparison plots:
  1. Learning curve (average reward)
  2. Success rate over time
  3. Episode length over time
  4. Reward distribution (last 100 episodes)
- ✅ Calculates sample efficiency (episodes to 90% success)
- ✅ Performs statistical tests (t-test)
- ✅ Saves results to `analysis_results/`

**Output files:**
- `learning_curves_comparison.png` - Main comparison plots
- `sample_efficiency.png` - Sample efficiency plot
- `statistical_comparison.txt` - Statistical test results

---

### **2. Batch Script (Windows)**

**File:** `run_analysis.bat`

**Usage:**
1. Edit `run_analysis.bat` and set the correct paths:
   ```batch
   set DDQN_LOG=C:\path\to\ddqn_episodes.csv
   set GAT_CVD_LOG=C:\path\to\gat_cvd_episodes.csv
   ```

2. Double-click `run_analysis.bat` or run:
   ```bash
   run_analysis.bat
   ```

**What it does:**
- ✅ Checks if log files exist
- ✅ Installs required packages
- ✅ Runs analysis script
- ✅ Saves results to `analysis_results/`

---

## 📋 **Step-by-Step Analysis Workflow**

### **Step 1: Train Both Models**

**DDQN (Baseline):**
```bash
cd "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"
C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --timesteps 15000 --grid_size 4 --num_cubes 9
```

**GAT+CVD (Proposed):**
```bash
cd "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd"
C:\isaacsim\python.bat train_gat_cvd_isaacsim.py --timesteps 20000 --grid_size 4 --num_cubes 9
```

---

### **Step 2: Locate Log Files**

**DDQN logs:**
```
C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs\
└── ddqn_rrt_isaacsim_grid4_cubes9_TIMESTAMP_episodes.csv
```

**GAT+CVD logs:**
```
C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd\logs\
└── gat_cvd_isaacsim_grid4_cubes9_TIMESTAMP_episodes.csv
```

---

### **Step 3: Run Analysis**

**Option A: Using Python script directly**
```bash
cd "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd"

C:\isaacsim\python.bat analyze_results.py \
    --ddqn_log "..\..\..\..\doubleDQN_script\logs\ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_episodes.csv" \
    --gat_cvd_log "logs\gat_cvd_isaacsim_grid4_cubes9_20260123_120000_episodes.csv" \
    --output_dir "analysis_results"
```

**Option B: Using batch script**
1. Edit `run_analysis.bat` with correct paths
2. Run: `run_analysis.bat`

---

### **Step 4: Review Results**

**Check output directory:**
```
analysis_results/
├── learning_curves_comparison.png    ← Main comparison plots
├── sample_efficiency.png             ← Sample efficiency plot
└── statistical_comparison.txt        ← Statistical test results
```

**Example output:**
```
STATISTICAL COMPARISON
============================================================

Final Performance (Last 100 Episodes):
       Metric  DDQN Mean  DDQN Std  GAT+CVD Mean  GAT+CVD Std  Improvement (%)
  Avg Reward     145.23     12.45        152.34        10.23             4.90
Success Rate       0.95      0.08          0.97         0.06             2.11
Episode Length     8.50      1.20          8.20         1.10            -3.53

T-Test Results:
  Reward: t=3.4567, p=0.0012 ✅ Significant
  Success: t=2.1234, p=0.0345 ✅ Significant
```

---

### **Step 5: Create Thesis Figures**

**Use the generated plots:**
1. `learning_curves_comparison.png` → Figure 1 in thesis
2. `sample_efficiency.png` → Figure 2 in thesis
3. Create additional plots as needed

**Customize plots:**
Edit `analyze_results.py` to:
- Change colors, line styles
- Add annotations
- Adjust figure size
- Add more metrics

---

## 🎯 **Expected Results**

### **Scenario 1: GAT+CVD Outperforms DDQN**

**Indicators:**
- ✅ Higher final success rate (>95%)
- ✅ Higher average reward
- ✅ Faster convergence (fewer episodes to 90% success)
- ✅ Lower variance (more stable)
- ✅ Statistically significant improvement (p < 0.05)

**Thesis claim:**
> "GAT+CVD achieves X% higher success rate and converges Y% faster than DDQN baseline, demonstrating the effectiveness of graph-based spatial reasoning and explicit credit assignment."

---

### **Scenario 2: GAT+CVD Matches DDQN**

**Indicators:**
- ✅ Similar final success rate (~95%)
- ✅ Similar average reward
- ✅ Similar convergence speed
- ❌ No statistically significant difference (p > 0.05)

**Thesis claim:**
> "GAT+CVD achieves comparable performance to DDQN while providing additional benefits such as explicit credit assignment and better interpretability through attention mechanisms."

---

### **Scenario 3: GAT+CVD Underperforms DDQN (Unlikely)**

**Possible reasons:**
- Hyperparameters not tuned
- Training time too short
- Implementation issues

**Action:**
- Tune hyperparameters in `config_gat_cvd.yaml`
- Increase training time (--timesteps 30000)
- Run ablation studies to identify issues

---

## 📊 **Additional Analysis Ideas**

### **1. Attention Visualization**

Create script to visualize GAT attention weights:
```python
# Extract attention weights from trained model
attention_weights = agent.gat_encoder.get_attention_weights(graph)

# Visualize which objects the robot attends to
import networkx as nx
G = nx.Graph()
# Add nodes and edges with attention weights
# Plot with node sizes proportional to attention
```

**Thesis contribution:** Show that GAT learns to attend to relevant objects (e.g., blocking objects, target objects)

---

### **2. Credit Assignment Analysis**

Analyze CVD credit assignment:
```python
# Extract counterfactual values
V_total, V_ddqn, V_masac = agent.cvd_module.compute_counterfactual_values(graph)

# Plot contribution of each agent over time
plt.plot(episodes, V_ddqn_contributions, label='DDQN Contribution')
plt.plot(episodes, V_masac_contributions, label='MASAC Contribution')
```

**Thesis contribution:** Show that CVD correctly assigns credit to DDQN (object selection) vs MASAC (spatial manipulation)

---

### **3. Scalability Analysis**

Train on multiple problem sizes:
```bash
# Small
C:\isaacsim\python.bat train_gat_cvd_isaacsim.py --grid_size 3 --num_cubes 4 --timesteps 10000

# Medium
C:\isaacsim\python.bat train_gat_cvd_isaacsim.py --grid_size 4 --num_cubes 6 --timesteps 15000

# Large
C:\isaacsim\python.bat train_gat_cvd_isaacsim.py --grid_size 4 --num_cubes 9 --timesteps 20000
```

**Plot:** Success rate vs problem size for DDQN and GAT+CVD

**Thesis contribution:** Show that GAT+CVD scales better to larger problems

---

## 📝 **Thesis Writing Tips**

### **Results Section Structure:**

1. **Experimental Setup**
   - Environment description
   - Hyperparameters
   - Training procedure

2. **Main Results**
   - Learning curves (Figure 1)
   - Success rate comparison (Figure 2)
   - Sample efficiency (Figure 3)
   - Statistical significance

3. **Ablation Studies**
   - GAT vs RNN
   - CVD vs QMIX
   - Component contributions

4. **Qualitative Analysis**
   - Attention visualization
   - Credit assignment
   - Example episodes

5. **Discussion**
   - Why GAT+CVD works
   - Limitations
   - Future work

---

## ✅ **Summary**

### **Training Results:**
1. ✅ **Per-step log:** `logs/*_training.csv` (12 columns)
2. ✅ **Per-episode log:** `logs/*_episodes.csv` (6 columns)
3. ✅ **Checkpoints:** `models/*_step_*.pt` (every 5000 steps)
4. ✅ **Metadata:** `models/*_metadata.json`

### **Comparison Baseline:**
1. ✅ **DDQN (Isaac Sim RRT)** - Main baseline
   - Location: `doubleDQN_script/logs/ddqn_rrt_isaacsim_grid4_cubes9_*_episodes.csv`
   - Expected performance: ~95% success rate, ~145 avg reward

### **Analysis Tools:**
1. ✅ **analyze_results.py** - Main analysis script
2. ✅ **run_analysis.bat** - Batch script for Windows
3. ✅ **Output:** Plots + statistical tests

### **Key Metrics:**
1. ✅ Success rate (most important)
2. ✅ Average reward
3. ✅ Sample efficiency (episodes to 90% success)
4. ✅ Statistical significance (p-value)

---

**Ready to analyze! Train both models, then run the analysis scripts!** 📊🎓

