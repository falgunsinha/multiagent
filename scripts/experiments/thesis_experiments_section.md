# Experiments Section - Master Thesis

## 5. Experiments

### 5.1 Experimental Setup

#### 5.1.1 Environment Configuration

**Task Description:**
The experimental task involves intelligent object selection for robotic pick-and-place operations in a cluttered workspace. The robot must select and pick objects in an optimal sequence that minimizes path length, avoids collisions, and maximizes task efficiency.

**Workspace Configuration:**
- **Grid-based workspace**: 4×4 grid layout (16 possible object positions)
- **Number of objects**: 9 cubes per episode
- **Object distribution**: Random placement on grid cells
- **Container**: Fixed placement area for picked objects
- **Obstacles**: Dynamic obstacles including unpicked cubes and workspace boundaries

**Episode Parameters:**
- **Maximum steps per episode**: 50
- **Success criterion**: All 9 cubes successfully picked and placed
- **Early termination**: Episode ends if no valid actions remain

#### 5.1.2 State Representation

The observation space consists of a 6-dimensional feature vector for each object (max 16 objects):

1. **Distance to End-Effector** (normalized): Euclidean distance from robot gripper to object
2. **Distance to Container** (normalized): Distance from object to next placement position
3. **Obstacle Proximity Score** [0-1]: Density of obstacles around object (includes unpicked cubes)
4. **Reachability Flag** {0,1}: Binary indicator if object is reachable via path planning
5. **Path Clearance Score** [0-1]: Free space along straight-line path to object
6. **Picked Flag** {0,1}: Binary indicator if object already picked

**Total observation dimension**: 16 objects × 6 features = 96-dimensional continuous state space

#### 5.1.3 Action Space

- **Type**: Discrete action space
- **Size**: 16 actions (one per grid cell)
- **Action masking**: Invalid actions masked using 3-layer reachability check:
  - **Layer 1**: Already picked objects (picked_flag = 1)
  - **Layer 2**: Objects with insufficient path clearance (clearance < 0.25)
  - **Layer 3**: Objects in highly cluttered areas (obstacle_score > 0.65)

#### 5.1.4 Reward Function

The reward function combines multiple objectives to guide learning:

**Base Rewards:**
- Successful pick: +10.0
- Episode completion: +20.0 + time_bonus (0.5 × remaining_steps)

**Path-based Rewards (max +10.0):**
- RRT path length reward: Inversely proportional to path length
- Penalty for path planning failure: -10.0 if RRT fails

**Proximity Rewards:**
- Distance to container (max +3.0): Prioritizes objects closer to placement area
- Obstacle avoidance (max +7.0): Rewards picking from less cluttered areas
- Path clearance (max +4.0): Rewards picks with clear paths

**Penalties:**
- Unreachable object: -10.0
- Risky picks (near masking thresholds): -5.0
- Time penalty: -2.0 per step
- Invalid action: -50.0

**Total reward range**: Approximately [-50, +54] per step

### 5.2 Path Planning Methods

Three path planning methods were evaluated as observation features:

#### 5.2.1 Heuristic (Baseline)
- **Method**: Euclidean distance estimation
- **Computation**: O(1) - instant
- **Accuracy**: Low (ignores obstacles)
- **Use case**: Fast baseline for comparison

#### 5.2.2 A* (Grid-based)
- **Method**: A* search on 2D occupancy grid
- **Grid resolution**: 0.05m cells
- **Heuristic**: Manhattan distance
- **Computation**: O(n log n) - fast (~1-5ms)
- **Accuracy**: Medium (2D only, no kinematic constraints)

#### 5.2.3 RRT (Sampling-based)
- **Method**: Rapidly-exploring Random Tree
- **Sampling**: 2D workspace sampling
- **Max iterations**: 500
- **Step size**: 0.1m
- **Computation**: O(n) - moderate (~10-50ms)
- **Accuracy**: High (considers obstacles and path feasibility)

**Selected Method for Comparison**: RRT Visualization (rrt_viz)
- Provides realistic path length estimates
- Balances accuracy and computational efficiency
- Suitable for both PPO and DDQN training

### 5.3 Reinforcement Learning Algorithms

#### 5.3.1 Proximal Policy Optimization (PPO)

**Algorithm**: On-policy actor-critic method with clipped surrogate objective

**Hyperparameters:**
- Learning rate: 3×10⁻⁴
- Batch size: 64
- Number of steps per update (n_steps): 2048
- Number of epochs per update: 10
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip range (ε): 0.2
- Value function coefficient: 0.5
- Entropy coefficient: 0.01

**Network Architecture:**
- Policy network: MLP [96 → 64 → 64 → 16]
- Value network: MLP [96 → 64 → 64 → 1]
- Activation: Tanh
- Action masking: Integrated via MaskablePPO (SB3-Contrib)

**Training Configuration:**
- Total timesteps: 50,000
- Checkpoint frequency: Every 10,000 steps
- Logging: TensorBoard
- Framework: Stable-Baselines3 v2.0

#### 5.3.2 Double Deep Q-Network (DDQN)

**Algorithm**: Off-policy value-based method with double Q-learning and experience replay

**Hyperparameters:**
- Learning rate: 1×10⁻³
- Batch size: 64
- Replay buffer size: 100,000
- Discount factor (γ): 0.99
- Epsilon (exploration):
  - Initial (ε_start): 1.0
  - Final (ε_end): 0.01
  - Decay type: Exponential
  - Decay rate: 2500 steps
- Target network update:
  - Method: Soft update
  - Tau (τ): 0.005 (0.5% of online network weights per update)
- Warmup steps: 1000 (random exploration before training)

**Network Architecture:**
- Q-network: MLP [96 → 128 → 128 → 16]
- Target network: Same architecture (soft-updated copy)
- Activation: ReLU
- Action masking: Applied during action selection (mask invalid Q-values to -∞)

**Training Configuration:**
- Total timesteps: 50,000
- Checkpoint frequency: Every 5,000 steps
- Logging: CSV files (training metrics and episode statistics)
- Framework: PyTorch 2.0

**Key Differences from PPO:**
- Off-policy learning (can reuse past experiences)
- Experience replay buffer for sample efficiency
- Epsilon-greedy exploration vs. stochastic policy
- Single Q-network vs. separate actor-critic
- More frequent updates (every step vs. every 2048 steps)

### 5.4 Evaluation Metrics

#### 5.4.1 Performance Metrics

1. **Average Reward (100-episode moving average)**
   - Primary performance indicator
   - Measures overall task success and efficiency
   - Higher is better

2. **Success Rate**
   - Percentage of episodes where all 9 cubes were picked
   - Binary success criterion
   - Target: ≥95%

3. **Episode Length**
   - Number of steps to complete episode
   - Lower is better (more efficient)
   - Ideal: 9 steps (one per cube)

4. **Convergence Speed**
   - Timesteps to reach reward threshold (>180)
   - Measures learning efficiency
   - Lower is better

#### 5.4.2 Learning Metrics

1. **Explained Variance** (PPO only)
   - Measures value function prediction accuracy
   - Range: [-∞, 1.0], target: >0.9
   - Formula: 1 - Var(returns - values) / Var(returns)

2. **TD Error Loss** (DDQN only)
   - Temporal difference error
   - Measures Q-value prediction accuracy
   - Lower is better

3. **Reward Stability**
   - Standard deviation of rewards
   - Lower indicates more stable learning
   - Calculated over 100-episode windows

### 5.5 Baselines

#### 5.5.1 Random Policy
- **Method**: Uniformly random action selection from valid actions
- **Purpose**: Lower bound performance baseline
- **Expected performance**: ~50-100 average reward

#### 5.5.2 Greedy Heuristic
- **Method**: Always pick nearest object to end-effector
- **Features**: Distance-based selection, no learning
- **Purpose**: Simple rule-based baseline
- **Expected performance**: ~120-150 average reward

#### 5.5.3 A*-based Heuristic
- **Method**: Pick object with shortest A* path length
- **Features**: Grid-based path planning, deterministic
- **Purpose**: Path-aware baseline
- **Expected performance**: ~150-170 average reward

**Note**: The primary comparison is between PPO and DDQN using the same RRT-based environment (rrt_viz), as both algorithms learn from the same observation space and reward structure.


