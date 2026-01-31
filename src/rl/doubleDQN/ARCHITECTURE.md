# Double DQN Architecture - Action Space, Observation Space & Rewards

## Action Space

**Type**: `Discrete(max_objects)`
- **Size**: Depends on grid size (e.g., 16 for 4x4 grid, 36 for 6x6 grid)
- **Actions**: Select which object to pick next (0 to max_objects-1)
- **Action Masking**: Invalid actions are masked out:
  - Already picked objects (picked flag = 1.0)
  - Non-existent objects (beyond total_objects)
  - **IMPORTANT**: Agent CANNOT select unreachable objects due to:
    - Reachability flag in observation (0.0 = unreachable)
    - Negative reward penalty (-5.0) for unreachable objects
    - Path planning failure penalty (-5.0) if A*/RRT cannot find path

## Observation Space

**Type**: `Box(shape=(max_objects × 6,), dtype=float32)`

Each object has **6 features**:

### 1. Distance to End-Effector (EE)
- **Range**: [0.0, ∞)
- **Calculation**: `np.linalg.norm(object_position - ee_position)`
- **Purpose**: Measures how far the object is from robot's current position

### 2. Distance to Container
- **Range**: [0.0, ∞)
- **Calculation**: `np.linalg.norm(object_position - container_position)`
- **Purpose**: Prioritizes objects closer to container (less travel distance after pick)

### 3. Obstacle Proximity Score
- **Range**: [0.0, 1.0]
- **Calculation**: 
  ```python
  min_distance = min(distance to all obstacles + unpicked cubes)
  if min_distance < 0.10m: score = 1.0  # Very close
  elif min_distance > 0.30m: score = 0.0  # Far away
  else: score = 1.0 - (min_distance - 0.10) / 0.20  # Linear interpolation
  ```
- **Obstacles Include**:
  - Random external obstacles (red cubes)
  - **Other unpicked cubes** (except target itself)
- **Purpose**: Penalizes objects surrounded by obstacles

### 4. Reachability Flag
- **Range**: [0.0, 1.0]
- **Calculation**:
  - **Heuristic**: `1.0 if 0.3m ≤ distance ≤ 0.9m else 0.0`
  - **A* Method**: `1.0 if A* finds path else 0.0`
  - **RRT Method**: `1.0 if RRT finds collision-free path else 0.0`
- **Purpose**: **PREVENTS SELECTING UNREACHABLE OBJECTS**
  - A* checks grid-based pathfinding with obstacles
  - RRT checks actual robot kinematics and collision avoidance
  - **Unreachable objects get 0.0 flag → Agent learns to avoid them**

### 5. Path Clearance Score
- **Range**: [0.0, 1.0]
- **Calculation**: 
  ```python
  # Sample 5 points along straight-line path from EE to object
  min_clearance = min(distance to obstacles at each sample point)
  score = min(min_clearance / 0.3, 1.0)  # 0cm = 0.0, 30cm+ = 1.0
  ```
- **Purpose**: Measures free space around path (higher = more clearance)

### 6. Picked Flag
- **Range**: [0.0, 1.0]
- **Calculation**: `1.0 if object already picked else 0.0`
- **Purpose**: Prevents re-selecting picked objects (also enforced by action masking)

## Reward Function

### Base Reward Components

#### 1. Path Length Reward (max 5 points)
- **A* Method**: Uses A* path length
- **RRT Method**: Uses RRT path length
- **Heuristic Method**: Uses Euclidean distance
- **Calculation**:
  ```python
  normalized_path = (path_length - 0.3) / 0.6  # Normalize to [0, 1]
  path_reward = 5.0 * (1.0 - normalized_path)  # Shorter path = higher reward
  ```

#### 2. Container Proximity Reward (max 3 points)
- **Calculation**: `3.0 * exp(-distance_to_container)`
- **Purpose**: Prioritizes objects closer to container

#### 3. Obstacle Avoidance Reward (max 3 points)
- **Calculation**: `3.0 * (1.0 - obstacle_proximity_score)`
- **Purpose**: Rewards picking objects far from obstacles
- **Includes**: External obstacles + other unpicked cubes

#### 4. Reachability Penalty (-5 points if unreachable)
- **Calculation**: `-5.0 if reachability_flag == 0.0`
- **Purpose**: **STRONG PENALTY FOR UNREACHABLE OBJECTS**
- **Effect**: Agent learns to avoid objects blocked by obstacles

#### 5. Path Clearance Reward (max 2 points)
- **Calculation**: `2.0 * path_clearance_score`
- **Purpose**: Rewards paths with more free space

#### 6. Time Penalty (-1 per step)
- **Calculation**: `-1.0` per action
- **Purpose**: Encourages faster completion

### Additional Rewards/Penalties

#### Path Planning Failure Penalty (-5 points)
- **Trigger**: When A*/RRT cannot find path (returns 2.0 × Euclidean distance)
- **Calculation**: `-5.0`
- **Purpose**: **PREVENTS SELECTING OBJECTS BLOCKED BY OBSTACLES**

#### First Pick Bonus (+5 points)
- **Trigger**: First object picked has shortest path among all objects
- **Calculation**: `+5.0 if action == argmin(all_path_lengths)`
- **Purpose**: Rewards optimal first selection

#### Completion Bonus (+20 + time bonus)
- **Trigger**: All objects successfully picked
- **Calculation**: `+20.0 + max(0, max_steps - current_step)`
- **Purpose**: Rewards successful completion and speed

#### Invalid Action Penalty (-10 points)
- **Trigger**: Selecting already-picked object (should not happen with action masking)
- **Calculation**: `-10.0`

### Total Reward Range
- **Typical range**: -15 to +35 per action
- **Episode range**: Varies based on number of objects and efficiency

## How Agent Avoids Unreachable Objects

The Double DQN agent learns to avoid unreachable objects through:

1. **Observation Signal**: Reachability flag (0.0 for unreachable objects)
2. **Reward Penalty**: -5.0 penalty for selecting unreachable objects
3. **Path Planning Penalty**: -5.0 penalty when A*/RRT fails to find path
4. **Obstacle Proximity**: High obstacle score (close to 1.0) for blocked objects
5. **Path Clearance**: Low clearance score (close to 0.0) for blocked paths

**Combined Effect**: Unreachable objects receive **-10.0 penalty** (reachability + path failure), making them highly undesirable. The agent learns through experience replay to avoid these objects.

## Obstacle Handling

### External Obstacles
- Red fixed cubes placed randomly in empty grid cells
- Number varies by grid size: 3x3→1, 4x4→2, 6x6→3-5
- Treated as static obstacles in path planning

### Other Unpicked Cubes as Obstacles
- **CRITICAL**: All unpicked cubes (except target) are treated as obstacles
- Updated dynamically as cubes are picked
- Included in:
  - Obstacle proximity score calculation
  - A* grid obstacle map
  - RRT collision checking
  - Path clearance calculation

### Dynamic Updates
- A* grid updated before each reachability/reward calculation
- Target cube excluded from obstacles when planning path TO it
- Picked cubes removed from obstacle list
- Ensures accurate reachability assessment at each step

