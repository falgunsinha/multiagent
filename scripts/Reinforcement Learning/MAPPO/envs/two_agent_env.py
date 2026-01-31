"""
Two-Agent Environment Wrapper

Integrates:
- Agent 1 (DDQN): Selects optimal pick sequence
- Agent 2 (MAPPO): Decides when and how to reshuffle cubes

This wrapper coordinates both agents and provides a unified interface for training.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rl.doubleDQN import DoubleDQNAgent

# Import reshuffling modules using absolute path to avoid import issues
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get MAPPO envs directory
mappo_envs = project_root / "scripts" / "Reinforcement Learning" / "MAPPO" / "envs"

# Import reshuffling modules
reshuffling_decision_module = import_module_from_path(
    "reshuffling_decision",
    mappo_envs / "reshuffling_decision.py"
)
ReshufflingDecisionModule = reshuffling_decision_module.ReshufflingDecisionModule
ReshuffleDecision = reshuffling_decision_module.ReshuffleDecision
ReshuffleReason = reshuffling_decision_module.ReshuffleReason

reshuffling_action_module = import_module_from_path(
    "reshuffling_action_space",
    mappo_envs / "reshuffling_action_space.py"
)
ReshufflingActionSpace = reshuffling_action_module.ReshufflingActionSpace
ReshuffleAction = reshuffling_action_module.ReshuffleAction


class TwoAgentEnv:
    """
    Two-agent environment for cube manipulation with reshuffling.
    
    Agent 1 (DDQN): Selects next cube to pick
    Agent 2 (MAPPO): Decides when to reshuffle and where to move cubes
    
    The environment alternates between:
    1. Agent 1 selects next cube
    2. Agent 2 checks if reshuffling is needed
    3. If reshuffling: Agent 2 moves cubes
    4. Agent 1 executes pick
    5. Repeat
    """
    
    def __init__(
        self,
        base_env,  # Base environment (ObjectSelectionEnvRRT, ObjectSelectionEnvAStar, etc.)
        ddqn_agent: DoubleDQNAgent,
        grid_size: int = 4,
        num_cubes: int = 9,
        max_reshuffles_per_episode: int = 5,
        reshuffle_reward_scale: float = 1.0,
        max_episode_steps: int = 50,  # NEW: Prevent infinite loops
        verbose: bool = False,  # NEW: Control logging verbosity
    ):
        """
        Initialize two-agent environment.

        Args:
            base_env: Base RL environment (provides cube positions, RRT, etc.)
            ddqn_agent: Pre-trained DDQN agent for pick sequence
            grid_size: Grid size
            num_cubes: Number of cubes
            max_reshuffles_per_episode: Maximum reshuffles allowed per episode
            reshuffle_reward_scale: Scaling factor for reshuffle rewards
            max_episode_steps: Maximum steps per episode (prevents infinite loops from invalid actions)
        """
        self.base_env = base_env
        self.ddqn_agent = ddqn_agent
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.max_reshuffles_per_episode = max_reshuffles_per_episode
        self.reshuffle_reward_scale = reshuffle_reward_scale
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose  # Control logging verbosity

        # Initialize reshuffling modules
        self.reshuffle_decision = ReshufflingDecisionModule()
        # CRITICAL: cube_spacing MUST match the base environment's spacing!
        # A* env uses: 0.26 for grid>3, 0.28 for grid<=3
        # RRT env uses: 0.13 for grid>3, 0.15 for grid<=3
        # Detect which environment we're using
        env_type = type(base_env).__name__
        if "AStar" in env_type:
            cube_spacing = 0.26 if grid_size > 3 else 0.28
        else:  # RRT or other
            cube_spacing = 0.13 if grid_size > 3 else 0.15

        self.reshuffle_action_space = ReshufflingActionSpace(
            grid_size=grid_size,
            num_cubes=num_cubes,
            cube_spacing=cube_spacing
        )
        
        # Episode tracking
        self.reshuffles_performed = 0
        self.reshuffle_history = []  # List of (reason, cube_idx, target_pos)
        self.total_reshuffle_reward = 0.0
        self.total_pick_reward = 0.0  # NEW: Track total Agent 1 (pick) reward
        self.episode_steps = 0  # NEW: Track steps to prevent infinite loops
        self.total_distance_reduced = 0.0  # NEW: Track total distance improvement from reshuffling
        self.total_time_saved = 0.0  # NEW: Track estimated time saved from reshuffling
        self.reshuffle_count_per_cube = {}  # NEW: Track reshuffles per cube (max 2 per cube)

        # NEW: Track actual distance/time for efficiency calculation
        self.total_distance_traveled = 0.0  # Actual robot movement distance
        self.total_time_taken = 0.0  # Actual time taken (episode duration)
        self.episode_start_time = None  # Track episode start time
        
        # Agent 2 (MAPPO) observation space
        # Observation includes:
        # - Cube positions (num_cubes * 3)
        # - Robot position (3)
        # - Picked cubes mask (num_cubes)
        # - Grid cell distances to robot (grid_size * grid_size) - NEW!
        # - Reshuffling decision features (10)
        self.agent2_obs_dim = num_cubes * 3 + 3 + num_cubes + (grid_size * grid_size) + 10
        
        # Agent 2 action space
        self.agent2_action_dim = self.reshuffle_action_space.action_dim
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Returns:
            observation: Initial observation for Agent 2
            info: Additional information
        """
        # Reset base environment
        base_obs, base_info = self.base_env.reset()
        
        # Reset episode tracking
        self.reshuffles_performed = 0
        self.reshuffle_history = []
        self.total_reshuffle_reward = 0.0
        self.total_pick_reward = 0.0  # NEW: Reset Agent 1 reward
        self.episode_steps = 0  # NEW: Reset step counter
        self.total_distance_reduced = 0.0  # NEW: Reset distance tracking
        self.total_time_saved = 0.0  # NEW: Reset time tracking
        self.reshuffle_count_per_cube = {i: 0 for i in range(self.num_cubes)}  # NEW: Reset reshuffle counts
        self.reshuffle_decision.reset()

        # NEW: Reset actual distance/time tracking
        self.total_distance_traveled = 0.0
        self.total_time_taken = 0.0
        import time
        self.episode_start_time = time.time()
        
        # Get initial observation for Agent 2
        agent2_obs = self._get_agent2_observation()
        
        info = {
            **base_info,
            "reshuffles_performed": 0,
            "reshuffle_history": [],
            "cubes_picked": 0,  # No cubes picked at start
        }
        
        return agent2_obs, info
    
    def step(self, agent2_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Flow:
        1. Agent 1 (DDQN) selects next cube
        2. Check if reshuffling is needed
        3. If yes: Execute Agent 2 (MAPPO) reshuffling action
        4. Execute Agent 1 pick action
        5. Return rewards and next state
        
        Args:
            agent2_action: Reshuffling action from Agent 2 (MAPPO)
            
        Returns:
            observation: Next observation for Agent 2
            reward: Reward for Agent 2
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Additional information
        """
        # NEW: Increment step counter
        self.episode_steps += 1

        # NEW: Check for timeout (prevents infinite loops from invalid actions)
        if self.episode_steps >= self.max_episode_steps:
            # Episode timeout - return immediately
            next_agent2_obs = self._get_agent2_observation()
            info = {
                "reshuffles_performed": self.reshuffles_performed,
                "reshuffle_history": self.reshuffle_history,
                "reshuffled_this_step": False,
                "reshuffle_reason": "none",
                "agent1_action": -1,
                "agent2_action": agent2_action,
                "pick_reward": 0.0,
                "reshuffle_reward": 0.0,
                "total_pick_reward": self.total_pick_reward,  # NEW: Total Agent 1 reward
                "total_reshuffle_reward": self.total_reshuffle_reward,  # NEW: Total Agent 2 reward
                "timeout": True,
                "total_distance_reduced": self.total_distance_reduced,
                "total_time_saved": self.total_time_saved,
                "cubes_picked": len(self.base_env.objects_picked),  # Number of cubes picked before timeout
            }
            return next_agent2_obs, 0.0, False, True, info  # truncated=True

        # STEP 1: Agent 1 selects next cube
        base_obs = self._get_base_observation()
        # OPTIMIZED: Skip expensive reachability checks during testing
        # DDQN agent already learned to avoid unreachable/unsafe cubes during training
        action_mask = self.base_env.action_masks(skip_reachability_check=True)

        # DEBUG: Print action mask
        if self.verbose or True:  # Always print for debugging
            print(f"  [ACTION MASK] Valid actions: {np.where(action_mask)[0].tolist()}")
            print(f"  [ACTION MASK] Cubes picked: {len(self.base_env.objects_picked)}/{self.num_cubes}")

        # NEW: Check if all cubes are picked (all actions masked)
        if not np.any(action_mask):
            # All cubes picked - episode complete!
            print(f"  [EPISODE END] All cubes picked or no valid actions!")
            next_agent2_obs = self._get_agent2_observation()
            info = {
                "reshuffles_performed": self.reshuffles_performed,
                "reshuffle_history": self.reshuffle_history,
                "reshuffled_this_step": False,
                "reshuffle_reason": "none",
                "agent1_action": -1,
                "agent2_action": agent2_action,
                "pick_reward": 0.0,
                "reshuffle_reward": 0.0,
                "total_pick_reward": self.total_pick_reward,  # NEW: Total Agent 1 reward
                "total_reshuffle_reward": self.total_reshuffle_reward,  # NEW: Total Agent 2 reward
                "timeout": False,
                "total_distance_reduced": self.total_distance_reduced,
                "total_time_saved": self.total_time_saved,
                "cubes_picked": len(self.base_env.objects_picked),
            }
            return next_agent2_obs, 0.0, True, False, info  # terminated=True

        agent1_action = self.ddqn_agent.select_action(base_obs, action_mask)

        # STEP 2: Check if reshuffling is needed
        reshuffle_decision = self._check_reshuffling_needed(agent1_action)

        # DEBUG: Print reshuffling decision details
        if self.verbose or True:  # Always print for debugging
            cube_positions = self.base_env.get_cube_positions()
            robot_pos = self.base_env.get_robot_position()
            target_dist = np.linalg.norm(cube_positions[agent1_action] - robot_pos)
            print(f"  [RESHUFFLE CHECK] Target cube {agent1_action}, dist={target_dist:.3f}m")
            print(f"    Decision: should_reshuffle={reshuffle_decision.should_reshuffle}, reason={reshuffle_decision.reason.value}, priority={reshuffle_decision.priority}")
            if reshuffle_decision.metadata:
                print(f"    Metadata: {reshuffle_decision.metadata}")

        # STEP 3: Execute reshuffling if needed
        reshuffle_reward = 0.0
        reshuffled = False

        if reshuffle_decision.should_reshuffle and self.reshuffles_performed < self.max_reshuffles_per_episode:
            # Check if this is batch reshuffle
            if reshuffle_decision.reason.value == "batch_reshuffle":
                # Execute batch reshuffle (multiple cubes at once)
                reshuffle_reward = self._execute_batch_reshuffle(agent2_action, reshuffle_decision)
                reshuffled = True
                self.reshuffles_performed += 1  # Count as 1 batch reshuffle
            else:
                # Execute individual reshuffle
                reshuffle_reward = self._execute_reshuffle(agent2_action, reshuffle_decision)
                reshuffled = True
                self.reshuffles_performed += 1

        # STEP 4: Execute Agent 1 pick action
        # Track distance traveled to target cube
        cube_positions = self.base_env.get_cube_positions()
        robot_pos = self.base_env.get_robot_position()
        distance_to_target = np.linalg.norm(cube_positions[agent1_action] - robot_pos)
        self.total_distance_traveled += distance_to_target

        next_base_obs, pick_reward, terminated, truncated, base_info = self.base_env.step(agent1_action)

        # STEP 4.5: Remove picked cube from environment (testing optimization)
        # This prevents picked cubes from being obstacles in future path planning
        # IMPORTANT: Only for testing! Training needs picked cubes as obstacles
        # FIXED: Only remove cube if pick was successful
        if hasattr(self.base_env, 'remove_picked_cube') and base_info.get('pick_success', False):
            self.base_env.remove_picked_cube(agent1_action)

        # STEP 5: Calculate total reward
        # Total reward = pick reward + reshuffle reward
        # This allows the test script to properly track Agent1 and Agent2 contributions
        total_reward = pick_reward + reshuffle_reward
        self.total_reshuffle_reward += reshuffle_reward
        self.total_pick_reward += pick_reward  # NEW: Accumulate Agent 1 reward

        # Get next observation for Agent 2
        next_agent2_obs = self._get_agent2_observation()

        # NEW: Update total time taken (wall-clock time)
        import time
        self.total_time_taken = time.time() - self.episode_start_time

        # Compile info
        info = {
            **base_info,
            "reshuffles_performed": self.reshuffles_performed,
            "reshuffle_history": self.reshuffle_history,
            "reshuffled_this_step": reshuffled,
            "reshuffle_reason": reshuffle_decision.reason.value if reshuffle_decision.should_reshuffle else "none",
            "agent1_action": agent1_action,
            "agent2_action": agent2_action,
            "pick_reward": pick_reward,
            "reshuffle_reward": reshuffle_reward,
            "total_pick_reward": self.total_pick_reward,  # NEW: Total Agent 1 reward
            "total_reshuffle_reward": self.total_reshuffle_reward,  # Total Agent 2 reward
            "timeout": False,
            "total_distance_reduced": self.total_distance_reduced,  # NEW: Cumulative distance improvement
            "total_time_saved": self.total_time_saved,  # NEW: Estimated time saved
            "total_distance_traveled": self.total_distance_traveled,  # NEW: Actual distance traveled
            "total_time_taken": self.total_time_taken,  # NEW: Actual time taken
            "distance_to_target": distance_to_target,  # NEW: Distance to target this step
            "cubes_picked": len(self.base_env.objects_picked),  # Number of cubes picked so far
        }

        return next_agent2_obs, total_reward, terminated, truncated, info

    def _get_base_observation(self) -> np.ndarray:
        """Get observation for Agent 1 (DDQN) from base environment"""
        # This depends on the base environment's observation format
        # For ObjectSelectionEnv, it's typically flattened cube features
        return self.base_env._get_observation()

    def _get_agent2_observation(self) -> np.ndarray:
        """
        Get observation for Agent 2 (MAPPO).

        Observation includes:
        - Cube positions (num_cubes * 3)
        - Robot position (3)
        - Picked cubes mask (num_cubes)
        - Grid cell distances to robot (grid_size * grid_size) - NEW!
        - Reshuffling decision features (10):
          - Average distance to cubes
          - Number of far cubes
          - Number of crowded cubes
          - Recent RRT failure rate
          - Average path clearance
          - Reshuffles remaining
          - Current episode progress
          - Number of cubes picked
          - Number of cubes remaining
          - Priority of last reshuffle decision
        """
        # Get cube positions
        cube_positions = self.base_env.get_cube_positions()
        cube_features = cube_positions.flatten()  # (num_cubes * 3,)

        # Get robot position
        robot_pos = self.base_env.get_robot_position()  # (3,)

        # Get picked cubes mask
        picked_mask = np.zeros(self.num_cubes, dtype=np.float32)
        for idx in self.base_env.objects_picked:
            picked_mask[idx] = 1.0

        # NEW: Calculate distance from robot to each grid cell
        # This explicitly tells Agent 2 which cells are nearer to the robot
        grid_distances = self._calculate_grid_cell_distances(robot_pos)  # (grid_size * grid_size,)

        # Calculate reshuffling decision features
        stats = self.reshuffle_decision.get_statistics()
        avg_distance = np.mean([np.linalg.norm(pos - robot_pos) for pos in cube_positions]) if len(cube_positions) > 0 else 0.0

        decision_features = np.array([
            avg_distance,
            self._count_far_cubes(cube_positions, robot_pos, avg_distance),
            self._count_crowded_cubes(cube_positions),
            stats.get("failure_rate", 0.0),
            stats.get("avg_clearance", 0.0),
            self.max_reshuffles_per_episode - self.reshuffles_performed,
            self.base_env.current_step / self.base_env.max_steps,
            len(self.base_env.objects_picked),
            self.num_cubes - len(self.base_env.objects_picked),
            0.0,  # Priority of last decision (updated during step)
        ], dtype=np.float32)

        # Check for NaN/Inf in all observation components
        if np.isnan(cube_features).any() or np.isinf(cube_features).any():
            print(f"[WARNING] NaN/Inf detected in cube_features! Replacing with zeros.")
            cube_features = np.nan_to_num(cube_features, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(robot_pos).any() or np.isinf(robot_pos).any():
            print(f"[WARNING] NaN/Inf detected in robot_pos! Replacing with zeros.")
            robot_pos = np.nan_to_num(robot_pos, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(grid_distances).any() or np.isinf(grid_distances).any():
            print(f"[WARNING] NaN/Inf detected in grid_distances! Replacing with zeros.")
            grid_distances = np.nan_to_num(grid_distances, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(decision_features).any() or np.isinf(decision_features).any():
            print(f"[WARNING] NaN/Inf detected in decision_features! Replacing with zeros.")
            decision_features = np.nan_to_num(decision_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate all features (including grid distances)
        observation = np.concatenate([
            cube_features,
            robot_pos,
            picked_mask,
            grid_distances,  # NEW: Explicit distance information
            decision_features
        ])

        # NEW: Check for NaN/Inf values in observation
        if np.isnan(observation).any() or np.isinf(observation).any():
            print(f"[ERROR] NaN/Inf detected in Agent 2 observation!")
            print(f"  cube_features: NaN={np.isnan(cube_features).sum()}, Inf={np.isinf(cube_features).sum()}")
            print(f"  robot_pos: NaN={np.isnan(robot_pos).sum()}, Inf={np.isinf(robot_pos).sum()}")
            print(f"  grid_distances: NaN={np.isnan(grid_distances).sum()}, Inf={np.isinf(grid_distances).sum()}")
            print(f"  decision_features: NaN={np.isnan(decision_features).sum()}, Inf={np.isinf(decision_features).sum()}")
            # Replace NaN/Inf with safe values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def _check_reshuffling_needed(self, target_cube_idx: int) -> ReshuffleDecision:
        """Check if reshuffling is needed for target cube"""
        cube_positions = self.base_env.get_cube_positions()
        robot_pos = self.base_env.get_robot_position()
        obstacle_positions = self.base_env.get_obstacle_positions()

        # Get path planning info if available (RRT or A*)
        # NOTE: We don't actually use path planning for reshuffling decisions anymore
        # This is kept for backward compatibility but always skipped
        rrt_path = None
        rrt_success = True
        path_clearance = None

        # OPTIMIZATION: Skip path planning entirely for reshuffling checks
        # All 3 environments (A*, RRT Viz, Isaac Sim RRT) use distance-based checks only
        # This makes reshuffling checks ~100x faster (~0.01s instead of ~1s per check)
        #
        # Previously tried to call:
        # - self.base_env.rrt_estimator.estimate_path_cost() <- Method doesn't exist!
        # - self.base_env.astar_estimator.estimate_path_cost() <- Method doesn't exist!
        #
        # Correct method is estimate_path_length(), but we skip it entirely for speed

        # Check reshuffling decision
        decision = self.reshuffle_decision.check_reshuffling_needed(
            cube_positions=cube_positions,
            target_cube_idx=target_cube_idx,
            robot_position=robot_pos,
            obstacle_positions=obstacle_positions,
            rrt_path=rrt_path,
            rrt_success=rrt_success,
            path_clearance=path_clearance,
        )

        return decision

    def _execute_reshuffle(self, agent2_action: int, decision: ReshuffleDecision) -> float:
        """
        Execute reshuffling action and calculate reward.

        Args:
            agent2_action: Reshuffling action from Agent 2
            decision: Reshuffling decision that triggered this action

        Returns:
            Reward for reshuffling action
        """
        # Decode action
        reshuffle_action = self.reshuffle_action_space.decode_action(agent2_action)

        # Check if cube is already picked
        if reshuffle_action.cube_idx in self.base_env.objects_picked:
            print(f"[WARNING] Cannot move already picked cube: {reshuffle_action.cube_idx}")
            return -10.0  # Penalty for invalid action

        # Check if cube has been reshuffled max times
        if self.reshuffle_count_per_cube.get(reshuffle_action.cube_idx, 0) >= 2:
            print(f"[WARNING] Cube {reshuffle_action.cube_idx} already reshuffled 2 times, skipping")
            return -5.0  # Penalty for invalid action

        # Get old position and distance BEFORE moving
        robot_pos = self.base_env.get_robot_position()
        cube_positions = self.base_env.get_cube_positions()
        old_position = cube_positions[reshuffle_action.cube_idx].copy()
        old_distance = np.linalg.norm(old_position - robot_pos)

        # Execute reshuffle in base environment
        self.base_env.move_cube(reshuffle_action.cube_idx, reshuffle_action.target_world_pos)

        # Update reshuffle count for this cube
        self.reshuffle_count_per_cube[reshuffle_action.cube_idx] = self.reshuffle_count_per_cube.get(reshuffle_action.cube_idx, 0) + 1

        # Get new distance AFTER moving
        new_distance = np.linalg.norm(reshuffle_action.target_world_pos - robot_pos)

        # Print detailed reshuffle info (only if verbose)
        distance_improvement = old_distance - new_distance
        improvement_pct = (distance_improvement / old_distance * 100) if old_distance > 0 else 0
        if self.verbose:
            print(f"[RESHUFFLE] Cube {reshuffle_action.cube_idx}: {decision.reason.value.upper()} (priority={decision.priority})")
            print(f"            Old: {old_position} (dist={old_distance:.3f}m)")
            print(f"            New: {reshuffle_action.target_world_pos} (dist={new_distance:.3f}m)")
            print(f"            Improvement: {distance_improvement:+.3f}m ({improvement_pct:+.1f}%)")

        # NEW: Track cumulative distance reduction
        self.total_distance_reduced += distance_improvement

        # NEW: Estimate time saved (assuming 1m distance = 2 seconds of robot movement)
        time_saved = distance_improvement * 2.0  # seconds
        self.total_time_saved += time_saved

        # Calculate reward based on priority and outcome
        # Higher priority reshuffles get higher rewards
        base_reward = {
            3: 40.0,  # Essential (increased from 2.0)
            2: 20.0,  # Efficiency (increased from 1.0)
            1: 10.0,  # Strategic (increased from 0.5)
            0: -10.0,  # Unnecessary reshuffle (penalty, increased from -0.5)
        }.get(decision.priority, 0.0)

        # Bonus if reshuffle actually improves reachability
        improvement_bonus, reward_components = self._calculate_improvement_bonus_detailed(
            reshuffle_action.cube_idx,
            reshuffle_action.target_world_pos,
            old_distance,
            new_distance
        )

        total_reward = (base_reward + improvement_bonus) * self.reshuffle_reward_scale

        # Record reshuffle with detailed information
        self.reshuffle_history.append({
            "reason": decision.reason.value,
            "cube_idx": reshuffle_action.cube_idx,
            "old_position": old_position.tolist(),
            "new_position": reshuffle_action.target_world_pos.tolist(),
            "old_distance": float(old_distance),
            "new_distance": float(new_distance),
            "distance_improvement": float(old_distance - new_distance),
            "priority": decision.priority,
            "base_reward": float(base_reward),
            "improvement_bonus": float(improvement_bonus),
            "total_reward": float(total_reward),
            "reward_components": reward_components,
        })

        return total_reward

    def _execute_batch_reshuffle(self, agent2_action: int, decision: ReshuffleDecision) -> float:
        """
        Execute batch reshuffling for multiple far cubes at once.

        Args:
            agent2_action: Base action from Agent 2 (used as seed for multiple actions)
            decision: Reshuffling decision that triggered batch reshuffle

        Returns:
            Total reward for batch reshuffling
        """
        # Get far cube indices that are eligible for reshuffling
        robot_pos = self.base_env.get_robot_position()
        cube_positions = self.base_env.get_cube_positions()
        avg_distance = np.mean([np.linalg.norm(pos - robot_pos) for pos in cube_positions]) if len(cube_positions) > 0 else 0.0

        far_cube_indices = self.reshuffle_decision.get_far_cube_indices(
            cube_positions=cube_positions,
            robot_position=robot_pos,
            avg_distance=avg_distance,
            picked_cubes=set(self.base_env.objects_picked),
            reshuffle_count=self.reshuffle_count_per_cube,
            max_reshuffles=2
        )

        if len(far_cube_indices) == 0:
            print(f"[BATCH RESHUFFLE] No eligible cubes to reshuffle")
            return 0.0

        print(f"[BATCH RESHUFFLE] Reshuffling {len(far_cube_indices)} cubes: {far_cube_indices}")

        total_reward = 0.0

        # Reshuffle each far cube
        for cube_idx in far_cube_indices:
            # Get old position and distance
            old_position = cube_positions[cube_idx].copy()
            old_distance = np.linalg.norm(old_position - robot_pos)

            # Find closest available grid position for this cube
            target_world_pos = self._find_closest_available_position(cube_idx, robot_pos)

            # Execute reshuffle
            self.base_env.move_cube(cube_idx, target_world_pos)

            # Update reshuffle count
            self.reshuffle_count_per_cube[cube_idx] = self.reshuffle_count_per_cube.get(cube_idx, 0) + 1

            # Get new distance
            new_distance = np.linalg.norm(target_world_pos - robot_pos)

            # Calculate reward for this cube
            distance_improvement = old_distance - new_distance
            improvement_pct = (distance_improvement / old_distance * 100) if old_distance > 0 else 0

            print(f"  [RESHUFFLE] Cube {cube_idx}: {old_distance:.3f}m → {new_distance:.3f}m ({improvement_pct:+.1f}%)")

            # Track cumulative distance reduction
            self.total_distance_reduced += distance_improvement
            time_saved = distance_improvement * 2.0
            self.total_time_saved += time_saved

            # Calculate reward (strategic priority = 1)
            base_reward = 10.0  # Strategic reshuffle
            improvement_bonus, reward_components = self._calculate_improvement_bonus_detailed(
                cube_idx,
                target_world_pos,
                old_distance,
                new_distance
            )

            cube_reward = (base_reward + improvement_bonus) * self.reshuffle_reward_scale
            total_reward += cube_reward

            # Record reshuffle
            self.reshuffle_history.append({
                "reason": "batch_reshuffle",
                "cube_idx": cube_idx,
                "old_position": old_position.tolist(),
                "new_position": target_world_pos.tolist(),
                "old_distance": float(old_distance),
                "new_distance": float(new_distance),
                "distance_improvement": float(distance_improvement),
                "priority": 1,
                "base_reward": float(base_reward),
                "improvement_bonus": float(improvement_bonus),
                "total_reward": float(cube_reward),
                "reward_components": reward_components,
            })

        print(f"[BATCH RESHUFFLE] Total reward: {total_reward:.2f}")
        return total_reward

    def _find_closest_available_position(self, cube_idx: int, robot_pos: np.ndarray) -> np.ndarray:
        """
        Find the closest available grid position for a cube.

        Args:
            cube_idx: Index of cube to reshuffle
            robot_pos: Robot position

        Returns:
            Target world position (3D)
        """
        grid_size = self.reshuffle_action_space.grid_size
        cube_positions = self.base_env.get_cube_positions()

        best_position = None
        best_distance = float('inf')

        # Try all grid positions
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                # Convert to world position (3D)
                # Use decode_action to get proper 3D position
                action_idx = self.reshuffle_action_space.encode_action(cube_idx, grid_x, grid_y)
                reshuffle_action = self.reshuffle_action_space.decode_action(action_idx)
                world_pos = reshuffle_action.target_world_pos

                # Check if position is occupied by another cube
                occupied = False
                for other_idx, other_pos in enumerate(cube_positions):
                    if other_idx != cube_idx and other_idx not in self.base_env.objects_picked:
                        if np.linalg.norm(world_pos[:2] - other_pos[:2]) < 0.05:  # 5cm threshold
                            occupied = True
                            break

                if not occupied:
                    # Calculate distance to robot (both are 3D now)
                    distance = np.linalg.norm(world_pos - robot_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_position = world_pos

        # If no position found, use current position
        if best_position is None:
            best_position = cube_positions[cube_idx]

        return best_position

    def _calculate_improvement_bonus(self, cube_idx: int, new_position: np.ndarray) -> float:
        """
        Calculate bonus reward if reshuffling improves reachability.

        IMPROVED: Now includes both optimal distance deviation AND absolute distance improvement.
        This encourages moving cubes closer to robot while also targeting optimal range.
        """
        robot_pos = self.base_env.get_robot_position()
        cube_positions = self.base_env.get_cube_positions()

        # Old distance
        old_distance = np.linalg.norm(cube_positions[cube_idx] - robot_pos)

        # New distance
        new_distance = np.linalg.norm(new_position - robot_pos)

        # COMPONENT 1: Optimal distance deviation (existing logic)
        optimal_distance = 0.6  # Optimal reachable distance
        old_deviation = abs(old_distance - optimal_distance)
        new_deviation = abs(new_distance - optimal_distance)

        deviation_bonus = 0.0
        if new_deviation < old_deviation:
            deviation_bonus = 1.0  # Improvement bonus
        else:
            deviation_bonus = -0.5  # Penalty for making it worse

        # COMPONENT 2: Absolute distance improvement (NEW - addresses Question 2)
        # Reward for moving cube closer to robot (regardless of optimal distance)
        distance_improvement = old_distance - new_distance  # Positive if moved closer
        distance_bonus = 2.0 * distance_improvement  # Scale by 2.0 for stronger signal

        # Clip distance bonus to prevent extreme values
        distance_bonus = np.clip(distance_bonus, -2.0, 2.0)

        # COMPONENT 3: Reachability bonus (NEW)
        # Extra bonus if cube moved from unreachable to reachable range
        reachability_bonus = 0.0
        if old_distance < 0.3 or old_distance > 0.9:  # Was unreachable
            if 0.3 <= new_distance <= 0.9:  # Now reachable
                reachability_bonus = 3.0  # Large bonus for making unreachable cube reachable

        # Total improvement bonus
        total_bonus = deviation_bonus + distance_bonus + reachability_bonus

        return total_bonus

    def _calculate_improvement_bonus_detailed(
        self,
        cube_idx: int,
        new_position: np.ndarray,
        old_distance: float,
        new_distance: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate bonus reward with detailed component breakdown.

        Returns:
            Tuple of (total_bonus, reward_components_dict)
        """
        # COMPONENT 1: Optimal distance deviation
        optimal_distance = 0.6  # Optimal reachable distance
        old_deviation = abs(old_distance - optimal_distance)
        new_deviation = abs(new_distance - optimal_distance)

        deviation_bonus = 0.0
        if new_deviation < old_deviation:
            deviation_bonus = 1.0  # Improvement bonus
        else:
            deviation_bonus = -0.5  # Penalty for making it worse

        # COMPONENT 2: Absolute distance improvement (SYMMETRIC SCALING)
        distance_improvement = old_distance - new_distance  # Positive if moved closer

        # Symmetric scaling: 2x for both positive and negative improvements
        distance_bonus = 2.0 * distance_improvement

        distance_bonus = np.clip(distance_bonus, -2.0, 2.0)  # Symmetric clipping

        # COMPONENT 3: Reachability bonus
        reachability_bonus = 0.0
        if old_distance < 0.3 or old_distance > 0.9:  # Was unreachable
            if 0.3 <= new_distance <= 0.9:  # Now reachable
                reachability_bonus = 3.0  # Large bonus for making unreachable cube reachable

        # Total improvement bonus
        total_bonus = deviation_bonus + distance_bonus + reachability_bonus

        # Component breakdown
        reward_components = {
            "deviation_bonus": float(deviation_bonus),
            "distance_bonus": float(distance_bonus),
            "reachability_bonus": float(reachability_bonus),
        }

        return total_bonus, reward_components

    def _calculate_grid_cell_distances(self, robot_pos: np.ndarray) -> np.ndarray:
        """
        Calculate distance from robot to each grid cell.

        This explicitly tells Agent 2 which cells are nearer to the robot,
        enabling faster learning of "move cubes to nearer cells" strategy.

        Args:
            robot_pos: Robot end-effector position (3,)

        Returns:
            Flattened array of distances (grid_size * grid_size,)
        """
        grid_size = self.reshuffle_action_space.grid_size
        distances = np.zeros((grid_size, grid_size), dtype=np.float32)

        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                # Convert grid coordinates to world position
                cell_world_pos = self.reshuffle_action_space._grid_to_world(grid_x, grid_y)

                # Calculate distance from robot to cell center
                distance = np.linalg.norm(cell_world_pos - robot_pos[:2])  # Only XY distance
                distances[grid_y, grid_x] = distance

        # Flatten to 1D array
        return distances.flatten()

    def _count_far_cubes(self, cube_positions: np.ndarray, robot_pos: np.ndarray, avg_distance: float) -> int:
        """Count cubes that are far from robot"""
        count = 0
        for pos in cube_positions:
            dist = np.linalg.norm(pos - robot_pos)
            if dist > 1.2 * avg_distance:
                count += 1
        return count

    def _count_crowded_cubes(self, cube_positions: np.ndarray) -> int:
        """Count cubes in crowded areas"""
        count = 0
        for i, pos in enumerate(cube_positions):
            nearby = 0
            for j, other_pos in enumerate(cube_positions):
                if i != j and np.linalg.norm(pos - other_pos) < 0.15:
                    nearby += 1
            if nearby >= 3:
                count += 1
        return count

    def get_agent2_action_mask(self) -> np.ndarray:
        """Get action mask for Agent 2 (MAPPO)"""
        cube_positions = self.base_env.get_cube_positions()
        picked_cubes = list(self.base_env.objects_picked)
        obstacle_positions = self.base_env.get_obstacle_positions()
        robot_position = self.base_env.get_robot_position()

        return self.reshuffle_action_space.get_action_mask(
            cube_positions=cube_positions,
            picked_cubes=picked_cubes,
            obstacle_positions=obstacle_positions,
            robot_position=robot_position,
            base_env=self.base_env
        )

    # Wrapper methods for heuristic agents
    def get_robot_position(self) -> np.ndarray:
        """Get robot end-effector position (delegates to base_env)"""
        return self.base_env.get_robot_position()

    def get_cube_positions(self) -> np.ndarray:
        """Get cube positions (delegates to base_env)"""
        return self.base_env.get_cube_positions()

    def get_obstacle_positions(self) -> np.ndarray:
        """Get obstacle positions (delegates to base_env)"""
        return self.base_env.get_obstacle_positions()

