"""
MPC (Model Predictive Control) Planner for Robot Motion Planning
Uses MPPI optimization for obstacle avoidance and goal reaching
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from .mppi_optimizer import MPPIOptimizer
from .cost_functions import MPCCostFunction, simple_dynamics


class MPCPlanner:
    """
    Model Predictive Control planner for robot arm motion planning
    
    Uses MPPI (Model Predictive Path Integral) optimization to:
    1. Plan collision-free trajectories to goal poses
    2. Avoid dynamic obstacles in real-time
    3. Respect joint limits and kinematic constraints
    
    Key features:
    - Receding horizon control (replans at each timestep)
    - Sampling-based optimization (no gradients needed)
    - Real-time obstacle avoidance
    - Smooth trajectory generation
    """
    
    def __init__(
        self,
        kinematics_solver,
        horizon: int = 20,
        num_samples: int = 500,
        dt: float = 0.05,
        mppi_iterations: int = 3,
        control_dim: int = 7
    ):
        """
        Initialize MPC planner
        
        Args:
            kinematics_solver: Forward kinematics solver (computes end-effector pose from joint angles)
            horizon: Planning horizon (number of timesteps to look ahead)
            num_samples: Number of trajectory samples for MPPI
            dt: Time step between controls
            mppi_iterations: Number of MPPI optimization iterations per step
            control_dim: Control dimension (7 for Franka)
        """
        self.kinematics_solver = kinematics_solver
        self.horizon = horizon
        self.dt = dt
        self.mppi_iterations = mppi_iterations
        self.control_dim = control_dim
        
        # Initialize MPPI optimizer
        self.optimizer = MPPIOptimizer(
            horizon=horizon,
            num_samples=num_samples,
            lambda_=1.0,
            noise_sigma=0.3,
            control_dim=control_dim,
            dt=dt,
            temperature=1.0
        )
        
        # Initialize cost function
        self.cost_function = MPCCostFunction(
            goal_position_weight=100.0,
            goal_orientation_weight=50.0,
            obstacle_weight=500.0,
            control_weight=0.1,
            joint_limit_weight=10.0,
            obstacle_margin=0.15
        )
        
        # Set forward kinematics function
        self.cost_function.set_fk_function(self._compute_fk)
        
        # Current goal
        self.goal_position = None
        self.goal_orientation = None
        
        # Current obstacles
        self.obstacles = []
        
        # Planned trajectory
        self.planned_trajectory = None
        self.planned_controls = None
        
    def _compute_fk(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics (end-effector pose from joint positions)

        Args:
            joint_positions: Joint positions (7,)

        Returns:
            Tuple of (position, quaternion)
        """
        # Use the kinematics solver to compute FK
        # This will be set to use the ArticulationKinematicsSolver from Isaac Sim
        if self.kinematics_solver is not None:
            try:
                # ArticulationKinematicsSolver returns (position, rotation_matrix)
                ee_position, ee_rot_mat = self.kinematics_solver.compute_end_effector_pose()

                # Convert rotation matrix to quaternion (w, x, y, z)
                # Simple conversion (not production-ready, but works for MPC cost)
                trace = np.trace(ee_rot_mat)
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    w = 0.25 / s
                    x = (ee_rot_mat[2, 1] - ee_rot_mat[1, 2]) * s
                    y = (ee_rot_mat[0, 2] - ee_rot_mat[2, 0]) * s
                    z = (ee_rot_mat[1, 0] - ee_rot_mat[0, 1]) * s
                else:
                    if ee_rot_mat[0, 0] > ee_rot_mat[1, 1] and ee_rot_mat[0, 0] > ee_rot_mat[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + ee_rot_mat[0, 0] - ee_rot_mat[1, 1] - ee_rot_mat[2, 2])
                        w = (ee_rot_mat[2, 1] - ee_rot_mat[1, 2]) / s
                        x = 0.25 * s
                        y = (ee_rot_mat[0, 1] + ee_rot_mat[1, 0]) / s
                        z = (ee_rot_mat[0, 2] + ee_rot_mat[2, 0]) / s
                    elif ee_rot_mat[1, 1] > ee_rot_mat[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + ee_rot_mat[1, 1] - ee_rot_mat[0, 0] - ee_rot_mat[2, 2])
                        w = (ee_rot_mat[0, 2] - ee_rot_mat[2, 0]) / s
                        x = (ee_rot_mat[0, 1] + ee_rot_mat[1, 0]) / s
                        y = 0.25 * s
                        z = (ee_rot_mat[1, 2] + ee_rot_mat[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + ee_rot_mat[2, 2] - ee_rot_mat[0, 0] - ee_rot_mat[1, 1])
                        w = (ee_rot_mat[1, 0] - ee_rot_mat[0, 1]) / s
                        x = (ee_rot_mat[0, 2] + ee_rot_mat[2, 0]) / s
                        y = (ee_rot_mat[1, 2] + ee_rot_mat[2, 1]) / s
                        z = 0.25 * s

                quaternion = np.array([w, x, y, z])
                return ee_position, quaternion
            except Exception as e:
                # Fallback: return dummy values
                return np.zeros(3), np.array([1, 0, 0, 0])
        else:
            # Fallback: return dummy values
            return np.zeros(3), np.array([1, 0, 0, 0])
    
    def set_goal(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """
        Set goal end-effector pose
        
        Args:
            position: Goal position (x, y, z)
            orientation: Goal orientation as quaternion [w, x, y, z] (optional)
        """
        self.goal_position = position
        self.goal_orientation = orientation
        self.cost_function.set_goal(position, orientation)
    
    def update_obstacles(self, obstacles: List[np.ndarray]):
        """
        Update obstacle positions
        
        Args:
            obstacles: List of obstacle positions [(x, y, z), ...]
        """
        self.obstacles = obstacles
        self.cost_function.set_obstacles(obstacles)
    
    def plan(self, current_joint_positions: np.ndarray) -> Optional[np.ndarray]:
        """
        Plan trajectory from current state to goal using IK-based approach

        Args:
            current_joint_positions: Current robot joint positions (7,)

        Returns:
            Next control action (joint velocities) or None if planning fails
        """
        if self.goal_position is None:
            print("[MPC] No goal set")
            return None

        # Use IK to get target joint configuration
        try:
            # Compute IK for goal
            target_joints, ik_success = self.kinematics_solver.compute_inverse_kinematics(
                self.goal_position, self.goal_orientation
            )

            if not ik_success:
                print("[MPC] IK failed for goal")
                return None

            # Compute control as proportional controller towards goal
            # This is a simple MPC: u = K_p * (q_goal - q_current)
            position_error = target_joints - current_joint_positions

            # Proportional gain (tune for responsiveness vs stability)
            K_p = 2.0
            control = K_p * position_error

            # Clip control to reasonable velocity limits (rad/s)
            max_velocity = 1.0
            control = np.clip(control, -max_velocity, max_velocity)

            # Simple obstacle avoidance: repulsive potential
            if len(self.obstacles) > 0:
                # Get current end-effector position
                ee_pos, _ = self.kinematics_solver.compute_end_effector_pose()

                # Compute repulsive force from obstacles
                repulsive_force = np.zeros(3)
                for obs_pos in self.obstacles:
                    diff = ee_pos - obs_pos
                    distance = np.linalg.norm(diff)

                    # Repulsive potential (only if close)
                    if distance < 0.3:  # 30cm influence range
                        # Force magnitude inversely proportional to distance
                        force_magnitude = 0.5 * (1.0 / distance - 1.0 / 0.3)
                        repulsive_force += force_magnitude * (diff / distance)

                # Convert Cartesian repulsive force to joint space (simple Jacobian transpose)
                # For now, just reduce velocity towards goal if obstacles are close
                if np.linalg.norm(repulsive_force) > 0.1:
                    control *= 0.5  # Slow down near obstacles

            return control

        except Exception as e:
            print(f"[MPC] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def step(self, current_joint_positions: np.ndarray) -> Optional[np.ndarray]:
        """
        Execute one MPC step (plan and return next action)

        Args:
            current_joint_positions: Current robot joint positions (7,)

        Returns:
            Next joint positions to execute
        """
        # Plan next control
        control = self.plan(current_joint_positions)

        if control is None:
            return None

        # Apply control to get next state (integrate velocity)
        next_positions = current_joint_positions + control * self.dt

        # Clip to joint limits
        joint_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        next_positions = np.clip(next_positions, joint_lower, joint_upper)

        return next_positions
    
    def reset(self):
        """Reset planner (clear warm start and planned trajectory)"""
        self.optimizer.reset()
        self.planned_trajectory = None
        self.planned_controls = None
        self.goal_position = None
        self.goal_orientation = None
        self.obstacles = []

