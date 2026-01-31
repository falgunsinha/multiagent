"""
Scene Setup Module for RRT Pick and Place
Handles creation of Franka robot, container, and ground plane
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path (go up 2 levels from RRT folder to cobotproject)
project_root = None

# Method 1: Try __file__ if available
try:
    if '__file__' in globals():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # RRT -> scripts -> cobotproject
except:
    pass

# Method 2: Use hardcoded path as fallback
if project_root is None:
    project_root = Path(r"C:\isaacsim\cobotproject")

# Method 3: Try from current working directory
if not project_root.exists():
    try:
        project_root = Path(os.getcwd()) / "cobotproject"
        if not project_root.exists():
            project_root = Path(r"C:\isaacsim\cobotproject")
    except:
        project_root = Path(r"C:\isaacsim\cobotproject")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper


class SceneSetup:
    """Manages the static scene elements: Franka, Container, Ground Plane"""
    
    def __init__(self):
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None
        self.container_translation = np.array([-0.2, -0.5, 0.0])
        self.container_height = 0.64 * 0.3  # 0.192m (original height * scale)
        
    def setup_world(self, physics_dt=1.0/60.0, rendering_dt=1.0/60.0):
        """Create or get World instance with optimized physics"""
        # Clear any existing world
        if World.instance():
            World.clear_instance()
        
        # Create new world with optimized physics
        self.world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=1.0
        )
        
        print("World created with optimized physics settings")
        return self.world
    
    def add_ground_plane(self):
        """Add default ground plane to the scene"""
        self.world.scene.add_default_ground_plane()
        print("Ground plane added")
    
    def add_franka(self, franka_name="franka", franka_prim_path="/World/Franka"):
        """Add Franka robot to the scene"""
        print(f"Adding Franka robot: {franka_name}")
        
        # Add Franka manipulator
        self.franka = self.world.scene.add(
            SingleManipulator(
                prim_path=franka_prim_path,
                name=franka_name,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        
        # Add gripper
        self.gripper = ParallelGripper(
            end_effector_prim_path=f"{franka_prim_path}/panda_hand",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([-0.01, -0.01]),
        )
        
        print(f"Franka robot '{franka_name}' added successfully")
        return self.franka, self.gripper
    
    def add_container(self, container_name="container", container_prim_path="/World/Container"):
        """Add warehouse container to the scene"""
        print(f"Adding container: {container_name}")
        
        # Get container USD path
        assets_root_path = get_assets_root_path()
        container_usd_path = assets_root_path + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
        
        # Add container reference to stage
        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)
        
        # Create container XForm with scale and translation
        self.container = SingleXFormPrim(
            prim_path=container_prim_path,
            name=container_name,
            scale=np.array([0.3, 0.3, 0.3]),
            translation=self.container_translation
        )
        
        # Add physics to container (static rigid body with collision)
        from isaacsim.core.utils.prims import get_prim_at_path
        container_prim = get_prim_at_path(container_prim_path)
        
        # Add rigid body API (kinematic/static)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)  # Static body
        
        # Add collision API
        UsdPhysics.CollisionAPI.Apply(container_prim)
        
        print(f"Container '{container_name}' added at {self.container_translation}")
        return self.container
    
    def initialize_franka(self):
        """Initialize Franka with default joint positions for RRT"""
        # Set default joint positions for 7 arm joints + 2 gripper joints
        # From robot_descriptor.yaml: [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]
        default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04, 0.04])
        
        # Set default state for joints
        self.franka.set_joints_default_state(positions=default_joint_positions)
        
        # Set gripper default state
        self.gripper.set_default_state(self.gripper.joint_opened_positions)
        
        print("Franka initialized with default configuration")
    
    def setup_complete_scene(self):
        """Setup complete scene with all static elements"""
        print("\n=== Setting up scene ===")
        
        # Create world
        self.setup_world()
        
        # Add ground plane
        self.add_ground_plane()
        
        # Add Franka
        self.add_franka()
        
        # Add container
        self.add_container()
        
        # Initialize physics
        print("Initializing physics...")
        self.world.initialize_physics()
        
        # Reset world to apply default states
        print("Resetting world...")
        self.world.reset()
        
        # Initialize Franka
        self.initialize_franka()
        
        print("=== Scene setup complete ===\n")
        
        return self.world, self.franka, self.gripper, self.container
    
    def get_container_info(self):
        """Get container position and dimensions for placing objects"""
        return {
            'translation': self.container_translation,
            'height': self.container_height,
            'placing_height': 0.08  # Offset above container bottom
        }

