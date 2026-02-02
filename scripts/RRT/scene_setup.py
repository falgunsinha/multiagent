import numpy as np
from pathlib import Path
import sys
import os

project_root = None

try:
    if '__file__' in globals():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
except:
    pass

if project_root is None or not project_root.exists():
    try:
        cwd = Path(os.getcwd())
        if cwd.name == "multiagent":
            project_root = cwd
        else:
            for parent in [cwd] + list(cwd.parents):
                if parent.name == "multiagent":
                    project_root = parent
                    break
    except:
        pass

if project_root is None:
    project_root = Path.cwd()

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
    """Manages the static scene elements: Cobot, Container, Ground Plane"""

    def __init__(self):
        self.world = None
        self.cobot = None
        self.gripper = None
        self.container = None
        self.container_translation = np.array([-0.2, -0.5, 0.0])
        self.container_height = 0.64 * 0.3

    def setup_world(self, physics_dt=1.0/60.0, rendering_dt=1.0/60.0):
        """Create or get World instance with optimized physics"""
        if World.instance():
            World.clear_instance()

        self.world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=1.0
        )

        return self.world

    def add_ground_plane(self):
        """Add default ground plane to the scene"""
        self.world.scene.add_default_ground_plane()
    
    def add_cobot(self, cobot_name="cobot", cobot_prim_path="/World/Cobot"):
        """Add cobot to the scene"""
        self.cobot = self.world.scene.add(
            SingleManipulator(
                prim_path=cobot_prim_path,
                name=cobot_name,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        self.gripper = ParallelGripper(
            end_effector_prim_path=f"{cobot_prim_path}/panda_hand",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([-0.01, -0.01]),
        )

        return self.cobot, self.gripper
    
    def add_container(self, container_name="container", container_prim_path="/World/Container"):
        """Add container to the scene"""
        assets_root_path = get_assets_root_path()
        container_usd_path = assets_root_path + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"

        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

        self.container = SingleXFormPrim(
            prim_path=container_prim_path,
            name=container_name,
            scale=np.array([0.3, 0.3, 0.3]),
            translation=self.container_translation
        )

        from isaacsim.core.utils.prims import get_prim_at_path
        container_prim = get_prim_at_path(container_prim_path)

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)

        UsdPhysics.CollisionAPI.Apply(container_prim)

        return self.container
    
    def initialize_cobot(self):
        """Initialize cobot with default joint positions for RRT"""
        default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04, 0.04])

        self.cobot.set_joints_default_state(positions=default_joint_positions)
        self.gripper.set_default_state(self.gripper.joint_opened_positions)

    def setup_complete_scene(self):
        """Setup complete scene with all static elements"""
        self.setup_world()
        self.add_ground_plane()
        self.add_cobot()
        self.add_container()

        self.world.initialize_physics()
        self.world.reset()

        self.initialize_cobot()

        return self.world, self.cobot, self.gripper, self.container
    
    def get_container_info(self):
        """Get container position and dimensions for placing objects"""
        return {
            'translation': self.container_translation,
            'height': self.container_height,
            'placing_height': 0.08
        }

