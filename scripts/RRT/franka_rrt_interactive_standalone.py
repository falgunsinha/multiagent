"""
Franka RRT Interactive Pick and Place - Standalone Version
All-in-one script for Isaac Sim Script Editor
Creates pyramid stack of cuboids and picks/places them using Lula RRT
"""

import asyncio
import time
import numpy as np
import os
from pathlib import Path
import sys
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from pxr import UsdPhysics

# Add project root to path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper


# ============================================================================
# SCENE SETUP CLASS
# ============================================================================

class SceneSetup:
    """Manages the static scene elements: Franka, Container, Ground Plane"""
    
    def __init__(self):
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None
        self.container_position = np.array([-0.3, -0.5, 0.0])
        self.container_scale = np.array([0.3, 0.3, 0.3])
    
    def setup_world(self):
        """Create World with optimized physics settings"""
        print("\n=== Setting up World ===")
        
        # Clear any existing world
        if World.instance():
            World.instance().clear_instance()
        
        # Create new world with optimized physics
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,  # 60 Hz physics
            rendering_dt=1.0/60.0  # 60 Hz rendering
        )
        
        print("World created successfully!")
        return self.world
    
    def add_ground_plane(self):
        """Add default ground plane"""
        print("\n=== Adding Ground Plane ===")
        self.world.scene.add_default_ground_plane()
        print("Ground plane added!")
    
    def add_franka(self):
        """Add Franka robot and gripper"""
        print("\n=== Adding Franka Robot ===")

        franka_name = f"franka_{int(time.time() * 1000)}"
        franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

        # Add the USD reference to the stage (use the standard franka.usd)
        franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)

        # Set variant selections for gripper and mesh quality
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

        # Create gripper (use panda_rightfinger as end effector)
        self.gripper = ParallelGripper(
            end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01])
        )

        # Now wrap it as a SingleManipulator
        self.franka = self.world.scene.add(
            SingleManipulator(
                prim_path=franka_prim_path,
                name=franka_name,
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                gripper=self.gripper,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
        )

        print(f"Franka added: {franka_name}")
        return self.franka, self.gripper
    
    def add_container(self):
        """Add warehouse container with physics"""
        print("\n=== Adding Container ===")
        
        container_prim_path = "/World/Container"
        
        # Add container USD reference
        container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
        
        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)
        
        # Create XForm prim for container
        self.container = self.world.scene.add(
            SingleXFormPrim(
                prim_path=container_prim_path,
                name="container",
                translation=self.container_position,
                scale=self.container_scale
            )
        )
        
        # Add physics to container
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        container_prim = stage.GetPrimAtPath(container_prim_path)
        
        # Add rigid body (kinematic/static)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        
        # Add collision
        UsdPhysics.CollisionAPI.Apply(container_prim)
        
        print(f"Container added at position: {self.container_position}")
        return self.container
    
    def initialize_franka(self):
        """Initialize Franka with default joint positions for RRT"""
        print("\n=== Initializing Franka ===")
        
        # Set default joint positions for 7 arm joints + 2 gripper joints
        default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04, 0.04])
        
        # Set default state for joints
        self.franka.set_joints_default_state(positions=default_joint_positions)
        
        # Set gripper default state
        self.gripper.set_default_state(self.gripper.joint_opened_positions)
        
        print("Franka initialized with default configuration")
    
    def setup_complete_scene(self):
        """Setup complete scene with all elements"""
        self.setup_world()
        self.add_ground_plane()
        self.add_franka()
        self.add_container()
        
        # Reset world to initialize everything
        self.world.reset()
        
        # Initialize Franka after reset
        self.initialize_franka()
        
        print("\n=== Scene Setup Complete! ===\n")
        return self.world, self.franka, self.gripper, self.container
    
    def get_container_info(self):
        """Get container position and dimensions"""
        return {
            'position': self.container_position,
            'scale': self.container_scale
        }


# ============================================================================
# OBJECT MANAGER CLASS
# ============================================================================

class ObjectManager:
    """Manages dynamic cuboid objects in pyramid-style stacking"""
    
    def __init__(self, world, base_layer_count=4, num_layers=4):
        """
        Initialize object manager with configurable pyramid stack
        
        Args:
            world: Isaac Sim World instance
            base_layer_count: Number of cuboids in the bottom layer (e.g., 4 for 2x2 grid)
            num_layers: Total number of layers in the pyramid (e.g., 4 for layers of 4,3,2,1)
        """
        self.world = world
        self.cuboids = []  # List of all created cuboids with metadata
        
        # Pyramid configuration
        self.base_layer_count = base_layer_count
        self.num_layers = num_layers
        self.base_position = np.array([0.5, 0.0, 0.0])
        
        # Stack state
        self.stack_added = False
        
        # Cube properties
        self.cube_size = 0.0515
        self.cube_half_size = self.cube_size / 2.0
    
    def _calculate_layer_positions(self, layer_index, cuboids_in_layer):
        """Calculate positions for cuboids in a specific layer"""
        positions = []
        z_position = self.cube_half_size + (layer_index * self.cube_size) + 0.01
        
        if cuboids_in_layer == 1:
            positions.append(self.base_position + np.array([0.0, 0.0, z_position]))
        elif cuboids_in_layer == 2:
            spacing = self.cube_size * 0.5
            positions.append(self.base_position + np.array([-spacing, 0.0, z_position]))
            positions.append(self.base_position + np.array([spacing, 0.0, z_position]))
        elif cuboids_in_layer == 3:
            spacing = self.cube_size * 0.5
            positions.append(self.base_position + np.array([-spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([0.0, spacing, z_position]))
        elif cuboids_in_layer == 4:
            spacing = self.cube_size * 0.5
            positions.append(self.base_position + np.array([-spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([-spacing, spacing, z_position]))
            positions.append(self.base_position + np.array([spacing, spacing, z_position]))
        else:
            grid_size = int(np.ceil(np.sqrt(cuboids_in_layer)))
            spacing = self.cube_size * 0.5
            offset = (grid_size - 1) * spacing / 2.0
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= cuboids_in_layer:
                        break
                    x = i * spacing - offset
                    y = j * spacing - offset
                    positions.append(self.base_position + np.array([x, y, z_position]))
                    idx += 1
                if idx >= cuboids_in_layer:
                    break
        
        return positions

    def add_stack(self):
        """Add pyramid stack of cuboids with configurable layers"""
        if self.stack_added:
            print("Pyramid stack has already been added!")
            return False

        print(f"\n=== Adding Pyramid Stack ===")
        print(f"Base layer: {self.base_layer_count} cuboids")
        print(f"Total layers: {self.num_layers}")

        total_cuboids = 0

        for layer_index in range(self.num_layers):
            cuboids_in_layer = self.base_layer_count - layer_index

            if cuboids_in_layer <= 0:
                break

            print(f"\nLayer {layer_index + 1}: {cuboids_in_layer} cuboid(s)")
            positions = self._calculate_layer_positions(layer_index, cuboids_in_layer)

            for pos_index, position in enumerate(positions):
                timestamp = int(time.time() * 1000) + total_cuboids
                cube_name = f"cube_L{layer_index}_P{pos_index}_{timestamp}"
                cube_prim_path = f"/World/Cube_Layer{layer_index}_Pos{pos_index}_{timestamp}"

                cuboid = self.world.scene.add(
                    DynamicCuboid(
                        name=cube_name,
                        position=position,
                        prim_path=cube_prim_path,
                        scale=np.array([self.cube_size, self.cube_size, self.cube_size]),
                        size=1.0,
                        color=np.array([0.0, 0.0, 1.0]),
                    )
                )

                self.cuboids.append({
                    'object': cuboid,
                    'name': cube_name,
                    'prim_path': cube_prim_path,
                    'layer': layer_index,
                    'position_in_layer': pos_index,
                    'picked': False
                })

                print(f"  Added: {cube_name} at position {position}")
                total_cuboids += 1

        self.stack_added = True
        print(f"\nPyramid stack added successfully!")
        print(f"Total cuboids in pyramid: {total_cuboids}\n")

        return True

    def get_next_unpicked_cuboid(self):
        """Get the next cuboid that hasn't been picked (pick from top to bottom by actual Z position)"""
        # Get all unpicked cuboids
        unpicked = [c for c in self.cuboids if not c['picked']]

        if not unpicked:
            return None

        # Sort by actual Z position (highest first) - read current position dynamically
        unpicked_with_z = []
        for cuboid_info in unpicked:
            current_pos = cuboid_info['object'].get_world_pose()[0]
            unpicked_with_z.append((cuboid_info, current_pos[2]))  # Store cuboid and its Z position

        # Sort by Z position (descending - highest first)
        unpicked_with_z.sort(key=lambda x: x[1], reverse=True)

        # Return the cuboid with highest Z position
        return unpicked_with_z[0][0]

    def mark_as_picked(self, cuboid_info):
        """Mark a cuboid as picked"""
        cuboid_info['picked'] = True

    def get_unpicked_count(self):
        """Get count of unpicked cuboids"""
        return sum(1 for c in self.cuboids if not c['picked'])

    def clear_all(self):
        """Clear all cuboids from the scene"""
        print("\n=== Clearing all cuboids ===")
        for cuboid_info in self.cuboids:
            try:
                prim_path = cuboid_info['prim_path']
                from omni.isaac.core.utils.stage import get_current_stage
                stage = get_current_stage()
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    stage.RemovePrim(prim_path)
                    print(f"  Removed: {cuboid_info['name']}")
            except Exception as e:
                print(f"  Error removing {cuboid_info['name']}: {e}")

        self.cuboids.clear()
        self.stack_added = False
        print("All cuboids cleared!\n")

    def get_stack_info(self):
        """Get information about current pyramid stack"""
        info = {
            'base_layer_count': self.base_layer_count,
            'num_layers': self.num_layers,
            'stack_added': self.stack_added,
            'total_cuboids': len(self.cuboids),
            'unpicked_cuboids': self.get_unpicked_count()
        }
        return info


# ============================================================================
# RRT CONTROLLER CLASS
# ============================================================================

class PickPlaceState:
    """State machine states for pick and place"""
    IDLE = 0
    MOVING_TO_PICK_APPROACH = 1  # Move above cube
    DESCENDING_TO_PICK = 2        # Descend to cube
    CLOSING_GRIPPER = 3           # Close gripper
    ASCENDING_FROM_PICK = 4       # Ascend with cube
    MOVING_TO_PLACE_APPROACH = 5  # Move above container
    DESCENDING_TO_PLACE = 6       # Descend to place
    OPENING_GRIPPER = 7           # Open gripper
    ASCENDING_FROM_PLACE = 8      # Ascend away
    DONE = 9


class RRTController:
    """Manages RRT motion planning and pick/place execution"""

    def __init__(self, world, franka, gripper, container_info):
        self.world = world
        self.franka = franka
        self.gripper = gripper
        self.container_info = container_info

        # RRT components
        self.rrt = None
        self.path_planner_visualizer = None

        # State machine
        self.state = PickPlaceState.IDLE
        self.current_plan = None
        self.plan_index = 0

        # Current target
        self.current_cuboid = None

        # Placing position tracking
        self.placed_count = 0

        # Setup RRT
        self.setup_rrt()

    def setup_rrt(self):
        """Setup Lula RRT motion planner"""
        print("\n=== Setting up RRT Motion Planner ===")

        # Get extension path
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        # Config file paths
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rmpflow")
        robot_description_file = os.path.join(rmp_config_dir, "robot_descriptor.yaml")
        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        # Create RRT planner
        self.rrt = RRT(
            robot_description_path=robot_description_file,
            urdf_path=urdf_path,
            rrt_config_path=rrt_config_file,
            end_effector_frame_name="right_gripper"
        )

        # Set max iterations after creation
        self.rrt.set_max_iterations(10000)

        # Create path planner visualizer
        self.path_planner_visualizer = PathPlannerVisualizer(
            robot_articulation=self.franka,
            path_planner=self.rrt
        )

        print("RRT planner initialized successfully!")
        print(f"RRT max iterations: 10000")
        print("Note: No obstacles added - planning in free space")

    def plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT"""
        # Set target
        self.rrt.set_end_effector_target(target_position, target_orientation)

        # Update world state (updates obstacle positions)
        self.rrt.update_world()

        # Compute plan using PathPlannerVisualizer
        # The visualizer automatically gets current joint positions from the robot
        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)

        return plan

    def start_pick_place(self, cuboid_info):
        """Start pick and place operation for a cuboid

        Args:
            cuboid_info: The cuboid to pick
        """
        self.current_cuboid = cuboid_info
        self.state = PickPlaceState.MOVING_TO_PICK_APPROACH
        self.plan_index = 0

        # Get cube center DYNAMICALLY (current position after physics settling)
        cube_center, _ = cuboid_info['object'].get_world_pose()

        # Store cube center for later use
        self.pick_cube_center = cube_center

        print(f"Current cube position: {cube_center}")

        # Approach from above: cube center + offset above
        # Add 15cm above cube center for safe approach
        pick_approach_offset = 0.15
        pick_approach_position = cube_center + np.array([0.0, 0.0, pick_approach_offset])
        pick_orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

        self.pick_orientation = pick_orientation

        print(f"\n=== Planning path to pick {cuboid_info['name']} ===")
        print(f"Cube center: {cube_center}")
        print(f"Approach position: {pick_approach_position}")

        # Update world before planning (important!)
        self.rrt.update_world()

        # Plan to approach position (above the cube)
        self.current_plan = self.plan_to_target(pick_approach_position, pick_orientation)

        if self.current_plan is None:
            print("Failed to compute plan to approach position")
            self.state = PickPlaceState.IDLE
            return False

        print(f"Plan computed with {len(self.current_plan)} steps")
        return True

    async def execute_step(self):
        """Execute one step of the pick and place state machine"""
        if self.state == PickPlaceState.IDLE or self.state == PickPlaceState.DONE:
            return False

        # State 1: Moving to pick approach position (above cube)
        if self.state == PickPlaceState.MOVING_TO_PICK_APPROACH:
            if self.current_plan and self.plan_index < len(self.current_plan):
                action = self.current_plan[self.plan_index]
                self.franka.apply_action(action)
                self.plan_index += 1
                await omni.kit.app.get_app().next_update_async()
                return True
            else:
                # Reached approach position - OPEN GRIPPER HERE
                print("Reached approach position, opening gripper...")
                self.gripper.open()

                # Wait for gripper to open
                for _ in range(15):
                    await omni.kit.app.get_app().next_update_async()

                print("Gripper opened, descending to cube...")
                self.state = PickPlaceState.DESCENDING_TO_PICK

                # Plan descent to cube TOP (cube is 5.15cm, so half is 2.575cm)
                # Descend to just above the top surface
                cube_half_height = 0.0515 / 2.0  # 2.575cm
                descend_position = self.pick_cube_center + np.array([0.0, 0.0, cube_half_height + 0.005])  # 5mm above top
                print(f"Descending to position: {descend_position} (cube top + 5mm)")
                self.rrt.update_world()
                self.current_plan = self.plan_to_target(descend_position, self.pick_orientation)

                if self.current_plan is None:
                    print("Failed to plan descent")
                    self.state = PickPlaceState.IDLE
                    return False

                self.plan_index = 0
                return True

        # State 2: Descending to pick position
        elif self.state == PickPlaceState.DESCENDING_TO_PICK:
            if self.current_plan and self.plan_index < len(self.current_plan):
                action = self.current_plan[self.plan_index]
                self.franka.apply_action(action)
                self.plan_index += 1
                await omni.kit.app.get_app().next_update_async()
                return True
            else:
                # Reached pick position, close gripper
                print("Reached pick position, closing gripper...")
                self.state = PickPlaceState.CLOSING_GRIPPER
                return True

        # State 3: Closing gripper
        elif self.state == PickPlaceState.CLOSING_GRIPPER:
            print("Closing gripper to pick cube...")
            self.gripper.close()
            await omni.kit.app.get_app().next_update_async()

            # Wait longer for gripper to close and grip the cube
            for _ in range(30):  # Increased from 15 to 30 frames
                await omni.kit.app.get_app().next_update_async()

            print("Gripper closed, ascending with cube...")
            self.state = PickPlaceState.ASCENDING_FROM_PICK

            # Plan ascent (back to approach height)
            ascend_position = self.pick_cube_center + np.array([0.0, 0.0, 0.15])
            self.rrt.update_world()
            self.current_plan = self.plan_to_target(ascend_position, self.pick_orientation)

            if self.current_plan is None:
                print("Failed to plan ascent")
                self.state = PickPlaceState.IDLE
                return False

            self.plan_index = 0
            return True

        # State 4: Ascending from pick
        elif self.state == PickPlaceState.ASCENDING_FROM_PICK:
            if self.current_plan and self.plan_index < len(self.current_plan):
                action = self.current_plan[self.plan_index]
                self.franka.apply_action(action)
                self.plan_index += 1
                await omni.kit.app.get_app().next_update_async()
                return True
            else:
                # Ascended, now plan to place approach position
                print("Ascended with cube, planning to place approach...")

                container_pos = self.container_info['position']
                cube_size = 0.0515
                cube_half_size = cube_size / 2.0
                base_offset = 0.08

                # Calculate final placing height
                final_placing_height = cube_half_size + base_offset + (self.placed_count * cube_size)

                # Approach position (15cm above final)
                place_approach_offset = 0.15
                place_approach_position = container_pos + np.array([0.0, 0.0, final_placing_height + place_approach_offset])
                place_orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

                self.place_final_height = final_placing_height
                self.place_orientation = place_orientation
                self.container_pos = container_pos

                print(f"Place approach position: {place_approach_position}")

                self.rrt.update_world()
                self.current_plan = self.plan_to_target(place_approach_position, place_orientation)

                if self.current_plan is None:
                    print("Failed to compute plan to place approach")
                    self.state = PickPlaceState.IDLE
                    return False

                self.plan_index = 0
                self.state = PickPlaceState.MOVING_TO_PLACE_APPROACH
                return True

        # State 5: Moving to place approach position
        elif self.state == PickPlaceState.MOVING_TO_PLACE_APPROACH:
            if self.current_plan and self.plan_index < len(self.current_plan):
                action = self.current_plan[self.plan_index]
                self.franka.apply_action(action)
                self.plan_index += 1
                await omni.kit.app.get_app().next_update_async()
                return True
            else:
                # Reached place approach, now descend to place
                print("Reached place approach, descending to place...")
                self.state = PickPlaceState.DESCENDING_TO_PLACE

                # Plan descent to final placing position (2cm above final)
                descend_position = self.container_pos + np.array([0.0, 0.0, self.place_final_height + 0.02])
                self.rrt.update_world()
                self.current_plan = self.plan_to_target(descend_position, self.place_orientation)

                if self.current_plan is None:
                    print("Failed to plan descent to place")
                    self.state = PickPlaceState.IDLE
                    return False

                self.plan_index = 0
                return True

        # State 6: Descending to place position
        elif self.state == PickPlaceState.DESCENDING_TO_PLACE:
            if self.current_plan and self.plan_index < len(self.current_plan):
                action = self.current_plan[self.plan_index]
                self.franka.apply_action(action)
                self.plan_index += 1
                await omni.kit.app.get_app().next_update_async()
                return True
            else:
                # Reached place position, open gripper
                print("Reached place position, opening gripper...")
                self.state = PickPlaceState.OPENING_GRIPPER
                return True

        # State 7: Opening gripper
        elif self.state == PickPlaceState.OPENING_GRIPPER:
            self.gripper.open()
            await omni.kit.app.get_app().next_update_async()

            # Wait for gripper to open
            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

            print("Gripper opened, ascending away...")
            self.state = PickPlaceState.ASCENDING_FROM_PLACE

            # Plan ascent away from container
            ascend_position = self.container_pos + np.array([0.0, 0.0, self.place_final_height + 0.15])
            self.rrt.update_world()
            self.current_plan = self.plan_to_target(ascend_position, self.place_orientation)

            if self.current_plan is None:
                print("Failed to plan ascent from place")
                self.state = PickPlaceState.IDLE
                return False

            self.plan_index = 0
            return True

        # State 8: Ascending from place
        elif self.state == PickPlaceState.ASCENDING_FROM_PLACE:
            if self.current_plan and self.plan_index < len(self.current_plan):
                action = self.current_plan[self.plan_index]
                self.franka.apply_action(action)
                self.plan_index += 1
                await omni.kit.app.get_app().next_update_async()
                return True
            else:
                # Done!
                self.placed_count += 1
                self.state = PickPlaceState.DONE
                print(f"Placed {self.current_cuboid['name']} successfully!")
                return True

        return False

    def is_done(self):
        """Check if current operation is done"""
        return self.state == PickPlaceState.DONE or self.state == PickPlaceState.IDLE

    def reset(self):
        """Reset controller state for next cuboid (keep placed_count)"""
        self.state = PickPlaceState.IDLE
        self.current_plan = None
        self.plan_index = 0
        self.current_cuboid = None
        # Don't reset placed_count - it tracks total cubes placed for stacking height


# ============================================================================
# MAIN INTERACTIVE UI CLASS
# ============================================================================

class FrankaRRTInteractive:
    """Main class for interactive RRT pick and place with UI"""

    def __init__(self, base_layer_count=4, num_layers=4):
        """
        Initialize interactive RRT system

        Args:
            base_layer_count: Number of cuboids in bottom layer of pyramid
            num_layers: Total number of layers in pyramid
        """
        # Pyramid configuration
        self.base_layer_count = base_layer_count
        self.num_layers = num_layers

        # Components
        self.scene_setup = None
        self.object_manager = None
        self.rrt_controller = None

        # World and robot
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None

        # UI components
        self.window = None
        self.status_label = None
        self.load_btn = None
        self.add_stack_btn = None
        self.pick_place_btn = None
        self.reset_btn = None

        # Performance metrics
        self.step_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_time_ms = 0.0
        self.fps_update_interval = 0.5
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.fps_label = None
        self.frame_time_label = None
        self.steps_label = None

        # Pick and place state
        self.is_picking = False
        self.pick_place_task = None

        # Timeline
        self.timeline = omni.timeline.get_timeline_interface()

    def build_ui(self):
        """Build the UI window with buttons"""
        self.window = ui.Window("Franka RRT Control", width=450, height=450)

        with self.window.frame:
            with ui.VStack(spacing=10):
                # Title
                ui.Label("Franka RRT Pick and Place",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})

                ui.Spacer(height=5)

                # Status label
                self.status_label = ui.Label("Status: Ready",
                                            alignment=ui.Alignment.CENTER,
                                            style={"font_size": 14, "color": 0xFF00FF00})

                ui.Spacer(height=10)

                # Buttons
                with ui.VStack(spacing=5):
                    self.load_btn = ui.Button("Load Scene",
                                              clicked_fn=self._on_load_scene,
                                              height=40)

                    self.add_stack_btn = ui.Button("Add Object Stack",
                                                   clicked_fn=self._on_add_stack,
                                                   height=40)
                    self.add_stack_btn.enabled = False

                    self.pick_place_btn = ui.Button("Pick and Place",
                                                    clicked_fn=self._on_pick_place,
                                                    height=40)
                    self.pick_place_btn.enabled = False

                    self.reset_btn = ui.Button("Reset",
                                               clicked_fn=self._on_reset,
                                               height=40)
                    self.reset_btn.enabled = False

                ui.Spacer(height=10)

                # Performance metrics frame
                with ui.Frame(style={"background_color": 0x33000000}):
                    with ui.VStack(spacing=5, height=0):
                        ui.Label("Performance Metrics",
                                alignment=ui.Alignment.CENTER,
                                style={"font_size": 12, "color": 0xFFFFFFFF})

                        with ui.HStack(spacing=10):
                            ui.Spacer(width=10)
                            self.fps_label = ui.Label("FPS: 0.0",
                                                     alignment=ui.Alignment.LEFT,
                                                     style={"font_size": 11, "color": 0xFF00FFFF})
                            ui.Spacer()
                            self.frame_time_label = ui.Label("Frame Time: 0.0 ms",
                                                            alignment=ui.Alignment.CENTER,
                                                            style={"font_size": 11, "color": 0xFF00FFFF})
                            ui.Spacer()
                            self.steps_label = ui.Label("Steps: 0",
                                                       alignment=ui.Alignment.RIGHT,
                                                       style={"font_size": 11, "color": 0xFF00FFFF})
                            ui.Spacer(width=10)

                ui.Spacer(height=10)

                # Instructions
                with ui.CollapsableFrame("Instructions", height=0):
                    with ui.VStack(spacing=3):
                        ui.Label("Instructions:", alignment=ui.Alignment.LEFT)
                        ui.Label("1. Load Scene - Creates Franka, Container, Ground",
                                alignment=ui.Alignment.LEFT, word_wrap=True)
                        ui.Label(f"2. Add Object Stack - Adds pyramid ({self.num_layers} layers: {self.base_layer_count}, {self.base_layer_count-1}, ..., 1)",
                                alignment=ui.Alignment.LEFT, word_wrap=True)
                        ui.Label("3. Pick and Place - Starts pick/place automation",
                                alignment=ui.Alignment.LEFT, word_wrap=True)
                        ui.Label("4. Reset - Stops simulation and clears everything",
                                alignment=ui.Alignment.LEFT, word_wrap=True)

    def _update_status(self, message, color=0xFF00FF00):
        """Update status label"""
        if self.status_label:
            self.status_label.text = f"Status: {message}"

    def _update_performance_metrics(self):
        """Update FPS, Frame Time, and Steps display"""
        current_time = time.time()

        # Calculate frame time
        self.frame_time_ms = (current_time - self.last_frame_time) * 1000.0
        self.last_frame_time = current_time

        # Increment frame counter
        self.frame_count += 1

        # Update FPS every interval
        if current_time - self.last_fps_update >= self.fps_update_interval:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

        # Update UI labels
        if self.fps_label:
            self.fps_label.text = f"FPS: {self.fps:.1f}"
        if self.frame_time_label:
            self.frame_time_label.text = f"Frame Time: {self.frame_time_ms:.2f} ms"
        if self.steps_label:
            self.steps_label.text = f"Steps: {self.step_count}"

    def _on_load_scene(self):
        """Load scene button callback"""
        print("\n" + "="*60)
        print("LOADING SCENE")
        print("="*60)

        try:
            # Create scene setup
            self.scene_setup = SceneSetup()

            # Setup complete scene
            self.world, self.franka, self.gripper, self.container = self.scene_setup.setup_complete_scene()

            # Create object manager with pyramid configuration
            self.object_manager = ObjectManager(self.world, self.base_layer_count, self.num_layers)

            # Create RRT controller
            container_info = self.scene_setup.get_container_info()
            self.rrt_controller = RRTController(self.world, self.franka, self.gripper, container_info)

            # Reset performance metrics
            self.step_count = 0
            self.last_frame_time = time.time()
            self.fps = 0.0
            self.frame_time_ms = 0.0
            self.last_fps_update = time.time()
            self.frame_count = 0

            # Enable next button
            self.add_stack_btn.enabled = True
            self.reset_btn.enabled = True

            self._update_status("Scene loaded successfully!")

        except Exception as e:
            self._update_status(f"Error loading scene: {e}", color=0xFFFF0000)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _on_add_stack(self):
        """Add stack button callback"""
        if not self.object_manager:
            self._update_status("Load scene first!", color=0xFFFF0000)
            return

        # Add pyramid stack
        success = self.object_manager.add_stack()

        if success:
            stack_info = self.object_manager.get_stack_info()
            self._update_status(f"Pyramid added! Total: {stack_info['total_cuboids']} cuboids")

            # Enable pick and place button if we have cuboids
            if stack_info['total_cuboids'] > 0:
                self.pick_place_btn.enabled = True
        else:
            self._update_status("Pyramid stack already added!", color=0xFFFFAA00)

    def _on_pick_place(self):
        """Pick and place button callback"""
        if not self.object_manager or not self.rrt_controller:
            self._update_status("Load scene and add objects first!", color=0xFFFF0000)
            return

        if self.is_picking:
            # Pause
            self.is_picking = False
            self._update_status("Paused")
            self.timeline.pause()
        else:
            # Start or resume
            self.is_picking = True
            self._update_status("Starting pick and place...")

            # Start async task
            self.pick_place_task = run_coroutine(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            # Start timeline
            self.timeline.play()

            # Wait for physics simulation view to initialize
            print("\n=== Waiting for physics to initialize ===")
            for _ in range(30):  # Wait ~0.5 seconds
                await omni.kit.app.get_app().next_update_async()

            # Wait for cubes to settle (important for accurate Z positions)
            print("=== Waiting for cubes to settle ===")
            for _ in range(120):  # Wait ~2 seconds at 60Hz
                await omni.kit.app.get_app().next_update_async()
            print("Cubes settled, starting pick and place...\n")

            while self.is_picking:
                # Get next unpicked cuboid (sorted by actual Z position)
                cuboid_info = self.object_manager.get_next_unpicked_cuboid()

                if cuboid_info is None:
                    # All done!
                    self._update_status("All cuboids placed!")
                    self.is_picking = False
                    self.timeline.stop()
                    break

                # Start pick and place for this cuboid (no obstacles needed)
                success = self.rrt_controller.start_pick_place(cuboid_info)

                if not success:
                    self._update_status(f"Failed to plan for {cuboid_info['name']}", color=0xFFFF0000)
                    self.is_picking = False
                    break

                # Execute until done
                while not self.rrt_controller.is_done() and self.is_picking:
                    await self.rrt_controller.execute_step()
                    self.step_count += 1
                    self._update_performance_metrics()

                # Mark as picked
                self.object_manager.mark_as_picked(cuboid_info)

                # Reset controller for next cuboid
                self.rrt_controller.reset()

                # Small delay between cuboids
                for _ in range(10):
                    if not self.is_picking:
                        break
                    await omni.kit.app.get_app().next_update_async()
                    self.step_count += 1
                    self._update_performance_metrics()

            if not self.is_picking:
                self.timeline.pause()

        except Exception as e:
            self._update_status(f"Error: {e}", color=0xFFFF0000)
            print(f"Error in pick and place: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self.timeline.stop()

    def _on_reset(self):
        """Reset button callback - Stops simulation and clears everything"""
        print("\n" + "="*60)
        print("RESET - CLEARING EVERYTHING")
        print("="*60)

        try:
            # Stop any ongoing operations
            self.is_picking = False

            # Stop timeline
            if self.timeline:
                self.timeline.stop()
                print("Timeline stopped")

            # Clear World instance (this removes all prims from stage)
            if World.instance():
                print("Clearing World instance...")
                World.clear_instance()
                print("World cleared - all prims deleted from stage")

            # Reset all component references
            self.world = None
            self.franka = None
            self.gripper = None
            self.container = None
            self.scene_setup = None
            self.object_manager = None
            self.rrt_controller = None

            # Reset performance metrics
            self.step_count = 0
            self.last_frame_time = time.time()
            self.fps = 0.0
            self.frame_time_ms = 0.0
            self.last_fps_update = time.time()
            self.frame_count = 0
            self._update_performance_metrics()

            # Disable all buttons except Load Scene
            self.load_scene_btn.enabled = True
            self.add_stack_btn.enabled = False
            self.pick_place_btn.enabled = False
            self.reset_btn.enabled = False

            self._update_status("Reset complete - everything cleared!")
            print("Reset complete!\n")

        except Exception as e:
            self._update_status(f"Error resetting: {e}", color=0xFFFF0000)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(base_layer_count=4, num_layers=4):
    """
    Main entry point

    Args:
        base_layer_count: Number of cuboids in bottom layer (default: 4 for 2x2 grid)
        num_layers: Total number of layers (default: 4 for layers of 4,3,2,1)
    """
    print("\n" + "="*60)
    print("Franka RRT Interactive Pick and Place - Standalone")
    print(f"Pyramid Configuration: {num_layers} layers")
    print(f"Base layer: {base_layer_count} cuboids")
    total_cuboids = sum(range(base_layer_count - num_layers + 1, base_layer_count + 1))
    print(f"Total cuboids: {total_cuboids}")
    print("="*60 + "\n")

    # Create and show UI with configuration
    app = FrankaRRTInteractive(base_layer_count, num_layers)
    app.build_ui()

    print("UI loaded! Use the buttons to control the robot.")
    print("="*60 + "\n")


# Run the application
if __name__ == "__main__":
    # You can customize the pyramid configuration here:
    # main(base_layer_count=4, num_layers=4)  # Default: 4,3,2,1 pyramid
    # main(base_layer_count=5, num_layers=5)  # 5,4,3,2,1 pyramid
    # main(base_layer_count=3, num_layers=3)  # 3,2,1 pyramid
    main()

