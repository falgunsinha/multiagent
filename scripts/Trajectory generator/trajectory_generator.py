import carb
import numpy as np
import os

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
import omni.timeline
from omni.kit.async_engine import run_coroutine

from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

import lula

class TrajectoryGeneration():
    def __init__(self):
        self._c_space_trajectory_generator = None
        self._taskspace_trajectory_generator = None
        self._kinematics_solver = None

        self._action_sequence = []
        self._action_sequence_index = 0

        self._articulation = None
        self._end_effector_name = "ee_link"

    def load_example_assets(self, world):
        """Load the UR10 robot to the stage"""
        robot_prim_path = "/ur10"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        world.scene.add(self._articulation)

        return self._articulation

    def setup(self):
        """Setup trajectory generators and kinematics solver"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf"
        )

        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf"
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf"
        )

    def setup_advanced_trajectory(self):
        """Setup a complex trajectory combining c-space and task-space movements"""
        initial_c_space_robot_pose = np.array([0, 0, 0, 0, 0, 0])

        composite_path_spec = lula.create_composite_path_spec(initial_c_space_robot_pose)

        r0 = lula.Rotation3(np.pi / 2, np.array([1.0, 0.0, 0.0]))
        t0 = np.array([.3, -.1, .3])
        task_space_spec = lula.create_task_space_path_spec(lula.Pose3(r0, t0))

        t1 = np.array([.3, -.1, .5])
        r1 = lula.Rotation3(np.pi / 3, np.array([1, 0, 0]))
        task_space_spec.add_linear_path(lula.Pose3(r1, t1))

        task_space_spec.add_translation(t0)
        task_space_spec.add_rotation(r0)

        t2 = np.array([.3, .3, .3, ])
        midpoint = np.array([.3, 0, .5])
        task_space_spec.add_three_point_arc(t2, midpoint, constant_orientation=True)
        task_space_spec.add_three_point_arc(t0, midpoint, constant_orientation=False)
        task_space_spec.add_three_point_arc_with_orientation_target(lula.Pose3(r1, t2), midpoint)
        task_space_spec.add_tangent_arc(t0, constant_orientation=True)
        task_space_spec.add_tangent_arc(t2, constant_orientation=False)
        task_space_spec.add_tangent_arc_with_orientation_target(lula.Pose3(r0, t0))
        c_space_spec = lula.create_c_space_path_spec(np.array([0, 0, 0, 0, 0, 0]))
        c_space_spec.add_c_space_waypoint(np.array([0, 0.5, -2.0, -1.28, 5.13, -4.71]))
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_task_space_path_spec(task_space_spec, transition_mode)
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_c_space_path_spec(c_space_spec, transition_mode)
        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
            composite_path_spec, self._end_effector_name
        )

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)

            self._action_sequence = articulation_trajectory.get_action_sequence()

    def update(self):
        """Update function to be called each frame"""
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= len(self._action_sequence) + 10
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])
        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10

    def reset(self):
        """Reset the trajectory execution"""
        if get_prim_at_path("/visualized_frames"):
            delete_prim("/visualized_frames")

        self._action_sequence = []
        self._action_sequence_index = 0

    def _teleport_robot_to_position(self, articulation_action):
        """Teleport robot to the initial position of the trajectory"""
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))

async def setup_scene():
    """Setup the scene with UR10 robot"""
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    World.clear_instance()
    await omni.kit.app.get_app().next_update_async()
    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    await omni.kit.app.get_app().next_update_async()
    my_world.scene.add_default_ground_plane()
    trajectory_example = TrajectoryGeneration()
    trajectory_example.load_example_assets(my_world)
    trajectory_example.setup()
    trajectory_example.setup_advanced_trajectory()
    my_world.initialize_physics()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    my_world.reset()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    return my_world, trajectory_example

async def run_trajectory_generation():
    """Main async function to run the trajectory generation example"""
    my_world, trajectory_example = await setup_scene()

    step_count = 0
    max_steps = 10000
    reset_needed = False

    while step_count < max_steps:
        await omni.kit.app.get_app().next_update_async()

        if my_world.is_stopped() and not reset_needed:
            reset_needed = True

        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                trajectory_example.reset()
                reset_needed = False

            step_count += 1

            trajectory_example.update()

            if step_count % 100 == 0:
                print(f"Step {step_count}, Action index: {trajectory_example._action_sequence_index}")
        else:
            carb.log_warn("World is not playing")
            break

    print(f"Trajectory execution completed (Steps: {step_count})")

run_coroutine(run_trajectory_generation())

