import carb
import numpy as np
import os

# Isaac Sim Core APIs
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
import omni.timeline
from omni.kit.async_engine import run_coroutine

# Motion generation imports
from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

import lula


class UR10TrajectoryGenerationExample():
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

        # Add articulation to the world scene
        world.scene.add(self._articulation)

        return self._articulation

    def setup(self):
        """Setup trajectory generators and kinematics solver"""
        # Config files for supported robots are stored in the motion_generation extension
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        # Initialize a LulaCSpaceTrajectoryGenerator object
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
        # The following code demonstrates how to specify a complicated cspace and taskspace path
        # using the lula.CompositePathSpec object

        initial_c_space_robot_pose = np.array([0, 0, 0, 0, 0, 0])

        # Combine a cspace and taskspace trajectory
        composite_path_spec = lula.create_composite_path_spec(initial_c_space_robot_pose)

        #############################################################################
        # Demonstrate all the available movements in a taskspace path spec:

        # Lula has its own classes for Rotations and 6 DOF poses: Rotation3 and Pose3
        r0 = lula.Rotation3(np.pi / 2, np.array([1.0, 0.0, 0.0]))
        t0 = np.array([.3, -.1, .3])
        task_space_spec = lula.create_task_space_path_spec(lula.Pose3(r0, t0))

        # Add path linearly interpolating between r0,r1 and t0,t1
        t1 = np.array([.3, -.1, .5])
        r1 = lula.Rotation3(np.pi / 3, np.array([1, 0, 0]))
        task_space_spec.add_linear_path(lula.Pose3(r1, t1))

        # Add pure translation. Constant rotation is assumed
        task_space_spec.add_translation(t0)

        # Add pure rotation.
        task_space_spec.add_rotation(r0)

        # Add three-point arc with constant orientation.
        t2 = np.array([.3, .3, .3, ])
        midpoint = np.array([.3, 0, .5])
        task_space_spec.add_three_point_arc(t2, midpoint, constant_orientation=True)

        # Add three-point arc with tangent orientation.
        task_space_spec.add_three_point_arc(t0, midpoint, constant_orientation=False)

        # Add three-point arc with orientation target.
        task_space_spec.add_three_point_arc_with_orientation_target(lula.Pose3(r1, t2), midpoint)

        # Add tangent arc with constant orientation. Tangent arcs are circles that connect two points
        task_space_spec.add_tangent_arc(t0, constant_orientation=True)

        # Add tangent arc with tangent orientation.
        task_space_spec.add_tangent_arc(t2, constant_orientation=False)

        # Add tangent arc with orientation target.
        task_space_spec.add_tangent_arc_with_orientation_target(lula.Pose3(r0, t0))

        ###################################################
        # Demonstrate the usage of a c_space path spec:
        c_space_spec = lula.create_c_space_path_spec(np.array([0, 0, 0, 0, 0, 0]))

        c_space_spec.add_c_space_waypoint(np.array([0, 0.5, -2.0, -1.28, 5.13, -4.71]))

        ##############################################################
        # Combine the two path specs together into a composite spec:

        # specify how to connect initial_c_space and task_space points with transition_mode option
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_task_space_path_spec(task_space_spec, transition_mode)

        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_c_space_path_spec(c_space_spec, transition_mode)

        # Transition Modes:
        # lula.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE:
        #      Connect cspace to taskspace points linearly through task space. This mode is only available when adding a task_space path spec.
        # lula.CompositePathSpec.TransitionMode.FREE:
        #      Put no constraints on how cspace and taskspace points are connected
        # lula.CompositePathSpec.TransitionMode.SKIP:
        #      Skip the first point of the path spec being added, using the last pose instead

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
            composite_path_spec, self._end_effector_name
        )

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)

            # Get a sequence of ArticulationActions that are intended to be passed to the robot at 1/60 second intervals
            self._action_sequence = articulation_trajectory.get_action_sequence()

    def update(self):
        """Update function to be called each frame"""
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])

        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories

    def reset(self):
        """Reset the trajectory execution"""
        # Delete any visualized frames
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

    # Stop timeline first
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    # Wait for timeline to stop
    await omni.kit.app.get_app().next_update_async()

    # Clear any existing World instance to ensure clean state
    print("Creating fresh World instance...")
    World.clear_instance()

    # Wait for cleanup to complete
    await omni.kit.app.get_app().next_update_async()

    # Create new World instance with 60Hz physics
    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)

    # Wait for World to initialize
    await omni.kit.app.get_app().next_update_async()

    print("Adding ground plane...")
    my_world.scene.add_default_ground_plane()

    # Create trajectory example instance
    trajectory_example = UR10TrajectoryGenerationExample()

    # Load UR10 robot
    print("Loading UR10 robot...")
    trajectory_example.load_example_assets(my_world)

    # Setup trajectory generators
    print("Setting up trajectory generators...")
    trajectory_example.setup()

    # Setup the advanced trajectory
    print("Computing advanced trajectory...")
    trajectory_example.setup_advanced_trajectory()

    print("Initializing physics...")
    my_world.initialize_physics()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Resetting world...")
    my_world.reset()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Playing simulation...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    return my_world, trajectory_example


async def run_trajectory_generation():
    """Main async function to run the trajectory generation example"""

    # Setup scene
    my_world, trajectory_example = await setup_scene()

    # Task state variables
    step_count = 0
    max_steps = 10000  # Run for a long time to see multiple trajectory repetitions
    reset_needed = False

    print("Starting trajectory execution...")
    print(f"World is playing: {my_world.is_playing()}")
    print(f"Trajectory has {len(trajectory_example._action_sequence)} actions")

    # Main simulation loop
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

            # Update trajectory execution
            trajectory_example.update()

            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}, Action index: {trajectory_example._action_sequence_index}")
        else:
            print("Warning: World is not playing")
            break

    print(f"Trajectory execution completed! (Steps: {step_count})")


# Run the async function
print("=" * 80)
print("UR10 Trajectory Generation Example - Script Editor Version")
print("=" * 80)
run_coroutine(run_trajectory_generation())

