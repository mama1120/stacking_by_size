import time
import os
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine
import json
from scipy.spatial.transform import Rotation as R

### Creates a dataset with noise in the position and orientation of the robots end-effector
### The cubes are found in random positions

# Define some action labels:
ACTIONS = {
    "move_to_pre_grasp": 0,
    "move_to_grasp": 1,
    "lift_cube": 2,
    "move_to_stack_position": 3,
    "stack_cube": 4,
    "above_stack": 5,
    "return_home": 6

}

#Folder to save the demonstrations
demo_folder = "noise_dataset"

# Define the BulletEnvironment class to get the current state of the environment and save it
class BulletEnvironment:
    def __init__(self, bullet_client, robot):
        self.bullet_client = bullet_client
        self.robot = robot
        self.objects = {}  # Dictionary to store object IDs and their names
        self.gripper_state = False  # Initialize gripper state (False means closed, True means open)

    def add_object(self, object_id, name, size):
        """Register an object with its ID, name, and size."""
        self.objects[object_id] = {"name": name, "size": size}

    def get_cube_positions(self):
        """Get positions of all registered cubes."""
        cube_positions = {}
        for object_id, obj in self.objects.items():
            pos, ori = self.bullet_client.getBasePositionAndOrientation(object_id)
            cube_positions[obj["name"]] = {"position": pos, "orientation": ori}
        return cube_positions

    def get_robot_state(self):
        """Get the robot's end-effector pose."""
        eef_pose = self.robot.get_eef_pose()
        return {
            "eef_position": eef_pose.translation.tolist(),
            "eef_orientation": eef_pose.quat.tolist(),
        }

    def get_gripper_state(self):
        """Get the gripper's open/close state."""
        return self.gripper_state

    def set_gripper_state(self, state):
        """Manually set the gripper state (open/close)."""
        self.gripper_state = state

    def get_cube_sizes(self):
        """Get the sizes of all cubes."""
        return {obj["name"]: obj["size"] for obj in self.objects.values()}

    def save_demonstration(self, folder, filename, action_label, stacking_cube):
        """Save the current state of the environment and the action to a file."""
        data = {
            "action_label": action_label,
            "robot_state": self.get_robot_state(),
            "cube_positions": self.get_cube_positions(),
            "gripper_state": self.get_gripper_state(),
            "cube_sizes": self.get_cube_sizes(),
            "stacking_cube": stacking_cube
        }
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Demonstration saved to {filepath}")

def generate_action_noise(mean=0.0, std_dev=0.015):
    """Generate random noise for actions based on a normal distribution."""
    return np.random.normal(mean, std_dev, size=3)

def rotation_matrix_to_euler(rotation_matrix, order="xyz"):
    """
    Convert a 3x3 rotation matrix to Euler angles.
    :param rotation_matrix: 3x3 numpy array representing the rotation matrix.
    :param order: The sequence of axes for Euler angles (e.g., "xyz", "zyx").
    :return: Euler angles as a list [roll, pitch, yaw].
    """
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_euler(order)

def apply_orientation_noise(rotation_matrix, std_dev=[0.005, 0.005, 0.1]):
    """
    Apply noise to a rotation represented as a quaternion.
    :param rotation: List or array representing the quaternion [x, y, z, w].
    :param std_dev: List of standard deviations for noise in Euler angles [x, y, z].
    :return: A new quaternion with added noise.
    """
    order = "xyz"
    # Convert rotation matrix to Euler angles
    euler_angles = rotation_matrix_to_euler(rotation_matrix, order)

    # Add noise to Euler angles
    noisy_euler = [
        angle + noise for angle, noise in zip(euler_angles, generate_action_noise(std_dev=std_dev))
    ]

    # Convert back to a rotation matrix
    noisy_rotation = R.from_euler(order, noisy_euler).as_matrix()
    return noisy_rotation

def stack_cubes(
    bullet_client, robot, gripper, urdf_paths, cube_positions, cube_sizes, cube_colors, env, dataset
):
    """Stacks cubes with data augmentation and saves the demonstration with action labels."""
    # Define the home pose for the robot, as the current pose, when the simulation starts
    home_pose = robot.get_eef_pose()

    # Load cubes with augmented positions (sizes remain unchanged)
    cube_ids = []
    for i, (position, size, urdf_path) in enumerate(zip(cube_positions, cube_sizes, urdf_paths)):
        cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        cube_ids.append(cube_id)
        env.add_object(cube_id, f"cube_{i}", size)  # Save cube size here

    for _ in range(100):
        bullet_client.stepSimulation()
        time.sleep(1 / 100)

    first_cube_position = None
    for i, cube_id in enumerate(cube_ids):
        position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
        cube_pose = Affine(position, quat)

        if first_cube_position is None:
            first_cube_position = position
            continue

        # Save pre-grasp action
        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_0pre_grasp.json", ACTIONS["move_to_pre_grasp"], i)

        # Pre-grasp position with noise
        gripper_rotation = Affine(rotation=[0, np.pi, 0])
        target_pose = cube_pose * gripper_rotation
        pre_grasp_offset = Affine(translation=[0, 0, -0.35])

        pre_grasp_pose = target_pose * pre_grasp_offset
        current_translation = pre_grasp_pose.translation
        current_translation += generate_action_noise(std_dev=0.01)  # Add noise to pre-grasp position
        current_rotation = apply_orientation_noise(pre_grasp_pose.rotation, std_dev=[0.004, 0.004, 0.1])  # Add noise (Z-axis emphasized)
        pre_grasp_pose = Affine(translation=current_translation, rotation=current_rotation)
        robot.ptp(pre_grasp_pose)
        gripper.open()
        env.set_gripper_state(True)


        # Move to grasp position
        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_1move_to_grasp.json", ACTIONS["move_to_grasp"], i)
        grasp_pose = target_pose
        current_translation = grasp_pose.translation
        current_translation[0] += generate_action_noise(std_dev=0.003)[0]  # Add noise to X
        current_translation[1] += generate_action_noise(std_dev=0.003)[1]  # Add noise to Y
        current_rotation = apply_orientation_noise(grasp_pose.rotation, std_dev=[0.004, 0.004, 0.1])  # Smaller noise for grasp
        grasp_pose = Affine(translation=current_translation, rotation=current_rotation)
        robot.lin(grasp_pose)
        gripper.close()
        env.set_gripper_state(False)

        # Lift the cube with noise
        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_2lift.json", ACTIONS["lift_cube"], i)
        lift_pose = target_pose * Affine(translation=[0, 0, -0.2])
        current_translation = lift_pose.translation
        current_translation += generate_action_noise(std_dev=0.01)  # Add noise to position
        current_rotation = apply_orientation_noise(lift_pose.rotation, std_dev=[0.004, 0.004, 0.1])  # Add noise to rotation
        lift_pose = Affine(translation=current_translation, rotation=current_rotation)
        robot.lin(lift_pose)

        # Move to stacking position dynamically
        current_cube_positions = env.get_cube_positions()
        stack_position = list(current_cube_positions['cube_0']['position'])
        # In case the cube falls off the table
        if stack_position[2] < 0:
            break   
        print("Stack position:", stack_position)
        stack_position[2] += 0.05 + cube_sizes[0] / 2
        stack_position[2] += sum(cube_sizes[1:i])  # Adjust for the cubes stacked already

        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_3move_to_stack.json", ACTIONS["move_to_stack_position"], i)
        stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])
        current_translation = stack_target.translation
        current_translation += generate_action_noise(std_dev=0.007)  # Add noise to position
        current_rotation = apply_orientation_noise(stack_target.rotation, std_dev=[0.004, 0.004, 0.1])  # Add noise (Z-axis emphasized)
        above_stack = Affine(translation=current_translation, rotation=current_rotation)
        above_stack = stack_target * Affine(translation=[0, 0, -0.25])
        robot.lin(above_stack)

        # Descend and stack cube
        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_4stack.json", ACTIONS["stack_cube"], i)
        current_translation = stack_target.translation
        current_translation[0] += generate_action_noise(std_dev=0.003)[0]  # Add noise to X
        current_translation[1] += generate_action_noise(std_dev=0.003)[1]  # Add noise to Y
        current_rotation = apply_orientation_noise(stack_target.rotation, std_dev=[0.004, 0.004, 0.1])  # Smaller noise for stack
        stack_target = Affine(translation=current_translation, rotation=current_rotation)
        robot.lin(stack_target)
        gripper.open()
        env.set_gripper_state(True)

        # Move above the stack to avoid collisions
        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_5above_stack.json", ACTIONS["above_stack"], i)
        robot.lin(above_stack)

        # Return to home
        env.save_demonstration(demo_folder, f"demo_{len(dataset)+288}_cube{i}_6home.json", ACTIONS["return_home"], i)
        current_translation = home_pose.translation
        current_translation += generate_action_noise(std_dev=0.01)  # Add noise to position
        current_rotation = apply_orientation_noise(home_pose.rotation, std_dev=[0.004, 0.004, 0.1])  # Add noise to rotation
        home_pose = Affine(translation=current_translation, rotation=current_rotation)
        robot.ptp(home_pose)

    # Check success criteria
    last_cube_id = cube_ids[-1]
    last_cube_position, _ = bullet_client.getBasePositionAndOrientation(last_cube_id)
    expected_height = first_cube_position[2] + sum(cube_sizes)
    tolerance = 0.08
    return abs(last_cube_position[2] - expected_height) <= tolerance


def main():
    RENDER = True
# Create a BulletClient and configure the visualizer
    bullet_client = BulletClient(connection_mode=p.GUI)
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    if not RENDER:
        bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# Create an empty dataset to store the demonstrations
    dataset = []

    # Loop to create and save demonstrations
    while True:
        bullet_client.resetSimulation()

        robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
        gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
        robot.home()
        
        # Create a BulletEnvironment instance
        env = BulletEnvironment(bullet_client, robot)

        # Generate positions, sizes, and colors for the cubes
        cube_positions = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(5)]
        cube_sizes = [0.08 - i * 0.01 for i in range(5)]
        cube_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 1 0 1", "1 0 1 1"]
        CUBE_URDF_PATHS = [f"/home/jovyan/workspace/src/cubes_urdf/cube{i}.urdf" for i in range(5)]

        success = stack_cubes(bullet_client, robot, gripper, CUBE_URDF_PATHS, cube_positions, cube_sizes, cube_colors, env, dataset)
        #Wait 5 seconds before starting a new scene, only for debugging
        #time.sleep(5)
        if success:
            # Save the scene to the dataset
            dataset.append((cube_positions, cube_sizes, cube_colors))
            print("Scene saved to dataset.")
        else:
            print("Stacking failed. Restarting scene.")

if __name__ == "__main__":
    main()
