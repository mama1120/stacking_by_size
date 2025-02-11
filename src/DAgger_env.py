import numpy as np
import pybullet as p
from transform import Affine
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

"""This is a helper file that contains the functions and classes used in the DAgger environment."""
# Function to test the model
def test_model(model, test_input, input_scaler, output_scaler_position,output_scaler_orientation ):
    """
    Test the trained model with a given input.

    Args:
    - model: The trained Keras model.
    - test_input: A numpy array of shape (1, X.shape[1]) representing a single test sample.

    Returns:
    - predicted_position: The predicted EEF position in the original range.
    - predicted_orientation: The predicted EEF orientation in the original range.
    - predicted_gripper_state: The predicted gripper state as a binary value.
    """
    try:
        # Scale input
        test_input_scaled = input_scaler.transform(np.array(test_input).reshape(1, -1))
        # Get prediction in scaled space
        predicted_output_scaled = model.predict(test_input_scaled)

        # Split predictions
        predicted_position_scaled = predicted_output_scaled[0, :3]
        predicted_orientation_scaled = predicted_output_scaled[0, 3:7]
        predicted_gripper_state = predicted_output_scaled[0, 7]

        # Inverse scale EEF position and orientation
        predicted_position = output_scaler_position.inverse_transform(predicted_position_scaled.reshape(1, -1))
        predicted_orientation = output_scaler_orientation.inverse_transform(predicted_orientation_scaled.reshape(1, -1))

        # Convert gripper state to binary (0 or 1)
        predicted_gripper_state = int(round(predicted_gripper_state))

        return predicted_position, predicted_orientation, predicted_gripper_state
    except Exception as e:
        raise ValueError(f"Error in test_model: {e}")
    
def test_model_bin(model, test_input, input_scaler, output_scaler_position,output_scaler_orientation ):


    test_input_scaled = input_scaler.transform(np.array(test_input).reshape(1, -1))
    predicted_position_orientation, predicted_gripper = model.predict(test_input_scaled)
    
    predicted_position = output_scaler_position.inverse_transform(predicted_position_orientation[:, :3])
    predicted_orientation = output_scaler_orientation.inverse_transform(predicted_position_orientation[:, 3:])
    predicted_gripper_state = int(round(predicted_gripper[0, 0]))

    return predicted_position, predicted_orientation, predicted_gripper_state

# Define the Bullet environment and its functions to interact with the simulation and get the current state
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

    def get_cube_sizes(self):
        """Get the sizes of all cubes."""
        return {obj["name"]: obj["size"] for obj in self.objects.values()}
    
    def check_cubes_on_table(self, table_height=0.0, threshold=0.03):
        """
        Check which cubes are still on the table and which are not.

        Args:
            table_height: The height of the table in the simulation (default: 0.0).
            threshold: Tolerance for determining if a cube is on the table (default: 0.01).

        Returns:
            on_table: List of object names that are still on the table.
            off_table: List of object names that are not on the table.
        """
        on_table = []
        off_table = []

        for object_id, obj in self.objects.items():
            # Get the position of the cube
            cube_position, _ = self.bullet_client.getBasePositionAndOrientation(object_id)
            cube_height = cube_position[2]  # Z-coordinate (height)

            # Check if the cube's height is within the threshold of the table height
            if abs(cube_height - table_height) <= threshold:
                on_table.append(obj["name"])
            else:
                off_table.append(obj["name"])

        return on_table, off_table


class Expert:
    def __init__(self):
        self.position = None
        self.orientation = None
        self.gripper_state = None
        self.linptp = None

    def expert_policy(self, sample_input):
        stacking_cube = int(sample_input[-1])  # Stacking cube index
        action = int(sample_input[-2])  # Action label
        cube_sizes = sample_input[42:47]

        # Extract the position and orientation of the stacking cube
        stacking_cube_arr_index = 7 + stacking_cube * 7
        stacking_cube_position = sample_input[stacking_cube_arr_index:stacking_cube_arr_index + 3]
        stacking_cube_orientation = sample_input[stacking_cube_arr_index + 3:stacking_cube_arr_index + 7]
        cube_pose = Affine(stacking_cube_position, stacking_cube_orientation)

        # Get the position of the base stack (largest cube) and add the height of the other cubes
        base_cube_position = sample_input[7:10]
        base_cube_position[2] += 0.05 + cube_sizes[0] / 2
        base_cube_position[2] += sum(cube_sizes[1:stacking_cube-1])

        match action:
            case 0:  # Move to pre-grasp position
                gripper_rotation = Affine(rotation=[0, np.pi, 0])
                target_pose = cube_pose * gripper_rotation
                pre_grasp_offset = Affine(translation=[0, 0, -0.35])
                pre_grasp_pose = target_pose * pre_grasp_offset
                self.position = pre_grasp_pose.translation
                self.orientation = R.from_matrix(pre_grasp_pose.rotation).as_quat()  # Convert to quaternion
                self.gripper_state = 1
                self.linptp = 0

            case 1:  # Move to grasp
                gripper_rotation = Affine(rotation=[0, np.pi, 0])
                target_pose = cube_pose * gripper_rotation
                self.position = target_pose.translation
                self.orientation = R.from_matrix(target_pose.rotation).as_quat()  # Convert to quaternion
                self.gripper_state = 0
                self.linptp = 1

            case 2:  # Lift cube
                gripper_rotation = Affine(rotation=[0, np.pi, 0])
                target_pose = cube_pose * gripper_rotation
                lift_pose = target_pose * Affine(translation=[0, 0, -0.2])
                self.position = lift_pose.translation
                self.orientation = R.from_matrix(lift_pose.rotation).as_quat()  # Convert to quaternion
                self.gripper_state = 0
                self.linptp = 1

            case 3:  # Move to stack position
                stack_position = list(base_cube_position)
                stack_position[2] += 0.05 + sample_input[-3] / 2  # Add base cube height
                stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])
                above_stack = stack_target * Affine(translation=[0, 0, -0.2])
                self.position = above_stack.translation
                self.orientation = R.from_matrix(above_stack.rotation).as_quat()  # Convert to quaternion
                self.gripper_state = 0
                self.linptp = 0

            case 4:  # Stack cube
                stack_position = list(base_cube_position)
                stack_position[2] += 0.05 + sample_input[-3] / 2  # Add base cube height
                stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])
                self.position = stack_target.translation
                self.orientation = R.from_matrix(stack_target.rotation).as_quat()  # Convert to quaternion
                self.gripper_state = 1
                self.linptp = 1

            case 5:  # Return home
                self.position = [0.6913298964500427, 0.1742745339870453,0.4565303921699524]
                self.orientation = [0.7071565638830705, -0.7070569645181409,-0.0001979143341199308, -6.255715782012504e-05]
                self.gripper_state = 0
                self.linptp = 0


        return self.position, self.orientation, self.gripper_state,  self.linptp


class Expert_position:
    def __init__(self):
        self.position = None
        self.gripper_state = None
        self.linptp = None

    def expert_policy(self, sample_input):
        stacking_cube = int(sample_input[-1])  # Stacking cube index
        action = int(sample_input[-2])  # Action label
        cube_sizes = sample_input[15:20]

        # Extract the position and orientation of the stacking cube
        stacking_cube_arr_index = stacking_cube * 3
        stacking_cube_position = sample_input[stacking_cube_arr_index:stacking_cube_arr_index + 3]
        home_orientation = [0.7071565638830705, -0.7070569645181409,-0.0001979143341199308, -6.255715782012504e-05]
        cube_pose = Affine(stacking_cube_position, home_orientation)

        # Get the position of the base stack (largest cube) and add the height of the other cubes
        base_cube_position = sample_input[0:3]
        base_cube_position[2] += 0.05 + cube_sizes[0] / 2
        base_cube_position[2] += sum(cube_sizes[1:stacking_cube-1])

        match action:
            case 0:  # Move to pre-grasp position
                target_pose = cube_pose 
                pre_grasp_offset = Affine(translation=[0, 0, -0.35])
                pre_grasp_pose = target_pose * pre_grasp_offset
                self.position = pre_grasp_pose.translation
                self.gripper_state = 1
                self.linptp = 0

            case 1:  # Move to grasp
                target_pose = cube_pose 
                self.position = target_pose.translation
                self.gripper_state = 0
                self.linptp = 1

            case 2:  # Lift cube
                target_pose = cube_pose
                lift_pose = target_pose * Affine(translation=[0, 0, -0.2])
                self.position = lift_pose.translation
                self.gripper_state = 0
                self.linptp = 1

            case 3:  # Move to stack position
                stack_position = list(base_cube_position)
                stack_position[2] += 0.05 + sample_input[-3] / 2  # Add base cube height
                stack_target = Affine(translation=stack_position)
                above_stack = stack_target * Affine(translation=[0, 0, 0.2])
                self.position = above_stack.translation
                self.gripper_state = 0
                self.linptp = 0

            case 4:  # Stack cube
                stack_position = list(base_cube_position)
                stack_position[2] += 0.05 + sample_input[-3] / 2  # Add base cube height
                stack_target = Affine(translation=stack_position)
                print("stack_target", stack_target)
                self.position = stack_target.translation
                self.gripper_state = 1
                self.linptp = 1

            case 5:  # Return home
                self.position = [0.6913298964500427, 0.1742745339870453,0.4565303921699524]
                self.gripper_state = 0
                self.linptp = 0


        return self.position, self.gripper_state,  self.linptp
    