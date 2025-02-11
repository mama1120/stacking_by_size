import time
import os
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine
from tensorflow.keras.models import load_model
import tensorflow as tf
from joblib import load
from DAgger_env import BulletEnvironment

# Load the saved scalers
input_scaler_path = "../assets/models/position/input_scaler_position.pkl"
input_scaler = load(input_scaler_path)
output_scaler_path = "../assets/models/position/output_scaler_position.pkl"
output_scaler_position = load(output_scaler_path)

# Load the saved model
model_path = "../assets/models/position/position_model.keras"
model = tf.keras.models.load_model(model_path)

# Function to test the model
def test_model_pos(model, test_input, input_scaler, output_scaler_position):
    """
    Test the trained model with a given input.
    """
    # Reshape input to 2D array
    test_input_scaled = input_scaler.transform(np.array(test_input).reshape(1, -1))
    
    # Get prediction in scaled space
    predicted_output_scaled = model.predict(test_input_scaled)

    # Split predictions
    predicted_position_scaled = predicted_output_scaled[0, :3]
    predicted_gripper_state = predicted_output_scaled[0, 3]

    # Inverse scale EEF position
    predicted_position = output_scaler_position.inverse_transform(predicted_position_scaled.reshape(1, -1))
    
    # Convert gripper state to binary (0 or 1)
    predicted_gripper_state = int(round(predicted_gripper_state))

    return predicted_position, predicted_gripper_state

def test_model_pos_bin(model, test_input, input_scaler, output_scaler_position):


    test_input_scaled = input_scaler.transform(np.array(test_input).reshape(1, -1))
    predicted_position_orientation, predicted_gripper = model.predict(test_input_scaled)
    
    predicted_position = output_scaler_position.inverse_transform(predicted_position_orientation[:, :3])
    predicted_gripper_state = int(round(predicted_gripper[0, 0]))

    return predicted_position, predicted_gripper_state

def stack_cubes(bullet_client, robot, gripper, urdf_path, cube_positions, cube_sizes, cube_colors, env, model):
    """Stacks cubes ."""
    cube_ids = []
    global iterations

    Top_down_pose = robot.get_eef_pose()
    top_down_orientation = Top_down_pose.quat


    # Load cubes
    for i, (position, size, urdf_path) in enumerate(zip(cube_positions, cube_sizes, urdf_path)):
        cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        cube_ids.append(cube_id)
        env.add_object(cube_id, f"cube_{i}", size)

    for _ in range(100):
        bullet_client.stepSimulation()
        time.sleep(1 / 100)

    for _ in range(len(cube_ids) - 1):
        current_cube_attempt = 0
        stacking_success = False

        while not stacking_success:
            cubes_on_table, _ = env.check_cubes_on_table()
            cube_sizes = env.get_cube_sizes()

            cube_sizes_on_table = {cube: cube_sizes[cube] for cube in cubes_on_table}
            largest_cube = max(cube_sizes_on_table, key=cube_sizes_on_table.get)
            second_largest_cube = max(
                (cube for cube in cube_sizes_on_table if cube != largest_cube),
                key=cube_sizes_on_table.get,
                default=None
            )

            if not second_largest_cube:
                print("No second-largest cube found. Stopping.")
                return False

            j = int(second_largest_cube.split('_')[-1])
            cube_id = cube_ids[j]

            for i in range(7):
                print(f"Iteration: {iterations}")

                position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
                #In case the cube falls off the table
                cube_pose = Affine(position, quat)

                eef_pose = robot.get_eef_pose()
                cube_positions = env.get_cube_positions()
                cube_sizes = env.get_cube_sizes()

                #Check if a cube fell off the table. if so --> Stop the loop. Otherwise the simulation will drag
                for cube_key, position_data in cube_positions.items():
                    z_position = position_data['position'][2]  # Get the z-axis value
                    if z_position < 0:
                        print(f"{cube_key} has fallen off the table! Z position: {z_position}")
                        time.sleep(1)
                        return False  # Signal to stop the loop

                sample_input= [
                    item
                    for k in range(5)
                    for item in (
                        cube_positions[f'cube_{k}']['position'][0],
                        cube_positions[f'cube_{k}']['position'][1],
                        cube_positions[f'cube_{k}']['position'][2],
                    )
                ] + [0.08, 0.07, 0.06, 0.05, 0.04] + [i, j]


                predicted_position, predicted_gripper = test_model_pos(
                    model, sample_input, input_scaler, output_scaler_position
                )
                predicted_position = np.squeeze(predicted_position)

                
                #target_pose = Affine(predicted_position, predicted_orientation)
                target_pose = Affine(predicted_position, top_down_orientation)

                if i == 6:
                    target_pose = Top_down_pose

                if i == 0 or 6:
                    robot.ptp(target_pose)
                else:
                    robot.lin(target_pose)

                if predicted_gripper == 0:
                    gripper.close()
                else:
                    gripper.open()
                

                if i == 1:
                    gripper_pose = robot.get_eef_pose()
                    distance_to_cube = np.linalg.norm(
                        np.array(position) - np.array(gripper_pose.translation)
                    )
                    cube_gripped_successfully = distance_to_cube < 0.015  # Define a tolerance for gripping
   


            position, _ = bullet_client.getBasePositionAndOrientation(cube_id)
            expected_height = sum([cube_sizes[f'cube_{i}'] for i in range(j)]) + cube_sizes[f'cube_{j}'] / 2
            tolerance = 0.02
            if abs(position[2] - expected_height) <= tolerance:
                stacking_success = True
                print(f"Cube {j} stacked successfully.")
            else:
                current_cube_attempt += 1
                print(f"Cube {j} not stacked correctly. Attempt {current_cube_attempt}")

            if current_cube_attempt >= 3:
                print(f"Failed to stack cube {j} after {current_cube_attempt} attempts.")
                return False
    return True

iterations = 0

def main():
    RENDER = True
    bullet_client = BulletClient(connection_mode=p.GUI)
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    if not RENDER:
        bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    attempts = 0
    global iterations
    while iterations < 10:
        attempts += 1
        iterations += 1

        bullet_client.resetSimulation()

        robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
        gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
        robot.home()

        env = BulletEnvironment(bullet_client, robot)

        cube_positions = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(5)]
        cube_sizes = [0.08 - i * 0.01 for i in range(5)]
        cube_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 1 0 1", "1 0 1 1"]
        CUBE_URDF_PATHS = [f"/home/jovyan/workspace/src/cubes_urdf/cube{i}.urdf" for i in range(5)]

        success = stack_cubes(bullet_client, robot, gripper, CUBE_URDF_PATHS, cube_positions, cube_sizes, cube_colors, env, model)
        if success:
            print("Stacking successful.")
            attempts = 0
            time.sleep(1)
            break
        else:
            print("Stacking failed.")
            print(f"Attempts: {attempts}")
            attempts = 0


if __name__ == "__main__":
    main()
