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
from DAgger_env import Expert, BulletEnvironment, test_model_bin, test_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input



# Load the trained model
training_cycles= 0
training_step = 0
dataset = []


# Load the saved scalers
input_scaler = load("input_scaler.pkl")
output_scaler_path_pos = "../../../assets/models/position_orientation/output_scaler_position.pkl"
output_scaler_position = load(output_scaler_path_pos)
output_scaler_path_ori = "../../../assets/models/position_orientation/output_scaler_orientation.pkl"
output_scaler_orientation = load(output_scaler_path_ori)

def calculate_difference(predicted, expert):
    """Calculate the difference between the predicted and expert outputs."""
    position_diff = np.linalg.norm(np.array(predicted[:3]) - np.array(expert[:3]))
    orientation_diff = np.linalg.norm(np.array(predicted[3:7]) - np.array(expert[3:7]))
    gripper_diff = abs(predicted[7] - expert[7])  # Gripper state (0 or 1)
    return position_diff, orientation_diff, gripper_diff


def train_online(model, dataset, input_scaler, output_scaler_position, output_scaler_orientation, epochs=2):
    """Train the model online using the aggregated dataset. This is meant to be used with the regression only model"""
    global training_cycles 
    
    if len(dataset) == 0:
        return model  # Skip training if no data is collected yet
    
    # Convert dataset to numpy arrays
    inputs, expert_outputs = zip(*dataset)
    inputs = np.array(inputs)
    expert_outputs = np.array(expert_outputs)

    # Normalize inputs using the input scaler
    inputs = input_scaler.transform(inputs)
    print("Shape of inputs:", inputs.shape)

    # Split expert outputs into position, orientation, and gripper state
    expert_positions = expert_outputs[:, :3]
    expert_orientations = expert_outputs[:, 3:7]
    expert_grippers = expert_outputs[:, 7:]

    # Normalize outputs using the corresponding scalers
    expert_positions = output_scaler_position.transform(expert_positions)
    expert_orientations = output_scaler_orientation.transform(expert_orientations)

    y_combined = np.concatenate([expert_positions, expert_orientations, expert_grippers.reshape(-1, 1)], axis=1)
    print("Shape of y_combined:", y_combined.shape)


    # Train the model on the new dataset
    model.compile(optimizer=Adam(learning_rate=0.00025), loss='mse', metrics=['mae'])
    model.fit(inputs, y_combined, epochs=epochs, verbose=1)
    print("Model trained online with new data.")
    print("Model evaluation:", model.evaluate(inputs, y_combined))
    model.save("DAgger_ori.keras")            #break
    training_cycles += 1
    time.sleep(2)


    return model

def train_online_bin(model, dataset, input_scaler, output_scaler_position, output_scaler_orientation, epochs=2):
    """Train the model online using the aggregated dataset.This is meant to be used with the regression and binary model"""

    global training_cycles # Keep track of the number of training cycles

    if len(dataset) == 0:
        return model  # Skip training if no data is collected yet
    
    # Convert dataset to numpy arrays
    inputs, expert_outputs = zip(*dataset)
    inputs = np.array(inputs)
    expert_outputs = np.array(expert_outputs)

    # Normalize inputs using the input scaler
    inputs = input_scaler.transform(inputs)
    print("Shape of inputs:", inputs.shape)

    # Split expert outputs into position, orientation, and gripper state
    expert_positions = expert_outputs[:, :3]
    expert_orientations = expert_outputs[:, 3:7]
    expert_grippers = expert_outputs[:, 7:]

    # Normalize outputs using the corresponding scalers
    expert_positions = output_scaler_position.transform(expert_positions)
    expert_orientations = output_scaler_orientation.transform(expert_orientations)

    y_position_orientation = np.concatenate([expert_positions, expert_orientations], axis=1)
    y_gripper = expert_grippers.reshape(-1, 1)  # Gripper state (reshape to match model input)

    print("Shape of y_position_orientation:", y_position_orientation.shape)
    print("Shape of y_gripper:", y_gripper.shape)

    # Train the model on the new dataset
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss={'regression_output': 'mse', 'classification_output': 'binary_crossentropy'},
                  metrics={'regression_output': 'mae', 'classification_output': 'accuracy'})

    model.fit(inputs, 
              {'regression_output': y_position_orientation, 'classification_output': y_gripper},
              epochs=epochs,
              verbose=1)

    print("Model trained online with new data.")
    print("Model evaluation:", model.evaluate(inputs, 
                                               {'regression_output': y_position_orientation, 'classification_output': y_gripper}))

    # Save the updated model
    model.save("DAgger_ori.keras")
    training_cycles += 1
    time.sleep(3)
    print("Model saved as DAgger_ori.keras.")
    print("Training cycles:", training_cycles)

    return model

def stack_cubes(bullet_client, robot, gripper, urdf_path, cube_positions, cube_sizes, cube_colors, env, model):
    """Stacks cubes and trains online using DAgger."""
    cube_ids = []
    global training_step
    global dataset

    # Create expert instance
    expert = Expert()
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
            
            print("Second largest cube: ", second_largest_cube)

            j = int(second_largest_cube.split('_')[-1])
            cube_id = cube_ids[j]
            expert_usage = 0
            for i in range(7):  # Loop through all actions
                # Get state features
                position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
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

                # Prepare input for model prediction
                sample_input = [
                    eef_pose.translation[0], eef_pose.translation[1], eef_pose.translation[2],
                    eef_pose.quat[0], eef_pose.quat[1], eef_pose.quat[2], eef_pose.quat[3],
                ] + [
                    item
                    for k in range(5)
                    for item in (
                        cube_positions[f'cube_{k}']['position'][0],
                        cube_positions[f'cube_{k}']['position'][1],
                        cube_positions[f'cube_{k}']['position'][2],
                        cube_positions[f'cube_{k}']['orientation'][0],
                        cube_positions[f'cube_{k}']['orientation'][1],
                        cube_positions[f'cube_{k}']['orientation'][2],
                        cube_positions[f'cube_{k}']['orientation'][3],
                    )
                ] + [0.08, 0.07, 0.06, 0.05, 0.04] + [i, j]

                predicted_position, predicted_orientation, predicted_gripper = test_model(model, sample_input, input_scaler, output_scaler_position, output_scaler_orientation)
                predicted_position = np.squeeze(predicted_position)
                predicted_orientation = np.squeeze(predicted_orientation)
                pred_output = np.concatenate([predicted_position, predicted_orientation, [predicted_gripper]])

                # Get expert action
                expert_position, expert_orientation, expert_gripper, linptp = expert.expert_policy(sample_input)
                expert_output = np.concatenate([expert_position, expert_orientation, [expert_gripper]])

                # Decide whether to use expert or model
                position_diff, orientation_diff, gripper_diff = calculate_difference(
                    np.concatenate([predicted_position, predicted_orientation, [predicted_gripper]]),
                    expert_output
                )

                pos_threshold = 0.1    #Choose a very high value to not use the expert policy
                                        #Choose a very low value to always use the expert policy
                                        #Default 0.05
                print("Position diff:", position_diff)
                print("Orientation diff:", orientation_diff)
                #If a cube falls of the table, the position difference will be very large, so we can use this as a threshold
                if position_diff > 15:
                    return False
                
                use_expert = position_diff > pos_threshold and position_diff < 1 
                if use_expert:
                    target_pose = Affine(expert_position, expert_orientation)
                    dataset.append((sample_input, expert_output))
                    expert_usage += 1
                    print("Using expert policy.")
                else:
                    target_pose = Affine(predicted_position, predicted_orientation)
                    dataset.append((sample_input, pred_output))

                # Move the robot
                if linptp:
                    robot.lin(target_pose)
                else:
                    robot.ptp(target_pose)

                # Operate gripper
                if use_expert:
                    if expert_gripper == 0:
                        gripper.close()
                    else:
                        gripper.open()
                else:
                    if predicted_gripper == 0:
                        gripper.close()
                    else:
                        gripper.open()

                time.sleep(0)
                print("Current Training Step: ", training_step)
                print("Current Training cycle: ", training_cycles)

            # Check stacking success
            position, _ = bullet_client.getBasePositionAndOrientation(cube_id)
            expected_height = sum([cube_sizes[f'cube_{i}'] for i in range(j)]) + cube_sizes[f'cube_{j}'] / 2
            tolerance = 0.02
            if abs(position[2] - expected_height) <= tolerance:
                stacking_success = True
                print(f"Cube {j} stacked successfully.")
                print(f"Expert used {expert_usage} times.")
                expert_usage = 0
                #Only expand the dataset if the cube 2 or higher is stacked correctly
                #Otherwise the model will only learn how to stack the first cube
                if j >= 2:
                    training_step += 1
                    print("Size of Dataset = ", len(dataset))
                #Only train the model after the 8th training step, so the model has enough data to learn from
                if training_step >= 8 and j >= 2:
                    model = train_online(model, dataset, input_scaler, output_scaler_position, output_scaler_orientation)
                    dataset = [] 
                    training_step = 0
                    return True
                    
            else:
                current_cube_attempt += 1
                print(f"Cube {j} not stacked correctly. Attempt {current_cube_attempt}")
                #If the cube is not stacked correctly, delete the last 6 appended data from the dataset
                for _ in range(7):
                    if dataset:
                     dataset.pop()
                if current_cube_attempt >= 2:
                    print(f"Failed to stack cube {j} after {current_cube_attempt} attempts.")
                    return False
    return True


def main():
    RENDER = True
# Create a BulletClient and configure the visualizer
    bullet_client = BulletClient(connection_mode=p.GUI)
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    if not RENDER:
        bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# Create an empty dataset to store the demonstrations
    
    attempts = 0
    # Loop to create and save demonstrations
    while True:
        attempts += 1
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

        if os.path.exists("DAgger_ori.keras"):
            model = load_model("DAgger_ori.keras")
        else:
            model = load_model("base_model.keras")
            print("First Model loaded")

        success = stack_cubes(bullet_client, robot, gripper, CUBE_URDF_PATHS, cube_positions, cube_sizes, cube_colors, env, model)
        #Wait 5 seconds before starting a new scene, only for debugging
        #time.sleep(5)
        if success:
            print("Stacking successful. After attempts: ", attempts)
            attempts = 0
            #time.sleep(2)
            # Fine-tune model after each cube stacking attempt
        else:
            print("Stacking failed. Restarting scene.")
            print("Attempts: ", attempts)
            #if(attempts > 6):
                #break
if __name__ == "__main__":
    main()
