import os
import json

# Function to combine future robot and gripper states to the last state of each demonstration

def update_demonstrations(demo_folder):
    # Get a sorted list of all demonstration files
    demo_files = sorted([f for f in os.listdir(demo_folder) if f.endswith('.json')])

    # Group files by demonstration
    demos = {}
    for demo_file in demo_files:
        demo_id = demo_file.split('_')[0]
        if demo_id not in demos:
            demos[demo_id] = []
        demos[demo_id].append(demo_file)

    # Process each demonstration
    for demo_id, files in demos.items():
        files.sort()  # Ensure files are sorted in the correct order

        for i in range(len(files) - 1):
            current_file = files[i]
            next_file = files[i + 1]

            # Load current and next state
            with open(os.path.join(demo_folder, current_file), 'r') as f:
                current_data = json.load(f)
            with open(os.path.join(demo_folder, next_file), 'r') as f:
                next_data = json.load(f)

            # Copy next robot state and gripper state to current data
            current_data['next_robot_state'] = next_data['robot_state']
            current_data['next_gripper_state'] = next_data['gripper_state']

            # Save updated current data
            with open(os.path.join(demo_folder, current_file), 'w') as f:
                json.dump(current_data, f, indent=4)

        # Handle the last file in the demonstration
        last_file = files[-1]
        first_file = files[0]

        # Load last and first state
        with open(os.path.join(demo_folder, last_file), 'r') as f:
            last_data = json.load(f)
        with open(os.path.join(demo_folder, first_file), 'r') as f:
            first_data = json.load(f)

        # Copy first robot state and gripper state to last data
        last_data['next_robot_state'] = first_data['robot_state']
        last_data['next_gripper_state'] = first_data['gripper_state']

        # Save updated last data
        with open(os.path.join(demo_folder, last_file), 'w') as f:
            json.dump(last_data, f, indent=4)

if __name__ == "__main__":
    demo_folder = "noise_dataset"  # Update this to your actual folder path
    update_demonstrations(demo_folder)