import os
import json
import numpy as np
import random
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from joblib import dump, load
import matplotlib.pyplot as plt


# Define CubeStackDataset
class CubeStackDataset:
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.sequences = []  # Store sequences
        self.load_data()

    def load_data(self):
        for data_dir in self.data_dirs:
            files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            sequences = {}
            for file in files:
                if "_cube" in file:
                    prefix = file.split("_cube")[0]
                else:
                    print(f"Skipping file {file} due to invalid naming convention.")
                    continue

                if prefix not in sequences:
                    sequences[prefix] = []
                sequences[prefix].append(file)
            print(f"Found {len(sequences)} sequences in {data_dir}")

            for prefix, sequence_files in sequences.items():
                sequence_data = []
                for file in sorted(sequence_files):
                    filepath = os.path.join(data_dir, file)
                    with open(filepath, 'r') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as e:
                            print(f"Skipping file {file} due to JSON error: {e}")
                            continue
                        sequence_data.append({
                            "features": self.extract_features(data),
                            "position": data["next_robot_state"]["eef_position"],
                            "gripper": int(data["next_gripper_state"])
                        })
                if sequence_data:
                    self.sequences.append(sequence_data)

    def extract_features(self, data):
        features = []
        for cube in data["cube_positions"].values():
            features.extend(cube["position"])
        features.extend(data["cube_sizes"].values())
        features.append(data["action_label"])  # 1 Feature
        features.append(data["stacking_cube"])  # 1 Feature
        return features

    def get_data(self):
        inputs, positions, grippers = [], [], []
        for sequence in self.sequences:
            seq_inputs, seq_positions, seq_grippers = [], [], []
            for step in sequence:
                seq_inputs.append(step["features"])
                seq_positions.append(step["position"])
                seq_grippers.append(step["gripper"])
            inputs.append(seq_inputs)
            positions.append(seq_positions)
            grippers.append(seq_grippers)
        return np.array(inputs), np.array(positions), np.array(grippers)


# Split sequences into training and testing sets
def train_test_split_sequences(sequences, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    random.shuffle(sequences)
    split_idx = int(len(sequences) * (1 - test_size))

    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]

    return train_sequences, test_sequences


# Prepare sequences
def prepare_sequences(sequences):
    inputs, positions, grippers = [], [], []
    for sequence in sequences:
        for step in sequence:
            inputs.append(step["features"])
            positions.append(step["position"])
            grippers.append(step["gripper"])

    return np.array(inputs), np.array(positions), np.array(grippers)


# Load datasets
data_dirs = ["./noise_dataset", "./new_dataset"]
dataset = CubeStackDataset(data_dirs)

# Split the sequences
train_sequences, test_sequences = train_test_split_sequences(dataset.sequences, test_size=0.2, random_state=42)
print(f"Found {len(train_sequences)} training sequences and {len(test_sequences)} testing sequences")

# Prepare training and testing data
X_train, y_train_pos, y_train_grip = prepare_sequences(train_sequences)
X_test, y_test_pos, y_test_grip = prepare_sequences(test_sequences)

# Normalize features and labels
input_scaler = StandardScaler()
output_scaler_position = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

y_train_pos_scaled = output_scaler_position.fit_transform(y_train_pos)
y_test_pos_scaled = output_scaler_position.transform(y_test_pos)

y_train_combined = np.concatenate([y_train_pos_scaled, y_train_grip.reshape(-1, 1)], axis=1)
y_test_combined = np.concatenate([y_test_pos_scaled, y_test_grip.reshape(-1, 1)], axis=1)

dump(input_scaler, "input_scaler_position.pkl")
dump(output_scaler_position, "output_scaler_position.pkl")
print(f"Input shape is {X_train.shape[1]} and Output shape is {y_train_combined.shape[1]}")
# Define neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(4, activation='linear')  # 3 for EEF position + 1 for gripper state
])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0018), loss='mse', metrics=['mae'])

# Define learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 38:
        return lr
    elif epoch < 50:
        rate = lr * 0.8
        return rate
    else:
        return lr * 0.5

# Train the model
history = model.fit(X_train_scaled, y_train_combined, validation_data=(X_test_scaled, y_test_combined), 
                    epochs=45, batch_size=64, 
                    callbacks=[early_stopping, LearningRateScheduler(lr_scheduler)])

# Save the trained model
model.save("position_model.keras")

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test_scaled, y_test_combined)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Function to test the model
def test_model(model, test_input, input_scaler_path, output_scaler_position_path):
    """
    Test the trained model with a given input.
    """
    input_scaler = load(input_scaler_path)
    output_scaler_position = load(output_scaler_position_path)

    test_input_scaled = input_scaler.transform(test_input)
    predicted_output = model.predict(test_input_scaled)
    # Split predictions
    predicted_position_scaled = predicted_output[0, :3]
    predicted_gripper_state = predicted_output[0, 3]

    # Inverse scale EEF position and orientation
    predicted_position = output_scaler_position.inverse_transform(predicted_position_scaled.reshape(1, -1))
    
    # Convert gripper state to binary (0 or 1)
    predicted_gripper_state = int(round(predicted_gripper_state))

    return predicted_position, predicted_gripper_state

# Test the model
test_input = X_test_scaled[0].reshape(1, -1)
print(f"Test Input: {test_input}")
predicted_position, predicted_gripper_state = test_model(model, test_input, "input_scaler_position.pkl", "output_scaler_position.pkl")
print(f"Predicted Position: {predicted_position}, Predicted Gripper State: {predicted_gripper_state}")
