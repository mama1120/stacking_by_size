import os
import json
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
import matplotlib.pyplot as plt
import random

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
                            "orientation": data["next_robot_state"]["eef_orientation"],
                            "gripper": int(data["next_gripper_state"])
                        })
                if sequence_data:
                    self.sequences.append(sequence_data)

    def extract_features(self, data):
        features = []
        features.extend(data["robot_state"]["eef_position"])
        features.extend(data["robot_state"]["eef_orientation"])
        for cube in data["cube_positions"].values():
            features.extend(cube["position"])
            features.extend(cube["orientation"])
        features.extend(data["cube_sizes"].values())
        features.append(data["action_label"])  # 1 Feature
        features.append(data["stacking_cube"])  # 1 Feature
        return features

    def get_data(self):
        inputs, positions, orientations, grippers = [], [], [], []
        for sequence in self.sequences:
            seq_inputs, seq_positions, seq_orientations, seq_grippers = [], [], [], []
            for step in sequence:
                seq_inputs.append(step["features"])
                seq_positions.append(step["position"])
                seq_orientations.append(step["orientation"])
                seq_grippers.append(step["gripper"])
            inputs.append(seq_inputs)
            positions.append(seq_positions)
            orientations.append(seq_orientations)
            grippers.append(seq_grippers)
        return (np.array(inputs), 
                np.array(positions), 
                np.array(orientations), 
                np.array(grippers))

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
    inputs, positions, orientations, grippers = [], [], [], []
    for sequence in sequences:
        for step in sequence:
            inputs.append(step["features"])
            positions.append(step["position"])
            orientations.append(step["orientation"])
            grippers.append(step["gripper"])

    inputs = np.array(inputs)
    positions = np.array(positions)
    orientations = np.array(orientations)
    grippers = np.array(grippers)

    return inputs, positions, orientations, grippers

# Load datasets
data_dirs = ["./noise_dataset", "./new_dataset"]
dataset = CubeStackDataset(data_dirs)

# Split the sequences
train_sequences, test_sequences = train_test_split_sequences(dataset.sequences, test_size=0.2, random_state=42)

# Prepare training and testing data
X_train, y_train_pos, y_train_ori, y_train_grip = prepare_sequences(train_sequences)
X_test, y_test_pos, y_test_ori, y_test_grip = prepare_sequences(test_sequences)

# Normalize features and labels
input_scaler = StandardScaler()
output_scaler_position = StandardScaler()
output_scaler_orientation = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

y_train_pos_scaled = output_scaler_position.fit_transform(y_train_pos)
y_test_pos_scaled = output_scaler_position.transform(y_test_pos)

y_train_ori_scaled = output_scaler_orientation.fit_transform(y_train_ori)
y_test_ori_scaled = output_scaler_orientation.transform(y_test_ori)

y_train_combined = np.concatenate([y_train_pos_scaled, y_train_ori_scaled, y_train_grip.reshape(-1, 1)], axis=1)
y_test_combined = np.concatenate([y_test_pos_scaled, y_test_ori_scaled, y_test_grip.reshape(-1, 1)], axis=1)

# Create results directory
def create_results_dir():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

results_dir = create_results_dir()

# Save scalers
dump(input_scaler, os.path.join(results_dir, 'input_scaler.pkl'))
dump(output_scaler_position, os.path.join(results_dir, 'output_scaler_position.pkl'))
dump(output_scaler_orientation, os.path.join(results_dir, 'output_scaler_orientation.pkl'))

# Grid Search for Feedforward model
architectures = [
    {"layers": [512, 256, 128, 64], "dropout": [0.2, 0.2, 0.2, 0.2]},
    {"layers": [128, 128, 128, 64], "dropout": [0.2, 0.3, 0.1, 0.1]},
    {"layers": [512, 256, 128, 128], "dropout": [0.2, 0.4, 0.2, 0.3]},
    {"layers": [256, 256, 128, 64], "dropout": [0.2, 0.2, 0.2, 0.2]},
    {"layers": [128, 128, 64], "dropout": [0.2, 0.3, 0.1]},
    {"layers": [256, 256, 128], "dropout": [0.1, 0.3, 0.2]}
]
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]

best_val_loss = float('inf')
best_model = None
best_architecture = None
all_results = []

for arch in architectures:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Testing architecture: {arch['layers']}, learning rate: {lr}, batch size: {batch_size}")

            # Build the model
            model = Sequential()
            for i, units in enumerate(arch['layers']):
                if i == 0:
                    model.add(Dense(units, activation='relu', input_shape=(X_train_scaled.shape[1],)))
                else:
                    model.add(Dense(units, activation='relu'))
                model.add(Dropout(arch['dropout'][i]))
            model.add(Dense(y_train_combined.shape[1], activation='linear'))

            model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

            # Train the model
            early_stopping = EarlyStopping(monitor='val_loss', patience=11, restore_best_weights=True)

            history = model.fit(
                X_train_scaled, y_train_combined,
                validation_data=(X_test_scaled, y_test_combined),
                epochs=1000,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )

            # Evaluate the model
            val_loss = min(history.history['val_loss'])
            all_results.append({
                'architecture': arch,
                'learning_rate': lr,
                'batch_size': batch_size,
                'val_loss': val_loss
            })

            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_architecture = {
                    'architecture': arch,
                    'learning_rate': lr,
                    'batch_size': batch_size
                }

# Save the best model
best_model_path = os.path.join(results_dir, 'best_model_feedforward_gridsearch.keras')
best_model.save(best_model_path)

# Save architecture and results
results_path = os.path.join(results_dir, 'gridsearch_results.json')
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=4)

best_architecture_path = os.path.join(results_dir, 'best_architecture.json')
with open(best_architecture_path, 'w') as f:
    json.dump(best_architecture, f, indent=4)

# Plot loss curve for the best model
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(results_dir, 'loss_curve.png')
plt.savefig(loss_plot_path)
plt.close()

print(f"Best model saved at: {best_model_path}")
print(f"Results saved at: {results_path}")
print(f"Best architecture saved at: {best_architecture_path}")
