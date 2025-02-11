import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Load the dataset from CSV, ignoring the first label row.
df = pd.read_csv('percentage_changes.csv', header=0)

# Drop the 'Date' and 'Symbol' columns if they exist
df = df.drop(columns=['Date', 'Symbol'], errors='ignore')

# Reverse the order of the rows (start from bottom to top).
df = df.iloc[::-1]

# Convert all columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Replace infinity values with NaN and then fill NaN values with 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Normalize the data using StandardScaler.
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df)

# Create input (X) and output (y) sequences.
# Each row is a timestep: X is the current timestep and y is the next timestep.
X, y = [], []
for i in range(len(data_normalized) - 1):
    X.append(data_normalized[i])
    y.append(data_normalized[i + 1])

X, y = np.array(X), np.array(y)

# Convert to PyTorch tensors.
# For an LSTM with batch_first=True, inputs must be 3D: [num_samples, seq_len, num_features].
# Here we use a sequence length of 1.
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Shape: [num_samples, 1, num_features]
y_tensor = torch.tensor(y, dtype=torch.float32)               # Shape: [num_samples, num_features]

print(f"Input shape: {X_tensor.shape}")   # Expected: (num_samples, 1, num_features)
print(f"Output shape: {y_tensor.shape}")    # Expected: (num_samples, num_features)

# Optionally, you might want to save the scaler to use during inference.
# For example:
# import pickle
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)