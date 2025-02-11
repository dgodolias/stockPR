import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# 🔹 Load the dataset from CSV
data = pd.read_csv('percentage_changes.csv').values

# 🔹 Reverse the order of the rows to start from the most recent
data = data[::-1]

# 🔹 Create input (X) and output (y) sequences
X, y = [], []
for i in range(len(data) - 1):
    X.append(data[i])  # Input: current timestep
    y.append(data[i + 1])  # Output: next timestep

X, y = np.array(X), np.array(y)

# 🔹 Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


print(f"Input shape: {X_tensor.shape}")  # Expected: (num_samples, num_features)
print(f"Output shape: {y_tensor.shape}")  # Expected: (num_samples, num_features)
