import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_preprocess import X_tensor, y_tensor  # Import preprocessed tensors
import numpy as np

# Remove the extra dimension to treat each row as one timestep
X_raw = X_tensor.squeeze(1)  # New shape: [num_samples, num_features]

window = 1  # Number of timesteps in the input sequence
new_X = []
new_y = []
for i in range(len(X_raw) - window):
    # Get window sequence: shape: [window, num_features]
    window_seq = X_raw[i : i + window]
    new_X.append(window_seq)
    
    # For target, check if next close price is greater than 0
    next_close = X_raw[i + window, 3]  # Next timestep's close price (index 3)
    new_y.append(next_close)

new_X = torch.stack(new_X)  # Final shape: [new_num_samples, window, num_features]
new_y = torch.tensor(new_y).unsqueeze(1)  # Final shape: [new_num_samples, 1]

# Normalize new_X
new_X_mean = new_X.mean(dim=0, keepdim=True)
new_X_std = new_X.std(dim=0, keepdim=True)
new_X = (new_X - new_X_mean) / new_X_std

# Normalize new_y to be between 0 and 1
new_y_min = new_y.min()
new_y_max = new_y.max()
new_y = (new_y - new_y_min) / (new_y_max - new_y_min)

#print 3 rows of X and y
print(f"X: {new_X[:3]}")
print(f"y: {new_y[:3]}")

# Create dataset and dataloaders
dataset = TensorDataset(new_X, new_y)

# Split the dataset into training and testing sets.
train_size = int(0.8 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

# Use your desired batch size.
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Maps hidden state to output (high price)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim]
        # Use the output from the final time step.
        last_out = lstm_out[:, -1, :]  # Shape: [batch, hidden_dim]
        out = self.fc(last_out)        # Shape: [batch, output_dim]
        out = self.sigmoid(out)        # Apply sigmoid activation
        return out

# Model hyperparameters.
INPUT_DIM = new_X.shape[2]  # Number of features (e.g., 7)
HIDDEN_DIM = 20            # LSTM hidden layer size
OUTPUT_DIM = 1             # Predict binary outcome
NUM_LAYERS = 2             # Stacked LSTM layers

# Initialize the model.
model = LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)

# Define Loss & Optimizer.
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training Loop
EPOCHS = 10
epsilon = 1e-8  # Small value to avoid division by zero
new_y_min_np = new_y_min.numpy()
new_y_max_np = new_y_max.numpy()
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch_X, batch_y in train_dataloader:
        optimizer.zero_grad()            # Reset gradients
        output = model(batch_X)          # Forward pass
        loss = criterion(output, batch_y)  # Compute loss
        loss.backward()                  # Backpropagation
        optimizer.step()                 # Update weights
        epoch_loss += loss.item()
        
        # Compute accuracy
        pred_y = output.detach().numpy()
        real_y = batch_y.detach().numpy()

        
        # Denormalize pred_y and real_y
        denorm_pred_y = pred_y * (new_y_max_np - new_y_min_np) + new_y_min_np
        denorm_real_y = real_y * (new_y_max_np - new_y_min_np) + new_y_min_np
        
        percentage_diff = np.abs((denorm_pred_y - denorm_real_y) / (denorm_real_y + epsilon)) * 100
        correct_predictions += (percentage_diff <= 0.2).sum()
        total_predictions += batch_y.size(0)
        
    avg_loss = epoch_loss / len(train_dataloader)
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}, Accuracy: {accuracy*100:.2f}%")

print("Training complete!")

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        test_loss += loss.item()
        
        pred_y = output.numpy()
        real_y = batch_y.numpy()
        percentage_diff = np.abs((pred_y - real_y) / (real_y + epsilon)) * 100
        test_correct += (percentage_diff <= 10).sum()
        test_total += batch_y.size(0)

        # Denormalize pred_y and real_y
        denorm_pred_y = pred_y * (new_y_max_np - new_y_min_np) + new_y_min_np
        denorm_real_y = real_y * (new_y_max_np - new_y_min_np) + new_y_min_np
        print(f"Denormalized pred_y: {denorm_pred_y}")
        print(f"Denormalized real_y: {denorm_real_y}")

test_loss /= len(test_dataloader)
test_accuracy = test_correct / test_total
print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy*100:.2f}%")

# Function to predict the next direction
def predict_direction(model, last_sequence):
    model.eval()
    with torch.no_grad():
        last_seq_tensor = last_sequence.unsqueeze(0)
        prob = model(last_seq_tensor)
        prediction = 1 if prob.item() >= 0.5 else 0
        return prediction, prob.item()

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')
print("Model saved as 'lstm_model.pth'.")