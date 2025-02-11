import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_preprocess import X_tensor, y_tensor  # Import preprocessed tensors

# 🔹 Define the LSTM Model.
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer maps from hidden_dim to output_dim

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim]
        # Instead of pooling across the batch dimension, we take the output of the final timestep.
        last_out = lstm_out[:, -1, :]  # Shape: [batch, hidden_dim]
        out = self.fc(last_out)        # Shape: [batch, output_dim]
        return out

# 🔹 Model Hyperparameters.
INPUT_DIM = X_tensor.shape[2]  # Number of features (e.g. 7)
HIDDEN_DIM = 32                # LSTM hidden layer size
OUTPUT_DIM = X_tensor.shape[2] # Predicting all features (7)
NUM_LAYERS = 2                 # Stacked LSTM layers

# 🔹 Initialize the model.
model = LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)

# 🔹 Define Loss & Optimizer.
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔹 Create DataLoader.
dataset = TensorDataset(X_tensor, y_tensor)

# Split the dataset into training and testing sets.
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 🔹 Training Loop.
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        optimizer.zero_grad()            # Reset gradients
        output = model(batch_X)          # Forward pass
        loss = criterion(output, batch_y)  # Compute loss
        loss.backward()                  # Backpropagation

        # 🔹 Apply gradient clipping to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                 # Update weights
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

print("Training complete!")

# 🔹 Evaluate the model on the test set.
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        test_loss += loss.item()
test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.6f}")

# 🔹 Function to predict the next timestep based on the last seen sequence.
def predict_next(model, last_sequence):
    model.eval()
    with torch.no_grad():
        # last_sequence should be a 1D array/tensor of features.
        # Reshape to [1, 1, num_features] (batch size 1, sequence length 1).
        last_seq_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        predicted = model(last_seq_tensor)
    return predicted.numpy()

# Take the last sequence from the preprocessed X_tensor (squeezing the seq_len dimension).
last_seq = X_tensor[-1].squeeze(0)  # Shape: [num_features]
predicted_next = predict_next(model, last_seq)
print("Predicted Next % Change (normalized):", predicted_next)

# 🔹 Save the trained model.
torch.save(model.state_dict(), 'lstm_model.pth')
print("Model has been saved to 'lstm_model.pth'.")
