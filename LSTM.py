import torch.nn as nn
from data_preprocess import *
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


# 🔹 Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer

    def forward(self, x):
        h_out, _ = self.lstm(x)  # LSTM output
        print(f"h_out shape:", h_out.shape)
        print(f"h_out:", h_out)
        if torch.isnan(h_out).any():
            raise ValueError("NaN encountered in h_out")
            
        # Apply global max pooling across the batch dimension (for each hidden feature)
        gmp_pooled, _ = torch.max(h_out, dim=0)
        print(f"gmp_pooled shape:", gmp_pooled.shape)
        print(f"gmp_pooled:", gmp_pooled)
        if torch.isnan(gmp_pooled).any():
            raise ValueError("NaN encountered in gmp_pooled")
            
        out = self.fc(gmp_pooled)  # Use pooled output
        print(f"out shape:", out.shape)
        if torch.isnan(out).any():
            raise ValueError("NaN encountered in fc output")
            
        return out

# 🔹 Model Hyperparameters
INPUT_DIM = X.shape[1]  # Number of features (7)
print("Input Dim:",INPUT_DIM)
HIDDEN_DIM = 32  # LSTM hidden layer size
OUTPUT_DIM = X.shape[1]  # Predicting all 7 features
print("Output Dim:",OUTPUT_DIM)
NUM_LAYERS = 2  # Stacked LSTMs

# 🔹 Initialize Model
model = LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
# 🔹 Define Loss & Optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



print(X_tensor)
print(y_tensor)

# 🔹 Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)


# 🔹 Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])


train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# 🔹 Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in train_dataloader:
        optimizer.zero_grad()  # Reset gradients
        print(f"Batch X shape:",batch_X.shape)
        print(f"Batch X:",batch_X)
        output = model(batch_X)  # Forward pass
        print(f"Output:",output)
        loss = criterion(output, batch_y)  # Compute loss
        print(f"Loss:",loss)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    if epoch % 1 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.6f}")

print("Training complete!")

# 🔹 Evaluate the model on the test set
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        test_loss += loss.item()

test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.6f}")

# 🔹 Predict the next timestep based on the last seen sequence
def predict_next(model, last_sequence):
    model.eval()
    with torch.no_grad():
        last_seq_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        predicted = model(last_seq_tensor).numpy()
    return predicted

# Take the last SEQ_LENGTH known values as input
last_seq = X[-1]  # Last seen sequence
predicted_next = predict_next(model, last_seq)

print("Predicted Next % Change:", predicted_next)

# 🔹 Save the trained model
torch.save(model.state_dict(), 'lstm_model.pth')
print("Model has been saved to 'lstm_model.pth'.")