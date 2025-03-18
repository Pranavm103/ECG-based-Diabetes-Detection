import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


dataset_path = "ecg_dataset_1000.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: Dataset '{dataset_path}' not found!")

df = pd.read_csv(dataset_path)


df.columns = df.columns.str.strip().str.lower()
if "label (diabetes)" in df.columns:
    df.rename(columns={"label (diabetes)": "label"}, inplace=True)

if "label" not in df.columns:
    raise ValueError("Error: 'label' column not found in dataset!")


X = df.drop(columns=["label"]).values
y = df["label"].astype(int).values  # Ensure labels are integers


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch, seq_len, features)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#  LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])  # Take output from last time step
        return self.sigmoid(x)


input_size = X_train.shape[1]
model = LSTMModel(input_size)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 50
batch_size = 32

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


torch.save(model.state_dict(), "lstm_model.pth")
print("✅ Model training complete. Saved as 'lstm_model.pth'.")
