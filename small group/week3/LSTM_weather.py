# -*- coding: gbk -*-

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

filepath = "C:/Users/32673/Desktop/工作室作业/小组/week3/weather_data.csv"
batch_size = 64
test_size=0.2
seq_length = 24

# Load the data
df = pd.read_csv(filepath)

# datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df.set_index('Date/Time', inplace=True)

# 又没说要预测多少个那就选最简单的温度
features = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']
target = 'Temp_C'


# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df[features])

X, y = create_sequences(scaled_data, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train[:, features.index(target)])
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test[:, features.index(target)])



class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)  # output shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

# chao参数
input_size = len(features)
hidden_size = 64
num_layers = 2
output_size = 1  # Temp_C
learning_rate = 0.001
num_epochs = 50

# Initialize model
model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        test_loss += criterion(outputs.squeeze(), batch_y).item()

    print(f'Test Loss: {test_loss / len(test_loader):.4f}')

    predictions = []
    actuals = []
    for batch_X, batch_y in test_loader:
        preds = model(batch_X)
        predictions.extend(preds.squeeze().tolist())
        actuals.extend(batch_y.tolist())


import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
plt.plot(actuals[:100], label='Actual Temperature')
plt.plot(predictions[:100], label='Predicted Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (normalized)')
plt.title('Temperature Prediction')
plt.legend()
plt.show()