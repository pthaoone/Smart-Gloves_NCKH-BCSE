
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Cấu hình


PROJECT_ROOT = Path(__file__).resolve().parent.parent

X_PATH = PROJECT_ROOT / "outputs" / "X_windows.npy"
Y_PATH = PROJECT_ROOT / "outputs" / "y_windows.npy"

EPOCHS = 300
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# Chỉ sử dụng một tập nhỏ để kiểm tra overfit
MAX_SAMPLES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# 2. Đọc dữ liệu

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X ban đầu:", X.shape)
print("y ban đầu:", y.shape)


# Chỉ lấy một số window đầu tiên
X = X[:MAX_SAMPLES]
y = y[:MAX_SAMPLES]

# Chuyển sang Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print("X dùng để train:", X.shape)
print("y dùng để train:", y.shape)
print("Labels:", y.tolist())

# 3. Tạo DataLoader


dataset = TensorDataset(X, y)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 4. Xây dựng LSTM model


class BaselineLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # output shape:
        # (batch_size, sequence_length, hidden_size)
        output, (hidden, cell) = self.lstm(x)

        # Lấy output tại bước thời gian cuối cùng
        last_output = output[:, -1, :]

        # Phân loại
        logits = self.classifier(last_output)

        return logits


# 5. Khởi tạo model


input_size = X.shape[2]       # 69 features
hidden_size = 32
num_layers = 1
num_classes = int(y.max().item()) + 1

model = BaselineLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes
).to(DEVICE)

print("\nModel:")
print(model)

# 6. Loss và Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 7. Training

print("\n=== TRAINING ===")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # Forward
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tính loss
        total_loss += loss.item()

        # Tính accuracy
        predictions = outputs.argmax(dim=1)

        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    accuracy = correct / total
    average_loss = total_loss / len(dataloader)

    if (epoch + 1) % 10 == 0 or epoch == 0:

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

# 8. Kiểm tra kết quả

model.eval()

with torch.no_grad():

    X_test = X.to(DEVICE)
    y_test = y.to(DEVICE)

    outputs = model(X_test)

    predictions = outputs.argmax(dim=1)

    accuracy = (predictions == y_test).float().mean()

print("\n=== FINAL RESULT ===")
print("Labels thật:", y_test.cpu().tolist())
print("Labels dự đoán:", predictions.cpu().tolist())
print(f"Accuracy cuối: {accuracy.item() * 100:.2f}%")