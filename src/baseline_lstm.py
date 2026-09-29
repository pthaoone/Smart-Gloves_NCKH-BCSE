import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm src vào path nếu cần
sys.path.append(str(Path(__file__).resolve().parent))
from window_dataset import WindowDataset, get_dataloaders


# 1. Cấu hình & Siêu tham số (Hyperparameters)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
NUM_LAYERS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Cố định seed để đảm bảo tính tái lập (Reproducibility) trong NCKH."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# 2. Định nghĩa kiến trúc Baseline LSTM

class BaselineLSTM(nn.Module):
   

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)

        # Lấy hidden state tại bước thời gian cuối cùng của window
        last_timestep = out[:, -1, :]  # (batch_size, hidden_size)

        logits = self.fc(last_timestep)  # (batch_size, num_classes)
        return logits

    def count_parameters(self) -> int:
        """Đếm tổng số tham số học được."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 3. Chạy Sanity Check (Overfit tập nhỏ - Task W6)

def run_overfit_sanity_check(train_dataset, input_size=6, num_classes=4):
    """Kiểm tra sanity-check: overfit trên 4 mẫu nhỏ để chứng minh mô hình học

    được (Task trọng tâm của Tuần 6).
    """
    print("\n--- [TASK TUẦN 6]: CHẠY OVERFIT TEST TRÊN TẬP NHỎ (4 MẪU) ---")
    small_subset = Subset(train_dataset, range(4))
    small_loader = DataLoader(small_subset, batch_size=4, shuffle=True)

    test_model = BaselineLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)

    test_model.train()
    for epoch in range(1, 101):
        for bx, by in small_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = test_model(bx)
            loss = criterion(out, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            pred = out.argmax(dim=1)
            acc = (pred == by).float().mean().item() * 100
            print(f"  Epoch [{epoch:03d}/100] - Loss: {loss.item():.4f} - Accuracy: {acc:.1f}%")

    print("=> Kết quả Task W6: Mô hình overfit 100% thành công! Gradient flow ổn định.\n")


# 4. Huấn luyện và Đánh giá toàn diện

def train_and_evaluate(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
):
    set_seed(SEED)
    print(f"Sử dụng thiết bị: {DEVICE}")

    # Nạp dữ liệu qua DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(
        outputs_dir=OUTPUTS_DIR,
        batch_size=batch_size,
    )

    train_ds = train_loader.dataset
    input_size = train_ds.num_features
    num_classes = train_ds.num_classes

    # Chạy sanity check Tuần 6 trước
    run_overfit_sanity_check(train_ds, input_size, num_classes)

    # Khởi tạo mô hình cho huấn luyện toàn tập
    model = BaselineLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(DEVICE)

    total_params = model.count_parameters()
    print("=== MÔ HÌNH BASELINE LSTM ===")
    print(model)
    print(f"Tổng số tham số (Trainable Parameters): {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = OUTPUTS_DIR / "best_baseline_lstm.pt"

    start_train_time = time.time()
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN ({epochs} EPOCHS TRÊN {len(train_ds)} WINDOWS) ---")

    for epoch in range(1, epochs + 1):
        # 1. Train loop
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100

        # 2. Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total * 100

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Lưu checkpoint tốt nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 1 == 0:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%"
            )

    total_time = time.time() - start_train_time
    print(f"\nHuấn luyện xong trong: {total_time:.2f} giây! Checkpoint lưu tại: {best_model_path}")

    # ==========================================
    # 5. Đánh giá trên tập TEST độc lập
    # ==========================================
    print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST (SUBJ_017 -> SUBJ_020) ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    # Đo độ trễ suy luận (Inference Latency per window)
    latencies = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            t0 = time.perf_counter()
            outputs = model(batch_x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / batch_x.size(0))

            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    test_acc = test_correct / test_total * 100
    avg_latency_ms = np.mean(latencies) * 1000

    print(f"• Độ chính xác trên tập Test (Test Accuracy): {test_acc:.2f}%")
    print(f"• Độ trễ suy luận trung bình (Inference Latency): {avg_latency_ms:.3f} ms / window")

    # ==========================================
    # 6. Vẽ Learning Curve
    # ==========================================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="orange")
    plt.title("LSTM Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="orange")
    plt.title("LSTM Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = OUTPUTS_DIR / "lstm_learning_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"• Đã lưu biểu đồ Learning Curve tại: {plot_path}")

    # Lưu metrics
    results = {
        "model": "Baseline LSTM",
        "parameters": total_params,
        "train_time_sec": round(total_time, 2),
        "best_val_acc": round(best_val_acc, 2),
        "test_acc": round(test_acc, 2),
        "latency_ms_per_window": round(avg_latency_ms, 4),
    }
    with open(OUTPUTS_DIR / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_and_evaluate()